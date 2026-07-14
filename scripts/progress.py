"""实时进度条：tail 导入日志，解析 chunk 抽取进度。

用法:
    python scripts/progress.py <日志文件路径>

若不传路径，自动查找 claude 任务目录下最新的 .output 文件。
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# 剥离 ANSI 颜色码
ANSI = re.compile(r"\x1b\[[0-9;]*m")
CHUNK_RE = re.compile(r"Chunk (\d+) of (\d+) extracted (\d+) Ent \+ (\d+) Rel")
ERR_RE = re.compile(r"ERROR|RateLimitError|Failed to extract")
PHASE_RE = re.compile(r"(Extracting|Merging|Completed processing|Phase \d)")


def strip_ansi(s: str) -> str:
    return ANSI.sub("", s)


def find_latest_log() -> Path | None:
    # 后台任务日志可能在两个位置，都搜
    bases = [
        Path(os.environ.get("TEMP", "")) / "claude",
        Path.home() / ".claude",
    ]
    candidates = []
    for base in bases:
        if not base.exists():
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".output"):
                    p = Path(root) / f
                    try:
                        candidates.append((p.stat().st_mtime, p))
                    except OSError:
                        pass
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def follow(thefile):
    """像 tail -f 一样迭代新增行。"""
    thefile.seek(0, 2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.3)
            continue
        yield line


def main() -> None:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_log()
    if not log_path or not log_path.exists():
        print("未找到日志文件。用法: python scripts/progress.py <日志文件路径>")
        sys.exit(1)

    print(f"监控: {log_path}")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        TextColumn("[green]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task("抽取进度", total=0)

    state = {
        "total": 0,
        "done": 0,
        "ent": 0,
        "rel": 0,
        "errs": 0,
        "phase": "等待中...",
        "last_line": "",
        "start": time.time(),
    }

    def render() -> Panel:
        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(style="cyan", no_wrap=True)
        tbl.add_column()
        tbl.add_row("已抽块数", f"[bold]{state['done']}[/bold] / {state['total'] or '?'}")
        tbl.add_row("累计实体", f"[green]{state['ent']}[/green]")
        tbl.add_row("累计关系", f"[green]{state['rel']}[/green]")
        tbl.add_row("错误数", f"[{'red' if state['errs'] else 'dim'}]{state['errs']}[/]")
        tbl.add_row("阶段", state["phase"])
        elapsed = int(time.time() - state["start"])
        tbl.add_row("已用时", f"{elapsed//60}m{elapsed%60}s")
        if state["done"] and state["total"]:
            eta = int(elapsed * (state["total"] - state["done"]) / max(state["done"], 1))
            tbl.add_row("预计剩余", f"{eta//60}m{eta%60}s")
        tbl.add_row("最新日志", state["last_line"][-70:])

        return Panel(
            Group(progress, tbl),
            title="[bold]Book → Knowledge Graph 导入进度[/bold]",
            border_style="blue",
        )

    with Live(render(), refresh_per_second=4, screen=False) as live:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            # 先读已有内容
            for line in f:
                line = strip_ansi(line.rstrip())
                if not line:
                    continue
                if "INFO:" in line:
                    state["last_line"] = line.split("INFO:", 1)[-1].strip()
                m = CHUNK_RE.search(line)
                if m:
                    n, total, ent, rel = map(int, m.groups())
                    state["total"] = total
                    state["done"] = max(state["done"], n)
                    state["ent"] += ent
                    state["rel"] += rel
                    progress.update(task_id, total=total, completed=n)
                if ERR_RE.search(line):
                    state["errs"] += 1
                pm = PHASE_RE.search(line)
                if pm:
                    state["phase"] = pm.group(1)
                if "Completed processing" in line:
                    state["phase"] = "✅ 完成"
                live.update(render())

            # 继续跟踪新内容
            for line in follow(f):
                line = strip_ansi(line.rstrip())
                if not line:
                    continue
                if "INFO:" in line:
                    state["last_line"] = line.split("INFO:", 1)[-1].strip()
                m = CHUNK_RE.search(line)
                if m:
                    n, total, ent, rel = map(int, m.groups())
                    state["total"] = total
                    state["done"] = max(state["done"], n)
                    state["ent"] += ent
                    state["rel"] += rel
                    progress.update(task_id, total=total, completed=n)
                if ERR_RE.search(line):
                    state["errs"] += 1
                pm = PHASE_RE.search(line)
                if pm:
                    state["phase"] = pm.group(1)
                if "Completed processing" in line:
                    state["phase"] = "✅ 完成"
                    state["done"] = state["total"] or state["done"]
                    progress.update(task_id, completed=state["done"])
                live.update(render())
                if "导入结果" in line or "Application startup" in line:
                    # 进程结束
                    live.update(render())
                    break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n停止监控（导入仍在后台继续）")
