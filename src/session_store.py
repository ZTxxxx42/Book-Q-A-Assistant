"""会话持久化存储（JSON 文件）。

为无状态的 ``/chat`` 增加跨刷新的会话历史：每个会话一个 JSON 文件，
外加一个扁平索引文件 ``index.json`` 供列表查询，避免读全部会话文件。

存储布局（``settings.session_dir`` 下）::

    sessions/
      <session_id>.json   # 单会话完整内容（含 messages）
      index.json          # [{id, user_id, title, book, mode, created_at, updated_at}, ...]

会话按 ``user_id`` 归属（前端 localStorage 生成的持久 UUID，无登录体系）。
所有写操作经模块级 ``asyncio.Lock`` 串行化，保证索引读-改-写的原子性。
日志走 ``book_kg.session`` 命名空间，由 ``logging_config`` 统一落 ``logs/app.log``。
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import settings

logger = logging.getLogger("book_kg.session")

# 索引读-改-写串行化锁（单进程 asyncio 足够）。
_lock = asyncio.Lock()


def _now_iso() -> str:
    """当前 UTC 时间 ISO8601 字符串（带时区）。"""
    return datetime.now(timezone.utc).isoformat()


def _session_path(session_id: str) -> Path:
    return settings.session_dir / f"{session_id}.json"


def _index_path() -> Path:
    return settings.session_dir / "index.json"


def _read_index() -> list[dict]:
    """读索引文件；不存在/损坏返回空列表。"""
    p = _index_path()
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        logger.exception("index.json 解析失败，回退为空索引")
        return []


def _write_index(index: list[dict]) -> None:
    """原子写索引：先写临时文件再 replace，避免并发/中断产生半截文件。"""
    _ensure_dir()
    p = _index_path()
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def _write_session(session: dict) -> None:
    _ensure_dir()
    p = _session_path(session["id"])
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def _ensure_dir() -> None:
    """防御性确保会话目录存在（即使启动时未调 ensure_dirs 也能写）。"""
    settings.session_dir.mkdir(parents=True, exist_ok=True)


def _summary(session: dict) -> dict:
    """从完整会话字典提取索引项（不含 messages）。"""
    return {
        "id": session["id"],
        "user_id": session["user_id"],
        "title": session["title"],
        "book": session["book"],
        "mode": session["mode"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
    }


async def create_session(user_id: str, book: str, mode: str, title: str | None = None) -> dict:
    """创建新会话，返回完整会话字典（含空 messages）。"""
    sid = uuid.uuid4().hex
    now = _now_iso()
    session = {
        "id": sid,
        "user_id": user_id,
        "title": title or "",
        "book": book,
        "mode": mode,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    async with _lock:
        _write_session(session)
        index = _read_index()
        index.append(_summary(session))
        _write_index(index)
    logger.info("session created: id=%s user=%s book=%s mode=%s", sid, user_id, book, mode)
    return session


async def list_sessions(user_id: str) -> list[dict]:
    """列出某用户的会话索引项，按 updated_at 降序。"""
    index = _read_index()
    items = [it for it in index if it.get("user_id") == user_id]
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return items


async def get_session(session_id: str, user_id: str) -> dict | None:
    """读单会话完整内容；不存在或归属不符返回 None。"""
    p = _session_path(session_id)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            session = json.load(f)
    except Exception:
        logger.exception("session 文件解析失败: id=%s", session_id)
        return None
    if session.get("user_id") != user_id:
        # 越权访问：对调用方表现为不存在。
        return None
    return session


async def delete_session(session_id: str, user_id: str) -> bool:
    """删除会话；不存在或归属不符返回 False。"""
    async with _lock:
        p = _session_path(session_id)
        if not p.exists():
            return False
        try:
            with p.open("r", encoding="utf-8") as f:
                session = json.load(f)
        except Exception:
            logger.exception("session 文件解析失败（删除前）: id=%s", session_id)
            return False
        if session.get("user_id") != user_id:
            return False
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        index = _read_index()
        index = [it for it in index if it.get("id") != session_id]
        _write_index(index)
    logger.info("session deleted: id=%s user=%s", session_id, user_id)
    return True


async def append_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    refs: list | None = None,
) -> dict | None:
    """向会话追加一条消息，更新 updated_at 与索引。

    首条 user 消息时用其内容前缀设置会话标题（若标题为空）。
    不存在或归属不符返回 None。
    """
    async with _lock:
        p = _session_path(session_id)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                session = json.load(f)
        except Exception:
            logger.exception("session 文件解析失败（追加前）: id=%s", session_id)
            return None
        if session.get("user_id") != user_id:
            return None

        now = _now_iso()
        session["messages"].append({
            "role": role,
            "content": content,
            "refs": refs or [],
            "created_at": now,
        })
        # 首条 user 消息设置标题
        if not session.get("title") and role == "user":
            session["title"] = content[: settings.session_title_length]
        session["updated_at"] = now
        _write_session(session)

        # 同步索引
        index = _read_index()
        summ = _summary(session)
        for i, it in enumerate(index):
            if it.get("id") == session_id:
                index[i] = summ
                break
        else:
            index.append(summ)
        _write_index(index)

    logger.info(
        "session message appended: id=%s user=%s role=%s chars=%d",
        session_id, user_id, role, len(content),
    )
    return session
