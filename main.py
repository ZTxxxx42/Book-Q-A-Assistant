"""CLI 入口：ingest / query / stats / cypher。"""
from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.table import Table

from config import settings
from src.graph_builder import ingest_book
from src.loader import resolve_book_path
from src.maintenance import (
    delete_document,
    delete_entity,
    delete_relation,
    edit_entity,
    edit_relation,
    list_documents,
    refresh_document,
)
from src.query import ask, cypher_query, graph_stats

console = Console()


@click.group()
def cli() -> None:
    """Book → Knowledge Graph (LightRAG + Neo4j)。"""


@cli.command()
@click.option("--file", "-f", required=True, help="书籍路径（data/books 下文件名或绝对路径）")
@click.option("--max-chunks", type=int, default=None, help="最多导入的块数（测试用）")
def ingest(file: str, max_chunks: int | None) -> None:
    """导入一本书，构建知识图谱。"""
    path = resolve_book_path(file)
    console.print(f"[cyan]加载书籍:[/cyan] {path}")
    info = asyncio.run(ingest_book(str(path), max_chunks=max_chunks))
    table = Table(title="导入结果")
    table.add_column("项")
    table.add_column("值", style="green")
    for k, v in info.items():
        table.add_row(k, str(v))
    console.print(table)


@cli.command()
@click.option("--question", "-q", required=True, help="问题")
@click.option(
    "--mode",
    type=click.Choice(["local", "global", "hybrid", "naive"]),
    default="hybrid",
    help="检索模式",
)
@click.option("--book", required=True, help="书籍文件名（= workspace，仅检索该书图谱）")
def query(question: str, mode: str, book: str) -> None:
    """对图谱提问（限定一本书）。"""
    console.print(f"[cyan]问题[/cyan] ({mode}, book={book}): {question}")
    result = asyncio.run(ask(question, book=book, mode=mode))  # type: ignore[arg-type]
    console.print("\n[bold green]回答:[/bold green]")
    console.print(result["answer"])
    refs = result.get("references", [])
    if refs:
        console.print("\n[dim]引用出处:[/dim]")
        for r in refs:
            console.print(f"  - {r.get('file_path', '?')} ({r.get('reference_id', '?')})")


@cli.command()
@click.option("--book", default=None, help="书籍文件名（= workspace）；省略则统计全部")
def stats(book: str | None) -> None:
    """查看图谱统计。"""
    info = graph_stats(book=book)
    title = f"知识图谱统计 ({book})" if book else "知识图谱统计（全部）"
    table = Table(title=title)
    table.add_column("项")
    table.add_column("值", style="green")
    table.add_row("节点总数", str(info["total_nodes"]))
    table.add_row("关系总数", str(info["total_relationships"]))
    for lbl, cnt in info["node_counts_by_label"].items():
        table.add_row(f"  · {lbl}", str(cnt))
    console.print(table)


@cli.command()
@click.option("--cypher", "-c", required=True, help="Cypher 语句")
def cypher(cypher: str) -> None:
    """直接执行 Cypher 查询（只读）。"""
    rows = cypher_query(cypher)
    if not rows:
        console.print("[yellow]无结果[/yellow]")
        return
    table = Table(title="Cypher 结果")
    keys = list(rows[0].keys())
    for k in keys:
        table.add_column(k)
    for r in rows[:100]:
        table.add_row(*[str(r[k]) for k in keys])
    console.print(table)
    console.print(f"[dim]共 {len(rows)} 条记录（最多显示 100 条）[/dim]")


@cli.group()
def documents() -> None:
    """文档级维护：列表 / 删除 / 刷新。"""


@documents.command(name="list")
def documents_list() -> None:
    """列出所有已导入文档。"""
    rows = asyncio.run(list_documents())
    if not rows:
        console.print("[yellow]无文档[/yellow]")
        return
    table = Table(title="已导入文档")
    for col in ["doc_id", "file_path", "status", "chunks", "length"]:
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["doc_id"][:16] + "...",
            r["file_path"],
            r["status"],
            str(r["chunks_count"]),
            str(r["content_length"]),
        )
    console.print(table)


@documents.command()
@click.argument("doc_id")
def delete(doc_id: str) -> None:
    """删除一整本书。"""
    res = asyncio.run(delete_document(doc_id))
    console.print(res)


@documents.command()
@click.argument("doc_id")
@click.option("--file", "-f", required=True, help="重新导入的书籍路径")
@click.option("--max-chunks", type=int, default=None)
def refresh(doc_id: str, file: str, max_chunks: int | None) -> None:
    """刷新一本书：删旧后重新导入。"""
    res = asyncio.run(refresh_document(doc_id, file, max_chunks=max_chunks))
    console.print(res)


@cli.group()
def entity() -> None:
    """实体级维护：编辑 / 删除。"""


@entity.command(name="edit")
@click.argument("name")
@click.option("--description", "-d", default=None, help="新描述")
@click.option("--type", "entity_type", default=None, help="新类型")
@click.option("--rename", default=None, help="重命名实体")
def entity_edit(name: str, description: str | None, entity_type: str | None, rename: str | None) -> None:
    """编辑实体（自动重算 embedding）。"""
    updated: dict = {}
    if description:
        updated["description"] = description
    if entity_type:
        updated["entity_type"] = entity_type
    if rename:
        updated["entity_name"] = rename
    if not updated:
        console.print("[yellow]未提供修改字段（--description/--type/--rename）[/yellow]")
        return
    res = asyncio.run(edit_entity(name, updated))
    console.print(res)


@entity.command(name="delete")
@click.argument("name")
def entity_delete(name: str) -> None:
    """删除单个实体（含其关系）。"""
    res = asyncio.run(delete_entity(name))
    console.print(res)


@cli.group()
def relation() -> None:
    """关系级维护：编辑 / 删除。"""


@relation.command(name="delete")
@click.option("--source", "-s", required=True)
@click.option("--target", "-t", required=True)
def relation_delete(source: str, target: str) -> None:
    """删除一条关系（保留两端实体）。"""
    res = asyncio.run(delete_relation(source, target))
    console.print(res)


@relation.command(name="edit")
@click.option("--source", "-s", required=True)
@click.option("--target", "-t", required=True)
@click.option("--description", "-d", default=None)
@click.option("--weight", type=float, default=None)
def relation_edit(source: str, target: str, description: str | None, weight: float | None) -> None:
    """编辑一条关系。"""
    updated: dict = {}
    if description:
        updated["description"] = description
    if weight is not None:
        updated["weight"] = weight
    if not updated:
        console.print("[yellow]未提供修改字段[/yellow]")
        return
    res = asyncio.run(edit_relation(source, target, updated))
    console.print(res)


if __name__ == "__main__":
    cli()
