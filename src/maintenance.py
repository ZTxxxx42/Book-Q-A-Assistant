"""知识图谱维护：文档/实体/关系的增删改查。

每本书 = 独立 workspace（= 文件 basename）：Neo4j label、Qdrant workspace_id、
KV 子目录都按 workspace 隔离。基于 LightRAG 1.5.x 维护 API（adelete_by_doc_id 等），
内部同步向量库与 Neo4j。
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from config import settings
from src.graph_builder import _build_rag
from src.loader import resolve_book_path


@asynccontextmanager
async def _rag(workspace: str) -> AsyncIterator[Any]:
    """从实例池取一个 LightRAG（只读 / 轻量写用）。

    用完归还池（不 finalize）。重操作（ingest / delete_document）用 _rag_exclusive。
    """
    from src.rag_pool import get_pool

    pool = await get_pool()
    rag = await pool.acquire(workspace)
    try:
        yield rag
    finally:
        await pool.release(workspace)


async def _rag_exclusive(workspace: str):
    """独占构建一个 LightRAG（重写操作用），调用方负责 finalize_storages。"""
    rag = _build_rag(workspace=workspace)
    await rag.initialize_storages()
    return rag


async def _invalidate_pool(workspace: str) -> None:
    """重写后使池中该 workspace 的实例失效（删除其可能过期的缓存实例）。"""
    from src.rag_pool import get_pool

    pool = await get_pool()
    await pool.invalidate(workspace)


# ---------- 文档级 ----------

def _doc_status_path(workspace: str) -> Path:
    return settings.working_dir / workspace / "kv_store_doc_status.json"


def list_documents() -> list[dict]:
    """列出所有已导入文档：遍历 working_dir 下各 workspace 子目录的 doc_status JSON。

    每书一个 workspace 子目录，直接读 doc_status JSON 避免 N 次 LightRAG 初始化。
    """
    out: list[dict] = []
    if not settings.working_dir.exists():
        return out
    for sub in sorted(settings.working_dir.iterdir()):
        if not sub.is_dir():
            continue
        ds = sub / "kv_store_doc_status.json"
        if not ds.exists():
            continue
        try:
            data = json.loads(ds.read_text(encoding="utf-8"))
        except Exception:
            continue
        for doc_id, d in data.items():
            status_val = d.get("status", "")
            if hasattr(status_val, "value"):
                status_val = status_val.value
            out.append({
                "doc_id": doc_id,
                "file_path": d.get("file_path") or sub.name,
                "workspace": sub.name,
                "status": str(status_val),
                "content_length": d.get("content_length", 0),
                "chunks_count": d.get("chunks_count", 0),
                "content_summary": (d.get("content_summary") or "")[:120],
                "created_at": d.get("created_at", 0),
            })
    out.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return out


async def get_document(book: str) -> dict | None:
    """取单本书文档详情（按 workspace）。"""
    from lightrag.base import DocStatus

    async with _rag(book) as rag:
        for st in DocStatus:
            try:
                docs = await rag.get_docs_by_status(st)
            except Exception:
                continue
            for doc_id, d in docs.items():
                status_val = d.status.value if hasattr(d.status, "value") else str(d.status)
                return {
                    "doc_id": doc_id,
                    "file_path": d.file_path,
                    "workspace": book,
                    "status": status_val,
                    "content_length": d.content_length,
                    "chunks_count": d.chunks_count,
                    "content_summary": d.content_summary,
                    "created_at": d.created_at,
                }
    return None


async def delete_document(book: str) -> dict:
    """删除一整本书（= 其 workspace）：清 Neo4j label 节点 + Qdrant workspace 向量 + KV 子目录。"""
    # 1) 读 doc_id
    ds_path = _doc_status_path(book)
    doc_id = None
    if ds_path.exists():
        try:
            data = json.loads(ds_path.read_text(encoding="utf-8"))
            doc_id = next(iter(data.keys()), None) if data else None
        except Exception:
            pass

    deleted_info: dict[str, Any] = {"file_path": book, "doc_id": doc_id}
    # 2) LightRAG 原生删除（同步 Qdrant + Neo4j + KV 索引）— 独占实例，避免与查询实例冲突
    if doc_id:
        rag = await _rag_exclusive(book)
        try:
            res = await rag.adelete_by_doc_id(doc_id)
            deleted_info["status"] = res.status
            deleted_info["message"] = res.message
        except Exception as e:
            deleted_info["status"] = "failed"
            deleted_info["message"] = str(e)
        finally:
            try:
                await rag.finalize_storages()
            except Exception:
                pass

    # 3) 兜底：直接按 Neo4j label 删节点（防 LightRAG 漏删）+ 删 KV 子目录
    from src.query import cypher_query

    try:
        label = book.replace("`", "``")
        cypher_query(f"MATCH (n:`{label}`) DETACH DELETE n")
    except Exception:
        pass
    # 删 KV 子目录
    sub = settings.working_dir / book
    if sub.exists() and sub.is_dir():
        import shutil

        shutil.rmtree(sub, ignore_errors=True)

    deleted_info.setdefault("status", "deleted")
    deleted_info.setdefault("message", f"workspace {book} removed")
    # 4) 使池中该 workspace 的实例失效（已删，避免复用过期实例）
    await _invalidate_pool(book)
    return deleted_info


async def refresh_document(file: str, max_chunks: int | None = None) -> dict:
    """刷新一本书：先删除其 workspace，再重新导入。"""
    from src.graph_builder import ingest_book

    basename = Path(file).name
    deleted = await delete_document(basename)
    path = resolve_book_path(file)
    info = await ingest_book(str(path), max_chunks=max_chunks)
    return {"deleted": deleted, "reingested": info}


async def upsert_document(file: str, max_chunks: int | None = None) -> dict:
    """按文件名 upsert：同 basename workspace 已存在则先删后导，不存在直接导入。"""
    from src.graph_builder import ingest_book

    basename = Path(file).name
    deleted = None
    if _doc_status_path(basename).exists():
        deleted = await delete_document(basename)
    path = resolve_book_path(file)
    info = await ingest_book(str(path), max_chunks=max_chunks)
    return {"deleted": deleted, "reingested": info, "file": basename}


# ---------- 实体级 ----------

async def edit_entity(
    entity_name: str,
    updated_data: dict[str, str],
    allow_rename: bool = True,
    allow_merge: bool = False,
    book: str | None = None,
) -> dict:
    """编辑实体（按 workspace）。"""
    async with _rag(book or "") as rag:
        await rag.aedit_entity(
            entity_name, updated_data, allow_rename=allow_rename, allow_merge=allow_merge
        )
    return {"entity": entity_name, "updated": True}


async def delete_entity(entity_name: str, book: str | None = None) -> dict:
    """删除实体（按 workspace）。"""
    async with _rag(book or "") as rag:
        await rag.adelete_entity(entity_name)
    return {"entity": entity_name, "deleted": True}


async def edit_relation(
    source: str, target: str, updated_data: dict[str, str], book: str | None = None
) -> dict:
    """编辑关系（按 workspace）。"""
    async with _rag(book or "") as rag:
        await rag.aedit_relation(source, target, updated_data)
    return {"relation": f"{source}->{target}", "updated": True}


async def delete_relation(source: str, target: str, book: str | None = None) -> dict:
    """删除关系（按 workspace）。"""
    async with _rag(book or "") as rag:
        await rag.adelete_relation(source, target)
    return {"relation": f"{source}->{target}", "deleted": True}
