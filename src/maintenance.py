"""知识图谱维护：文档/实体/关系的增删改查。

基于 LightRAG 1.5.x 的维护 API（aedit_entity / adelete_by_* 等），
内部会同步向量库（nano_vdb）与 Neo4j，无需手动重算 embedding。
"""
from __future__ import annotations

from typing import Any

from src.graph_builder import _build_rag
from src.loader import resolve_book_path


async def _rag():
    rag = _build_rag()
    await rag.initialize_storages()
    return rag


# ---------- 文档级 ----------

async def list_documents() -> list[dict]:
    """列出所有已导入的文档（含状态、块数、摘要）。"""
    rag = await _rag()
    from lightrag.base import DocStatus

    out: list[dict] = []
    for st in DocStatus:
        try:
            docs = await rag.get_docs_by_status(st)
        except Exception:
            continue
        if not docs:
            continue
        for doc_id, d in docs.items():
            status_val = d.status.value if hasattr(d.status, "value") else str(d.status)
            out.append({
                "doc_id": doc_id,
                "file_path": d.file_path,
                "status": status_val,
                "content_length": d.content_length,
                "chunks_count": d.chunks_count,
                "content_summary": (d.content_summary or "")[:120],
                "created_at": d.created_at,
                "updated_at": getattr(d, "updated_at", None),
                "error_msg": d.error_msg,
            })
    # 按创建时间倒序
    out.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return out


async def get_document(doc_id: str) -> dict | None:
    """取单个文档详情。"""
    rag = await _rag()
    docs = await rag.aget_docs_by_ids(doc_id)
    d = docs.get(doc_id)
    if not d:
        return None
    status_val = d.status.value if hasattr(d.status, "value") else str(d.status)
    return {
        "doc_id": doc_id,
        "file_path": d.file_path,
        "status": status_val,
        "content_length": d.content_length,
        "chunks_count": d.chunks_count,
        "content_summary": d.content_summary,
        "created_at": d.created_at,
        "error_msg": d.error_msg,
    }


async def delete_document(doc_id: str) -> dict:
    """删除一整本书：清理其 chunks / 实体 / 关系 / 向量（处理共享引用）。"""
    rag = await _rag()
    res = await rag.adelete_by_doc_id(doc_id)
    return {
        "status": res.status,
        "doc_id": res.doc_id,
        "message": res.message,
        "file_path": res.file_path,
    }


async def refresh_document(doc_id: str, file: str, max_chunks: int | None = None) -> dict:
    """刷新一本书：先删除该 doc_id，再重新导入指定文件。"""
    deleted = await delete_document(doc_id)
    from src.graph_builder import ingest_book

    path = resolve_book_path(file)
    info = await ingest_book(str(path), max_chunks=max_chunks)
    return {"deleted": deleted, "reingested": info}


async def upsert_document(file: str, max_chunks: int | None = None) -> dict:
    """按文件名 upsert：同 basename 已存在则先删后导，不存在则直接导入。

    LightRAG 的 filename dedup 会拦下同 basename 的重复 ingest，故同书内容变更
    必须先 delete 再 ingest。本函数封装该"按文件名 upsert"语义。
    """
    from pathlib import Path

    from src.graph_builder import ingest_book

    basename = Path(file).name
    # 查同 basename 的现有文档
    existing = None
    for d in await list_documents():
        if d.get("file_path") == basename:
            existing = d
            break

    deleted = None
    if existing:
        deleted = await delete_document(existing["doc_id"])

    path = resolve_book_path(file)
    info = await ingest_book(str(path), max_chunks=max_chunks)
    return {"deleted": deleted, "reingested": info, "file": basename}


# ---------- 实体级 ----------

async def edit_entity(
    entity_name: str,
    updated_data: dict[str, str],
    allow_rename: bool = True,
    allow_merge: bool = False,
) -> dict[str, Any]:
    """编辑实体。updated_data 可含:
    - description / entity_type: 修改属性（自动重算 embedding）
    - entity_name: 重命名（allow_rename=True 时生效）
    """
    rag = await _rag()
    return await rag.aedit_entity(
        entity_name, updated_data,
        allow_rename=allow_rename, allow_merge=allow_merge,
    )


async def delete_entity(entity_name: str) -> dict:
    """删除单个实体（含其关系），同步清理向量库。"""
    rag = await _rag()
    res = await rag.adelete_by_entity(entity_name)
    return {"status": res.status, "message": res.message, "entity": entity_name}


# ---------- 关系级 ----------

async def edit_relation(
    source_entity: str, target_entity: str, updated_data: dict[str, Any],
) -> dict[str, Any]:
    """编辑关系。updated_data 可含 description / weight / keywords 等。"""
    rag = await _rag()
    return await rag.aedit_relation(source_entity, target_entity, updated_data)


async def delete_relation(source_entity: str, target_entity: str) -> dict:
    """删除一条关系（保留两端实体）。"""
    rag = await _rag()
    res = await rag.adelete_by_relation(source_entity, target_entity)
    return {
        "status": res.status,
        "message": res.message,
        "source": source_entity,
        "target": target_entity,
    }
