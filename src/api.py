"""FastAPI 查询接口：通过 HTTP 操作知识图谱。"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.graph_builder import ingest_book
from src.graph_view import get_subgraph, get_top_entities
from src.maintenance import (
    delete_document,
    delete_entity,
    delete_relation,
    edit_entity,
    edit_relation,
    get_document,
    list_documents,
    refresh_document,
)
from src.query import ask, ask_stream, cypher_query, graph_stats

QueryMode = Literal["local", "global", "hybrid", "naive"]


# ---------- 请求 / 响应模型 ----------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="问题")
    mode: QueryMode = Field("hybrid", description="检索模式")
    stream: bool = Field(False, description="流式返回（暂不支持，预留）")
    book: str = Field(..., description="书籍文件名（= workspace，仅检索该书图谱）")


class QueryResponse(BaseModel):
    answer: str
    mode: QueryMode
    references: list = Field(default_factory=list, description="引用出处 [{reference_id, file_path}]")


class IngestRequest(BaseModel):
    file: str = Field(..., description="书籍路径（data/books 下文件名或绝对路径）")
    max_chunks: int | None = Field(None, description="最多导入块数（测试用）")


class IngestResponse(BaseModel):
    file: str
    total_chars: int
    chunks: int
    working_dir: str


class CypherRequest(BaseModel):
    cypher: str = Field(..., min_length=1, description="Cypher 语句（只读查询）")
    limit: int = Field(100, ge=1, le=1000, description="返回行数上限")


class StatsResponse(BaseModel):
    total_nodes: int
    total_relationships: int
    node_counts_by_label: dict[str, int]


# ---------- 后台任务追踪 ----------

_tasks: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Book → Knowledge Graph API",
    version="0.1.0",
    description="基于 LightRAG + Neo4j 的书籍知识图谱查询接口",
    lifespan=lifespan,
)


# ---------- 路由 ----------

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/")
async def index():
    """返回可视化页面。"""
    return FileResponse(STATIC_DIR / "index.html")


# 静态资源（如有额外 js/css）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """对图谱提问，返回答案 + 引用出处。"""
    try:
        result = await ask(req.question, book=req.book, mode=req.mode, stream=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败：{e}")
    return QueryResponse(
        answer=result["answer"], mode=req.mode, references=result["references"]
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: QueryMode = Field("hybrid")
    history: list[ChatMessage] = Field(default_factory=list, description="多轮对话历史")
    book: str = Field(..., description="书籍文件名（= workspace，仅检索该书图谱）")


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """流式问答（SSE）：逐 token 返回，支持多轮对话历史。

    事件格式：`data: {"type":"token","content":"..."}\\n\\n`
    结束：`data: {"type":"done"}\\n\\n`；出错：`data: {"type":"error","content":"..."}\\n\\n`
    """
    history = [{"role": m.role, "content": m.content} for m in req.history]

    async def event_stream():
        try:
            async for evt in ask_stream(req.question, book=req.book, mode=req.mode, history=history):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    """同步导入一本书（小书/测试用）。大书建议用 /ingest/async。"""
    from src.loader import resolve_book_path

    path = resolve_book_path(req.file)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"书籍不存在：{path}")
    try:
        info = await ingest_book(str(path), max_chunks=req.max_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导入失败：{e}")
    return IngestResponse(**info)


@app.post("/ingest/async")
async def ingest_async_endpoint(req: IngestRequest) -> dict:
    """异步导入：立即返回 task_id，后台执行，用 /tasks/{id} 查进度。"""
    from src.loader import resolve_book_path

    path = resolve_book_path(req.file)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"书籍不存在：{path}")

    task_id = uuid.uuid4().hex
    _tasks[task_id] = {"status": "running", "file": str(path), "result": None, "error": None}

    async def _run() -> None:
        try:
            info = await ingest_book(str(path), max_chunks=req.max_chunks)
            _tasks[task_id] = {"status": "done", "file": str(path), "result": info, "error": None}
        except Exception as e:
            _tasks[task_id] = {"status": "failed", "file": str(path), "result": None, "error": str(e)}

    asyncio.create_task(_run())
    return {"task_id": task_id, "status": "running"}


class UpsertRequest(BaseModel):
    file: str = Field(..., description="书籍路径（data/books 下文件名或绝对路径）")
    max_chunks: int | None = Field(None, description="最多导入块数（测试用）")


@app.post("/documents/upsert")
async def upsert_document_endpoint(req: UpsertRequest) -> dict:
    """按文件名 upsert：同 basename 已存在则先删后导，不存在则直接导入。"""
    from src.loader import resolve_book_path
    from src.maintenance import upsert_document

    path = resolve_book_path(req.file)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"书籍不存在：{path}")
    try:
        return await upsert_document(str(path), max_chunks=req.max_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upsert 失败：{e}")


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _tasks[task_id]


@app.get("/stats", response_model=StatsResponse)
async def stats_endpoint(book: str | None = None) -> StatsResponse:
    """查看图谱统计。``book`` 给定时仅统计该书 workspace。"""
    try:
        info = graph_stats(book=book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"统计失败（Neo4j 是否已启动？）：{e}")
    return StatsResponse(**info)


class GraphRequest(BaseModel):
    entity: str = Field(..., min_length=1, description="中心实体名称（模糊匹配）")
    book: str = Field(..., description="书籍文件名（= workspace）")
    depth: int = Field(2, ge=1, le=3, description="跳数：1=直接邻居，2=邻居的邻居")
    limit: int = Field(50, ge=1, le=500, description="返回子图规模上限")


class TopEntitiesRequest(BaseModel):
    book: str = Field(..., description="书籍文件名（= workspace）")
    limit: int = Field(40, ge=1, le=200, description="返回的热门实体数量")


@app.post("/graph")
async def graph_endpoint(req: GraphRequest) -> dict:
    """返回某实体周围的关系子图（nodes + edges），供前端可视化。"""
    try:
        sub = get_subgraph(req.entity, book=req.book, depth=req.depth, limit=req.limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"子图查询失败（Neo4j 是否已启动？）：{e}")
    return sub


@app.post("/graph/top")
async def graph_top_endpoint(req: TopEntitiesRequest) -> dict:
    """返回度数最高的实体，用于初始展示。"""
    try:
        return get_top_entities(book=req.book, limit=req.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败：{e}")


# ---------- 维护：文档 / 实体 / 关系 ----------

class RefreshRequest(BaseModel):
    file: str | None = Field(None, description="重新导入的书籍路径；省略则用 book 名")
    max_chunks: int | None = Field(None)


class EditEntityRequest(BaseModel):
    description: str | None = None
    entity_type: str | None = None
    entity_name: str | None = Field(None, description="新实体名（重命名）")
    allow_rename: bool = True
    allow_merge: bool = False


class EditRelationRequest(BaseModel):
    description: str | None = None
    weight: float | None = None
    keywords: str | None = None


@app.get("/documents")
async def documents_endpoint() -> list[dict]:
    """列出所有已导入的文档（各 workspace）。"""
    try:
        return list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文档列表失败：{e}")


@app.get("/documents/{book}")
async def document_detail_endpoint(book: str) -> dict:
    """取单本书文档详情（按 workspace）。"""
    d = await get_document(book)
    if not d:
        raise HTTPException(status_code=404, detail="文档不存在")
    return d


@app.delete("/documents/{book}")
async def delete_document_endpoint(book: str) -> dict:
    """删除一整本书（其 workspace：Neo4j label 节点 + Qdrant 向量 + KV 子目录）。"""
    try:
        return await delete_document(book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}")


@app.post("/documents/{book}/refresh")
async def refresh_document_endpoint(book: str, req: RefreshRequest) -> dict:
    """刷新一本书：删旧 workspace 后重新导入指定文件。"""
    try:
        return await refresh_document(req.file or book, max_chunks=req.max_chunks)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刷新失败：{e}")


@app.post("/entities/{entity_id}/edit")
async def edit_entity_endpoint(entity_id: str, req: EditEntityRequest, book: str) -> dict:
    """编辑实体（属性 / 重命名），自动重算 embedding 并同步向量库。"""
    updated = {k: v for k, v in req.model_dump().items() if v is not None}
    updated.pop("allow_rename", None)
    updated.pop("allow_merge", None)
    if not updated:
        raise HTTPException(status_code=400, detail="未提供任何修改字段")
    try:
        return await edit_entity(
            entity_id, updated,
            allow_rename=req.allow_rename, allow_merge=req.allow_merge, book=book,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"编辑失败：{e}")


@app.delete("/entities/{entity_id}")
async def delete_entity_endpoint(entity_id: str, book: str) -> dict:
    """删除单个实体（含其关系），同步清理向量库。"""
    try:
        return await delete_entity(entity_id, book=book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}")


@app.post("/relations/edit")
async def edit_relation_endpoint(
    source: str, target: str, req: EditRelationRequest, book: str,
) -> dict:
    """编辑一条关系。通过 query 参数 ?source=&target=&book= 指定。"""
    updated = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updated:
        raise HTTPException(status_code=400, detail="未提供任何修改字段")
    try:
        return await edit_relation(source, target, updated, book=book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"编辑失败：{e}")


@app.delete("/relations")
async def delete_relation_endpoint(source: str, target: str, book: str) -> dict:
    """删除一条关系（保留两端实体）。"""
    try:
        return await delete_relation(source, target, book=book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}")


@app.post("/cypher")
async def cypher_endpoint(req: CypherRequest) -> dict:
    """直接执行 Cypher 查询（只读）。"""
    if not req.cypher.strip().lower().startswith(("match", "return", "with", "call", "unwind")):
        raise HTTPException(status_code=400, detail="仅允许只读查询（MATCH/RETURN/WITH/CALL）")
    try:
        rows = cypher_query(req.cypher)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cypher 执行失败：{e}")
    return {"rows": rows[: req.limit], "total": len(rows)}
