"""FastAPI 查询接口：通过 HTTP 操作知识图谱。"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from config import get_config_snapshot, save_config_updates, settings
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
    upsert_document,
)
from src.query import ask, ask_stream, cypher_query, graph_stats
from src.rag_pool import get_pool, shutdown_pool
from src import session_store

logger = logging.getLogger("book_kg.api")
logging.basicConfig(level=logging.INFO)

QueryMode = Literal["local", "global", "hybrid", "naive"]

# book 文件名白名单：防路径穿越 / Cypher 注入（允许字母数字 _ - . 及中文）
_BOOK_NAME_RE = re.compile(r"^[A-Za-z0-9_\-\.一-龥]+$")


def _validate_book(v: str) -> str:
    if not v or len(v) > settings.max_book_name_length:
        raise ValueError(f"book 长度需在 1..{settings.max_book_name_length} 之间")
    if not _BOOK_NAME_RE.match(v):
        raise ValueError("book 含非法字符")
    if "/" in v or "\\" in v or ".." in v:
        raise ValueError("book 含路径分隔符")
    return v


# ---------- 请求 / 响应模型 ----------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=settings.max_question_length, description="问题")
    mode: QueryMode = Field("hybrid", description="检索模式")
    stream: bool = Field(False, description="流式返回（暂不支持，预留）")
    book: str = Field(..., description="书籍文件名（= workspace，仅检索该书图谱）")
    decompose: bool = Field(False, description="A3：用 GLM 拆子问题扩展查询（默认关）")

    @field_validator("book")
    @classmethod
    def _check_book(cls, v: str) -> str:
        return _validate_book(v)


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

# 限流（B2）：HTTP 入口并发上限 + ingest 并发上限
_query_concurrency: asyncio.Semaphore | None = None
_ingest_concurrency: asyncio.Semaphore | None = None
# B4：当前在飞查询数（用于排队可见化）
_query_in_flight: int = 0


def _query_sem() -> asyncio.Semaphore:
    global _query_concurrency
    if _query_concurrency is None:
        _query_concurrency = asyncio.Semaphore(settings.max_concurrent_requests)
    return _query_concurrency


def _ingest_sem() -> asyncio.Semaphore:
    global _ingest_concurrency
    if _ingest_concurrency is None:
        _ingest_concurrency = asyncio.Semaphore(settings.max_concurrent_ingest)
    return _ingest_concurrency


async def _try_acquire(sem: asyncio.Semaphore) -> bool:
    """非阻塞获取信号量：成功 True，已满 False（用于立即返回 503）。"""
    if sem._value <= 0:  # noqa: SLF001
        return False
    try:
        sem._value -= 1  # noqa: SLF001
        return True
    except Exception:
        return False


def _release(sem: asyncio.Semaphore) -> None:
    sem.release()


async def _with_heartbeat(agen, interval: float = 15.0):
    """包装异步生成器：两条事件之间若超过 interval 秒，插入一条 ping 心跳。

    防 SSE 长连接被中间代理/浏览器因空闲超时断流。

    实现要点：用后台任务消费生成器并推入 queue，主循环只对 queue.get() 计时。
    绝不 ``wait_for`` 生成器的 ``__anext__`` —— 超时会 cancel 掉正在 await 的生成器
    帧从而终止它（经典坑）。
    """
    import asyncio as _aio

    queue: _aio.Queue = _aio.Queue()
    _SENTINEL = object()

    async def _producer():
        try:
            async for evt in agen:
                await queue.put(evt)
        except Exception as e:
            await queue.put(e)
        finally:
            await queue.put(_SENTINEL)

    task = _aio.create_task(_producer())
    try:
        while True:
            try:
                item = await _aio.wait_for(queue.get(), timeout=interval)
            except _aio.TimeoutError:
                yield {"type": "ping"}
                continue
            if item is _SENTINEL:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except Exception:
                pass


def _ensure_book_exists(book: str) -> None:
    """book workspace 存在性预检：避免后续空检索/初始化浪费。不存在 → 404。"""
    from src.maintenance import _doc_status_path

    if not _doc_status_path(book).exists():
        raise HTTPException(status_code=404, detail=f"书籍不存在：{book}")


def _safe_detail(prefix: str, e: Exception) -> str:
    """对外脱敏的错误消息；完整异常写日志。"""
    logger.exception("%s: %r", prefix, e)
    return f"{prefix}（详情见服务端日志）"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动即确保所有工作目录存在（working_dir / data/books / log_dir / session_dir）。
    # 否则裸 uvicorn 启动时 session_dir 未创建，首个 POST /sessions 会 500。
    settings.ensure_dirs()
    # B1：全局并发闸 —— 让 LightRAG 的 max_async 跨实例真正生效（零改源码）
    try:
        from lightrag.kg.shared_storage import initialize_share_data

        initialize_share_data(
            workers=1,
            global_concurrency_limits={
                "llm:extract": settings.global_extract_concurrency,
                "llm:keyword": settings.global_keyword_concurrency,
                "llm:query": settings.global_query_concurrency,
                "embedding": settings.global_embedding_concurrency,
                "rerank": settings.global_rerank_concurrency,
            },
        )
        logger.info("全局并发闸已启用: %s", {
            "llm:extract": settings.global_extract_concurrency,
            "llm:keyword": settings.global_keyword_concurrency,
            "llm:query": settings.global_query_concurrency,
            "embedding": settings.global_embedding_concurrency,
            "rerank": settings.global_rerank_concurrency,
        })
    except Exception as e:
        logger.warning("initialize_share_data 失败（并发闸未启用）: %r", e)
    # 预热实例池
    await get_pool()
    # 后台 TTL 清理任务
    cleanup_task = asyncio.create_task(_task_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        await shutdown_pool()


async def _task_cleanup_loop() -> None:
    """定期清理过期的 done/failed 任务（TTL）。"""
    while True:
        try:
            await asyncio.sleep(300)
            now = time.time()
            expired = [
                tid for tid, t in _tasks.items()
                if t.get("status") in ("done", "failed")
                and t.get("finished_at")
                and now - t["finished_at"] > settings.task_ttl
            ]
            for tid in expired:
                _tasks.pop(tid, None)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass


app = FastAPI(
    title="Book → Knowledge Graph API",
    version="0.1.0",
    description="基于 LightRAG + Neo4j 的书籍知识图谱查询接口",
    lifespan=lifespan,
)

# A4：CORS（若配置了 CORS_ORIGINS）。通配源 "*" 与 credentials 不兼容（CORS 规范
# 禁止），故检测到 "*" 时自动关闭 credentials；其余情况按显式来源允许 credentials。
if settings.cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    _cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    _cors_allow_credentials = "*" not in _cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
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
    _ensure_book_exists(req.book)
    sem = _query_sem()
    if not await _try_acquire(sem):
        raise HTTPException(
            status_code=503, detail="服务繁忙，请稍后重试", headers={"Retry-After": "5"}
        )
    try:
        result = await asyncio.wait_for(
            ask(req.question, book=req.book, mode=req.mode, stream=False,
                history=None, decompose=req.decompose),
            timeout=settings.query_overall_timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="查询超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("查询失败", e))
    finally:
        _release(sem)
    return QueryResponse(
        answer=result["answer"], mode=req.mode, references=result["references"]
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=settings.max_question_length)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=settings.max_question_length)
    mode: QueryMode = Field("hybrid")
    history: list[ChatMessage] = Field(
        default_factory=list, max_length=settings.max_history_turns, description="多轮对话历史"
    )
    book: str = Field(..., description="书籍文件名（= workspace，仅检索该书图谱）")
    decompose: bool = Field(False, description="A3：用 GLM 拆子问题扩展查询（默认关）")
    # 会话归属：提供 session_id + user_id 时，本次问答自动持久化到该会话。
    session_id: str | None = Field(None, description="归属会话 ID；提供则持久化本次问答")
    user_id: str | None = Field(None, description="用户标识（session_id 非空时必填）")

    @field_validator("book")
    @classmethod
    def _check_book(cls, v: str) -> str:
        return _validate_book(v)


# ---------- 会话管理 ----------

class SessionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, description="用户标识")
    book: str = Field(..., description="书籍文件名（= workspace）")
    mode: QueryMode = Field("hybrid")
    title: str | None = Field(None, max_length=200, description="会话标题；省略则取首条问题前缀")

    @field_validator("book")
    @classmethod
    def _check_book(cls, v: str) -> str:
        return _validate_book(v)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """流式问答（SSE）：逐 token 返回，支持多轮对话历史。

    事件格式：`data: {"type":"token","content":"..."}\\n\\n`
    拒答：`data: {"type":"refuse","content":"..."}\\n\\n`
    结束：`data: {"type":"done"}\\n\\n`；出错：`data: {"type":"error","content":"..."}\\n\\n`
    """
    _ensure_book_exists(req.book)
    sem = _query_sem()
    history = [{"role": m.role, "content": m.content} for m in req.history]
    # 会话持久化：仅当 session_id + user_id 同时提供才启用
    persist = bool(req.session_id and req.user_id)
    if persist:
        logger.info(
            "chat start: session=%s user=%s book=%s mode=%s",
            req.session_id, req.user_id, req.book, req.mode,
        )

    async def event_stream():
        # 信号量在生成器内获取/释放（纳入下方 try/finally）：客户端在 StreamingResponse
        # 开始迭代前断开时生成器根本不会启动，此时既未获取 sem 也未计数，无泄漏。
        # 计数器增量与所有 yield 都落在 try 内，断连引发的 GeneratorExit 会先走 finally
        # 归零计数器并释放 sem，再向上传播 —— 修复原先队列 yield 在 try 外导致的泄漏。
        global _query_in_flight
        if not await _try_acquire(sem):
            yield f"data: {json.dumps({'type': 'error', 'content': '服务繁忙，请稍后重试'}, ensure_ascii=False)}\n\n"
            return
        # B4：排队可见化 —— 跟踪 in-flight 计数，告知前方还有多少请求
        _query_in_flight += 1
        ahead = _query_in_flight - 1
        # 累积答复与引用，供会话持久化
        answer = ""
        refs: list = []
        # 先持久化 user 消息（即使后续失败也保留提问痕迹，便于管理员排查）
        if persist:
            try:
                await session_store.append_message(
                    req.session_id, req.user_id, "user", req.question,
                )
            except Exception:
                logger.exception("persist user message failed: session=%s", req.session_id)
        try:
            if ahead > 0:
                yield f"data: {json.dumps({'type': 'queue', 'ahead': ahead}, ensure_ascii=False)}\n\n"
            async for evt in _with_heartbeat(
                ask_stream(
                    req.question, book=req.book, mode=req.mode, history=history,
                    decompose=req.decompose,
                ),
                interval=settings.stream_heartbeat_interval,
            ):
                # 累积 references / token 以便持久化 assistant 消息
                if evt.get("type") == "references":
                    refs = evt.get("content", []) or []
                elif evt.get("type") == "token":
                    answer += evt.get("content", "")
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            if persist and answer:
                try:
                    await session_store.append_message(
                        req.session_id, req.user_id, "assistant", answer, refs=refs,
                    )
                    logger.info(
                        "chat done: session=%s user=%s answer_chars=%d",
                        req.session_id, req.user_id, len(answer),
                    )
                except Exception:
                    logger.exception("persist assistant message failed: session=%s", req.session_id)
        except Exception as e:
            logger.exception("chat 流式异常: %r", e)
            yield f"data: {json.dumps({'type': 'error', 'content': '答复服务暂不可用'}, ensure_ascii=False)}\n\n"
        finally:
            _query_in_flight -= 1
            _release(sem)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- 会话管理路由 ----------

@app.post("/sessions", status_code=201)
async def create_session_endpoint(req: SessionCreateRequest) -> dict:
    """创建新的聊天会话。``user_id`` 标识归属（前端 localStorage 持久 UUID）。"""
    try:
        session = await session_store.create_session(
            user_id=req.user_id, book=req.book, mode=req.mode, title=req.title,
        )
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("创建会话失败", e))


@app.get("/sessions")
async def list_sessions_endpoint(user_id: str) -> list[dict]:
    """列出某用户的会话（索引项，不含消息），按 updated_at 降序。"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 必填")
    try:
        return await session_store.list_sessions(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("查询会话列表失败", e))


@app.get("/sessions/{session_id}")
async def get_session_endpoint(session_id: str, user_id: str) -> dict:
    """取单会话完整内容（含消息）。不存在或越权返回 404。"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 必填")
    try:
        session = await session_store.get_session(session_id, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("查询会话失败", e))
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session_endpoint(session_id: str, user_id: str):
    """删除会话。不存在或越权返回 404。"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 必填")
    try:
        ok = await session_store.delete_session(session_id, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("删除会话失败", e))
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    """同步导入一本书（小书/测试用）。大书建议用 /ingest/async。"""
    from src.loader import resolve_book_path

    path = resolve_book_path(req.file)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"书籍不存在：{path}")
    sem = _ingest_sem()
    if not await _try_acquire(sem):
        raise HTTPException(
            status_code=503, detail="ingest 服务繁忙，请稍后重试", headers={"Retry-After": "10"}
        )
    try:
        info = await ingest_book(str(path), max_chunks=req.max_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("导入失败", e))
    finally:
        _release(sem)
    return IngestResponse(**info)


@app.post("/ingest/async")
async def ingest_async_endpoint(req: IngestRequest) -> dict:
    """异步导入：立即返回 task_id，后台执行，用 /tasks/{id} 查进度。"""
    from src.loader import resolve_book_path

    path = resolve_book_path(req.file)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"书籍不存在：{path}")
    return await _submit_ingest_task(str(path), req.max_chunks, kind="ingest")


class UpsertRequest(BaseModel):
    file: str = Field(..., description="书籍路径（data/books 下文件名或绝对路径）")
    max_chunks: int | None = Field(None, description="最多导入块数（测试用）")


@app.post("/documents/upsert")
async def upsert_document_endpoint(req: UpsertRequest) -> dict:
    """按文件名 upsert（异步）：同 basename 已存在则先删后导。立即返回 task_id。"""
    from src.loader import resolve_book_path

    path = resolve_book_path(req.file)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"书籍不存在：{path}")
    return await _submit_ingest_task(str(path), req.max_chunks, kind="upsert")


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    t = _tasks[task_id]
    # 不把内部字段（_task_ref）暴露给客户端
    return {k: v for k, v in t.items() if not k.startswith("_")}


async def _submit_ingest_task(
    file_path: str, max_chunks: int | None, kind: str
) -> dict:
    """统一的异步 ingest 类任务提交：限流 + 任务追踪 + 防 GC。

    ``kind``: "ingest" | "upsert" | "refresh"。
    """
    sem = _ingest_sem()
    if not await _try_acquire(sem):
        raise HTTPException(
            status_code=503, detail="ingest 服务繁忙，请稍后重试", headers={"Retry-After": "10"}
        )

    task_id = uuid.uuid4().hex
    _tasks[task_id] = {
        "status": "running", "kind": kind, "file": file_path,
        "result": None, "error": None,
        "started_at": time.time(), "finished_at": None, "_task_ref": None,
    }

    async def _run() -> None:
        try:
            if kind == "ingest":
                result = await ingest_book(file_path, max_chunks=max_chunks)
            elif kind == "upsert":
                result = await upsert_document(file_path, max_chunks=max_chunks)
            elif kind == "refresh":
                result = await refresh_document(file_path, max_chunks=max_chunks)
            else:
                raise ValueError(f"unknown kind: {kind}")
            _tasks[task_id].update(
                {"status": "done", "result": result, "error": None, "finished_at": time.time()}
            )
        except Exception as e:
            logger.exception("ingest 任务 (%s) 失败: %r", kind, e)
            _tasks[task_id].update(
                {"status": "failed", "result": None, "error": str(e), "finished_at": time.time()}
            )
        finally:
            _release(sem)

    task = asyncio.create_task(_run())
    _tasks[task_id]["_task_ref"] = task
    return {"task_id": task_id, "status": "running"}


@app.get("/stats", response_model=StatsResponse)
async def stats_endpoint(book: str | None = None) -> StatsResponse:
    """查看图谱统计。``book`` 给定时仅统计该书 workspace。"""
    if book:
        _validate_book(book)
        _ensure_book_exists(book)
    try:
        info = graph_stats(book=book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("统计失败", e))
    return StatsResponse(**info)


@app.get("/config")
async def config_endpoint() -> dict:
    """返回分组配置快照（敏感字段 *** 只读）。改动需重启生效。"""
    return get_config_snapshot()


@app.put("/config")
async def update_config_endpoint(updates: dict) -> dict:
    """更新非敏感配置字段并写回 config.yaml。敏感字段忽略，返回 needs_restart=true。"""
    if not isinstance(updates, dict) or not updates:
        raise HTTPException(status_code=422, detail="请求体需为非空 JSON 对象")
    try:
        return save_config_updates(updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("配置保存失败", e))


class GraphRequest(BaseModel):
    entity: str = Field(..., min_length=1, description="中心实体名称（模糊匹配）")
    book: str = Field(..., description="书籍文件名（= workspace）")
    depth: int = Field(2, ge=1, le=3, description="跳数：1=直接邻居，2=邻居的邻居")
    limit: int = Field(50, ge=1, le=500, description="返回子图规模上限")

    @field_validator("book")
    @classmethod
    def _check_book(cls, v: str) -> str:
        return _validate_book(v)


class TopEntitiesRequest(BaseModel):
    book: str = Field(..., description="书籍文件名（= workspace）")
    limit: int = Field(40, ge=1, le=200, description="返回的热门实体数量")

    @field_validator("book")
    @classmethod
    def _check_book(cls, v: str) -> str:
        return _validate_book(v)


@app.post("/graph")
async def graph_endpoint(req: GraphRequest) -> dict:
    """返回某实体周围的关系子图（nodes + edges），供前端可视化。"""
    _ensure_book_exists(req.book)
    try:
        sub = get_subgraph(req.entity, book=req.book, depth=req.depth, limit=req.limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("子图查询失败", e))
    return sub


@app.post("/graph/top")
async def graph_top_endpoint(req: TopEntitiesRequest) -> dict:
    """返回度数最高的实体，用于初始展示。"""
    _ensure_book_exists(req.book)
    try:
        return get_top_entities(book=req.book, limit=req.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("查询失败", e))


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
        raise HTTPException(status_code=500, detail=_safe_detail("读取文档列表失败", e))


@app.get("/documents/{book}")
async def document_detail_endpoint(book: str) -> dict:
    """取单本书文档详情（按 workspace）。"""
    _validate_book(book)
    try:
        d = await get_document(book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("读取文档失败", e))
    if not d:
        raise HTTPException(status_code=404, detail="文档不存在")
    return d


@app.delete("/documents/{book}")
async def delete_document_endpoint(book: str) -> dict:
    """删除一整本书（其 workspace：Neo4j label 节点 + Qdrant 向量 + KV 子目录）。"""
    _validate_book(book)
    try:
        return await delete_document(book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("删除失败", e))


@app.post("/documents/{book}/refresh")
async def refresh_document_endpoint(book: str, req: RefreshRequest) -> dict:
    """刷新一本书（异步）：删旧 workspace 后重新导入指定文件。立即返回 task_id。"""
    from src.loader import resolve_book_path

    target = req.file or book
    try:
        path = resolve_book_path(target)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"书籍不存在：{path}")
    return await _submit_ingest_task(str(path), req.max_chunks, kind="refresh")


@app.post("/entities/{entity_id}/edit")
async def edit_entity_endpoint(entity_id: str, req: EditEntityRequest, book: str) -> dict:
    """编辑实体（属性 / 重命名），自动重算 embedding 并同步向量库。"""
    _validate_book(book)
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
        raise HTTPException(status_code=500, detail=_safe_detail("编辑失败", e))


@app.delete("/entities/{entity_id}")
async def delete_entity_endpoint(entity_id: str, book: str) -> dict:
    """删除单个实体（含其关系），同步清理向量库。"""
    _validate_book(book)
    try:
        return await delete_entity(entity_id, book=book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("删除失败", e))


@app.post("/relations/edit")
async def edit_relation_endpoint(
    source: str, target: str, req: EditRelationRequest, book: str,
) -> dict:
    """编辑一条关系。通过 query 参数 ?source=&target=&book= 指定。"""
    _validate_book(book)
    updated = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updated:
        raise HTTPException(status_code=400, detail="未提供任何修改字段")
    try:
        return await edit_relation(source, target, updated, book=book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("编辑失败", e))


@app.delete("/relations")
async def delete_relation_endpoint(source: str, target: str, book: str) -> dict:
    """删除一条关系（保留两端实体）。"""
    _validate_book(book)
    try:
        return await delete_relation(source, target, book=book)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("删除失败", e))


@app.post("/cypher")
async def cypher_endpoint(req: CypherRequest) -> dict:
    """直接执行 Cypher 查询（只读）。"""
    if not req.cypher.strip().lower().startswith(("match", "return", "with", "call", "unwind")):
        raise HTTPException(status_code=400, detail="仅允许只读查询（MATCH/RETURN/WITH/CALL）")
    try:
        rows = cypher_query(req.cypher)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_detail("Cypher 执行失败", e))
    return {"rows": rows[: req.limit], "total": len(rows)}
