"""LightRAG + Neo4j + Qdrant 集成：构建知识图谱（适配 LightRAG 1.5.x）。

检索链路：文本分块(重叠)→SiliconFlow bge-m3 向量化→Qdrant 存储；查询时
LightRAG hybrid（向量+图+关键词三路融合）→ SiliconFlow bge-reranker 重排。

LLM 按角色分流（LightRAG role_llm_configs）：
- extract / keyword（实体抽取、关键词抽取）→ 远程 GLM-4.7（重活，不占本地 GPU）。
- query（最终答复生成）→ 本地 Ollama Qwen2.5-7B-Instruct（短答复流式，不持续满载）。
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from config import settings
from src.loader import load_book, split_into_chunks

logger = logging.getLogger("book_kg.graph_builder")


def _make_llm_func(
    api_key: str,
    base_url: str | None,
    model: str,
    streaming_enabled: bool,
    timeout: float = 60.0,
    first_token_timeout: float = 60.0,
    rate_limit_retries: int = 4,
    connection_retries: int = 3,
):
    """构造通用 LightRAG LLM callable（OpenAI 兼容端点）。

    支持流式：当 kwargs 含 stream=True 时返回 async generator 逐 token yield，
    让 LightRAG 的 aquery(stream=True) 判定 is_streaming=True、真正流式输出。
    非流式分支（实体抽取等）返回完整字符串。

    鲁棒性：
    - tenacity 重试 RateLimitError(429, 6 次) + 连接/超时错误(3 次)，指数退避 2-60s。
    - 显式 ``timeout`` 防请求永久挂起。
    - 非流式空响应抛 ``EmptyLLMResponseError``，不静默成功。
    - 流式首 token 超时（``first_token_timeout``）防 Ollama 卡死不 yield。
    """
    import asyncio

    from openai import (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    )
    from tenacity import (
        retry,
        retry_if_exception_type,
        wait_exponential,
    )

    from src.errors import EmptyLLMResponseError

    client = AsyncOpenAI(
        api_key=api_key or "token-abc", base_url=base_url, timeout=timeout
    )

    def _stop_by_type(retry_state):
        """429 重试 rate_limit_retries 次，连接/超时错误重试 connection_retries 次。"""
        exc = retry_state.outcome.exception()
        n = retry_state.attempt_number
        if isinstance(exc, RateLimitError):
            return n >= rate_limit_retries
        return n >= connection_retries

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError, asyncio.TimeoutError)
        ),
        stop=_stop_by_type,
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        **kwargs: Any,
    ):
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        stream = bool(kwargs.get("stream", False))
        create_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"temperature", "top_p", "max_tokens", "stream"}
        }
        create_kwargs["stream"] = stream

        if stream and streaming_enabled:
            # 流式：返回 async generator（LightRAG 会 async for 消费）
            return _stream_completion(
                client, model, messages, create_kwargs, first_token_timeout
            )

        # 非流式：返回完整字符串（实体抽取走这里）
        resp = await client.chat.completions.create(
            model=model, messages=messages, **create_kwargs
        )
        content = resp.choices[0].message.content
        if not content:
            raise EmptyLLMResponseError(f"{model} 返回空响应")
        return content

    return llm_model_func


async def _stream_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    create_kwargs: dict,
    first_token_timeout: float = 60.0,
) -> AsyncIterator[str]:
    """流式生成器：逐 delta yield 文本片。

    首 token 超时：若 ``first_token_timeout`` 秒内未 yield 任何内容，抛
    ``asyncio.TimeoutError``（被上层重试/兜底捕获），防 Ollama 卡死。
    """
    import asyncio

    resp = await client.chat.completions.create(
        model=model, messages=messages, **create_kwargs
    )

    async def _first_token():
        async for delta in resp:
            if not delta.choices:
                continue
            content = delta.choices[0].delta.content
            if content:
                return content, delta
        return None, None

    # 等 首 token（或流结束）
    first_content, _ = await asyncio.wait_for(_first_token(), timeout=first_token_timeout)
    if first_content:
        yield first_content
    async for delta in resp:
        if not delta.choices:
            continue
        content = delta.choices[0].delta.content
        if content:
            yield content


def _make_glm_func():
    """抽取 / 关键词角色 LLM：远程 GLM-4.7。"""
    return _make_llm_func(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        streaming_enabled=settings.llm_streaming,
        timeout=settings.llm_timeout,
        first_token_timeout=settings.stream_first_token_timeout,
        rate_limit_retries=settings.llm_rate_limit_retries,
        connection_retries=settings.llm_connection_retries,
    )


def _make_qwen_func():
    """答复角色 LLM：本地 Ollama Qwen。"""
    return _make_llm_func(
        api_key=settings.query_llm_api_key,
        base_url=settings.query_llm_base_url,
        model=settings.query_llm_model,
        streaming_enabled=settings.query_llm_streaming,
        timeout=settings.query_llm_timeout,
        first_token_timeout=settings.stream_first_token_timeout,
        rate_limit_retries=settings.llm_rate_limit_retries,
        connection_retries=settings.llm_connection_retries,
    )


# 答复语言标签映射（settings.language → 模板指令中的语言名）
_RESPONSE_LANG_LABELS = {
    "chinese": "简体中文",
    "english": "English",
    "japanese": "日本語",
    "french": "Français",
    "german": "Deutsch",
}


def _enforce_response_language() -> None:
    """覆盖 LightRAG ``rag_response`` 模板的"跟随问题语言"指令，强制用 ``settings.language`` 回答。

    LightRAG 默认模板写 "The response MUST be in the same language as the user query."
    —— 英文书 + 英文提问 → Qwen 用英文答复，但因模型中英双语常混杂中文，造成中英混杂。
    改为强制目标语言（默认简体中文），与问题/上下文语言无关。

    两处加固（小模型遵从性弱，单点指令常被忽略）：
    1. 替换原"跟随问题语言"行为目标语言的强指令；
    2. 模板末尾追加一条强制提醒（recency bias：模型对结尾指令更敏感）。
    幂等：已替换则跳过。
    """
    from lightrag.prompt import PROMPTS

    lang_key = settings.language.lower()
    label = _RESPONSE_LANG_LABELS.get(lang_key, settings.language)
    directives = {
        "chinese": "必须全部使用简体中文回复，无论用户问题或上下文是何种语言；严禁中英混杂（专有名词可保留原文）。",
        "english": "You MUST respond entirely in English, regardless of the language of the user query or context; do not mix languages.",
    }
    directive = directives.get(
        lang_key, f"The response MUST be written in {label}; do not mix languages."
    )

    p = PROMPTS.get("rag_response", "")
    old = "The response MUST be in the same language as the user query."
    new = f"The response MUST be written in {label}. {directive}"
    if old in p:
        p = p.replace(old, new)
    # 末尾强制提醒
    marker = "【强制输出语言】"
    if marker not in p:
        p = p + f"\n\n{marker}{directive}"
    PROMPTS["rag_response"] = p
    logger.info("rag_response 语言指令已覆盖为: %s", label)


def _build_rag(workspace: str = ""):
    """构造已配置 Qdrant + Neo4j + SiliconFlow embed/rerank + 双 LLM 的 LightRAG 实例。

    ``workspace`` 给定时，该实例的所有存储（图/向量/KV）都隔离在该 workspace：
    Neo4j 节点 label = workspace、Qdrant payload workspace_id = workspace、
    KV 落 working_dir/<workspace>/。每本书用各自 basename 作 workspace → 独立图谱。
    """
    import os

    from lightrag.llm_roles import RoleLLMConfig

    _enforce_response_language()

    # LightRAG 后端从环境变量读取连接信息
    os.environ.setdefault("NEO4J_URI", settings.neo4j_uri)
    os.environ.setdefault("NEO4J_USERNAME", settings.neo4j_username)
    os.environ.setdefault("NEO4J_PASSWORD", settings.neo4j_password)
    os.environ.setdefault("QDRANT_URL", settings.qdrant_url)
    if settings.qdrant_api_key:
        os.environ.setdefault("QDRANT_API_KEY", settings.qdrant_api_key)

    from lightrag import LightRAG

    from src.remote_models import make_embedding_func, make_reranker_func

    settings.ensure_dirs()

    return LightRAG(
        working_dir=str(settings.working_dir),
        workspace=workspace,                       # 每书独立 workspace = 独立图谱
        llm_model_func=_make_glm_func(),          # base = GLM（extract / keyword 角色）
        llm_model_name=settings.llm_model,
        embedding_func=make_embedding_func(),
        graph_storage="Neo4JStorage",              # 图存储：Neo4j
        vector_storage="QdrantVectorDBStorage",    # 向量存储：Qdrant（Docker，真 cosine）
        rerank_model_func=make_reranker_func(),
        chunk_token_size=settings.chunk_size,
        chunk_overlap_token_size=settings.chunk_overlap,
        entity_extract_max_gleaning=2,
        llm_model_max_async=settings.llm_model_max_async,
        max_parallel_insert=2,
        # query 角色（最终答复生成）覆盖为本地 Qwen
        role_llm_configs={
            "query": RoleLLMConfig(
                func=_make_qwen_func(),
                max_async=settings.query_llm_model_max_async,
            ),
        },
        addon_params={"language": settings.language},
    )


async def ingest_book(
    file_path: str,
    max_chunks: int | None = None,
) -> dict:
    """加载并导入一本书到其独立 workspace（= 文件 basename），返回统计信息。"""
    text = load_book(file_path)
    chunks = split_into_chunks(text)
    if max_chunks:
        chunks = chunks[:max_chunks]

    from pathlib import Path

    fname = Path(file_path).name
    logger.info("ingest 开始: %s（%d chunks, workspace=%s）", fname, len(chunks), fname)
    # 每本书用 basename 作 workspace → Neo4j 独立 label、Qdrant 独立 workspace_id、KV 独立子目录
    # ingest 是重写操作，独占实例（不入池），用完 finalize 避免连接泄漏。
    rag = _build_rag(workspace=fname)
    await rag.initialize_storages()
    try:
        full_text = "\n\n".join(chunks)
        await rag.ainsert(full_text, file_paths=[fname])
    finally:
        try:
            await rag.finalize_storages()
        except Exception:
            pass
    # ingest 后使池中该 workspace 的旧实例失效（内容已变）
    try:
        from src.rag_pool import get_pool

        pool = await get_pool()
        await pool.invalidate(fname)
    except Exception:
        pass

    return {
        "file": str(file_path),
        "total_chars": len(text),
        "chunks": len(chunks),
        "workspace": fname,
        "working_dir": str(settings.working_dir),
    }
