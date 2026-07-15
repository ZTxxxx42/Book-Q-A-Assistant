"""LightRAG + Neo4j + Qdrant 集成：构建知识图谱（适配 LightRAG 1.5.x）。

检索链路：文本分块(重叠)→SiliconFlow bge-m3 向量化→Qdrant 存储；查询时
LightRAG hybrid（向量+图+关键词三路融合）→ SiliconFlow bge-reranker 重排。

LLM 按角色分流（LightRAG role_llm_configs）：
- extract / keyword（实体抽取、关键词抽取）→ 远程 GLM-4.7（重活，不占本地 GPU）。
- query（最终答复生成）→ 本地 Ollama Qwen2.5-7B-Instruct（短答复流式，不持续满载）。
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from config import settings
from src.loader import load_book, split_into_chunks


def _make_llm_func(api_key: str, base_url: str | None, model: str, streaming_enabled: bool):
    """构造通用 LightRAG LLM callable（OpenAI 兼容端点）。

    支持流式：当 kwargs 含 stream=True 时返回 async generator 逐 token yield，
    让 LightRAG 的 aquery(stream=True) 判定 is_streaming=True、真正流式输出。
    非流式分支（实体抽取等）返回完整字符串。带 tenacity 重试（RateLimitError）。
    """
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    from openai import RateLimitError

    client = AsyncOpenAI(api_key=api_key or "token-abc", base_url=base_url)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(6),
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
            return _stream_completion(client, model, messages, create_kwargs)

        # 非流式：返回完整字符串（实体抽取走这里）
        resp = await client.chat.completions.create(
            model=model, messages=messages, **create_kwargs
        )
        return resp.choices[0].message.content or ""

    return llm_model_func


async def _stream_completion(
    client: AsyncOpenAI, model: str, messages: list[dict], create_kwargs: dict
) -> AsyncIterator[str]:
    """流式生成器：逐 delta yield 文本片。"""
    resp = await client.chat.completions.create(
        model=model, messages=messages, **create_kwargs
    )
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
    )


def _make_qwen_func():
    """答复角色 LLM：本地 Ollama Qwen。"""
    return _make_llm_func(
        api_key=settings.query_llm_api_key,
        base_url=settings.query_llm_base_url,
        model=settings.query_llm_model,
        streaming_enabled=settings.query_llm_streaming,
    )


def _build_rag():
    """构造已配置 Qdrant + Neo4j + SiliconFlow embed/rerank + 双 LLM 的 LightRAG 实例。"""
    import os

    from lightrag.llm_roles import RoleLLMConfig

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
    """加载并导入一本书，返回统计信息。"""
    text = load_book(file_path)
    chunks = split_into_chunks(text)
    if max_chunks:
        chunks = chunks[:max_chunks]

    rag = _build_rag()
    await rag.initialize_storages()
    full_text = "\n\n".join(chunks)
    from pathlib import Path

    fname = Path(file_path).name
    await rag.ainsert(full_text, file_paths=[fname])

    return {
        "file": str(file_path),
        "total_chars": len(text),
        "chunks": len(chunks),
        "working_dir": str(settings.working_dir),
    }
