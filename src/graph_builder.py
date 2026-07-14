"""LightRAG + Neo4j + Milvus 集成：构建知识图谱（适配 LightRAG 1.5.x）。

检索链路：文本分块(重叠)→bge-m3 向量化→Milvus 存储；查询时 LightRAG hybrid
（向量+图+关键词三路融合）→ bge-reranker 重排 → 本地 vLLM(Qwen2.5-7B) 生成。
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from config import settings
from src.loader import load_book, split_into_chunks


def _make_llm_func():
    """构造 LightRAG 的 LLM 调用函数，指向本地 vLLM（OpenAI 兼容）。

    支持流式：当 kwargs 含 stream=True 时返回 async generator 逐 token yield，
    让 LightRAG 的 aquery(stream=True) 判定 is_streaming=True、真正流式输出。
    非流式分支（实体抽取等）返回完整字符串。
    """
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    from openai import RateLimitError

    client = AsyncOpenAI(
        api_key=settings.llm_api_key or "token-abc",
        base_url=settings.llm_base_url,
    )

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
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

        if stream and settings.llm_streaming:
            # 流式：返回 async generator（LightRAG 会 async for 消费）
            return _stream_completion(client, messages, create_kwargs)

        # 非流式：返回完整字符串（实体抽取走这里）
        resp = await client.chat.completions.create(
            model=settings.llm_model, messages=messages, **create_kwargs
        )
        return resp.choices[0].message.content or ""

    return llm_model_func


async def _stream_completion(
    client: AsyncOpenAI, messages: list[dict], create_kwargs: dict
) -> AsyncIterator[str]:
    """vLLM 流式生成器：逐 delta yield 文本片。"""
    resp = await client.chat.completions.create(
        model=settings.llm_model, messages=messages, **create_kwargs
    )
    async for delta in resp:
        if not delta.choices:
            continue
        content = delta.choices[0].delta.content
        if content:
            yield content


def _build_rag():
    """构造已配置 Milvus + Neo4j + 本地 embed/rerank 的 LightRAG 实例。"""
    import os

    # LightRAG 后端从环境变量读取连接信息
    os.environ.setdefault("NEO4J_URI", settings.neo4j_uri)
    os.environ.setdefault("NEO4J_USERNAME", settings.neo4j_username)
    os.environ.setdefault("NEO4J_PASSWORD", settings.neo4j_password)

    # Milvus env 的两难：
    # - LightRAG check_storage_env_vars 要求 MILVUS_URI / MILVUS_DB_NAME 两个 key 存在
    # - 但 pymilvus 的 legacy connections 单例在 import 时解析 Config.MILVUS_URI，
    #   文件路径（milvus-lite）会触发 ConnectionConfigException
    # 解法：先在 env 缺省时 import pymilvus（单例以空值初始化），再设 env；
    # milvus_impl 运行时用 os.environ.get 读取（绕过 Config 单例）。
    import pymilvus  # noqa: F401  —— 触发 connections 单例初始化（env 空，不报错）
    os.environ.setdefault(
        "MILVUS_URI", str(settings.working_dir / "milvus_lite.db")
    )
    os.environ.setdefault("MILVUS_DB_NAME", "")  # 空串：满足 verify + 跳过 db 操作

    from lightrag import LightRAG

    from src.local_models import make_bge_m3_embedding_func, make_bge_reranker_func

    settings.ensure_dirs()

    return LightRAG(
        working_dir=str(settings.working_dir),
        llm_model_func=_make_llm_func(),
        llm_model_name=settings.llm_model,
        embedding_func=make_bge_m3_embedding_func(),
        graph_storage="Neo4JStorage",          # 图存储：Neo4j
        vector_storage="MilvusVectorDBStorage",  # 向量存储：Milvus（milvus-lite）
        rerank_model_func=make_bge_reranker_func(),
        chunk_token_size=settings.chunk_size,
        chunk_overlap_token_size=settings.chunk_overlap,
        entity_extract_max_gleaning=2,
        llm_model_max_async=4,
        max_parallel_insert=2,
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
