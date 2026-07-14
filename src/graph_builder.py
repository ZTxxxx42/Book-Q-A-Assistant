"""LightRAG + Neo4j 集成：构建知识图谱（适配 LightRAG 1.5.x）。

LightRAG 在插入文本时会调用 LLM 抽取实体与关系，并写入指定的 Neo4j 图后端。
"""
from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI

from config import settings
from src.loader import load_book, split_into_chunks


def _make_llm_func():
    """构造 LightRAG 使用的 LLM 调用函数（OpenAI 兼容，智谱 GLM）。

    带 429 速率限制重试退避——智谱 glm-4.7 的 RPM 较严，并发时易触发。
    """
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    from openai import RateLimitError

    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
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
    ) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            **{k: v for k, v in kwargs.items() if k in {"temperature", "top_p", "max_tokens"}},
        )
        return resp.choices[0].message.content or ""

    return llm_model_func


def _make_embedding_func():
    """构造 EmbeddingFunc 实例（含维度与模型名），用同步客户端探测维度。"""
    from lightrag.utils import EmbeddingFunc

    async def embedding_func(texts: list[str]):
        import numpy as np

        client = AsyncOpenAI(
            api_key=settings.embedding_api_key or settings.llm_api_key,
            base_url=settings.embedding_base_url or settings.llm_base_url,
        )
        resp = await client.embeddings.create(
            model=settings.embedding_model, input=texts
        )
        # LightRAG 期望 numpy 数组（访问 .size / .shape）
        return np.array([d.embedding for d in resp.data])

    # 用同步客户端探测维度，避免在已有事件循环中嵌套 run_until_complete
    sync_client = OpenAI(
        api_key=settings.embedding_api_key or settings.llm_api_key,
        base_url=settings.embedding_base_url or settings.llm_base_url,
    )
    probe = sync_client.embeddings.create(model=settings.embedding_model, input=["dim probe"])
    embedding_dim = len(probe.data[0].embedding)

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        func=embedding_func,
        model_name=settings.embedding_model,
        max_token_size=8192,
    )


def _build_rag():
    """构造已配置好 Neo4j 后端的 LightRAG 实例。"""
    import os

    # LightRAG 的 Neo4JStorage 从环境变量读取连接信息，确保已导出
    os.environ.setdefault("NEO4J_URI", settings.neo4j_uri)
    os.environ.setdefault("NEO4J_USERNAME", settings.neo4j_username)
    os.environ.setdefault("NEO4J_PASSWORD", settings.neo4j_password)

    from lightrag import LightRAG

    settings.ensure_dirs()

    return LightRAG(
        working_dir=str(settings.working_dir),
        llm_model_func=_make_llm_func(),
        llm_model_name=settings.llm_model,
        embedding_func=_make_embedding_func(),
        graph_storage="Neo4JStorage",  # 图存储后端：Neo4j（按名字注册）
        chunk_token_size=settings.chunk_size,
        chunk_overlap_token_size=settings.chunk_overlap,
        entity_extract_max_gleaning=2,  # 注：1.5.4 中 >0 即跑 1 轮 gleaning，已开启
        llm_model_max_async=4,   # flash RPM 宽松，并发4 加速；429 有重试兜底
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
    # 传入已切好的块拼接文本；LightRAG 内部会按 chunk_token_size 再做切块
    full_text = "\n\n".join(chunks)
    # 带 file_paths 让 doc_status 记录真实文件名，便于文档列表与刷新
    from pathlib import Path

    fname = Path(file_path).name
    await rag.ainsert(full_text, file_paths=[fname])

    return {
        "file": str(file_path),
        "total_chars": len(text),
        "chunks": len(chunks),
        "working_dir": str(settings.working_dir),
    }
