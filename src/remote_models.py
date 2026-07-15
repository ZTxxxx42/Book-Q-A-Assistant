"""远程 API 模型：SiliconFlow bge-m3 嵌入 + bge-reranker-v2-m3 重排。

替代旧的本地 bge 栈（src/local_models.py 已删除）：无 torch/CUDA、无单例、
无并发加载/推理锁问题。embedding 走 OpenAI 兼容 /v1/embeddings，
rerank 走 SiliconFlow 专用 /v1/rerank（非 OpenAI 兼容）。
"""
from __future__ import annotations

import logging
import math
from typing import Any

import httpx
import numpy as np
from openai import AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger("book_kg.remote_models")


# ---------- embedding func（LightRAG 期望）----------

def make_embedding_func():
    """构造 LightRAG 的 EmbeddingFunc，调用 SiliconFlow /v1/embeddings。"""
    from lightrag.utils import EmbeddingFunc

    client = AsyncOpenAI(
        api_key=settings.embedding_api_key or "EMPTY",
        base_url=settings.embedding_base_url,
    )

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    async def _embed(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, settings.embedding_dim), dtype=np.float32)
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        # OpenAI 兼容响应：resp.data[i].embedding，顺序与输入一致
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=8192,
        func=_embed,
        model_name=settings.embedding_model,
    )


# ---------- rerank func（LightRAG 期望）----------

def _retryable_http(exc: Exception) -> bool:
    """429/5xx/网络错误重试；4xx 业务错误直接抛。"""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_reranker_func():
    """返回 async rerank_model_func，调用 SiliconFlow /v1/rerank。

    返回 [{"index": int, "relevance_score": float}]（按相关度降序）。
    SiliconFlow 返回的 relevance_score 通常已在 [0,1]；若超出（原始 logit）
    则 sigmoid 归一化，避免 LightRAG 的 min_rerank_score 过滤掉负分。
    """
    base = (settings.rerank_base_url or "").rstrip("/")
    url = f"{base}/rerank"
    headers = {
        "Authorization": f"Bearer {settings.rerank_api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(30.0, connect=10.0)

    @retry(
        retry=retry_if_exception(_retryable_http),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    async def _rerank(
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        if not documents:
            return []
        body: dict[str, Any] = {
            "model": settings.rerank_model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        if top_n is not None:
            body["top_n"] = top_n
        async with httpx.AsyncClient(timeout=timeout) as cli:
            r = await cli.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()

        results = data.get("results", []) or []
        out: list[dict] = []
        for it in results:
            idx = int(it["index"])
            score = float(it["relevance_score"])
            # 范围兜底：超出 [0,1] 视为原始 logit，sigmoid 归一化
            if score < 0.0 or score > 1.0:
                score = _sigmoid(score)
            out.append({"index": idx, "relevance_score": score})
        # SiliconFlow 已按分数降序返回；兜底再排一次
        out.sort(key=lambda x: x["relevance_score"], reverse=True)
        return out

    return _rerank
