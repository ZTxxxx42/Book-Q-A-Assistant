"""本地模型：BAAI/bge-m3 向量化 + bge-reranker-v2-m3 重排。

在 Windows 原生 + GPU 上跑（torch CUDA），不依赖 WSL2。
模型懒加载、模块级单例缓存，避免重复载入。
"""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from config import settings


# ---------- 模型单例（懒加载）----------

_bge_m3 = None
_bge_m3_lock = asyncio.Lock()
_reranker = None
_reranker_lock = asyncio.Lock()


def _get_bge_m3():
    """懒加载 BAAI/bge-m3（FlagEmbedding）。"""
    global _bge_m3
    if _bge_m3 is None:
        from FlagEmbedding import BGEM3FlagModel

        # use_fp16=True 走 GPU 半精度；无 CUDA 自动回退 CPU
        _bge_m3 = BGEM3FlagModel(
            settings.embedding_model, use_fp16=True
        )
    return _bge_m3


def _get_reranker():
    """懒加载 BAAI/bge-reranker-v2-m3（FlagRerank）。"""
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagRerank

        _reranker = FlagRerank(settings.rerank_model, use_fp16=True)
    return _reranker


# ---------- embedding func（LightRAG 期望）----------

async def _bge_m3_embed(texts: list[str]) -> np.ndarray:
    """bge-m3 编码，返回 numpy 数组（LightRAG 访问 .size/.shape）。"""
    model = await asyncio.to_thread(_get_bge_m3)
    # BGEM3FlagModel.encode 同步且内部已批处理；返回 dict，取 dense_vecs
    out = await asyncio.to_thread(
        lambda: model.encode(texts, batch_size=32, max_length=8192)
    )
    # encode 返回 dict（含 dense_vecs/sparse_vecs/colbert_vecs）
    dense = out["dense_vecs"]
    return np.array(dense)


def make_bge_m3_embedding_func():
    """构造 LightRAG 的 EmbeddingFunc 实例。"""
    from lightrag.utils import EmbeddingFunc

    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        func=_bge_m3_embed,
        model_name=settings.embedding_model,
        max_token_size=8192,
    )


# ---------- rerank func（LightRAG 期望）----------

async def _bge_rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """bge-reranker 重排，返回 [{"index","relevance_score"}]（按相关度降序）。"""
    if not documents:
        return []
    model = await asyncio.to_thread(_get_reranker)

    pairs = [(query, doc) for doc in documents]
    scores = await asyncio.to_thread(lambda: model.compute_score(pairs, normalize=True))

    # compute_score 单条返回 float，多条返回 list
    if isinstance(scores, (int, float)):
        scores = [float(scores)]
    else:
        scores = [float(s) for s in scores]

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    if top_n:
        ranked = ranked[:top_n]
    return [{"index": idx, "relevance_score": score} for idx, score in ranked]


def make_bge_reranker_func():
    """返回 LightRAG 的 rerank_model_func（async 可调用）。"""
    return _bge_rerank
