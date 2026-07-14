"""本地模型：BAAI/bge-m3 向量化 + bge-reranker-v2-m3 重排。

在 Windows 原生 + GPU 上跑（torch CUDA），不依赖 WSL2。
模型懒加载、模块级单例缓存，避免重复载入。
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import numpy as np

from config import settings


# ---------- 模型单例（懒加载）----------

_bge_m3 = None
_bge_m3_lock = asyncio.Lock()
_reranker = None
_reranker_lock = asyncio.Lock()


def _cuda_available() -> bool:
    # BGE_FORCE_CPU=1 时强制 CPU（避免与 Ollama/vLLM 共用 GPU 时 CUDA context 冲突）
    if os.getenv("BGE_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _resolve_model(name: str) -> str:
    """返回模型的本地目录路径。

    解析顺序：
    1. 扫描项目内缓存目录 model_cache/models/{owner}--{name}/snapshots/{rev}/，
       命中含 config.json 的快照即返回 —— 零 import、零网络，避开 modelscope
       在 lightrag/FlagEmbedding 等先加载后 `from modelscope import snapshot_download`
       偶发失败（__init__ 未绑定 snapshot_download）的问题。
    2. 本地未命中 → 走 ModelScope 下载（cache_dir=项目内）。
    3. ModelScope 不可用 → 回退原始名称，交给 FlagEmbedding/CrossEncoder 走 HuggingFace。
    """
    # 1. 本地缓存快照
    snap_root = settings.model_cache_dir / "models" / name.replace("/", "--") / "snapshots"
    if snap_root.exists():
        for snap in sorted(snap_root.iterdir()):
            if snap.is_dir() and (snap / "config.json").exists():
                return str(snap)

    # 2. ModelScope 下载
    try:
        from modelscope import snapshot_download

        # 显式 cache_dir=项目内，不依赖 MODELSCOPE_CACHE env 穿透两层 legacy 调用。
        # 标准布局：{cache_dir}/models/{owner}--{name}/snapshots/{rev}/
        local = snapshot_download(name, cache_dir=str(settings.model_cache_dir))
        return local
    except Exception as e:  # noqa: BLE001
        print(f"[local_models] ModelScope 下载 {name} 失败，回退 HF: {e}")
        return name


def _get_bge_m3():
    """懒加载 BAAI/bge-m3（FlagEmbedding）。"""
    global _bge_m3
    if _bge_m3 is None:
        from FlagEmbedding import BGEM3FlagModel

        # use_fp16 仅在有 CUDA 时开启；CPU 上用 fp32
        _bge_m3 = BGEM3FlagModel(
            _resolve_model(settings.embedding_model), use_fp16=_cuda_available()
        )
    return _bge_m3


def _get_reranker():
    """懒加载 BAAI/bge-reranker-v2-m3（用 sentence_transformers.CrossEncoder）。

    FlagEmbedding.FlagReranker 在新版 transformers 下 tokenizer 缺 prepare_for_model，
    故改用 CrossEncoder（bge-reranker 兼容）。
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(
            _resolve_model(settings.rerank_model),
            max_length=512,
            device="cuda" if _cuda_available() else "cpu",
        )
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
    # 强转 float32：GPU 上 use_fp16=True 时 dense_vecs 是 float16，
    # pymilvus 搜索会据此推断 FLOAT16_VECTOR(102) placeholder，
    # milvus-lite 不支持（仅 101/104/21）→ /query、/chat 搜索报 code=6。
    # 插入路径按 collection schema(float_vector) 强转不受影响，仅搜索受影响。
    return np.asarray(dense, dtype=np.float32)


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
    """bge-reranker 重排，返回 [{"index","relevance_score"}]（按相关度降序）。

    CrossEncoder.predict 返回 logits（可能为负），用 sigmoid 归一化到 (0,1)，
    避免 LightRAG 的 min_rerank_score(默认 0) 过滤掉负分文档。
    """
    if not documents:
        return []
    model = await asyncio.to_thread(_get_reranker)

    pairs = [(query, doc) for doc in documents]
    logits = await asyncio.to_thread(lambda: model.predict(pairs))

    import math

    scores = [1.0 / (1.0 + math.exp(-float(s))) for s in logits]

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    if top_n:
        ranked = ranked[:top_n]
    return [{"index": idx, "relevance_score": score} for idx, score in ranked]


def make_bge_reranker_func():
    """返回 LightRAG 的 rerank_model_func（async 可调用）。"""
    return _bge_rerank
