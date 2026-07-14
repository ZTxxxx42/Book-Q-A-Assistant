"""组件冒烟测试：bge-m3 向量化、bge-reranker 重排、Milvus 连通性、LightRAG 构造。

用法：python scripts/smoke_local_models.py
首次运行会下载 bge-m3(~2.2GB)+bge-reranker-v2-m3(~568MB)，需联网（国内建议设 HF_ENDPOINT）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 允许从项目根 import src / config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def header(t: str) -> None:
    print(f"\n{'=' * 60}\n{t}\n{'=' * 60}")


async def test_embedding() -> None:
    header("1. bge-m3 向量化")
    from src.local_models import make_bge_m3_embedding_func

    ef = make_bge_m3_embedding_func()
    print(f"embedding_dim={ef.embedding_dim}, model={ef.model_name}")
    vec = await ef.func(["你好世界", "知识图谱构建"])
    print(f"返回类型: {type(vec)}, shape: {vec.shape}, dim: {vec.shape[1]}")
    assert vec.shape == (2, 1024), f"期望 (2,1024)，实际 {vec.shape}"
    print("✅ bge-m3 OK")


async def test_rerank() -> None:
    header("2. bge-reranker 重排")
    from src.local_models import make_bge_reranker_func

    rerank = make_bge_reranker_func()
    docs = [
        "Alice 是一个爱幻想的女孩。",
        "Python 是一种编程语言。",
        "白兔先生带着怀表匆匆跑过。",
        "今天天气不错。",
    ]
    res = await rerank(query="主角是谁？", documents=docs, top_n=2)
    print(f"重排结果(top2): {res}")
    assert len(res) == 2
    assert all("index" in r and "relevance_score" in r for r in res)
    # "Alice 是一个爱幻想的女孩" 应排第一
    assert res[0]["index"] == 0, f"期望 index=0 排首位，实际 {res}"
    print("✅ bge-reranker OK")


async def test_milvus() -> None:
    header("3. Milvus-lite 连通")
    import shutil
    from pymilvus import MilvusClient

    db = "rag_storage/smoke_milvus.db"
    Path("rag_storage").mkdir(exist_ok=True)
    # 清理上次残留（milvus-lite drop_collection 在 Windows 有 rename 竞态，直接删目录）
    shutil.rmtree(db, ignore_errors=True)
    client = MilvusClient(uri=db)
    print(f"Milvus client 创建成功: {db}")
    client.create_collection(
        "smoke_test",
        dimension=4,
        metric_type="COSINE",
        id_field_name="id",
        vector_field_name="vec",
    )
    client.insert("smoke_test", [{"id": i, "vec": [float(i)] * 4, "t": f"row{i}"} for i in range(3)])
    res = client.search("smoke_test", data=[[1.0] * 4], limit=2, output_fields=["t"])
    print(f"搜索结果: {res}")
    shutil.rmtree(db, ignore_errors=True)
    print("✅ Milvus-lite OK")


async def test_build_rag() -> None:
    header("4. LightRAG 构造（Milvus + Neo4j + 本地 embed/rerank）")
    from src.graph_builder import _build_rag

    rag = _build_rag()
    print(f"LightRAG 构造成功: vector={rag.vector_storage}, graph={rag.graph_storage}")
    await rag.initialize_storages()
    print("✅ initialize_storages OK")


async def main() -> None:
    await test_embedding()
    await test_rerank()
    await test_milvus()
    await test_build_rag()
    print("\n🎉 全部组件冒烟测试通过")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
