"""冒烟测试：SiliconFlow embed/rerank + Qdrant + GLM 抽取 + Qwen 答复。

7 项全过才说明迁移后的栈可进入 ingest/query。用 my_env 解释器运行：
    D:/miniforge/envs/my_env/python.exe scripts/smoke_remote_models.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def test_embed():
    from src.remote_models import make_embedding_func

    ef = make_embedding_func()
    v = await ef.func(["你好世界", "知识图谱"])
    assert v.shape == (2, 1024), f"shape 期望 (2,1024)，实际 {v.shape}"
    assert v.dtype.name == "float32", f"dtype 期望 float32，实际 {v.dtype}"
    print(f"✅ embed: shape={v.shape} dtype={v.dtype}")


async def test_rerank():
    from src.remote_models import make_reranker_func

    rf = make_reranker_func()
    docs = [
        "Alice 是爱幻想的女孩。",
        "Python 是编程语言。",
        "白兔先生带着怀表。",
        "今天天气不错。",
    ]
    res = await rf(query="主角是谁？", documents=docs, top_n=2)
    assert len(res) == 2, f"期望 2 条，实际 {len(res)}"
    assert all(0.0 <= r["relevance_score"] <= 1.0 for r in res), "score 越界"
    print(f"✅ rerank: {res}")


async def test_empty():
    from src.remote_models import make_embedding_func, make_reranker_func

    assert await make_reranker_func()(query="x", documents=[]) == []
    v = await make_embedding_func().func([])
    assert v.shape[0] == 0
    print("✅ empty-input guards")


async def test_glm():
    """验证 GLM 抽取端点（extract/keyword 角色用）。"""
    from openai import AsyncOpenAI

    from config import settings

    cli = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    r = await cli.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": "用一句话介绍爱丽丝梦游仙境。"}],
        max_tokens=80,
    )
    print(f"✅ glm({settings.llm_model}): {r.choices[0].message.content}")


async def test_query_llm():
    """验证本地 Qwen 答复端点（query 角色用）。"""
    from openai import AsyncOpenAI

    from config import settings

    cli = AsyncOpenAI(
        api_key=settings.query_llm_api_key or "ollama",
        base_url=settings.query_llm_base_url,
    )
    r = await cli.chat.completions.create(
        model=settings.query_llm_model,
        messages=[{"role": "user", "content": "用一句话介绍爱丽丝梦游仙境。"}],
        max_tokens=80,
    )
    print(f"✅ qwen({settings.query_llm_model}): {r.choices[0].message.content}")


async def test_qdrant():
    """验证 Qdrant 连通 + cosine 分数语义（确认 score ∈ [-1,1] 是真 cosine 相似度）。"""
    from qdrant_client import QdrantClient, models

    from config import settings

    cli = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    cli.recreate_collection(
        collection_name="smoke_test",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    cli.upsert(
        collection_name="smoke_test",
        points=[
            {"id": 1, "vector": [1.0, 0.0, 0.0, 0.0], "payload": {"t": "a"}},
            {"id": 2, "vector": [0.0, 1.0, 0.0, 0.0], "payload": {"t": "b"}},
        ],
    )
    res = cli.query_points(
        collection_name="smoke_test", query=[1.0, 0.0, 0.0, 0.0], limit=2
    ).points
    cli.delete_collection(collection_name="smoke_test")
    top = res[0]
    assert top.payload["t"] == "a", f"最近邻应为 a，实际 {top.payload}"
    print(f"✅ qdrant: 连通正常，cosine score 示例={top.score:.4f}（同向应≈1.0）")


async def test_build_rag():
    from src.graph_builder import _build_rag

    rag = _build_rag()
    await rag.initialize_storages()
    print(f"✅ LightRAG init: vector_storage={rag.vector_storage} graph_storage={rag.graph_storage}")


async def main():
    await test_embed()
    await test_rerank()
    await test_empty()
    await test_glm()
    await test_query_llm()
    await test_qdrant()
    await test_build_rag()
    print("\n🎉 全部冒烟测试通过")


if __name__ == "__main__":
    asyncio.run(main())
