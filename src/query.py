"""查询接口：基于已构建的 LightRAG + Neo4j 图谱。"""
from __future__ import annotations

from typing import Any, AsyncIterator, Literal

from config import settings
from src.graph_builder import _build_rag

QueryMode = Literal["local", "global", "hybrid", "naive"]


def _make_param(mode: QueryMode, stream: bool, history: list[dict] | None = None):
    """统一构造 QueryParam，显式传入所有检索旋钮 + enable_rerank（修 /query 非流式不对称 bug）。"""
    from lightrag import QueryParam

    return QueryParam(
        mode=mode,
        stream=stream,
        enable_rerank=settings.enable_rerank,
        top_k=settings.top_k,
        chunk_top_k=settings.chunk_top_k,
        include_references=settings.include_references,
        response_type=settings.response_type,
        conversation_history=history or [],
    )


async def ask(
    question: str,
    mode: QueryMode = "hybrid",
    stream: bool = False,
) -> dict[str, Any]:
    """对图谱提问，返回 ``{"answer": str, "references": list}``。

    用 aquery_llm 取完整结果（含 references），而非向后兼容的 aquery（丢弃 references）。
    """
    rag = _build_rag()
    await rag.initialize_storages()
    param = _make_param(mode, stream=False)

    result = await rag.aquery_llm(question, param=param)
    llm_resp = result.get("llm_response", {})
    content = llm_resp.get("content", "")
    refs = result.get("data", {}).get("references", [])
    return {"answer": content, "references": refs}


async def ask_stream(
    question: str,
    mode: QueryMode = "hybrid",
    history: list[dict] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """流式问答生成器：yield 事件 dict，并带入多轮对话历史。

    事件序列：
    - ``{"type": "references", "content": [...]}``（检索后、生成前，附引用出处）
    - ``{"type": "token", "content": "..."}``（逐 token）
    生成结束自然停止（调用方发 done）。
    """
    rag = _build_rag()
    await rag.initialize_storages()
    param = _make_param(mode, stream=True, history=history)

    result = await rag.aquery_llm(question, param=param)
    refs = result.get("data", {}).get("references", [])
    if refs:
        yield {"type": "references", "content": refs}

    llm_resp = result.get("llm_response", {})
    iterator = llm_resp.get("response_iterator")
    if iterator is not None:
        async for chunk in iterator:
            if chunk:
                yield {"type": "token", "content": chunk}
    else:
        # 未走流式分支的兜底：一次性 yield 完整内容
        content = llm_resp.get("content", "")
        if content:
            yield {"type": "token", "content": content}


def cypher_query(cypher: str) -> list[dict]:
    """直接对 Neo4j 执行 Cypher，返回记录列表（只读查询）。"""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        with driver.session() as session:
            result = session.run(cypher)
            return [r.data() for r in result]
    finally:
        driver.close()


def graph_stats() -> dict:
    """返回图谱基本统计。"""
    records = cypher_query(
        """
        MATCH (n) WITH labels(n) AS lbl, count(*) AS cnt
        UNWIND lbl AS label
        RETURN label, sum(cnt) AS count
        UNION ALL
        RETURN 'RELATIONSHIPS' AS label, 0 AS count
        """
    )
    # 关系数单独查
    rel_count = cypher_query("MATCH ()-[r]->() RETURN count(r) AS c")
    rel_total = rel_count[0]["c"] if rel_count else 0

    nodes = {r["label"]: r["count"] for r in records if r["label"] != "RELATIONSHIPS"}
    return {
        "node_counts_by_label": nodes,
        "total_nodes": sum(nodes.values()),
        "total_relationships": rel_total,
    }
