"""查询接口：基于已构建的 LightRAG + Neo4j 图谱（每书独立 workspace）。"""
from __future__ import annotations

from typing import Any, AsyncIterator, Literal

from config import settings
from src.graph_builder import _build_rag

QueryMode = Literal["local", "global", "hybrid", "naive"]


def _make_param(mode: QueryMode, stream: bool, history: list[dict] | None = None):
    """统一构造 QueryParam，显式传入检索旋钮 + enable_rerank。"""
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
    book: str,
    mode: QueryMode = "hybrid",
    stream: bool = False,
) -> dict[str, Any]:
    """对指定书的图谱提问，返回 ``{"answer": str, "references": list}``。

    ``book`` 为文件 basename，同时是 workspace 名 → 仅检索该书的图+向量+chunk。
    用 aquery_llm 取完整结果（含 references）。
    """
    rag = _build_rag(workspace=book)
    await rag.initialize_storages()
    param = _make_param(mode, stream=False)

    result = await rag.aquery_llm(question, param=param)
    llm_resp = result.get("llm_response", {})
    content = llm_resp.get("content", "")
    refs = result.get("data", {}).get("references", [])
    return {"answer": content, "references": refs}


async def ask_stream(
    question: str,
    book: str,
    mode: QueryMode = "hybrid",
    history: list[dict] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """流式问答生成器（限定书）：yield 事件 dict。事件序列：references → token…。"""
    rag = _build_rag(workspace=book)
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
        content = llm_resp.get("content", "")
        if content:
            yield {"type": "token", "content": content}


def cypher_query(cypher: str, book: str | None = None) -> list[dict]:
    """直接对 Neo4j 执行 Cypher，返回记录列表（只读查询）。

    ``book`` 给定时限定该 workspace 的节点 label。
    """
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


def graph_stats(book: str | None = None) -> dict:
    """返回图谱统计。``book`` 给定时仅统计该 workspace（Neo4j label）。"""
    if book:
        # 转义 backtick，构造安全 label
        label = book.replace("`", "``")
        records = cypher_query(
            f"MATCH (n:`{label}`) RETURN count(n) AS c", book=book
        )
        nodes = records[0]["c"] if records else 0
        rel_records = cypher_query(
            f"MATCH (n:`{label}`)-[r]->() RETURN count(r) AS c", book=book
        )
        rels = rel_records[0]["c"] if rel_records else 0
        return {
            "node_counts_by_label": {book: nodes},
            "total_nodes": nodes,
            "total_relationships": rels,
        }

    records = cypher_query(
        """
        MATCH (n) WITH labels(n) AS lbls, count(*) AS cnt
        UNWIND lbls AS label
        RETURN label, sum(cnt) AS count
        """
    )
    rel_count = cypher_query("MATCH ()-[r]->() RETURN count(r) AS c")
    rel_total = rel_count[0]["c"] if rel_count else 0

    nodes = {r["label"]: r["count"] for r in records}
    return {
        "node_counts_by_label": nodes,
        "total_nodes": sum(nodes.values()),
        "total_relationships": rel_total,
    }
