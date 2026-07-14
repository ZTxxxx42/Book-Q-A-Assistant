"""查询接口：基于已构建的 LightRAG + Neo4j 图谱。"""
from __future__ import annotations

from typing import Literal

from config import settings
from src.graph_builder import _build_rag

QueryMode = Literal["local", "global", "hybrid", "naive"]


async def ask(
    question: str,
    mode: QueryMode = "hybrid",
    stream: bool = False,
) -> str:
    """对图谱提问，返回答案文本。"""
    from lightrag import QueryParam

    rag = _build_rag()
    await rag.initialize_storages()
    param = QueryParam(mode=mode, stream=stream)

    if stream:
        result = ""
        async for chunk in await rag.aquery(question, param=param):
            result += chunk
        return result

    return await rag.aquery(question, param=param)  # type: ignore[return-value]


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
