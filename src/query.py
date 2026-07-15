"""查询接口：基于已构建的 LightRAG + Neo4j 图谱（每书独立 workspace）。"""
from __future__ import annotations

from typing import Any, AsyncIterator, Literal

from config import settings
from src.rag_pool import get_pool

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


def _hard_refuse() -> dict[str, Any]:
    """硬拒答：无关问题，不调 LLM。"""
    return {"answer": settings.hard_refuse_answer, "references": []}


async def _probe_hits(rag: Any, question: str, mode: QueryMode) -> list:
    """预探测（aquery_data, only_need_context=True，不调 LLM）取 references。

    用于硬拒答：命中数不足则不调答复 LLM，省下 Ollama 串行槽位。
    """
    probe_param = _make_param(mode, stream=False)
    data_result = await rag.aquery_data(question, param=probe_param)
    return data_result.get("data", {}).get("references", []) or []


async def ask(
    question: str,
    book: str,
    mode: QueryMode = "hybrid",
    stream: bool = False,
) -> dict[str, Any]:
    """对指定书的图谱提问，返回 ``{"answer": str, "references": list}``。

    ``book`` 为文件 basename，同时是 workspace 名 → 仅检索该书的图+向量+chunk。

    硬拒答双保险：
    1. 预探测（aquery_data, 不调 LLM）：references 命中数 < min_hit_count → 直接拒答。
    2. 事后兜底：aquery_llm 返回后 references 仍空 → 丢弃 LLM 答复，改返拒答文案。
    """
    pool = await get_pool()
    rag = await pool.acquire(book)
    try:
        # 1) 预探测
        refs = await _probe_hits(rag, question, mode)
        if len(refs) < settings.min_hit_count:
            return _hard_refuse()

        # 2) 检索 + 生成
        param = _make_param(mode, stream=False)
        result = await rag.aquery_llm(question, param=param)
        final_refs = result.get("data", {}).get("references", []) or []
        # 事后兜底：生成阶段被 rerank 全过滤
        if len(final_refs) < settings.min_hit_count:
            return _hard_refuse()
        llm_resp = result.get("llm_response", {})
        content = llm_resp.get("content", "") or ""
        if not content:
            return _hard_refuse()
        return {"answer": content, "references": final_refs}
    finally:
        await pool.release(book)


async def ask_stream(
    question: str,
    book: str,
    mode: QueryMode = "hybrid",
    history: list[dict] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """流式问答生成器（限定书）：yield 事件 dict。事件序列：references → token…。

    硬拒答：预探测命中不足 → yield 单个 refuse 事件，不调 LLM。
    """
    pool = await get_pool()
    rag = await pool.acquire(book)
    try:
        refs = await _probe_hits(rag, question, mode)
        if len(refs) < settings.min_hit_count:
            yield {"type": "refuse", "content": settings.hard_refuse_answer}
            return

        param = _make_param(mode, stream=True, history=history)
        result = await rag.aquery_llm(question, param=param)
        final_refs = result.get("data", {}).get("references", []) or []
        if len(final_refs) < settings.min_hit_count:
            yield {"type": "refuse", "content": settings.hard_refuse_answer}
            return
        if final_refs:
            yield {"type": "references", "content": final_refs}

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
            else:
                yield {"type": "refuse", "content": settings.hard_refuse_answer}
    finally:
        await pool.release(book)


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
