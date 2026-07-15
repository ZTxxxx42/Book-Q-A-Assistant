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


async def _retrieve_chunks_for_book(
    question: str, mode: QueryMode, book: str
) -> tuple[list[dict], list[dict]]:
    """直接查 Qdrant chunks collection，按 file_path payload 过滤到指定书。

    LightRAG QueryParam 无 doc_id/file_path 过滤字段，且 aquery_llm 把 chunks 压进
    上下文字符串、data.chunks 不返回结构化项，故绕过 LightRAG 检索，直接用 embedding
    查 Qdrant chunks 集合 + file_path 过滤 + rerank。返回 (chunks, references)。
    """
    from qdrant_client import models

    rag = _build_rag()
    await rag.initialize_storages()

    # 1) 问题向量化（用 LightRAG 的 embedding_func）
    emb = await rag.embedding_func([question], context="query")
    query_vec = emb[0].tolist()

    # 2) 直查 chunks collection，加 file_path 过滤
    vdb = rag.chunks_vdb
    res = vdb._client.query_points(
        collection_name=vdb.final_namespace,
        query=query_vec,
        limit=max(settings.chunk_top_k, 10),
        with_payload=True,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_path", match=models.MatchValue(value=book)
                )
            ]
        ),
    ).points
    chunks = [
        {
            "content": p.payload.get("content", ""),
            "file_path": p.payload.get("file_path", ""),
            "reference_id": str(p.id),
        }
        for p in res
    ]

    # 3) rerank + 分数门槛
    if settings.enable_rerank and chunks:
        ranked = await rag.rerank_model_func(
            query=question,
            documents=[c["content"] for c in chunks],
            top_n=len(chunks),
        )
        kept = [r for r in ranked if r["relevance_score"] >= settings.min_rerank_score]
        if not kept and ranked:
            kept = ranked[:3]  # 兜底：阈值全不过时保留 top3，避免空上下文
        chunks = [chunks[r["index"]] for r in kept]

    refs = [
        {"reference_id": c["reference_id"], "file_path": c["file_path"]} for c in chunks
    ]
    return chunks, refs


def _book_answer_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    """构造按书过滤后的生成 prompt，返回 (system_prompt, user_prompt)。"""
    ctx = "\n\n".join(
        f"[Chunk {i+1}] {c.get('content', '')}" for i, c in enumerate(chunks)
    )
    system = (
        "You are a helpful assistant. Answer the user's question based ONLY on the "
        "provided context chunks from the same book. If the context does not contain "
        "the answer, say you don't have enough information. Keep it concise."
    )
    user = f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    return system, user


async def ask(
    question: str,
    mode: QueryMode = "hybrid",
    stream: bool = False,
    book: str | None = None,
) -> dict[str, Any]:
    """对图谱提问，返回 ``{"answer": str, "references": list}``。

    ``book`` 给定文件名时，仅用该书 chunk 生成答复（跨全库检索后按 file_path 过滤）。
    """
    if book:
        from src.graph_builder import _make_qwen_func

        chunks, refs = await _retrieve_chunks_for_book(question, mode, book)
        if not chunks:
            return {"answer": f"No relevant context found in '{book}'.", "references": []}
        system, user = _book_answer_prompt(question, chunks)
        qwen = _make_qwen_func()
        answer = await qwen(user, system_prompt=system)
        return {"answer": answer or "", "references": refs}

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
    book: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """流式问答生成器：yield 事件 dict，并带入多轮对话历史。

    ``book`` 给定时仅用该书 chunk 流式生成。事件序列：references → token…。
    """
    if book:
        from src.graph_builder import _make_qwen_func

        chunks, refs = await _retrieve_chunks_for_book(question, mode, book)
        if refs:
            yield {"type": "references", "content": refs}
        if not chunks:
            yield {"type": "token", "content": f"No relevant context found in '{book}'."}
            return
        system, user = _book_answer_prompt(question, chunks)
        qwen = _make_qwen_func()
        result = await qwen(user, system_prompt=system, stream=True)
        if isinstance(result, str):
            if result:
                yield {"type": "token", "content": result}
            return
        async for chunk in result:
            if chunk:
                yield {"type": "token", "content": chunk}
        return

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
