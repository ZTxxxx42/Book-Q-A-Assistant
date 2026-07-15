"""查询接口：基于已构建的 LightRAG + Neo4j 图谱（每书独立 workspace）。"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Literal

from config import settings
from src.rag_pool import get_pool

logger = logging.getLogger("book_kg.query")

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


# ---------- A3：query 改写 / 分解（GLM，走 llm:extract 全局组，不占 Ollama query 槽）----------

_rewrite_llm = None


async def _get_rewrite_llm():
    """惰性构造 GLM 改写 LLM，用 priority_limit_async_func_call 接入 llm:extract 全局并发组。"""
    global _rewrite_llm
    if _rewrite_llm is None:
        from lightrag.utils import priority_limit_async_func_call

        from src.graph_builder import _make_glm_func

        limiter = priority_limit_async_func_call(
            max_size=settings.global_extract_concurrency,
            concurrency_group="llm:extract",
        )
        _rewrite_llm = limiter(_make_glm_func())
    return _rewrite_llm


async def _rewrite_query(question: str, history: list[dict] | None) -> str:
    """用 GLM 做指代消解 + 历史压缩：把带历史的追问改写成独立完整查询。

    仅当 history 非空时触发；任何失败回退原 question（绝不阻塞主流程）。
    """
    if not history:
        return question
    glm = await _get_rewrite_llm()
    sys_prompt = (
        "你是查询改写助手。根据对话历史，把用户最新问题改写成一个独立、完整、无指代代词的"
        "检索查询。只输出改写后的查询，不要解释、不要引号。"
    )
    hist_text = "\n".join(
        f"{m.get('role','user')}: {m.get('content','')}" for m in history[-6:]
    )
    prompt = f"对话历史:\n{hist_text}\n\n最新问题: {question}\n\n改写后的独立查询:"
    try:
        rewritten = await glm(prompt, system_prompt=sys_prompt)
        rewritten = (rewritten or "").strip().strip('"').strip("'").strip()
        if rewritten and 1 <= len(rewritten) <= settings.max_question_length:
            logger.info("query 改写: %r -> %r", question, rewritten)
            return rewritten
    except Exception as e:
        logger.warning("query 改写失败，回退原问题: %r", e)
    return question


def _extract_json_array(text: str) -> list | None:
    """从 LLM 输出里稳健提取 JSON 数组。

    GLM 常把 JSON 包在 ```json ... ``` 代码块里，或前后带解释文字。
    依次尝试：剥 markdown fence → 直接解析 → 正则提取首个 [...]。
    """
    import re

    if not text:
        return None
    s = text.strip()
    # 1) 剥 markdown 代码块围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    # 2) 直接解析
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except json.JSONDecodeError:
        pass
    # 3) 正则提取首个 [...]（跨行）
    m = re.search(r"\[.*?\]", s, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    return None


async def _decompose_query(question: str) -> str:
    """用 GLM 把复杂问题拆成 2-4 个子问题并合并为一条扩展查询。

    任何失败回退原 question。
    """
    glm = await _get_rewrite_llm()
    sys_prompt = (
        "你是查询分解助手。把用户问题拆成 2-4 个独立的子检索问题，覆盖问题的不同方面，"
        '用于知识图谱检索。输出 JSON 字符串数组，如 ["子问题1","子问题2"]。只输出 JSON。'
    )
    prompt = f"问题: {question}\n\n子问题(JSON 数组):"
    try:
        out = await glm(prompt, system_prompt=sys_prompt)
        subs = _extract_json_array(out)
        if subs:
            merged = " ; ".join(str(s).strip() for s in subs[:4] if str(s).strip())
            if merged:
                logger.info("query 分解: %r -> %r", question, merged)
                return merged
    except Exception as e:
        logger.warning("query 分解失败，回退原问题: %r", e)
    return question


async def _prepare_question(
    question: str, history: list[dict] | None, decompose: bool
) -> tuple[str, str]:
    """改写 + 分解的统一入口：先改写（消解指代），再分解（拆子问题）。

    返回 ``(probe_q, retrieve_q)``：
    - ``probe_q``：改写后的**单查询**，作为硬拒答预探测的信号。分解会把多个子问题
      用 `` ; `` 拼成一串，其 embedding 被稀释、naive 探测容易召回为 0，不适合做
      "是否有相关 chunk"的判据，故探测用分解前的改写查询。
    - ``retrieve_q``：最终送 ``aquery_llm`` 的查询（含分解拼串），让生成阶段检索
      覆盖多个子方面。
    """
    q = question
    if history:
        q = await _rewrite_query(q, history)
    probe_q = q
    if decompose:
        q = await _decompose_query(q)
    return probe_q, q


def _hard_refuse() -> dict[str, Any]:
    """硬拒答：无关问题，不调 LLM。"""
    return {"answer": settings.hard_refuse_answer, "references": []}


async def _probe_hits(rag: Any, question: str, mode: QueryMode) -> list:
    """预探测（aquery_data, only_need_context=True，不调答复 LLM）取 references。

    用于硬拒答：命中数不足则不调答复 LLM，省下 Ollama 串行槽位。

    用请求的 ``mode`` 探测（而非强制 naive）：naive 纯向量召回太弱，对 hybrid 借图
    谱可答的问题常召回为 0 → 误拒答。hybrid 探测会触发一次 GLM 关键词抽取，但走
    ``llm:extract`` 全局组（上限 2 + 重试），可接受；换回准确信号。
    """
    probe_param = _make_param(mode, stream=False)
    data_result = await rag.aquery_data(question, param=probe_param)
    return data_result.get("data", {}).get("references", []) or []


async def _pick_gen_query(
    rag: Any, probe_q: str, retrieve_q: str, mode: QueryMode
) -> tuple[str, list]:
    """选用于生成的查询：对 probe_q 与 retrieve_q 各做一次预探测，取命中数多者。

    分解拼串（``retrieve_q``）embedding 被稀释，常比原始查询召回更差；若直接用它
    生成会把好查询搞坏。这里让分解"只能帮忙不能添乱"——命中更多才用拼串，否则回退
    到分解前的单查询。``decompose=False`` 时两者相同，仅探测一次，零额外开销。
    平局时偏向 probe_q（更贴近用户原意）。
    """
    refs_probe = await _probe_hits(rag, probe_q, mode)
    if retrieve_q == probe_q:
        return probe_q, refs_probe
    refs_retr = await _probe_hits(rag, retrieve_q, mode)
    if len(refs_retr) > len(refs_probe):
        return retrieve_q, refs_retr
    return probe_q, refs_probe


async def ask(
    question: str,
    book: str,
    mode: QueryMode = "hybrid",
    stream: bool = False,
    history: list[dict] | None = None,
    decompose: bool = False,
) -> dict[str, Any]:
    """对指定书的图谱提问，返回 ``{"answer": str, "references": list}``。

    ``book`` 为文件 basename，同时是 workspace 名 → 仅检索该书的图+向量+chunk。

    硬拒答双保险：
    1. 预探测（aquery_data, 不调 LLM）：references 命中数 < min_hit_count → 直接拒答。
    2. 事后兜底：aquery_llm 返回后 references 仍空 → 丢弃 LLM 答复，改返拒答文案。

    A3：``history`` 非空时先用 GLM 改写（指代消解）；``decompose=True`` 时先拆子问题。
    预探测 probe_q 与 retrieve_q 取命中多者用于生成（``_pick_gen_query``），分解只能
    帮忙不能添乱。
    """
    probe_q, retrieve_q = await _prepare_question(question, history, decompose)
    pool = await get_pool()
    rag = await pool.acquire(book)
    try:
        # 1) 预探测 + 选生成查询（分解拼串命中更多才用，否则回退单查询）
        gen_q, refs = await _pick_gen_query(rag, probe_q, retrieve_q, mode)
        if len(refs) < settings.min_hit_count:
            return _hard_refuse()

        # 2) 检索 + 生成
        param = _make_param(mode, stream=False)
        result = await rag.aquery_llm(gen_q, param=param)
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
    decompose: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """流式问答生成器（限定书）：yield 事件 dict。事件序列：queue? → references → token…。

    硬拒答：预探测命中不足 → yield 单个 refuse 事件，不调 LLM。
    A3：``history`` 非空时先用 GLM 改写；``decompose=True`` 时先拆子问题。
    预探测 probe_q 与 retrieve_q 取命中多者用于生成（``_pick_gen_query``）。
    B4：``ahead>0`` 时先 yield queue 事件告知排队位置。
    """
    probe_q, retrieve_q = await _prepare_question(question, history, decompose)
    pool = await get_pool()
    rag = await pool.acquire(book)
    try:
        gen_q, refs = await _pick_gen_query(rag, probe_q, retrieve_q, mode)
        if len(refs) < settings.min_hit_count:
            yield {"type": "refuse", "content": settings.hard_refuse_answer}
            return

        param = _make_param(mode, stream=True, history=history)
        result = await rag.aquery_llm(gen_q, param=param)
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
