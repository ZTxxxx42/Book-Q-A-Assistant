# 阶段 2 改进报告：多书 + 按书过滤查询

> 日期：2026-07-15
> 范围：支持"只在某本书里查"+ 同书内容变更 upsert + 多书答复带出处。

## 改动摘要

| 文件 | 改动 |
|------|------|
| `src/query.py` | `ask()`/`ask_stream()` 加 `book` 参数；新增 `_retrieve_chunks_for_book()`（直查 Qdrant chunks 集合 + file_path payload 过滤 + rerank）与 `_book_answer_prompt()`（自生成 prompt，指示仅用所给上下文、无关则拒答） |
| `src/maintenance.py` | 新增 `upsert_document(file, max_chunks)`：按 basename 查现有文档，存在则先 delete 再 ingest，不存在直接 ingest |
| `src/api.py` | `QueryRequest`/`ChatRequest` 加 `book` 字段；新增 `POST /documents/upsert` 端点 + `UpsertRequest` |
| `main.py` | `query` 命令加 `--book` 选项 |

## 关键设计决策：按书过滤的实现路径

**最初方案（失败）**：用 `QueryParam(only_need_context=True)` 取上下文后按 chunk 的 file_path 过滤。实测发现 LightRAG 1.5.4 在 `only_need_context=True`（乃至正常模式）下把 chunks 压进 `llm_response.content` 上下文字符串，`data.chunks` 返回空——拿不到带 file_path 的结构化 chunk，无法过滤。

**最终方案**：绕过 LightRAG 检索，**直接查 Qdrant chunks collection**：
1. 用 `rag.embedding_func` 把问题向量化。
2. `qdrant_client.query_points` 查 chunks 集合，`query_filter` 加 `file_path == book` 的 payload 过滤，取 top_k。
3. 用 `rag.rerank_model_func` 重排 + `min_rerank_score` 门槛（全不过时兜底 top3，避免空上下文）。
4. 用 `_make_qwen_func()`（query 角色 Qwen）+ 自定义 prompt 生成答复，prompt 明确"仅依据所给上下文，无信息则说明"。

chunks collection 的 payload 字段：`id / workspace_id / created_at / content / full_doc_id / file_path`（`file_path` 即 basename，与 ingest 时 `file_paths=[fname]` 一致）。

## 验证结果

导入第二本书 `ocean_tale.txt`（1094 字符 / 1 chunk），图谱从 91→118 实体。按书过滤测试矩阵：

| 问题 | book=alice | book=ocean | book=不限定 |
|------|-----------|-----------|------------|
| Who is the captain? | 拒答（不在 Alice）✅ | Captain Marlow ✅ | Captain Marlow (Sea Star) ✅ |
| Who is Dinah? | Alice 的猫 ✅ | 拒答（不在 Ocean）✅ | Alice 的猫 ✅ |

- `/chat` SSE 带 `book`：先发 ocean_tale 的 references 事件，再流式输出 "Elena"。✅
- `POST /documents/upsert` 同名重导：deleted=True，reingested 1 chunk。✅（先删旧 doc 再导，绕过 LightRAG filename dedup）

## 遗留问题与下一步

1. **rerank 阈值兜底**：当某书所有 chunk 的 rerank 分数都 < 0.3（问题与该书无关），代码兜底保留 top3 喂给 Qwen，靠 prompt 拒答。实测拒答正确，但 refs 仍显示这 3 个 chunk（"看过但无信息"）。可接受；若要更干净可在兜底时返回空 + "No relevant context"。
2. **跨书共享实体**：两本不同书里同名实体会被 LightRAG 合并为一个 Neo4j 节点。删除其中一本时该实体被 rebuild（用剩余 source chunks）。本例两书无同名实体，未触发。
3. **按书过滤不走 LightRAG 图检索**：当前仅向量召回 + rerank，不利用实体/关系图。对"某书内"的复杂关系问题可能不如 hybrid 全库。属预期取舍（按书隔离优先）。
4. **`doc_status` 残留 dup 记录**：早期重复 ingest alice 留下一条 status=failed 的 `dup-` 记录，无害但列表里可见。后续可过滤展示。

## 下一步

进入阶段 3（前端管理面板）：把 `/documents`、`/documents/upsert`、`/ingest`、`/cypher`、按书过滤 UI、references 展示对接到 `static/index.html`。
