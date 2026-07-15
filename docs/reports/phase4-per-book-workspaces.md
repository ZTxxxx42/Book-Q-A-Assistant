# 阶段 4 改进报告：每书独立知识图谱（LightRAG workspace 隔离）

> 日期：2026-07-15
> 上一版：阶段 2 的"按书过滤"用 chunk file_path 过滤 hack 实现；本阶段将其推翻，改为 LightRAG 原生 workspace 隔离，并砍掉跨书查询。

## 背景

阶段 2 让多书能共存，但"按书查询"是 hack：把所有书塞进一个 LightRAG 实例（共享图），查询时用 `only_need_context` 取 chunks 再按 `file_path` 过滤。问题：

1. **图是合并的**——两本书的同名实体（如 "Alice"）会被 LightRAG 合并成一个节点，跨书关系混乱，违背"一本书一张知识图谱"的自然语义。
2. **过滤 hack 脆弱**——`data.chunks` 在 `only_need_context` 下为空（见 TROUBLESHOOTING），不得不直接查 Qdrant payload，绕过 LightRAG 检索管线，rerank/上下文组装都要自己重做。
3. **删除不干净**——删一本书要按 doc_id 清 Qdrant 点 + 重建共享实体，复杂且易残留。

用户明确："这是 2 本书，所以要两张独立的知识图谱"，"直接砍掉"（跨书查询）。

## 改动

### 核心思路
`workspace = 书籍 basename`。LightRAG 1.5.4 的 `workspace` 参数原生隔离三处存储：

| 存储 | 隔离方式 |
|------|---------|
| Neo4j | 节点 label = workspace 名（`alice_en.txt`、`ocean_tale.txt` 各一个 label） |
| Qdrant | 向量 payload 带 `workspace_id`，检索按 workspace 过滤 |
| KV (JSON) | `working_dir/<workspace>/kv_store_*.json` 子目录 |

每本书一个独立的 LightRAG 实例（按需构建），天然一张图，无需 hack。

### 代码
- **`src/graph_builder.py`**：`_build_rag(workspace="")` 把 `workspace` 透传给 `LightRAG(workspace=...)`；`ingest_book()` 用 `workspace=fname`（basename）。其余 embed/rerank/role_llm_configs 不变。
- **`src/query.py`**：`ask`/`ask_stream` 必传 `book`，用它构建对应 workspace 的 LightRAG 实例检索——原生 hybrid，不再需要 `_retrieve_chunks_for_book` hack（已删除）。`cypher_query`/`graph_stats` 接受 `book`，用反引号转义的 label 查询 `MATCH (n:\`<book>\`)`。
- **`src/graph_view.py`**：`_label(book)` 反引号转义；`get_subgraph`/`get_top_entities` 按 `MATCH (n:\`<book>\`)-[r*1..{depth}]-...` 限定单书。
- **`src/maintenance.py`**：`list_documents()` 同步迭代 `working_dir` 子目录读 `kv_store_doc_status.json`；`delete_document(book)` 读 doc_id → `adelete_by_doc_id` → `MATCH (n:\`<book>\`) DETACH DELETE n` → `shutil.rmtree(子目录)`，干净利落。`_rag(workspace)` 按书构建。
- **`src/api.py` / `main.py`**：`QueryRequest`/`ChatRequest`/`GraphRequest`/`TopEntitiesRequest` 中 `book` 必填；CLI `query --book`、`stats --book`。
- **`static/index.html`**：标题区 `#book-select` 下拉驱动一切；`onBookChange()` 清图 + 重载 stats/top；问答/图谱/统计均带 `book`；`loadDocs` 填充下拉并自动选第一本。移除"不限书籍"选项。
- **`CLAUDE.md`**：架构说明改为每书独立工作区；`docs/TROUBLESHOOTING.md` 加条目。

## 验证

```
alice_en.txt   ingest 15 chunks → 96 节点 / 106 关系（label=alice_en.txt）
ocean_tale.txt ingest 1 chunk   → 26 节点 / 26 关系（label=ocean_tale.txt）
全局 stats     → 122 节点 / 132 关系（96 + 26，两个 label 独立计数）
```

按书查询：

| 问题 | book | 结果 |
|------|------|------|
| Who is Dinah? | alice_en.txt | ✅ "Alice's cat"，引用 alice_en.txt (1) |
| Who is the captain? | ocean_tale.txt | ✅ "Captain Marlow of the Sea Star"，引用 ocean_tale.txt |

两书互不干扰：alice 的查询只命中 alice 的实体/关系/chunks，反之亦然。Neo4j label、Qdrant workspace_id、KV 子目录三层全部按书隔离。

## 改前 vs 改后

| 维度 | 改前（阶段 2 hack） | 改后（workspace 隔离） |
|------|---------------------|------------------------|
| 图结构 | 两书合并一张图，同名实体混淆 | 每书独立一张图 |
| 按书查询 | `only_need_context` + 手动 Qdrant payload 过滤 | LightRAG 原生 hybrid，workspace 自动过滤 |
| 删除一本书 | 按 doc_id 清点 + 重建共享实体 | 删 label + workspace 点 + KV 子目录，无副作用 |
| 跨书查询 | 支持（但语义混乱） | 不支持（by design，砍掉） |
| 代码复杂度 | `_retrieve_chunks_for_book` 绕过管线 | 删除 hack，走原生管线 |

## 遗留与下一步
- **小书实体偏多**：ocean_tale.txt 仅 1094 字 / 1 chunk 抽出 26 实体，GLM 抽取偏激进；后续可调抽取 prompt 或合并小实体。
- **API 重启**：若 API 在改代码前已运行，需重启加载 workspace 代码（端口 8010）。
- **前端实测**：`node --check` 通过，浏览器交互需用户确认按书下拉切换图谱/问答正常。
