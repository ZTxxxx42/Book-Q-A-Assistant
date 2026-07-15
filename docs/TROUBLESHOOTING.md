# Troubleshooting & Iteration Log

> 实时文档。记录 SiliconFlow + NanoVectorDB + Ollama 迁移过程中遇到的每个非平凡问题及解决方案。
> 最新在最上。每条结构：Symptom / Root cause / Fix / Files touched。

## Open risks / to-verify

- **SiliconFlow `/v1/rerank` 响应格式** ✅ 已验证：`documents` 接受纯字符串数组，返回 `results:[{index, document:null, relevance_score}]`，score 已在 [0,1]（实测 0.212 / 0.0000166），sigmoid 兜底未触发。
- **SiliconFlow `/v1/embeddings`** ✅ 已验证：1024 维，OpenAI 兼容。批量上限 `EMBEDDING_BATCH_NUM=32` 待 ingest 实测，若被拒（400）则下调。
- **GLM 429**：`LLM_MODEL_MAX_ASYNC=2` + 6 次 retry（2-60s）。15-chunk demo 应可控；若仍 429 降到 1。
- **本地 Qwen 答复稳定性**：query 角色单次答复几百 token，远短于引发崩溃的 8 分钟抽取；若 SSE 异常则改非流式或换 3b。
- **网络延迟**：每次 query 多 2 个 SiliconFlow RTT（~100-300ms），demo 可接受。

## Resolved（当前轮：阶段 4 每书独立工作区）

### 2026-07-15 — 多书合并图语义混乱 + 按书过滤 hack 脆弱
- **Symptom:** 阶段 2 把所有书塞进一个 LightRAG 实例，同名实体跨书合并、关系混淆；按书查询靠 `only_need_context` + 直接查 Qdrant payload 过滤 chunks，绕过 LightRAG 检索管线，rerank/上下文都要自己重做，且 `data.chunks` 在 only_need_context 下为空。
- **Root cause:** 一个实例装多书违背"一本书一张图"的自然语义；LightRAG 1.5.4 原生支持 `workspace` 参数隔离 Neo4j label / Qdrant workspace_id / KV 子目录，之前没用。
- **Fix:** `workspace = basename`。`graph_builder._build_rag(workspace)` 透传给 `LightRAG(workspace=...)`；`ingest_book` 设 `workspace=fname`。`query`/`graph_view`/`maintenance` 全部按 book 构建 workspace 实例，走原生 hybrid 检索；删除 `_retrieve_chunks_for_book` hack。`delete_document(book)` 删 label 节点 + workspace 向量 + KV 子目录。API/CLI `book` 必填，砍掉跨书查询。
- **验证:** alice 96 节点（label=alice_en.txt）、ocean 26 节点（label=ocean_tale.txt），全局 122=96+26；按书查询各自命中、引用各自出处。
- **Files:** `src/graph_builder.py`、`src/query.py`、`src/graph_view.py`、`src/maintenance.py`、`src/api.py`、`main.py`、`static/index.html`、`CLAUDE.md`。
- **报告:** `docs/reports/phase4-per-book-workspaces.md`。

## Resolved（当前轮：阶段 3 前端管理面板）

### 2026-07-15 — 前端缺管理面板，仅图谱+问答
- **Symptom:** `static/index.html` 只对接 /graph /graph/top /stats /chat，无文档管理、Cypher、按书过滤、引用展示、类型图例。
- **Fix:** 单文件增量重写——侧栏三 tab（概览/文档/Cypher）、文档管理（/documents + /ingest + /documents/upsert + refresh + delete）、Cypher 控制台、按书下拉（/chat 带 book）、references 事件解析展示、node_counts_by_label 图例过滤、节点 description、toast 替代 alert。`node --check` 通过。
- **Files:** `static/index.html`。
- **遗留:** 浏览器交互实测需用户确认；dup-failed 残留文档可见；实体编辑 UI 未加。

## Resolved（当前轮：阶段 2 多书 + 按书过滤）

### 2026-07-15 — only_need_context 拿不到结构化 chunks，按书过滤方案失败
- **Symptom:** 用 `QueryParam(only_need_context=True)` 取上下文后按 `data.chunks` 的 file_path 过滤，但 `data.chunks` 始终为空（0 项），过滤后无 chunk。
- **Root cause:** LightRAG 1.5.4 在 only_need_context（及正常模式）下把 chunks 合并进 `llm_response.content` 上下文字符串，`data.chunks` 不返回结构化项；`data.references` 仅在 chunk 通过 rerank 时才填充，且只有 reference_id/file_path 无 content。
- **Fix:** 绕过 LightRAG 检索，`_retrieve_chunks_for_book()` 直接 `qdrant_client.query_points` 查 chunks 集合 + `file_path` payload 过滤，再 rerank + Qwen 自生成。
- **Files:** `src/query.py`。

### 2026-07-15 — 同书内容变更被 filename dedup 拦下
- **Symptom:** 同 basename 再 ingest 不更新（LightRAG filename + content_hash 双重 dedup 判重跳过）。
- **Fix:** `maintenance.upsert_document()` 按 basename 查现有文档，存在则先 `delete_document` 再 `ingest_book`，包成 upsert 语义；API `POST /documents/upsert`。
- **Files:** `src/maintenance.py`、`src/api.py`。

## Resolved（当前轮：阶段 1 查询/检索质量优化）

### 2026-07-15 — /query 非流式 enable_rerank 与 /chat 不对称
- **Symptom:** `ask()`（/query + CLI）构造 `QueryParam` 未传 `enable_rerank`，读 LightRAG 自身 `RERANK_BY_DEFAULT` env，与项目 `.env` 的 `ENABLE_RERANK` 脱节；`ask_stream()`（/chat）则正确读 `settings.enable_rerank`。两者当前都 true 未暴露，但设置不一致。
- **Root cause:** 两个函数分别手写 `QueryParam`，非流式分支漏传。
- **Fix:** 抽 `_make_param()` 统一显式传 `enable_rerank=settings.enable_rerank` 及所有检索旋钮。
- **Files:** `src/query.py`。

### 2026-07-15 — rerank 形同不过滤（MIN_RERANK_SCORE=0.0）
- **Symptom:** rerank 开启但低相关 chunk 仍进上下文，无关问题（如 weather）会被拼凑答复。
- **Root cause:** LightRAG `MIN_RERANK_SCORE` 默认 0.0，rerank 只排序不设门槛。
- **Fix:** `.env` 设 `MIN_RERANK_SCORE=0.3`；实测无关问题返回 0 引用且拒答。
- **Files:** `.env`、`.env.example`、`config.py`（记录字段）。

### 2026-07-15 — 答复无出处（include_references 未开）
- **Symptom:** `/query` 只返回 answer，不知来源 chunk。
- **Root cause:** 项目用向后兼容的 `rag.aquery`（只返回 LLM 文本，丢弃 references）；`QueryParam.include_references` 未设。
- **Fix:** `ask()` 改用 `rag.aquery_llm` 取完整结果，提取 `data.references`；`QueryParam` 显式传 `include_references=True`；`QueryResponse` 加 `references` 字段；`/chat` SSE 增加 `references` 事件。
- **Files:** `src/query.py`、`src/api.py`、`main.py`、`config.py`、`.env`。

## Resolved（当前轮：LLM 分流）

### 2026-07-15 — Qwen-7b 本地实体抽取 8 分钟失控生成拖崩笔记本 GPU
- **Symptom:** `ingest alice_en.txt --max-chunks 15` 报 `TimeoutError: extract LLM func: Worker execution timeout after 480s`；Ollama 日志显示请求以 12.5 tok/s 持续 7m59s 后返回 500；`nvidia-smi` 报 `Unable to determine the device handle ... Unknown Error`（驱动级卡死）。
- **Root cause:** Qwen2.5-7b 在 LightRAG 实体抽取时未正确触发停止符，ramble 失控生成 ≈6000 token；30W 功耗限制的笔记本 GPU（RTX 4060 Laptop）长时间满载致驱动挂死。重启 `NVDisplayContainerLocalSystem` 服务后 GPU 恢复。
- **Fix:** LLM 按角色分流（LightRAG `role_llm_configs`）—— `extract`/`keyword` 走远程 GLM-4.7（重活不占本地 GPU），仅 `query`（最终答复，短流式）走本地 Ollama Qwen。
- **Files:** `config.py`（拆 `llm_*`=GLM + `query_llm_*`=Qwen）、`src/graph_builder.py`（`_make_llm_func` 通用工厂 + `_make_glm_func`/`_make_qwen_func` + `role_llm_configs={"query":...}`）、`.env`/`.env.example`。
- **验证:** 7/7 冒烟通过（含 GLM + Qwen 分别测）。

## Resolved（当前轮：迁移到 Qdrant）

### 2026-07-15 — Qdrant 容器启动 bind 6333 失败（Windows 保留端口段）
- **Symptom:** `docker compose up qdrant` 报 `ports are not available ... listen tcp 0.0.0.0:6333: bind: An attempt was made to access a socket in a way forbidden by its access permissions`。
- **Root cause:** Windows Hyper-V 动态端口保留段含 6248-6347，6333/6334 都在其中。`netsh interface ipv4 show excludedportrange protocol=tcp` 可查。
- **Fix:** compose 端口映射改为宿主 `16333:6333` / `16334:6334`；`QDRANT_URL=http://localhost:16333`；`config.py` 默认值同步。
- **Files:** `docker-compose.yml`、`.env.example`、`config.py`。

### 2026-07-15 — Ollama 默认端口 11434 同样在 Windows 保留段
- **Symptom:** `ollama serve` 报 `listen tcp 127.0.0.1:11434: bind: ... forbidden by its access permissions`。
- **Root cause:** 11434 落在保留段 11417-11516。
- **Fix:** 启动前设 `OLLAMA_HOST=127.0.0.1:21434`；`.env` `LLM_BASE_URL=http://localhost:21434/v1`。本机其他保留段避坑：6248-6847、8526-8625、10101-11616、28385/28390。
- **Files:** `.env`、`.env.example`、`CLAUDE.md`、`docs/LOCAL_LLM_SETUP.md`。

### 2026-07-15 — qdrant-client 连本地 Qdrant 返回 502 Bad Gateway
- **Symptom:** `QdrantClient(url='http://localhost:16333').get_collections()` 抛 `UnexpectedResponse: 502 Bad Gateway`，但 `curl http://localhost:16333/collections` 正常返回 200。
- **Root cause:** 本机运行 FlClash（Clash 系统代理）；`qdrant-client` 底层 `requests` 在 Windows 上读注册表系统代理设置，把 `localhost:16333` 也路由进代理，代理回 502。curl 对 localhost 默认绕过故正常。LightRAG 内部构造 `QdrantClient` 不传 `trust_env=False`，同样受影响。
- **Fix:** `config.py` 导入时强制 `NO_PROXY` / `no_proxy` 包含 `localhost,127.0.0.1`（覆盖所有入口：api/CLI/ingest/query），LightRAG 内部 client 随之绕过代理。
- **Files:** `config.py`。
- **验证:** smoke `test_qdrant` 在不手动设 NO_PROXY 的情况下通过。

### 2026-07-15 — qdrant-client 与 server 版本差过大告警
- **Symptom:** `UserWarning: Qdrant client version 1.18.0 is incompatible with server version 1.12.4. Minor version difference must not exceed 1.`
- **Root cause:** pip 默认装最新 client 1.18.0，Docker 镜像是 v1.12.4。
- **Fix:** `requirements.txt` pin `qdrant-client>=1.12.0,<1.13.0` 与服务端 1.12.x 对齐。
- **Files:** `requirements.txt`。

### 2026-07-15 — 确认 Qdrant cosine 语义（决定 COSINE_THRESHOLD）
- **结论:** Qdrant `Distance.COSINE` 返回**真余弦相似度**（同向 1.0、正交 0.0，范围 [-1,1]），`score_threshold` 保留 `score >= threshold`。与 NanoVectorDB 一致。
- **决策:** `COSINE_THRESHOLD=0.2` 无需调整。milvus-lite 的 `1.0` hack 彻底作废。
- **验证:** smoke `test_qdrant` 实测。

## Resolved（历史，本地 bge + milvus 栈时期）

### 2026-07-15 — bge-m3 GPU fp16 → milvus 搜索 type 102 崩溃
- **Symptom:** `/query`、`/chat` 500，`MilvusException: PlaceholderValue type 102 not supported`。
- **Root cause:** bge-m3 `use_fp16=True` 返回 float16 dense_vecs；`np.array(dense)` 保留 float16，pymilvus 据此推断 FLOAT16_VECTOR(102) placeholder，milvus-lite 不支持。
- **Fix:** 强转 `np.asarray(dense, dtype=np.float32)`。
- **Files:** `src/local_models.py`（已删除，迁移后不再适用）。
- **迁移后状态:** 整条本地 bge 栈移除，问题不复存在。

### 2026-07-15 — 8 worker 并发加载模型 → meta tensor
- **Symptom:** ingest 时 `Cannot copy out of meta tensor; no data!`。
- **Root cause:** LightRAG 8 个 embedding worker 并发调 `from_pretrained`，互相干扰致权重未物化。`asyncio.Lock` 定义了但未使用。
- **Fix:** `threading.Lock` 双重检查锁保护 `_get_bge_m3`/`_get_reranker`。
- **Files:** `src/local_models.py`（已删除）。
- **迁移后状态:** 本地模型移除，无加载并发问题。

### 2026-07-15 — 8 worker 并发推理 → 段错误 exit 139
- **Symptom:** ingest flush 阶段进程崩溃 exit 139。
- **Root cause:** 8 worker 并发 `model.encode`/`CrossEncoder.predict` 非线程安全 + 单卡 GPU 并发。
- **Fix:** 推理锁串行化 encode/predict。
- **Files:** `src/local_models.py`（已删除）。
- **迁移后状态:** 推理走 API，无本地并发。

### 2026-07-15 — GLM 429 限流
- **Symptom:** ingest `RateLimitError 429 您的账户已达到速率限制`。
- **Root cause:** 远程 GLM 限流 + `llm_model_max_async=4` 并发过高。
- **Fix:** `llm_model_max_async` 4→1，tenacity 重试 4→6 次、max wait 60s。
- **Files:** `src/graph_builder.py`。
- **迁移后状态:** LLM 换本地 Ollama，无限流；并发保守设 1。

### 2026-07-15 — COSINE_THRESHOLD 误判（milvus-lite 语义）
- **Symptom:** `/query` 返回 `[no-context]`。
- **Root cause:** milvus-lite COSINE 返回 `distance=1-sim`（越小越相似），range search 语义 `distance < radius`（radius 是上界，L2 风格，与标准 Milvus server 相反）。`1.0` 正确（distance<1.0 保留正相似度），中途误改 0.0（distance<0 永空）造成回归。
- **Fix:** 回退 `.env` `COSINE_THRESHOLD=1.0`，修正 CLAUDE.md 说明。
- **Files:** `.env`、`CLAUDE.md`。
- **迁移后状态:** 换 NanoVectorDB（真 cosine，`scores >= threshold`），`COSINE_THRESHOLD=0.2`，milvus 语义陷阱彻底消失。

## Environment notes
- Python: `D:/miniforge/envs/my_env`（3.12）
- LightRAG: 1.5.4；Qdrant: server v1.12.4 / client 1.12.2
- GPU: RTX 4060 Laptop 8GB（bge/rerank 已走 API，GPU 仅 Ollama Qwen 使用）
- Ollama: 0.31.2；模型: qwen2.5:7b-instruct
- Embedding/Rerank: SiliconFlow `https://api.siliconflow.cn/v1`
