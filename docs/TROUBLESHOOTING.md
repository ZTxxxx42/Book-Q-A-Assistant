# Troubleshooting & Iteration Log

> 实时文档。记录 SiliconFlow + NanoVectorDB + Ollama 迁移过程中遇到的每个非平凡问题及解决方案。
> 最新在最上。每条结构：Symptom / Root cause / Fix / Files touched。

## Open risks / to-verify

- **SiliconFlow `/v1/rerank` 响应格式** ✅ 已验证：`documents` 接受纯字符串数组，返回 `results:[{index, document:null, relevance_score}]`，score 已在 [0,1]（实测 0.212 / 0.0000166），sigmoid 兜底未触发。
- **SiliconFlow `/v1/embeddings`** ✅ 已验证：1024 维，OpenAI 兼容。批量上限 `EMBEDDING_BATCH_NUM=32` 待 ingest 实测，若被拒（400）则下调。
- **Ollama 并发**：`LLM_MODEL_MAX_ASYNC=1` 先跑通；调 2 需 `OLLAMA_NUM_PARALLEL=2` 且 8GB 显存够 2 路 Qwen-7B 上下文。
- **网络延迟**：每次 query 多 2 个 SiliconFlow RTT（~100-300ms），demo 可接受。

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
