# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目是什么

书籍 → 知识图谱：导入一整本书（PDF/TXT/EPUB/MD/DOCX），用 **LightRAG** 借 LLM 抽取实体/关系，图谱存 **Neo4j**、向量存 **Qdrant**（Docker 容器），对外提供混合检索问答 API（FastAPI + SSE 流式）。**每本书是一个独立知识图谱** —— LightRAG `workspace` = 书的 basename，借此隔离 Neo4j label、Qdrant `workspace_id` 与 KV 子目录。不做跨书合并，不支持跨书查询。**SiliconFlow** API 提供 bge-m3 嵌入 + bge-reranker-v2-m3 重排。LLM **按角色拆分**（LightRAG `role_llm_configs`）：实体/关键词**抽取** → 远程 **GLM-4.7**（重活，让本地 GPU 空出来）；最终**答复生成** → 本地 **Ollama** Qwen2.5-7B-Instruct（OpenAI 兼容，只做短答复流式生成）。

> 为什么按书隔离 workspace："书籍知识图谱"天然是一书一图；合并图谱会混淆实体边界。`workspace=basename` 给出真隔离（独立 Neo4j label、干净的单书删除、原生单书混合检索，而非 chunk-filter 取巧）。跨书查询作为超范围功能已弃用。

> 迁移说明：技术栈此前用过 本地 bge-m3/reranker + milvus-lite + 远程 GLM，再 NanoVectorDB，再 合并 Qdrant 图 + chunk-filter 单书取巧。现为：API 嵌入/重排 + Qdrant + GLM 抽取 / Ollama 答复 + 单书 workspace。完整迭代史见 `docs/TROUBLESHOOTING.md`。

## 命令

所有命令在 `book_knowledge_graph/` 下执行，用 miniforge `my_env` 解释器：

```bash
# 前置：Ollama 运行中 + 已 pull Qwen，Neo4j + Qdrant 已起
# 容器路径（推荐，GPU 直通 + 自动拉 Qwen）：
docker compose up -d               # Neo4j + Qdrant + Ollama（ollama-pull 自动拉 qwen2.5:7b-instruct）+ app
# —— 以下为裸机路径（不用容器 Ollama 时）——
ollama pull qwen2.5:7b-instruct
# Ollama 默认端口 11434 落在 Windows 保留端口段 → 改用 21434
$env:OLLAMA_HOST="127.0.0.1:21434"   # PowerShell
ollama serve                       # 保持运行（OpenAI API 在 http://localhost:21434/v1）
docker compose up -d neo4j qdrant  # 仅起 Neo4j + Qdrant（裸机 Ollama 不入容器）

# FastAPI 服务（端口 8010，不是 8000 —— 8000 在本机被"幽灵占用"）
D:/miniforge/envs/my_env/python.exe -m uvicorn src.api:app --port 8010 --host 127.0.0.1
# 或：D:/miniforge/envs/my_env/python.exe serve.py（带统一日志 dictConfig）

# CLI
D:/miniforge/envs/my_env/python.exe main.py ingest --file alice_en.txt [--max-chunks N]
D:/miniforge/envs/my_env/python.exe main.py query  --question "..." --mode hybrid --book alice_en.txt
D:/miniforge/envs/my_env/python.exe main.py stats [--book alice_en.txt]
D:/miniforge/envs/my_env/python.exe main.py cypher -c "MATCH (n) RETURN labels(n), count(*)"

# 脚本
D:/miniforge/envs/my_env/python.exe scripts/smoke_remote_models.py  # 冒烟：SiliconFlow 嵌入/重排 + Qdrant + Ollama + LightRAG 构建
D:/miniforge/envs/my_env/python.exe scripts/check_model.py           # 验证 .env 的 LLM 端点可达
D:/miniforge/envs/my_env/python.exe scripts/e2e_test.py              # 导入 alice_en.txt（max 15 chunks）+ 流式查询
```

**pytest 套件**（会话管理 + 纯逻辑层，不依赖外部服务）：
```bash
D:/miniforge/envs/my_env/python.exe -m pytest -q          # tests/ 下全部用例
```
依赖：`pip install -r requirements-dev.txt`（pytest + pytest-asyncio）。`asyncio_mode=auto`（见 `pytest.ini`）。`tests/test_session_routes.py` 用 `TestClient(app)` 且不进入 lifespan，故无需 Neo4j/Qdrant/Ollama。端到端（触图/LLM）仍靠冒烟脚本 + 直连运行中的 API：
```bash
curl http://localhost:8010/health
curl -N -X POST http://localhost:8010/chat -H "Content-Type: application/json" \
  -d '{"question":"Who is the main character?","book":"alice_en.txt","mode":"hybrid"}'
```

查询模式：`local`（具体实体/事实）、`global`（跨章节）、`hybrid`（默认，推荐）、`naive`（纯向量 RAG，不走图）。

**按书独立图谱**：所有触图的 API 都带必填 `book`（文件名 = workspace）：`/query`、`/chat`、`/graph`、`/graph/top`、`/stats?book=`、`/documents/{book}`（GET/DELETE）、`/documents/{book}/refresh`、实体/关系编辑端点（`?book=`）。CLI：`query --book`、`stats --book`。每本书的实体/关系/chunk 各自住在独立 Neo4j label + Qdrant workspace_id + `working_dir/<book>/` KV 子目录。跨书查询设计上不支持。

**文档 upsert**：`POST /documents/upsert` 处理同 basename 重新导入 —— 先删该 workspace 再重新导入。

## 架构

数据流是线性的，代码在 `src/`：

```
loader.py ──load_book/split_into_chunks──▶ graph_builder._build_rag() ──▶ LightRAG
                                              │        │           │
                                   Neo4JStorage  QdrantVectorDB  KV(JSON) in working_dir
                                              │
                                              ▼
                                    query.ask / ask_stream / cypher_query
                                              │
                              api.py (FastAPI) ── /query /chat /ingest /stats /graph /documents /entities /relations /cypher /config /sessions
```

- `config.py` —— 单一 `Settings` dataclass；取值优先级 **环境变量 > `config.yaml` > 代码默认**（`_cfg()` 辅助函数）。敏感项（`*_api_key`/`neo4j_password`/`qdrant_api_key`）只走环境变量，绝不写入 YAML。`config.yaml` 被 gitignore（本地覆盖）；`config.example.yaml` 是带注释的模板。`settings` 是 import 时构造的单例 → **任何改动都需重启进程才生效**（配置 UI 据此返回 `needs_restart: true`）。`settings.ensure_dirs()` 创建 `working_dir` + `data/books` + `log_dir` + `session_dir`。`NO_PROXY=localhost,127.0.0.1` 在 import 时强制注入（见下）。
- `src/loader.py` —— 按格式读取（`_read_pdf`/`_read_epub`/`_read_docx`/`_read_plain`）→ `_normalize` → `split_into_chunks`（按字符切，中文友好，不依赖 tokenizer）。`_read_docx` 把段落 + 表格行（制表符分隔）序列化。`resolve_book_path` 把裸文件名映射到 `data/books/`。
- `src/graph_builder.py` —— **集成核心**。`_build_rag()` 构造 `LightRAG` 实例，配 `Neo4JStorage` + `QdrantVectorDBStorage` + SiliconFlow 嵌入/重排函数。它把 `QDRANT_URL`/`NO_PROXY` 注入进程环境（LightRAG 内部的 `QdrantClient` 读它们）。LLM 经 `role_llm_configs={"query": ...}` 拆角色：基座 `llm_model_func` = `_make_glm_func()`（GLM-4.7，用于 `extract`/`keyword` 角色），`query` 角色覆写为 `_make_qwen_func()`（本地 Ollama Qwen，最终答复）。`_make_llm_func(api_key, base_url, model, streaming_enabled)` 是共享工厂；按 `kwargs["stream"]` 分支 —— 流式请求返回异步生成器（让 LightRAG 识别 `is_streaming=True`），非流式（实体抽取）返回完整字符串。`ingest_book()` 加载、切块、拼接，调 `rag.ainsert(full_text, file_paths=[fname])`。**答复语言**：`_enforce_response_language()` 在 `_build_rag` 内覆盖 LightRAG `rag_response` 模板的"跟随问题语言"指令，强制用 `settings.language`（默认简体中文）回答 —— 否则英文书+英文提问时 Qwen 会中英混杂。`addon_params={"language":...}` 对该模板无效（无 `{language}` 占位符），故需此补丁。
- `src/remote_models.py` —— `make_embedding_func()` 把 SiliconFlow `/v1/embeddings`（OpenAI 兼容，AsyncOpenAI）包成 LightRAG `EmbeddingFunc`（返回 float32 ndarray）。`make_reranker_func()` 调 SiliconFlow `/v1/rerank`（httpx，非 OpenAI），返回 `[{index, relevance_score}]`，对越界分数做 sigmoid 兜底。两者都对 429/5xx 做 tenacity 重试。
- `src/query.py` —— `ask`（非流式）、`ask_stream`（异步生成器，多轮 `conversation_history`）、`cypher_query`（只读原生 Neo4j）、`graph_stats`。**查询改写（A3）**：`_prepare_question` 返回 `(probe_q, retrieve_q)`；`_rewrite_query` 用 GLM 消解指代（仅当历史非空，任何失败回退原问题），`_decompose_query` 拆分多部分问题（JSON 由 `_extract_json_array` 解析，先剥 markdown fence 再正则提 `[...]`），`_pick_gen_query` 探测两个候选、取命中数多者，使分解只能帮忙不会添乱。**硬拒答（A1）**：先用 `aquery_data(only_need_context)` 探测（不调 LLM）；引用 < `MIN_HIT_COUNT` → 返回 `HARD_REFUSE_ANSWER`，SSE 发 `{"type":"refuse"}`。
- `src/maintenance.py` —— LightRAG 1.5.x 维护 API（`aedit_entity`/`adelete_by_doc_id`/`aedit_relation`…）的薄封装；这些会自动同步向量库 + Neo4j。每个 helper 调 `_rag()` 从池取实例（只读/轻写）；重写操作走 `_rag_exclusive()` 独占 build + init + finalize。
- `src/rag_pool.py` —— `RagPool` 按 workspace LRU 缓存 LightRAG 实例（淘汰时 `finalize_storages` 关连接）。`query`/`maintenance._rag` 用池；`ingest_book`/`delete_document` 独占构建 + finalize + `invalidate` 池中条目。lifespan 启停池。
- `src/graph_view.py` —— `get_subgraph`/`get_top_entities` 供前端可视化。知道 Neo4j 存储约定：节点 label = workspace 名，实体名在 `entity_id` 属性，边类型为 `DIRECTED`，语义在 `description`/`weight`。
- `src/api.py` —— FastAPI 应用。`/chat` 是 SSE 流式，带**心跳**（`stream_heartbeat_interval`；后台生产者把生成器排空入 queue，主循环 `wait_for` queue、绝不 `wait_for` 生成器的 `__anext__`）与**排队可见化**（有查询在跑时发 `{"type":"queue","ahead":N}`；`_query_in_flight` 计数器的自增 + yield 全在 `try/finally` 内，客户端断开也不会泄漏或触发 503 雪崩）。`/chat` 接受可选 `session_id`+`user_id`：提供时流式开始前持久化 user 消息、完成后持久化 assistant 消息（仅 answer 非空），关键节点打 INFO 日志。CORS 走 `cors_origins`（通配源自动关 `Allow-Credentials`）。`/ingest/async` + `/tasks/{id}` 用进程内 `_tasks` dict 跟踪长导入（无 DB —— 重启即丢）。`/cypher` 仅允许 `MATCH/RETURN/WITH/CALL/UNWIND` 前缀。**`GET /config`** 返回分组快照（敏感项掩码 `***`）；**`PUT /config`** 把非敏感字段写回 `config.yaml`（跳过敏感 + 未知字段）；两者都返回 `needs_restart: true`。**`/sessions`**（POST 创建 / GET 列表 / GET 详情 / DELETE）—— 会话 CRUD，按 `user_id` 归属，越权返回 404；详见 `src/session_store.py`。
- `src/logging_config.py` —— `build_log_config(level, log_file)` 返回 `dictConfig`：`RotatingFileHandler` → `<log_dir>/app.log`（10MB × 5 备份）+ 控制台 `StreamHandler`，`book_kg.*` 命名空间按配置级别，第三方库（httpx/openai/httpcore/uvicorn.access）压到 WARNING。`serve.py` 把它作为 uvicorn `log_config` 传入；`main.py` CLI 调 `setup_logging()`。级别由 `LOG_LEVEL` 控制（env 或 `config.yaml`）。
- `src/session_store.py` —— 会话持久化（JSON 文件，无 DB）。布局：`session_dir/<id>.json`（单会话含 messages）+ `index.json`（扁平索引，供列表查询）。按 `user_id` 归属（前端 localStorage 持久 UUID，无登录体系），越权访问返回 None。模块级 `asyncio.Lock` 串行化索引读-改-写；写文件走 tmp + replace 原子写。`create_session`/`list_sessions`/`get_session`/`delete_session`/`append_message`（首条 user 消息生成标题，取 `session_title_length` 字）。日志走 `book_kg.session` 命名空间 → `logs/app.log`。`session_dir` 独立于按书隔离的 `working_dir` KV（会话是跨书全局用户数据）。
- `main.py` —— Click CLI，镜像 API 能力。
- `static/index.html` —— 挂在 `/` 的图谱可视化 UI。单文件（ECharts + 原生 JS，无构建）。四个侧栏 tab：**概览**（统计 + 可点类型图例来自 `node_counts_by_label` + 节点详情含描述）、**文档管理**（`/documents` 列表 + `/ingest` + `/documents/upsert` + 刷新 + 删除）、**Cypher**（`/cypher` 控制台带结果表）、**配置**（`GET /config` 分组表单，敏感项只读 `***`，保存 → `PUT /config` 写 `config.yaml` + toast "需重启生效"；tab 切换时懒加载）。聊天区有书域下拉（`/chat` 带 `book`）+ 会话栏（新会话/历史下拉/删除，`localStorage` 持久 `user_id`，选中历史会话回显消息并切书，提问自动归入当前会话），引用渲染在每条答复下，toast 替代 alert。

## 关键环境说明

完整迭代日志见 `docs/TROUBLESHOOTING.md`。以下是反直觉的点：

1. **`COSINE_THRESHOLD=0.2`** —— LightRAG 读它 → `cosine_better_than_threshold` → Qdrant 的 `score_threshold`。Qdrant `Distance.COSINE` 返回**真余弦相似度**（已验证：相同=1.0、正交=0.0、范围 [-1,1]），保留 `score >= threshold`。所以 `0.2` 是合理下限。（与旧 milvus-lite 的 `1.0` hack 无关 —— 那个已删。）
2. **Qdrant 端口是 `16333`，不是默认 `6333`** —— `6333`/`6334` 落在 Windows Hyper-V 保留端口段，Docker bind 失败。compose 把宿主 `16333/16334` → 容器 `6333/6334`。`QDRANT_URL=http://localhost:16333`。
3. **`NO_PROXY=localhost,127.0.0.1` 在 `config.py` 强制注入** —— 本机有系统代理（FlClash/Clash）；不绕过的话 `qdrant-client` 的 `requests` 会把 localhost 走代理，Qdrant 返回 `502 Bad Gateway`。`config.py` 在 import 时注入，覆盖所有入口（LightRAG 内部建 `QdrantClient` 时没传 `trust_env=False`）。
4. **Neo4j Community 版**不能建命名数据库，故 LightRAG 请求 `chunk-entity-relation` DB 时会日志 "not found... Fallback to use the default database" —— 无害，回退到 `neo4j` 默认库。
5. **Ollama 并发**：默认 `OLLAMA_NUM_PARALLEL=1`，故 `QUERY_LLM_MODEL_MAX_ASYNC=1`。要调高需以 `OLLAMA_NUM_PARALLEL=N` 启动 Ollama 且显存够 N 个并发 Qwen 上下文。
6. **本地 Qwen 只做答复**：别把 `extract`/`keyword` 角色路由到本地 Qwen —— 长抽取生成会拖崩这个 30W 笔记本 GPU（见 TROUBLESHOOTING）。抽取留在远程 GLM。
7. **生产硬化（phase 5/6）** —— 查询容错 + 并发：
   - **硬拒答（A1）**：`query.ask`/`ask_stream` 先用 `rag.aquery_data`（only_need_context，**不调 LLM**）探测；引用 < `MIN_HIT_COUNT` → 不碰 Ollama 直接返回 `HARD_REFUSE_ANSWER`。后置检查：`aquery_llm` 后引用/内容为空也拒答。SSE 发 `{"type":"refuse"}`。
   - **LLM 鲁棒性（A2）**：`_make_llm_func` 带 `timeout`（GLM 60s / Qwen 120s）+ 429 重试 6 次 + 连接/超时错误重试 3 次 + 空响应抛 `EmptyLLMResponseError` + 流式首 token 超时。`/query` 整体超时 180s → 504。
   - **实例池（B0）**：`src/rag_pool.py` `RagPool` 按 workspace 缓存 LightRAG（LRU，淘汰时 `finalize_storages`）。`query`/`maintenance._rag` 用池；`ingest_book`/`delete_document` 独占构建 + finalize + `invalidate` 池条目。lifespan 启停池。
   - **全局并发闸（B1）**：`lifespan` 调一次 `initialize_share_data(global_concurrency_limits={llm:extract:2, llm:keyword:2, llm:query:1, embedding:4, rerank:4})` —— 让 LightRAG 每实例的 `max_async` 信号量参与跨实例全局限额（零源码改动）。`llm:query=1` 跨请求串行化 Ollama。
   - **限流（B2）**：`Semaphore(MAX_CONCURRENT_REQUESTS=8)` 非阻塞 → 503 + Retry-After；ingest `Semaphore(1)`。输入校验：`book` 正则白名单 + `max_length` + `_ensure_book_exists` 预检；错误经 `_safe_detail` 脱敏。
   - **异步导入（B3）**：`/documents/upsert` 和 `/documents/{book}/refresh` 立即返回 `task_id`；`_submit_ingest_task` 持 `_task_ref`（防 GC）+ TTL 清理；前端 `pollTask` 轮询 `/tasks/{id}`。
   - **查询改写/分解（A3，phase 6）**：`_prepare_question` → `(probe_q, retrieve_q)`；GLM 按历史消解指代、拆分多部分问题，`_pick_gen_query` 取命中多者。改写 LLM 挂在 `llm:extract` 全局并发组（非 `llm:query`），不与 Ollama 抢槽。
   - **SSE 硬化（A4，phase 6）**：心跳走 后台生产者 + queue（绝不 `wait_for` 生成器）；CORS 走 `cors_origins`（通配源自动关 credentials）。
   - **排队可见化（B4，phase 6）**：`/chat` 在有查询等待时发 `{"type":"queue","ahead":N}`；在飞计数器在 `try/finally` 内，断开干净归零（无 503 雪崩）。范围：仅 `/chat` 流式，不含 `/query`。
8. **SiliconFlow 重排分数范围**：代码假定 `relevance_score ∈ [0,1]`，对越界值做 sigmoid 归一（原始 logit 兜底）。已冒烟验证在 [0,1]。
9. **GLM 429**：`LLM_MODEL_MAX_ASYNC=2` + 6 次退避重试。若 429 持续，降到 1。

## 端口

| 服务 | 端口 |
|---------|------|
| Ollama（OpenAI API） | 21434（默认 11434 在 Windows 保留端口段） |
| FastAPI | 8010 |
| Neo4j Bolt / Browser | 7687 / 7474 |
| Qdrant HTTP / gRPC | 16333 / 16334 |

## 重新导入

每本书隔离在各自 workspace，重导一本不影响其它：
- **更新一本书**：`POST /documents/{book}/refresh` 或 `POST /documents/upsert`（删该书 workspace，再重导）。
- **删除一本书**：`DELETE /documents/{book}` —— 删其 Neo4j label 节点、Qdrant workspace_id 点、`working_dir/<book>/` KV 子目录。
- **全量清空**（所有书）：drop Qdrant collections + Neo4j 里 `MATCH (n) DETACH DELETE n` + `rm -rf rag_storage/*`，再逐本重导。
