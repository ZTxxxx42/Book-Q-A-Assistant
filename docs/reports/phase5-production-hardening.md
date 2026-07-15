# 阶段 5 改进报告：生产硬化 — Query 容错 + 高并发（P0）

> 日期：2026-07-15
> 范围：针对"无关 query 容错"与"高并发"两个实际业务问题，实施 P0 级硬化。

## 背景

阶段 1-4 后栈已跑通，但两路探查发现应用层缺几乎所有生产护栏：
- rerank 过滤后 0 chunk 仍调 LLM，靠 prompt 自觉拒答 → 幻觉；
- LLM 无 timeout、仅重试 429、空响应静默成功 → Ollama 卡死则请求永久挂起；
- 每请求新建 LightRAG + driver/client 且不 finalize → 连接泄漏；
- max_async 信号量 per-instance，从未调 `initialize_share_data` → 跨请求并发限制失效，N 并发 = N×2 GLM（429）+ N 并发 Ollama；
- 无输入校验、无限流、无超时、错误 detail 直拼 `str(e)`、`_tasks` 无加固、长任务同步阻塞事件循环。

## 改动（P0）

### A — Query 容错
- **A0 输入校验 + book 预检 + 错误脱敏**（`src/api.py`）：
  - `QueryRequest`/`ChatRequest`/`GraphRequest`/`TopEntitiesRequest` 的 `question` 加 `max_length=2000`、`book` 加正则白名单 `^[A-Za-z0-9_\-\.一-龥]+$` + 长度上限 + 路径分隔符拒绝；`history` 加 `max_length=10`、每条 content `max_length`。
  - `_ensure_book_exists(book)` 复用 `maintenance._doc_status_path` 预检，不存在 → 404，避免空检索浪费。
  - 所有 `except Exception` 端点改用 `_safe_detail(prefix, e)`：对外分类消息，`repr(e)` 写 logger，不再泄露内部栈。
- **A1 硬拒答双保险**（`src/query.py`）：
  - 预探测：调 `rag.aquery_data(question, param)`（LightRAG 内部 `only_need_context=True`，**不调 LLM**）取 `data.references`；命中数 < `min_hit_count` → 直接返 `hard_refuse_answer`，**不占用 Ollama 槽位**。
  - 事后兜底：`aquery_llm` 返回后 references 仍空或 content 空 → 改返拒答文案。
  - 流式新增 `{"type":"refuse"}` 事件，前端渲染 ⚠️。
  - `config.py` 新增 `min_hit_count`(1) / `hard_refuse_threshold`(0.3) / `hard_refuse_answer`。
- **A2 LLM 鲁棒性**（`src/graph_builder.py` + 新建 `src/errors.py`）：
  - `_make_llm_func` 加 `timeout` 形参（GLM 60s / Qwen 120s），传给 `AsyncOpenAI`。
  - tenacity 重试扩展为 `(RateLimitError, APIConnectionError, APITimeoutError, asyncio.TimeoutError)`；自定义 `_stop_by_type`：429 重试 6 次、连接/超时错误 3 次。
  - 非流式空响应抛 `EmptyLLMResponseError`，不静默成功。
  - `_stream_completion` 加首 token 超时（`asyncio.wait_for`，60s），防 Ollama 卡死不 yield。

### B — 高并发
- **B0 实例池化**（新建 `src/rag_pool.py`）：
  - `RagPool`：`OrderedDict[workspace, _Entry]` LRU，容量 `rag_pool_size`(5)。
  - `acquire`/`release`：命中复用（move_to_end），未命中 `_build_rag`+`initialize_storages`（per-workspace lock 防双 init），超容淘汰 in-flight=0 的最久未用实例并 `finalize_storages`。
  - `invalidate(workspace)`：重写操作后使旧实例失效。
  - `query.ask`/`ask_stream` 改用池；`maintenance._rag` 改为 async context manager（只读/轻写用池）；`ingest_book`/`delete_document` 独占 build+init+finalize（不入池），完成后 `invalidate`。
  - `lifespan` 启动 `get_pool()`、退出 `shutdown_pool()`。
- **B1 全局并发闸**（`src/api.py` lifespan）：
  - 启动时调一次 `initialize_share_data(workers=1, global_concurrency_limits={...})`，让 LightRAG 的 `priority_limit_async_func_call` 信号量参与跨实例全局限额（零改源码）：
    `llm:extract=2, llm:keyword=2, llm:query=1, embedding=4, rerank=4`。
  - `llm:query=1` → N 并发请求排队打 Ollama 而非雪崩。
- **B2 限流 + 超时**（`src/api.py`）：
  - `_query_sem`(`Semaphore(8)`) + `_try_acquire` 非阻塞：超限立即 503 + `Retry-After`。
  - `_ingest_sem`(`Semaphore(1)`)：单机 GPU 一次一本 ingest。
  - `/query` 整体 `asyncio.wait_for(..., 180s)` → 504。
- **B3 长任务异步化 + 任务加固**（`src/api.py` + `serve.py`）：
  - `_submit_ingest_task`：统一 ingest/upsert/refresh 异步提交，`asyncio.create_task` 返回值存 `_task_ref` 防 GC，记录 `started_at`/`finished_at`。
  - `/documents/upsert`、`/documents/{book}/refresh` 改为立即返回 `task_id`（不再阻塞事件循环）；前端 `pollTask` 轮询 `/tasks/{id}`。
  - `_task_cleanup_loop` 后台定期清理 done/failed 任务（TTL 1h）。
  - `get_task` 过滤内部 `_` 字段。
  - `serve.py`：port 8010、`reload=False`、`workers=1`（对齐 `initialize_share_data(workers=1)`）。

### 其他
- `config.py`：新增 17 个硬化字段（均带 env 默认值）。
- `.env.example`：补全部新 env。
- `static/index.html`：处理 `refuse` 事件；`pollTask` 助手 + upsert/refresh 轮询。

## 验证结果

| 项 | 测试 | 结果 |
|---|---|---|
| A0 | 空 question | 422 ✅ |
| A0 | 路径穿越 book (`../../../etc/passwd`) | 422 "book 含非法字符" ✅ |
| A0 | 不存在 book | 404 ✅ |
| A0 | 超长 question (3000 字) | 422 ✅ |
| A1 | 无关问题（"量子力学原理…"，alice） | 200 + "未找到足够信息" + references=[]，未调 Ollama ✅ |
| A1 | 正常问题（"Who is Dinah?"） | 200 + 答案 + 引用 alice_en.txt ✅ |
| B0 | 连续两次同书查询 | 均成功，池复用 ✅ |
| B1 | lifespan 日志 | "全局并发闸已启用: {llm:extract:2, llm:keyword:2, llm:query:1, embedding:4, rerank:4}" ✅ |
| B2 | 15 并发 /query | 7×503 + 1×200 + 超时（限流生效）✅ |
| B3 | /documents/upsert | 立即返回 task_id（非阻塞）✅ |
| B3 | 轮询 /tasks/{id} | done，含 deleted+reingested 结果 ✅ |
| B3 | re-ingest 后 stats | ocean 26 节点（池失效后新实例读到新数据）✅ |

## 改前 vs 改后

| 维度 | 改前 | 改后 |
|---|---|---|
| 无关 query | 仍调 Ollama，靠 prompt 自觉拒答（幻觉风险） | aquery_data 预探测硬拒答，不调 LLM |
| LLM 超时 | 无，永久挂起 | GLM 60s / Qwen 120s + 首 token 60s |
| LLM 重试 | 仅 429 | 429 + 连接/超时错误 |
| 空响应 | 静默返回 "" | 抛 EmptyLLMResponseError |
| 输入校验 | 仅 min_length=1 | max_length + book 白名单 + 预检 |
| 错误信息 | str(e) 泄露栈 | 脱敏 + 日志 |
| LightRAG 实例 | 每请求新建 + 泄漏 | LRU 池 + finalize |
| 并发限制 | per-instance 失效 | initialize_share_data 全局闸 |
| HTTP 限流 | 无 | Semaphore(8) → 503 |
| ingest | 同步阻塞 / _tasks 无加固 | 异步 task_id + 防 GC + TTL |
| /query 超时 | 无 | 180s → 504 |

## P1（后续未做）
- A3 query 改写（GLM 指代消解，仅 history 非空，走 `llm:extract` 组不占 query 槽）+ sub-query 拆分。
- A4 SSE 心跳（15s ping）+ CORS。
- B4 Ollama 排队可见化（`{"type":"queue","position":N}`）+ 实测 `OLLAMA_NUM_PARALLEL=2`。

## 遗留
- `_try_acquire` 用 `sem._value` 非阻塞判断，依赖 asyncio 单线程原子性（无 await 间隙，安全）；未用阻塞队列，超限即 503。
- 全局闸 `initialize_share_data` 限额进程启动后不可改（read-only），调参需重启 + 改 env。
- `_tasks` 仍进程内，重启即丢（单机可接受；持久化留 P2）。
