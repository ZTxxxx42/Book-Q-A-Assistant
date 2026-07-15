# 阶段 6 改进报告：查询改写/分解 + SSE 硬化 + 排队可见化（P1）

> 日期：2026-07-15
> 范围：实现 phase5 报告列为 "P1（后续未做）" 的 A3/A4/B4 三项，并修复审阅发现的 4 处问题。

## 背景

phase5（P0 硬化）交付了查询容错（A0–A2）与高并发（B0–B3）。本阶段处理其 P1 遗留：
- **A3** 多轮对话指代消解 + 复杂问题子问题分解；
- **A4** SSE 长连接心跳（防代理空闲断流）+ CORS；
- **B4** Ollama 排队可见化（告知客户端排队位置）。

## 改动

### A3 查询改写 / 分解（`src/query.py`）
- `_get_rewrite_llm()`：惰性构造 GLM 改写 LLM，用 `priority_limit_async_func_call(concurrency_group="llm:extract")` 接入 B1 全局闸（已查 LightRAG 1.5.x 源码确认：注册过全局限额的 group，新装饰器实例走跨进程 slot 闸，**不绕过 B1，不额外打爆 GLM**）。
- `_rewrite_query(question, history)`：`history` 非空时用 GLM 做指代消解 + 历史压缩，改写成独立完整查询；任何失败回退原 question。
- `_decompose_query(question)`：GLM 拆 2–4 个子问题，合并为 `" ; "` 拼串扩展查询；失败回退原 question。
- `_extract_json_array(text)`：稳健解析 GLM 输出（剥 ```json 围栏 → 直接解析 → 正则提取首个 `[...]`），修复 GLM 返回 markdown 包裹 JSON 导致 `JSONDecodeError` 静默回退的 bug。
- `_prepare_question` 返回 `(probe_q, retrieve_q)`：改写后的单查询用于预探测，分解拼串用于生成检索。
- `_pick_gen_query(rag, probe_q, retrieve_q, mode)`：对两个查询各做一次预探测，取命中数多者用于生成（平局偏 probe_q）。**保证分解只能帮忙不能添乱** —— 拼串 embedding 被稀释常比原查询召回更差，命中更多才用，否则回退单查询。`decompose=False` 时两者相同，仅探测一次，零额外开销。
- `ask`/`ask_stream` 加 `history`/`decompose` 参数；`/query`、`/chat` 请求体加 `decompose` 字段（默认关）。

### A4 SSE 心跳 + CORS（`src/api.py`）
- `_with_heartbeat(agen, interval)`：包装异步生成器，两条事件间隔超 `interval` 插入 `{"type":"ping"}`。用后台 producer 任务消费生成器推入 queue，主循环对 `queue.get()` 计时——**绝不 `wait_for` 生成器的 `__anext__`**（超时会 cancel 掉正在 await 的生成器帧从而终止它，经典坑）。
- CORS 中间件：`CORS_ORIGINS` 非空时启用；**含 `*` 时自动关闭 `allow_credentials`**（CORS 规范禁止通配源带凭证），否则按显式来源允许 credentials。
- `config.py` 新增 `stream_heartbeat_interval`(15s)、`cors_origins`。

### B4 排队可见化（`src/api.py` + `static/index.html`）
- `_query_in_flight` 计数：`/chat` 流式开始时 `+=1`，`ahead = in_flight - 1`，`ahead>0` 先 yield `{"type":"queue","ahead":N}`，结束 `-=1`。
- 前端处理 `queue`（显示"⏳ 排队中，前方还有 N 个请求"）与 `ping`（忽略）事件。
- **范围说明**：仅覆盖 `/chat` 流式，不含 `/query` 非流式；混合负载下 `ahead` 偏低（不计 /query 占用的 Ollama 槽）。单机个人用途可接受。

## 审阅修复的 4 处问题

1. 🔴 **计数器/信号量泄漏**：原 `_query_in_flight += 1` 与 queue yield 在 `try` 外，断连时 `GeneratorExit` 在该 yield 抛出不进 `finally` → 计数器单调增 + sem 不释放 → `Semaphore(8)` 耗光后所有 /chat 503。修复：sem 获取 + 计数器增量 + queue yield 全部挪进 `try/finally`，`finally` 只做 `-=1` + `_release(sem)`。`yield` 在 try 内合法，`GeneratorExit` 会先走 finally 再传播。
2. 🟡 **CORS 通配源 + credentials**：`allow_credentials=True` 固定写死，`CORS_ORIGINS=*` 时规范禁止。修复：检测到 `*` 自动 `allow_credentials=False`；`.env.example` 补说明。
3. 🔴 **分解 JSON 解析失败**：GLM 返回 ```json [...] ``` 包裹，`json.loads` 直接 `JSONDecodeError` → 分解静默失效。修复：`_extract_json_array` 剥围栏 + 正则兜底。
4. 🔴 **naive 探测误拒答**：原 diff 把 `_probe_hits` 从 `mode` 改为 `naive`（省 GLM 关键词抽取），但 naive 纯向量召回太弱，对 hybrid 借图谱可答的问题召回为 0 → 误拒答。修复：探测改回 `mode`；并加 `_pick_gen_query` 让分解安全回退（拼串命中更多才用）。

## 验证结果

| 项 | 测试 | 结果 |
|---|---|---|
| A3 改写 | `/chat` 带 history 追问 "Who is her cat?" | 日志 `query 改写: 'Who is her cat?' -> "Who is Alice's cat in Wonderland?"`，代词消解 ✅ |
| A3 分解 | `/query` decompose=true "Who is Alice?" | 日志 `query 分解: ...`，拼串召回更差时 `_pick_gen_query` 回退原查询，正常返回答案 ✅ |
| A3 分解解析 | GLM 返回 markdown 包裹 JSON | `_extract_json_array` 正确提取，不再 JSONDecodeError ✅ |
| A4 心跳 | 慢查询 /chat（GLM 429 重试期间） | 出现多条 `{"type":"ping"}` ✅ |
| A4 CORS | 代码审阅 | `*` → credentials=false；显式源 → 回显 ✅（代码审阅） |
| B4 排队 | 并发 2 个 /chat | 第二个首事件 `{"type":"queue","ahead":1}` ✅ |
| 🔴 回归 | /chat 3s 强制断开（decompose 慢路径，curl exit 28）+ 后续 /chat | 计数器归零，后续 chat `ahead=0`，无 503 雪崩 ✅ |

## 遗留（P2）
- **decompose 非真 fan-out**：当前拼串送单次检索，不是多路并行召回 + 合并。`_pick_gen_query` 已保证不退步，但增益有限。P2 可改真 fan-out（每子问题独立 aquery_data，union references 再生成）。
- **`_tasks` 仍进程内**：重启即丢（单机可接受；持久化留 P2）。
- **B4 不含 /query**：非流式查询不参与 in-flight 计数。
