# 阶段 1 改进报告：查询/检索质量优化

> 日期：2026-07-15
> 范围：让 rerank 真正过滤、答复带出处、修 enable_rerank 不对称 bug、检索旋钮收口 config。

## 改动摘要

| 文件 | 改动 |
|------|------|
| `config.py` | 新增 `top_k`/`chunk_top_k`/`min_rerank_score`/`include_references`/`response_type` 字段（env 可调） |
| `src/query.py` | 统一 `_make_param()` 显式传 `enable_rerank`/`top_k`/`chunk_top_k`/`include_references`/`response_type`；`ask()` 改用 `aquery_llm` 取 references，返回 `{answer, references}`；`ask_stream()` yield 事件 dict（先 `references` 后 `token`） |
| `src/api.py` | `QueryResponse` 加 `references` 字段；`/chat` SSE 直接转发事件 dict |
| `main.py` | `query` 命令打印 references |
| `.env` / `.env.example` | 加 `TOP_K=40`/`CHUNK_TOP_K=12`/`MIN_RERANK_SCORE=0.3`/`INCLUDE_REFERENCES=true`/`RESPONSE_TYPE=Single Paragraph` |

## 改前 → 改后

| 项 | 改前 | 改后 |
|----|------|------|
| `MIN_RERANK_SCORE` | 0.0（rerank 只排序不过滤） | 0.3（过滤低相关 chunk） |
| `chunk_top_k` | 20（默认） | 12（小库下调） |
| `include_references` | 未开（无出处） | true（答复附 reference_id + file_path） |
| `response_type` | `Multiple Paragraphs` | `Single Paragraph` |
| `/query` 非流式 `enable_rerank` | 读 LightRAG 的 `RERANK_BY_DEFAULT`（与项目 `ENABLE_RERANK` 脱节） | 显式读 `settings.enable_rerank`（对称） |
| 检索旋钮 | 散落 env，`QueryParam` 不显式传 | 收口 `config.py`，`_make_param()` 统一传 |

## 验证结果（5 个英文问题）

| 问题 | 答复 | refs |
|------|------|------|
| Who is Alice? | 正确（主角/掉兔子洞/仙境） | 1 |
| What does the White Rabbit carry? | 识别出怀表（从 waistcoat-pocket 取表） | 1 |
| What did Alice find on the three-legged table? | 正确（tiny golden key + 原文引用） | 1 |
| Who is Dinah? | 正确（Alice 的猫） | 1 |
| What is the weather like? | 正确拒答（无相关信息） | **0** |

关键：**`MIN_RERANK_SCORE=0.3` 让无关问题返回 0 引用且不胡编**（改前会拼凑上下文生成）。相关问题答复准确并附出处。

`/chat` SSE 验证：先发 `{"type":"references",...}` 事件，再逐 token 流式，前端忽略未知事件类型不受影响。

## 遗留问题与下一步

1. **Qwen 偶发中英混输出**：`Single Paragraph` 模式下 Qwen-7b 有时在英文答复末尾混入中文（如 "奇特的角色和奇异的世界"）。这是模型行为，非配置问题。后续可在 query prompt 加 "Respond in English" 约束，或换 `qwen2.5:7b-instruct-q5_K_M` 等量化变体观察。
2. **`chunk_top_k=12` 可能偏紧**：阶段 2 按书过滤时若上下文过少需回调大。实测 5 问均命中，暂保持。
3. **references 仅 1 条**：小库下多数查询只命中 1 个 chunk 的引用；多书后会更丰富。
4. **LLM 缓存**：`aquery_llm` 结果会被 LightRAG 缓存（`llm_response_cache`）；调参后用新问题验证以避开缓存。

## 下一步

进入阶段 2（多书 + 按书过滤）：`ask_with_scope` + `/documents/upsert`，复用本阶段的 references 机制做按书过滤与出处展示。
