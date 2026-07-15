# 阶段 3 改进报告：前端管理面板

> 日期：2026-07-15
> 范围：把单文件前端补全为完整管理 UI——文档管理、Cypher 控制台、按书过滤、引用展示、类型图例、节点描述、toast。

## 改动摘要

仅 `static/index.html` 一个文件（单文件全栈，ECharts + 原生 JS，无构建）。在保留原有图谱可视化 + 多轮 SSE 问答的基础上增量加面板。

### 新增功能

| 功能 | 实现 |
|------|------|
| **侧栏 Tab** | 概览 / 文档管理 / Cypher 三 tab 切换 |
| **文档管理面板** | `GET /documents` 列表（文件名/状态/块数/时间）、导入（`POST /ingest`）、Upsert（`POST /documents/upsert`，阶段 2）、刷新（`POST /documents/{id}/refresh`）、删除（`DELETE /documents/{id}`） |
| **Cypher 控制台** | 文本框 + 执行按钮 → `POST /cypher`，结果渲染成表格（Ctrl/Cmd+Enter 快捷执行） |
| **按书过滤 UI** | 问答区"限定书籍"下拉（从 `/documents` 的 processed 文档填充）；`/chat` 请求带 `book` 字段 |
| **引用展示** | `/chat` 的 `references` 事件解析后，答复下方显示"📎 出处：file_path"（按书去重） |
| **类型图例/过滤** | `/stats` 的 `node_counts_by_label`（之前未用）渲染成可点击图例，点击按 label 过滤画布节点 |
| **节点详情富化** | 侧栏点节点展示 `description`（之前未展示） |
| **toast** | `alert()` 全部换为右上角 toast（含 ok/error 样式） |

## 改前 → 改后

| 项 | 改前 | 改后 |
|----|------|------|
| 文档管理 | 无 UI，只能 CLI 导书 | 完整列表 + 导入/Upsert/刷新/删除 |
| Cypher | 无 UI | 控制台 + 结果表格 |
| 按书过滤 | 无 UI | 下拉选择，流式问答带 book |
| 引用出处 | 前端忽略 references 事件 | 答复下方展示出处 |
| 类型图例 | node_counts_by_label 未用 | 可点击图例 + 过滤 |
| 节点描述 | 未展示 | 侧栏展示 description |
| 错误提示 | alert | toast |

## 验证结果

后端端点全部就绪（前端数据源验证）：
- `GET /documents` → 3 文档（alice processed 5 chunks / ocean processed 1 chunk / 一条早期 dup-failed 残留）
- `POST /cypher` `MATCH (n) RETURN labels(n)[0], count(*)` → 1 行（base, 118）
- `POST /graph/top` → 热门实体正常
- 页面 `GET /` 正常返回

JS 语法校验：`node --check` 通过（14662 字符脚本无语法错误）。

**浏览器实测需用户在 `http://localhost:8010` 操作**：导入第二本书 → 文档列表 → 图谱节点增加 → 按书过滤问答仅命中该书 → Cypher 控制台出表 → 图例点击过滤 → 节点详情显示 description。

## 遗留问题与下一步

1. **dup-failed 残留文档**：早期重复 ingest alice 留下一条 `status=failed` 的 `dup-` 记录，文档列表里可见（状态标红）。无害但杂乱；后续可在 `list_documents` 过滤 `dup-` 前缀，或单独清理。
2. **导入用文件名输入**：当前是文本输入 `data/books` 下文件名，无文件选择器（浏览器无法直接读本地路径给后端）。可接受；如需上传需加 multipart 上传端点。
3. **图谱导航历史栈未加**：`expandNode` 仍替换整图，无回退。本轮范围外，后续可加。
4. **实体编辑/删除 UI 未加**：`/entities/{id}/edit` 等接口未对接到侧栏按钮。后续可补。
5. **浏览器实测**：本次仅做端点 + 语法校验，真实交互需用户在浏览器确认。

## 总结

三阶段迭代完成：查询优化（阶段 1）→ 多书按书过滤（阶段 2）→ 前端管理面板（阶段 3）。栈从"能跑通的 demo"进入"可交互管理 + 按书问答 + 带出处"的可用状态。改进报告见 `docs/reports/`，迭代排错记录见 `docs/TROUBLESHOOTING.md`。
