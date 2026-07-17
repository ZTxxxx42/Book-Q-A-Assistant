# 恢复预构建的知识图谱与向量库（v0.2.0）

本发行版（v0.2.0）附带了一份**已构建好的知识图谱快照**，含两本书：
`alice_en.txt`（《爱丽丝梦游仙境》，公有领域）与 `ocean_tale.txt`（作者自有）。

按本指南恢复后，**无需再花 LLM 调用重新抽取实体/关系**，启动即可直接对这两本书提问。

## 附带文件（从 Release v0.2.0 下载）

| 文件 | 内容 | 体积 |
|------|------|------|
| `neo4j_data.tar.gz` | Neo4j 图数据卷（实体/关系节点与边） | ~1.7M |
| `qdrant_data.tar.gz` | Qdrant 向量数据卷（bge-m3 嵌入） | ~2.7M |
| `rag_storage.tar.gz` | LightRAG KV（doc_status / chunks / 实体 / 关系 / 缓存） | ~57K |
| `books.tar.gz` | 两本书原文 | ~54K |

## 前置条件

- Docker Desktop（需 NVIDIA GPU + 驱动，因 Ollama 容器跑 Qwen2.5-7B）
- 智谱 GLM API Key（`LLM_API_KEY`）—— 提问时改写/分解与未来重导需要
- SiliconFlow API Key（`EMBEDDING_API_KEY` / `RERANK_API_KEY`）—— 未来重导需要；仅查询已恢复数据时实际不调用嵌入（向量已在 Qdrant 里），但启动校验仍需配置

> 仅对这两本已恢复的书提问，不会消耗你的 GLM/SiliconFlow 配额（答复走本地 Ollama Qwen，不调远程 API）。远程 API 只在**重新导入新书**或**开启查询改写**时才调用。

## 恢复步骤

### 1. 获取代码并配置

```bash
git clone https://github.com/ZTxxxx42/Book-Q-A-Assistant.git
cd Book-Q-A-Assistant
git checkout v0.2.0
cp .env.example .env
# 编辑 .env，至少填：
#   LLM_API_KEY=sk-你的智谱GLM密钥
#   EMBEDDING_API_KEY=sk-你的硅基流动密钥
#   RERANK_API_KEY=sk-你的硅基流动密钥
```

把上面 4 个 `.tar.gz` 文件下载到项目根目录（`Book-Q-A-Assistant/`）。

### 2. 解压书籍原文与 KV

```bash
mkdir -p data/books
tar xzf books.tar.gz -C data/books        # 解出 data/books/alice_en.txt、ocean_tale.txt
tar xzf rag_storage.tar.gz                 # 解出 rag_storage/<book>/kv_store_*.json
```

### 3. 首次启动 Neo4j + Qdrant 创建数据卷，然后停止

```bash
docker compose up -d neo4j qdrant
# 等 ~5 秒让卷初始化完成
docker compose stop neo4j qdrant
```

### 4. 把快照导入到对应数据卷

compose 会用「目录名 + 卷名」命名数据卷。clone 到 `Book-Q-A-Assistant` 时，卷名是
`book-q-a-assistant_neo4j_data` 与 `book-q-a-assistant_qdrant_data`。
**先用 `docker volume ls` 确认实际卷名**，再导入：

```bash
# 确认卷名（应看到 ..._neo4j_data 和 ..._qdrant_data）
docker volume ls

# Windows PowerShell（注意路径用绝对路径）：
docker run --rm `
  -v book-q-a-assistant_neo4j_data:/dst `
  -v "${PWD}:/src" alpine tar xzf /src/neo4j_data.tar.gz -C /dst

docker run --rm `
  -v book-q-a-assistant_qdrant_data:/dst `
  -v "${PWD}:/src" alpine tar xzf /src/qdrant_data.tar.gz -C /dst
```

> Linux / macOS / Git Bash：去掉反引号续行，写成单行即可。
> 若你的卷名前缀不同（比如 clone 到别的目录名），把 `book-q-a-assistant_` 替换为 `docker volume ls` 显示的实际前缀。

### 5. 启动全部服务

```bash
docker compose up -d
```

首次会拉 Ollama 镜像并自动 `ollama pull qwen2.5:7b-instruct`（约 4.7GB，需等待）。

### 6. 验证

```bash
# 健康检查
curl http://localhost:8010/health

# 列出已恢复的文档（应看到 alice_en.txt 与 ocean_tale.txt）
curl http://localhost:8010/documents

# 提问（流式）
curl -N -X POST http://localhost:8010/chat `
  -H "Content-Type: application/json" `
  -d '{"question":"主角是谁？","book":"alice_en.txt","mode":"hybrid"}'
```

浏览器打开 http://localhost:8010/ 可用可视化 UI，在「概览」tab 应能看到两本书的实体/关系统计。

## 常见问题

**Q: 导入后查询报错 / 看不到数据？**
A: 多半是卷名前缀对不上。`docker volume ls` 看实际卷名，确保导入到 compose 实际使用的那个卷。`rag_storage/` 也必须在项目根目录（app 容器 bind mount `./rag_storage`）。

**Q: workspace 名必须一致吗？**
A: 是。LightRAG `workspace` = 书的 basename。快照里 `alice_en.txt` 的图谱在 Neo4j label `alice_en.txt`、Qdrant workspace_id 同名、`rag_storage/alice_en.txt/` KV。所以**书文件名不能改**——`alice_en.txt`、`ocean_tale.txt` 必须保持原文件名，否则对不上。

**Q: 想重新导入其中一本书（更新内容）？**
A: 用 `POST /documents/upsert` 或 `POST /documents/{book}/refresh`，会删该书 workspace 再重导——此时会调用 GLM 抽取（消耗配额）。其它书不受影响。

**Q: 版本不匹配怎么办？**
A: 本快照由 v0.2.0 生成。请确保 `git checkout v0.2.0`，不要用更新的代码恢复旧快照，以免存储格式差异导致问题。

## 端口

| 服务 | 端口 |
|------|------|
| FastAPI / 前端 UI | 8010 |
| Neo4j Bolt / Browser | 7687 / 7474 |
| Qdrant HTTP / gRPC | 16333 / 16334 |
| Ollama | 21434（容器内，一般无需直连） |
