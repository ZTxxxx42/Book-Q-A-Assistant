# 部署指南：GLM 抽取 + Ollama 答复 + SiliconFlow Embed/Rerank + Qdrant

本项目 LLM 按角色分流：
- **实体抽取 / 关键词抽取** → 远程 **GLM-4.7**（重活，不占本地 GPU）。
- **最终答复生成** → 本地 **Ollama Qwen2.5-7B-Instruct**（短答复流式，GPU 稳定）。

embedding 与 rerank 走 **SiliconFlow（硅基流动）** API。向量库用 **Qdrant**（Docker 容器，真 cosine）；KV 缓存与文档状态仍用 LightRAG 原生 JSON 落 `working_dir`。

## 端口

| 服务 | 端口 |
|------|------|
| Ollama（OpenAI 兼容 API） | 21434（默认 11434 在 Windows 保留段） |
| FastAPI | 8010 |
| Neo4j Bolt / Browser | 7687 / 7474 |
| Qdrant HTTP / gRPC | 16333 / 16334 |

> Qdrant 用 16333 而非默认 6333：6333/6334 落在 Windows Hyper-V 保留端口段，Docker bind 会失败。

## 1. GLM-4.7（远程抽取 LLM）

1. 注册智谱开放平台 https://open.bigmodel.cn 获取 API Key。
2. 填入 `.env` 的 `LLM_API_KEY`。
3. 端点：`https://open.bigmodel.cn/api/paas/v4`（OpenAI 兼容），模型 `glm-4.7`。
4. 并发 `LLM_MODEL_MAX_ASYNC=2`（避免 429，6 次 retry 兜底）。

> 抽取是 ingest 时的重 LLM 工作。本地 7B 跑抽取会因长持续生成拖崩笔记本 GPU（见 TROUBLESHOOTING），故抽取走远程。

## 2. Ollama（本地 Qwen 答复 LLM）

安装 Ollama：https://ollama.com/download （Windows 版）。安装后：

```powershell
ollama pull qwen2.5:7b-instruct    # ~4.7GB，RTX 4060 8GB 足够
# 注意：默认端口 11434 落在 Windows Hyper-V 保留段，改用 21434
$env:OLLAMA_HOST="127.0.0.1:21434"
ollama serve                       # 后台常驻，OpenAI API 在 http://localhost:21434/v1
# 验证：
curl http://localhost:21434/v1/models
```

- 仅用于 `query` 角色（最终答复生成，流式 SSE）。短答复，不会持续满载。
- 默认 `OLLAMA_NUM_PARALLEL=1`，故 `.env` 设 `QUERY_LLM_MODEL_MAX_ASYNC=1`。
- `.env` 对应字段：`QUERY_LLM_API_KEY=ollama`、`QUERY_LLM_BASE_URL=http://localhost:21434/v1`、`QUERY_LLM_MODEL=qwen2.5:7b-instruct`。

## 3. SiliconFlow（Embedding + Rerank API）

1. 注册 https://siliconflow.cn 获取 API Key。
2. 填入 `.env` 的 `EMBEDDING_API_KEY` 与 `RERANK_API_KEY`（通常同一个 key）。
3. 模型：
   - embedding：`BAAI/bge-m3`（1024 维）
   - rerank：`BAAI/bge-reranker-v2-m3`
4. 端点：`https://api.siliconflow.cn/v1`（embedding 走 `/embeddings` OpenAI 兼容；rerank 走 `/rerank` 专用端点）。

> 国内直连、有免费额度。本小书 ingest 一次的 embedding/rerank 费用极低。

## 4. Neo4j + Qdrant（Docker）

```powershell
docker compose up -d      # 同时启动 neo4j + qdrant
```

- Neo4j Browser：http://localhost:7474 ，用 `neo4j / bookgraph123` 登录查看图谱。
- Qdrant Dashboard：http://localhost:16333/dashboard 查看 collection。
- 数据持久化在 docker volume（`neo4j_data`、`qdrant_data`），容器删除不丢数据。

> 若 Qdrant 启动报 `ports are not available ... 6333`，说明宿主端口被占用——确认 compose 映射到 16333。

## 5. 配置 `.env`

```bash
cp .env.example .env
# 填入：LLM_API_KEY（GLM）、EMBEDDING_API_KEY / RERANK_API_KEY（SiliconFlow）
# QUERY_LLM_* 默认指向本地 Ollama，无需 key
```

## 6. 安装依赖

```powershell
D:/miniforge/envs/my_env/python.exe -m pip install -r requirements.txt
```

## 7. 启动验证

```powershell
# 冒烟（5 项全过才继续）
D:/miniforge/envs/my_env/python.exe scripts/smoke_remote_models.py

# 启动 API
D:/miniforge/envs/my_env/python.exe -m uvicorn src.api:app --port 8010 --host 127.0.0.1
curl http://localhost:8010/health
```

## 排错

历史问题（本地 bge + milvus 栈时期）与迁移迭代过程见 `TROUBLESHOOTING.md`。
