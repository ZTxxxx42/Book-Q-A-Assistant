# 部署指南：Ollama（本地 LLM）+ SiliconFlow（Embedding/Rerank API）

本项目问答生成走本地 **Ollama**（Qwen2.5-7B-Instruct，OpenAI 兼容，GPU 推理）；
embedding 与 rerank 走 **SiliconFlow（硅基流动）** API，不占用本地 GPU。
向量库用 **Qdrant**（Docker 容器，真 cosine）；KV 缓存与文档状态仍用 LightRAG 原生 JSON 落 `working_dir`。

## 端口

| 服务 | 端口 |
|------|------|
| Ollama（OpenAI 兼容 API） | 11434 |
| FastAPI | 8010 |
| Neo4j Bolt / Browser | 7687 / 7474 |
| Qdrant HTTP / gRPC | 16333 / 16334 |

> Qdrant 用 16333 而非默认 6333：6333/6334 落在 Windows Hyper-V 保留端口段，Docker bind 会失败。

## 1. Ollama（本地 Qwen LLM）

安装 Ollama：https://ollama.com/download （Windows 版）。安装后：

```powershell
ollama pull qwen2.5:7b-instruct    # ~4.7GB，RTX 4060 8GB 足够
ollama serve                       # 后台常驻，OpenAI API 在 http://localhost:11434/v1
# 验证：
curl http://localhost:11434/v1/models
```

- Ollama 自动用 GPU 推理（CPU 回退）。
- 默认 `OLLAMA_NUM_PARALLEL=1`，故 `.env` 设 `LLM_MODEL_MAX_ASYNC=1`。需要并发时启动 Ollama 前设 `$env:OLLAMA_NUM_PARALLEL=2`（注意 8GB 显存需容纳 2 路 Qwen 上下文）。

## 2. SiliconFlow（Embedding + Rerank API）

1. 注册 https://siliconflow.cn 获取 API Key。
2. 填入 `.env` 的 `EMBEDDING_API_KEY` 与 `RERANK_API_KEY`（通常同一个 key）。
3. 模型：
   - embedding：`BAAI/bge-m3`（1024 维）
   - rerank：`BAAI/bge-reranker-v2-m3`
4. 端点：`https://api.siliconflow.cn/v1`（embedding 走 `/embeddings` OpenAI 兼容；rerank 走 `/rerank` 专用端点）。

> 国内直连、有免费额度。本小书 ingest 一次的 embedding/rerank 费用极低。

## 3. Neo4j + Qdrant（Docker）

```powershell
docker compose up -d      # 同时启动 neo4j + qdrant
```

- Neo4j Browser：http://localhost:7474 ，用 `neo4j / bookgraph123` 登录查看图谱。
- Qdrant Dashboard：http://localhost:16333/dashboard 查看 collection。
- 数据持久化在 docker volume（`neo4j_data`、`qdrant_data`），容器删除不丢数据。

> 若 Qdrant 启动报 `ports are not available ... 6333`，说明宿主端口被占用——确认 compose 映射到 16333。

## 4. 配置 `.env`

```bash
cp .env.example .env
# 填入 SiliconFlow key（EMBEDDING_API_KEY / RERANK_API_KEY）
```

## 5. 安装依赖

```powershell
D:/miniforge/envs/my_env/python.exe -m pip install -r requirements.txt
```

## 6. 启动验证

```powershell
# 冒烟（5 项全过才继续）
D:/miniforge/envs/my_env/python.exe scripts/smoke_remote_models.py

# 启动 API
D:/miniforge/envs/my_env/python.exe -m uvicorn src.api:app --port 8010 --host 127.0.0.1
curl http://localhost:8010/health
```

## 排错

历史问题（本地 bge + milvus 栈时期）与迁移迭代过程见 `TROUBLESHOOTING.md`。
