# Book → Knowledge Graph

将一整本书自动转化为知识图谱：使用 **LightRAG** 抽取实体与关系，存入 **Neo4j** 图数据库，并提供混合检索查询接口。

## 架构

```
书籍 (PDF / TXT / EPUB / Markdown)
        │
        ▼
   loader.py        加载 + 分块
        │
        ▼
 graph_builder.py   LightRAG 调用 LLM 抽取实体/关系
        │
        ├── KV Storage      本地 JSON (原文块 / 上下文)
        ├── Vector Storage  向量索引 (检索召回)
        └── Graph Storage   Neo4j (实体节点 + 关系边)  ← 核心
        │
        ▼
    query.py        混合检索 (local / global / hybrid / naive)
```

LightRAG 自带 `Neo4JStorage` 后端：抽取出的实体、关系会直接写入 Neo4j；原文块与向量索引由本地存储维护；查询时三路融合召回。

## 目录结构

```
book_knowledge_graph/
├── README.md
├── requirements.txt
├── .env.example
├── config.py              配置 (LLM / Neo4j / 路径)
├── docker-compose.yml     一键启动 Neo4j
├── main.py                CLI 入口 (ingest / query / stats)
├── src/
│   ├── __init__.py
│   ├── loader.py          书籍加载与分块
│   ├── graph_builder.py   LightRAG + Neo4j 集成
│   └── query.py           查询接口
├── scripts/
│   └── ingest.py          批量导入示例
└── data/books/            放置待处理书籍
```

## 快速开始

> 完整部署细节见 [`docs/LOCAL_LLM_SETUP.md`](docs/LOCAL_LLM_SETUP.md)。LLM 按角色拆分：**实体/关键词抽取 → 远程 GLM-4.7**；**最终答复生成 → 本地 Ollama Qwen2.5-7B-Instruct（需 NVIDIA GPU）**。embedding/rerank 走 SiliconFlow API。

### 0. 前置要求

- **NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)**：Ollama Qwen 答复模型走容器内 GPU 直通。无 GPU 无法即拉即用（CPU 退化为极慢）。
- **Docker**（用于 Neo4j + Qdrant + Ollama）。
- **Python 解释器**：推荐 miniforge 环境（`D:/miniforge/envs/my_env/python.exe`），含项目所需的 pinned 依赖；也可用自带 venv。

### 1. 配置环境变量

```bash
cp .env.example .env
# 填入：LLM_API_KEY（智谱 GLM）、EMBEDDING_API_KEY / RERANK_API_KEY（SiliconFlow）
# QUERY_LLM_* 默认指向本地 Ollama，无需 key
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 可选：pytest
```

### 3. 启动 Neo4j + Qdrant + Ollama（含自动拉取 Qwen 模型）

```bash
docker compose up -d
```

首次启动时 `ollama-pull` 服务会自动拉取 `qwen2.5:7b-instruct`（~4.7GB，幂等，已存在则秒过）；`app` 服务在模型拉取完成后才启动。Ollama API 在 http://localhost:21434/v1 。

> Neo4j Browser：http://localhost:7474 ，用 `neo4j / bookgraph123` 登录。
> Qdrant Dashboard：http://localhost:16333/dashboard 。

### 4. 冒烟验证（5 项全过才继续）

```bash
python scripts/smoke_remote_models.py   # SiliconFlow embed/rerank + Qdrant + GLM + Qwen + LightRAG init
# 单独验证某个 LLM 端点：
python scripts/check_model.py                 # 默认验证 GLM 抽取
python scripts/check_model.py --role query    # 验证本地 Qwen 答复
```

### 5. 启动 API

```bash
python -m uvicorn src.api:app --port 8010 --host 127.0.0.1
curl http://localhost:8010/health
```

### 6. 导入一本书并提问

```bash
python main.py ingest --file data/books/your_book.pdf
# 也支持 .txt / .md / .epub
python main.py query --question "书中的主角与谁有关系？" --mode hybrid --book your_book.pdf
python main.py stats --book your_book.pdf
```

## 查询模式说明

LightRAG 支持四种检索模式，通过 `--mode` 切换：

| 模式      | 适用场景                              |
|-----------|---------------------------------------|
| `local`   | 针对具体实体/事实提问                 |
| `global`  | 跨章节的整体性问题                    |
| `hybrid`  | local + global 融合（默认，推荐）     |
| `naive`   | 传统向量 RAG，不做图推理              |

## 在 Neo4j 中查看图谱

```cypher
// 查看所有实体节点（按类型）
MATCH (n) RETURN labels(n), count(*);

// 查看某实体的关系
MATCH (n:ENTITY {name: "某角色"})-[r]->(m) RETURN n, r, m;

// 关系最多的实体
MATCH (n:ENTITY)-[r]->() RETURN n.name, count(r) AS deg
ORDER BY deg DESC LIMIT 10;
```

## 注意事项

- 长书导入耗时较长（取决于 LLM 调用次数），可先用 `--max-chunks` 限制块数测试。
- 实体抽取走远程 GLM-4.7；**不要**把抽取角色路由到本地 Qwen——长抽取生成会拖崩笔记本 GPU（见 `docs/TROUBLESHOOTING.md`）。
- 重新导入同一本书用 `POST /documents/upsert` 或 `POST /documents/{book}/refresh`（先删该 workspace 再重导），不要手动清 `working_dir`。
- 每本书是独立知识图谱（LightRAG `workspace` = 书名 basename），不做跨书合并或跨书查询。
- Windows 端口坑：Ollama 用 `21434`（默认 11434 落在 Hyper-V 保留段）、Qdrant 用 `16333`（默认 6333 同因），均已在 compose 与 `.env.example` 对齐。
