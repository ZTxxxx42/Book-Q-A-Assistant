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

### 1. 安装依赖

```bash
cd book_knowledge_graph
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. 启动 Neo4j

```bash
docker compose up -d
```

浏览器访问 http://localhost:7474 ，用 `neo4j / bookgraph123` 登录查看图谱。

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 LLM API Key
```

### 4. 导入一本书

```bash
python main.py ingest --file data/books/your_book.pdf
# 也支持 .txt / .md / .epub
```

### 5. 提问

```bash
python main.py query --question "书中的主角与谁有关系？"
python main.py query --question "这本书的核心论点是什么" --mode hybrid
```

### 6. 查看统计

```bash
python main.py stats
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
- LightRAG 的实体类型标注依赖 LLM 提示，建议用能力较强的模型（如 `gpt-4o-mini` 或 Claude Sonnet）。
- 重新导入前清空 `working_dir` 与 Neo4j 中的数据，避免重复实体。
