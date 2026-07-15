"""全局配置：从环境变量读取，带默认值。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Qdrant 跑在本地 Docker，必须绕过系统代理（如 FlClash），否则 requests 走代理返回 502。
_no_proxy = os.environ.get("NO_PROXY", "")
for _h in ("localhost", "127.0.0.1"):
    if _h not in _no_proxy:
        _no_proxy = f"{_no_proxy},{_h}" if _no_proxy else _h
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "books"


@dataclass
class Settings:
    # --- LLM：实体抽取 / 关键词抽取（远程 GLM-4.7，OpenAI 兼容）---
    # 抽取是 ingest 时的重 LLM 工作，走远程 API 不占本地 GPU。
    llm_binding: str = field(default_factory=lambda: os.getenv("LLM_BINDING", "openai"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str | None = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL") or None
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "glm-4.7")
    )
    llm_streaming: bool = field(
        default_factory=lambda: os.getenv("LLM_STREAMING", "true").lower() == "true"
    )
    # GLM 并发：保守 2 避免 429（上一轮 4 并发触发限流）。
    llm_model_max_async: int = field(
        default_factory=lambda: int(os.getenv("LLM_MODEL_MAX_ASYNC", "2"))
    )

    # --- Query LLM：最终答复生成（本地 Ollama Qwen，OpenAI 兼容）---
    # 仅 query 角色用；短答复生成，不持续满载故不会拖崩笔记本 GPU。
    query_llm_api_key: str = field(
        default_factory=lambda: os.getenv("QUERY_LLM_API_KEY", "ollama")
    )
    query_llm_base_url: str | None = field(
        default_factory=lambda: os.getenv("QUERY_LLM_BASE_URL") or None
    )
    query_llm_model: str = field(
        default_factory=lambda: os.getenv("QUERY_LLM_MODEL", "qwen2.5:7b-instruct")
    )
    query_llm_streaming: bool = field(
        default_factory=lambda: os.getenv("QUERY_LLM_STREAMING", "true").lower() == "true"
    )
    # Ollama 默认 OLLAMA_NUM_PARALLEL=1，故并发 1。
    query_llm_model_max_async: int = field(
        default_factory=lambda: int(os.getenv("QUERY_LLM_MODEL_MAX_ASYNC", "1"))
    )

    # --- Embedding（SiliconFlow /v1/embeddings，OpenAI 兼容）---
    embedding_binding: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BINDING", "openai")
    )
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    embedding_base_url: str | None = field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL") or None
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))

    # --- Rerank（SiliconFlow /v1/rerank，非 OpenAI 兼容）---
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    )
    rerank_api_key: str = field(default_factory=lambda: os.getenv("RERANK_API_KEY", ""))
    rerank_base_url: str | None = field(
        default_factory=lambda: os.getenv("RERANK_BASE_URL") or None
    )
    enable_rerank: bool = field(
        default_factory=lambda: os.getenv("ENABLE_RERANK", "true").lower() == "true"
    )

    # --- 向量库：Qdrant（Docker 容器，真 cosine）---
    # LightRAG 的 QdrantVectorDBStorage 从 QDRANT_URL / QDRANT_API_KEY 环境变量读取连接。
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:16333"))
    qdrant_api_key: str | None = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY") or None
    )

    # --- Neo4j ---
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "bookgraph123"))

    # --- 存储 / 分块 ---
    working_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("WORKING_DIR", "rag_storage")
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1200")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    language: str = field(default_factory=lambda: os.getenv("LANGUAGE", "chinese"))

    # --- 查询 / 检索质量（QueryParam 旋钮，显式传入 LightRAG）---
    # 实体/关系召回数；小库可下调。
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "40")))
    # 向量召回 + rerank 后保留的 chunk 数。
    chunk_top_k: int = field(default_factory=lambda: int(os.getenv("CHUNK_TOP_K", "12")))
    # rerank 分数门槛；0.0 = 不过滤（默认 LightRAG 行为）。0.3 过滤低相关 chunk。
    # 注意：LightRAG 直接读 MIN_RERANK_SCORE env，这里仅作记录与统一管理。
    min_rerank_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_RERANK_SCORE", "0.3"))
    )
    # 答复是否附引用出处（reference_id + file_path）。
    include_references: bool = field(
        default_factory=lambda: os.getenv("INCLUDE_REFERENCES", "true").lower() == "true"
    )
    # 答复格式：Multiple Paragraphs / Single Paragraph / Bullet Points。
    response_type: str = field(
        default_factory=lambda: os.getenv("RESPONSE_TYPE", "Single Paragraph")
    )

    # --- Query 容错护栏（应用层，A1 硬拒答）---
    # 预探测（aquery_data, only_need_context）命中的最少 chunk 数；不足则硬拒答不调 LLM。
    min_hit_count: int = field(
        default_factory=lambda: int(os.getenv("MIN_HIT_COUNT", "1"))
    )
    # 硬拒答的 rerank 分数门槛（最高分低于此值视为无关问题）。
    hard_refuse_threshold: float = field(
        default_factory=lambda: float(os.getenv("HARD_REFUSE_THRESHOLD", "0.3"))
    )
    # 硬拒答时返回的文案。
    hard_refuse_answer: str = field(
        default_factory=lambda: os.getenv(
            "HARD_REFUSE_ANSWER", "知识库中未找到足够信息回答此问题。"
        )
    )

    # --- LLM 调用鲁棒性（A2）---
    # GLM（抽取/关键词）单次请求超时（秒）。
    llm_timeout: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "60"))
    )
    # 本地 Qwen（答复生成）单次请求超时（秒）；本地生成慢，放宽。
    query_llm_timeout: float = field(
        default_factory=lambda: float(os.getenv("QUERY_LLM_TIMEOUT", "120"))
    )
    # 流式首 token 超时（秒）；超时仍未 yield 首 token 视为卡死。
    stream_first_token_timeout: float = field(
        default_factory=lambda: float(os.getenv("STREAM_FIRST_TOKEN_TIMEOUT", "60"))
    )
    # SSE 心跳间隔（秒，A4）。
    stream_heartbeat_interval: float = field(
        default_factory=lambda: float(os.getenv("STREAM_HEARTBEAT_INTERVAL", "15"))
    )
    # CORS 允许的来源（A4，逗号分隔；* 为全部；空则不启用 CORS）。
    cors_origins: str = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "")
    )

    # --- 输入校验（A0）---
    max_question_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_QUESTION_LENGTH", "2000"))
    )
    max_book_name_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_BOOK_NAME_LENGTH", "200"))
    )
    max_history_turns: int = field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY_TURNS", "10"))
    )

    # --- 并发硬化（B0/B1/B2）---
    # LightRAG 实例池容量（按 workspace 缓存）。
    rag_pool_size: int = field(
        default_factory=lambda: int(os.getenv("RAG_POOL_SIZE", "5"))
    )
    # 全局并发闸限额（initialize_share_data global_concurrency_limits）。
    global_extract_concurrency: int = field(
        default_factory=lambda: int(os.getenv("GLOBAL_EXTRACT_CONCURRENCY", "2"))
    )
    global_keyword_concurrency: int = field(
        default_factory=lambda: int(os.getenv("GLOBAL_KEYWORD_CONCURRENCY", "2"))
    )
    global_query_concurrency: int = field(
        default_factory=lambda: int(os.getenv("GLOBAL_QUERY_CONCURRENCY", "1"))
    )
    global_embedding_concurrency: int = field(
        default_factory=lambda: int(os.getenv("GLOBAL_EMBEDDING_CONCURRENCY", "4"))
    )
    global_rerank_concurrency: int = field(
        default_factory=lambda: int(os.getenv("GLOBAL_RERANK_CONCURRENCY", "4"))
    )
    # HTTP 入口最大并发请求数（限流中间件）。
    max_concurrent_requests: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))
    )
    # ingest 最大并发（单机 GPU 一次一本）。
    max_concurrent_ingest: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_INGEST", "1"))
    )
    # /query 整体超时（秒）。
    query_overall_timeout: float = field(
        default_factory=lambda: float(os.getenv("QUERY_OVERALL_TIMEOUT", "180"))
    )
    # 完成任务保留时长（秒，TTL 清理）。
    task_ttl: float = field(
        default_factory=lambda: float(os.getenv("TASK_TTL", "3600"))
    )

    def ensure_dirs(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
