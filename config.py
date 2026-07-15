"""全局配置：YAML 主源 + 环境变量覆盖 + 代码默认值。

优先级：``环境变量 > config.yaml > 代码默认``。敏感项（api_key / password）继续走
``.env`` / 环境变量，不入 ``config.yaml``。``settings`` 为 import 时构造的单例，
故任何改动需重启进程才生效（配置 UI 据此提示"需重启"）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
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
_CONFIG_YAML = PROJECT_ROOT / "config.yaml"


def _load_yaml() -> dict:
    """读 config.yaml（若存在）→ dict；不存在或解析失败返回空 dict。"""
    if not _CONFIG_YAML.exists():
        return {}
    try:
        with _CONFIG_YAML.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        # 配置解析失败不应阻断启动——回退到 env + 默认值。
        return {}


_YAML = _load_yaml()


def _cfg(env_key: str, yaml_key: str, default, cast=str):
    """按 ``env > YAML > 默认`` 解析单个配置值。

    - ``cast`` 为 ``str``/``int``/``float`` 时对原始值做转换；
    - ``cast`` 为 ``bool`` 时按 ``"true"`` 字符串判定（YAML 原生 bool 直通）；
    - 原始值为空（None 或空串）时回退到下一级，最终回退 ``default``（可为 ``None``）。
    - 默认值同样过 cast（保证类型一致），但 ``None`` 默认原样返回。
    """
    raw = os.getenv(env_key)
    if raw is None or raw == "":
        raw = _YAML.get(yaml_key)
    if raw is None or raw == "":
        raw = default
    if raw is None:
        return None
    if cast is bool:
        return raw if isinstance(raw, bool) else str(raw).strip().lower() == "true"
    return cast(raw)


@dataclass
class Settings:
    # --- LLM：实体抽取 / 关键词抽取（远程 GLM-4.7，OpenAI 兼容）---
    # 抽取是 ingest 时的重 LLM 工作，走远程 API 不占本地 GPU。
    llm_binding: str = field(default_factory=lambda: _cfg("LLM_BINDING", "llm_binding", "openai"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str | None = field(default_factory=lambda: _cfg("LLM_BASE_URL", "llm_base_url", None))
    llm_model: str = field(default_factory=lambda: _cfg("LLM_MODEL", "llm_model", "glm-4.7"))
    llm_streaming: bool = field(default_factory=lambda: _cfg("LLM_STREAMING", "llm_streaming", True, bool))
    # GLM 并发：保守 2 避免 429（上一轮 4 并发触发限流）。
    llm_model_max_async: int = field(default_factory=lambda: _cfg("LLM_MODEL_MAX_ASYNC", "llm_model_max_async", 2, int))

    # --- Query LLM：最终答复生成（本地 Ollama Qwen，OpenAI 兼容）---
    # 仅 query 角色用；短答复生成，不持续满载故不会拖崩笔记本 GPU。
    query_llm_api_key: str = field(default_factory=lambda: os.getenv("QUERY_LLM_API_KEY", "ollama"))
    query_llm_base_url: str | None = field(default_factory=lambda: _cfg("QUERY_LLM_BASE_URL", "query_llm_base_url", None))
    query_llm_model: str = field(default_factory=lambda: _cfg("QUERY_LLM_MODEL", "query_llm_model", "qwen2.5:7b-instruct"))
    query_llm_streaming: bool = field(default_factory=lambda: _cfg("QUERY_LLM_STREAMING", "query_llm_streaming", True, bool))
    # Ollama 默认 OLLAMA_NUM_PARALLEL=1，故并发 1。
    query_llm_model_max_async: int = field(default_factory=lambda: _cfg("QUERY_LLM_MODEL_MAX_ASYNC", "query_llm_model_max_async", 1, int))

    # --- Embedding（SiliconFlow /v1/embeddings，OpenAI 兼容）---
    embedding_binding: str = field(default_factory=lambda: _cfg("EMBEDDING_BINDING", "embedding_binding", "openai"))
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    embedding_base_url: str | None = field(default_factory=lambda: _cfg("EMBEDDING_BASE_URL", "embedding_base_url", None))
    embedding_model: str = field(default_factory=lambda: _cfg("EMBEDDING_MODEL", "embedding_model", "BAAI/bge-m3"))
    embedding_dim: int = field(default_factory=lambda: _cfg("EMBEDDING_DIM", "embedding_dim", 1024, int))

    # --- Rerank（SiliconFlow /v1/rerank，非 OpenAI 兼容）---
    rerank_model: str = field(default_factory=lambda: _cfg("RERANK_MODEL", "rerank_model", "BAAI/bge-reranker-v2-m3"))
    rerank_api_key: str = field(default_factory=lambda: os.getenv("RERANK_API_KEY", ""))
    rerank_base_url: str | None = field(default_factory=lambda: _cfg("RERANK_BASE_URL", "rerank_base_url", None))
    enable_rerank: bool = field(default_factory=lambda: _cfg("ENABLE_RERANK", "enable_rerank", True, bool))

    # --- 向量库：Qdrant（Docker 容器，真 cosine）---
    # LightRAG 的 QdrantVectorDBStorage 从 QDRANT_URL / QDRANT_API_KEY 环境变量读取连接。
    qdrant_url: str = field(default_factory=lambda: _cfg("QDRANT_URL", "qdrant_url", "http://localhost:16333"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)

    # --- Neo4j ---
    neo4j_uri: str = field(default_factory=lambda: _cfg("NEO4J_URI", "neo4j_uri", "bolt://localhost:7687"))
    neo4j_username: str = field(default_factory=lambda: _cfg("NEO4J_USERNAME", "neo4j_username", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "bookgraph123"))

    # --- 存储 / 分块 ---
    working_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / _cfg("WORKING_DIR", "working_dir", "rag_storage")
    )
    chunk_size: int = field(default_factory=lambda: _cfg("CHUNK_SIZE", "chunk_size", 1200, int))
    chunk_overlap: int = field(default_factory=lambda: _cfg("CHUNK_OVERLAP", "chunk_overlap", 100, int))
    language: str = field(default_factory=lambda: _cfg("LANGUAGE", "language", "chinese"))

    # --- 查询 / 检索质量（QueryParam 旋钮，显式传入 LightRAG）---
    # 实体/关系召回数；小库可下调。
    top_k: int = field(default_factory=lambda: _cfg("TOP_K", "top_k", 40, int))
    # 向量召回 + rerank 后保留的 chunk 数。
    chunk_top_k: int = field(default_factory=lambda: _cfg("CHUNK_TOP_K", "chunk_top_k", 12, int))
    # rerank 分数门槛；0.0 = 不过滤（默认 LightRAG 行为）。0.3 过滤低相关 chunk。
    # 注意：LightRAG 直接读 MIN_RERANK_SCORE env，这里仅作记录与统一管理。
    min_rerank_score: float = field(default_factory=lambda: _cfg("MIN_RERANK_SCORE", "min_rerank_score", 0.3, float))
    # 答复是否附引用出处（reference_id + file_path）。
    include_references: bool = field(default_factory=lambda: _cfg("INCLUDE_REFERENCES", "include_references", True, bool))
    # 答复格式：Multiple Paragraphs / Single Paragraph / Bullet Points。
    response_type: str = field(default_factory=lambda: _cfg("RESPONSE_TYPE", "response_type", "Single Paragraph"))

    # --- Query 容错护栏（应用层，A1 硬拒答）---
    # 预探测（aquery_data, only_need_context）命中的最少 chunk 数；不足则硬拒答不调 LLM。
    min_hit_count: int = field(default_factory=lambda: _cfg("MIN_HIT_COUNT", "min_hit_count", 1, int))
    # 硬拒答的 rerank 分数门槛（最高分低于此值视为无关问题）。
    hard_refuse_threshold: float = field(default_factory=lambda: _cfg("HARD_REFUSE_THRESHOLD", "hard_refuse_threshold", 0.3, float))
    # 硬拒答时返回的文案。
    hard_refuse_answer: str = field(
        default_factory=lambda: _cfg("HARD_REFUSE_ANSWER", "hard_refuse_answer", "知识库中未找到足够信息回答此问题。")
    )

    # --- LLM 调用鲁棒性（A2）+ SSE/CORS（A4）---
    # GLM（抽取/关键词）单次请求超时（秒）。
    llm_timeout: float = field(default_factory=lambda: _cfg("LLM_TIMEOUT", "llm_timeout", 60, float))
    # 本地 Qwen（答复生成）单次请求超时（秒）；本地生成慢，放宽。
    query_llm_timeout: float = field(default_factory=lambda: _cfg("QUERY_LLM_TIMEOUT", "query_llm_timeout", 120, float))
    # 流式首 token 超时（秒）；超时仍未 yield 首 token 视为卡死。
    stream_first_token_timeout: float = field(default_factory=lambda: _cfg("STREAM_FIRST_TOKEN_TIMEOUT", "stream_first_token_timeout", 60, float))
    # SSE 心跳间隔（秒，A4）。
    stream_heartbeat_interval: float = field(default_factory=lambda: _cfg("STREAM_HEARTBEAT_INTERVAL", "stream_heartbeat_interval", 15, float))
    # CORS 允许的来源（A4，逗号分隔；* 为全部；空则不启用 CORS）。
    cors_origins: str = field(default_factory=lambda: _cfg("CORS_ORIGINS", "cors_origins", ""))

    # --- 输入校验（A0）---
    max_question_length: int = field(default_factory=lambda: _cfg("MAX_QUESTION_LENGTH", "max_question_length", 2000, int))
    max_book_name_length: int = field(default_factory=lambda: _cfg("MAX_BOOK_NAME_LENGTH", "max_book_name_length", 200, int))
    max_history_turns: int = field(default_factory=lambda: _cfg("MAX_HISTORY_TURNS", "max_history_turns", 10, int))

    # --- 并发硬化（B0/B1/B2）---
    # LightRAG 实例池容量（按 workspace 缓存）。
    rag_pool_size: int = field(default_factory=lambda: _cfg("RAG_POOL_SIZE", "rag_pool_size", 5, int))
    # 全局并发闸限额（initialize_share_data global_concurrency_limits）。
    global_extract_concurrency: int = field(default_factory=lambda: _cfg("GLOBAL_EXTRACT_CONCURRENCY", "global_extract_concurrency", 2, int))
    global_keyword_concurrency: int = field(default_factory=lambda: _cfg("GLOBAL_KEYWORD_CONCURRENCY", "global_keyword_concurrency", 2, int))
    global_query_concurrency: int = field(default_factory=lambda: _cfg("GLOBAL_QUERY_CONCURRENCY", "global_query_concurrency", 1, int))
    global_embedding_concurrency: int = field(default_factory=lambda: _cfg("GLOBAL_EMBEDDING_CONCURRENCY", "global_embedding_concurrency", 4, int))
    global_rerank_concurrency: int = field(default_factory=lambda: _cfg("GLOBAL_RERANK_CONCURRENCY", "global_rerank_concurrency", 4, int))
    # HTTP 入口最大并发请求数（限流中间件）。
    max_concurrent_requests: int = field(default_factory=lambda: _cfg("MAX_CONCURRENT_REQUESTS", "max_concurrent_requests", 8, int))
    # ingest 最大并发（单机 GPU 一次一本）。
    max_concurrent_ingest: int = field(default_factory=lambda: _cfg("MAX_CONCURRENT_INGEST", "max_concurrent_ingest", 1, int))
    # /query 整体超时（秒）。
    query_overall_timeout: float = field(default_factory=lambda: _cfg("QUERY_OVERALL_TIMEOUT", "query_overall_timeout", 180, float))
    # 完成任务保留时长（秒，TTL 清理）。
    task_ttl: float = field(default_factory=lambda: _cfg("TASK_TTL", "task_ttl", 3600, float))

    def ensure_dirs(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
