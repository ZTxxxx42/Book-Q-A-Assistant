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
    # --- LLM（本地 Ollama，OpenAI 兼容）---
    llm_binding: str = field(default_factory=lambda: os.getenv("LLM_BINDING", "openai"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str | None = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL") or None
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    )
    llm_streaming: bool = field(
        default_factory=lambda: os.getenv("LLM_STREAMING", "true").lower() == "true"
    )
    # LightRAG LLM 并发；Ollama 默认 OLLAMA_NUM_PARALLEL=1，故默认 1。
    llm_model_max_async: int = field(
        default_factory=lambda: int(os.getenv("LLM_MODEL_MAX_ASYNC", "1"))
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

    def ensure_dirs(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
