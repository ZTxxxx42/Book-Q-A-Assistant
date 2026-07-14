"""全局配置：从环境变量读取，带默认值。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "books"


@dataclass
class Settings:
    # --- LLM ---
    llm_binding: str = field(default_factory=lambda: os.getenv("LLM_BINDING", "openai"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str | None = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL") or None
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))

    # --- Embedding ---
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

    # --- Rerank（本地 bge-reranker）---
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    )
    enable_rerank: bool = field(
        default_factory=lambda: os.getenv("ENABLE_RERANK", "true").lower() == "true"
    )
    llm_streaming: bool = field(
        default_factory=lambda: os.getenv("LLM_STREAMING", "true").lower() == "true"
    )

    # --- Milvus 向量库 ---
    milvus_uri: str = field(
        default_factory=lambda: os.getenv("MILVUS_URI", "rag_storage/milvus_lite.db")
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

    # --- 本地模型缓存（bge-m3 / bge-reranker / Qwen 等，集中存项目内）---
    # MODEL_CACHE_DIR 可为相对名（相对 PROJECT_ROOT）或绝对路径；默认 model_cache/
    model_cache_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("MODEL_CACHE_DIR", "model_cache")
    )

    def ensure_dirs(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()

# 重定向模型下载到项目内（须在任何 modelscope / huggingface_hub import 之前生效）。
# - MODELSCOPE_CACHE：modelscope Config.cache_dir 的 default_factory 读此 env，
#   标准布局 {cache}/models/{owner}--{name}/snapshots/{rev}/。
# - HF_HOME / HUGGINGFACE_HUB_CACHE：_resolve_model 在 modelscope 失败时回退 HF，
#   让 HF 下载也落在项目内而非 ~/.cache/huggingface。
os.environ.setdefault("MODELSCOPE_CACHE", str(settings.model_cache_dir))
os.environ.setdefault("HF_HOME", str(settings.model_cache_dir / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(settings.model_cache_dir / "hf"))
