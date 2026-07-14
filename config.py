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
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
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
