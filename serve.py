"""FastAPI 服务入口：uvicorn src.api:app --reload"""
from __future__ import annotations

import uvicorn

from src.api import app  # noqa: F401  re-export for uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
