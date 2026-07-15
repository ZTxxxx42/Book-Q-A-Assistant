"""FastAPI 服务入口：uvicorn src.api:app --reload"""
from __future__ import annotations

import uvicorn

from config import settings
from src.api import app  # noqa: F401  re-export for uvicorn
from src.logging_config import build_log_config

if __name__ == "__main__":
    # 生产配置：端口 8010（与 CLAUDE.md 一致），单 worker（对齐 initialize_share_data(workers=1)），
    # 关闭 reload（生产不应用 reload）。日志走统一 dictConfig（文件轮转 + 控制台）。
    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8010,
        workers=1,
        reload=False,
        log_config=build_log_config(settings.log_level, settings.log_dir / "app.log"),
        access_log=True,
    )
