"""统一日志配置：RotatingFileHandler 落盘 + 控制台，按级别可配。

- ``book_kg`` 命名空间走配置级别（默认 INFO）；
- 第三方库（httpx / openai / uvicorn 访问日志等）压到 WARNING，避免噪声；
- 文件 ``logs/app.log`` 按 10MB 轮转、保留 5 份。
"""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_log_config(level: str = "INFO", log_file: Path | str = "logs/app.log") -> dict:
    """返回 ``logging.config.dictConfig`` 兼容的字典，可供 uvicorn ``log_config`` 复用。"""
    level = (level or "INFO").upper()
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": _LOG_FORMAT, "datefmt": _DATE_FORMAT},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
                "stream": "ext://sys.stderr",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "level": level,
                "filename": str(log_file),
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            # 项目代码：配置级别。
            "book_kg": {"handlers": ["console", "file"], "level": level, "propagate": False},
            # 第三方噪声压到 WARNING。
            "httpx": {"handlers": ["console", "file"], "level": "WARNING", "propagate": False},
            "openai": {"handlers": ["console", "file"], "level": "WARNING", "propagate": False},
            "httpcore": {"handlers": ["console", "file"], "level": "WARNING", "propagate": False},
            "uvicorn.access": {"handlers": ["console", "file"], "level": "WARNING", "propagate": False},
        },
        # root 兜底：INFO，走控制台+文件。
        "root": {"handlers": ["console", "file"], "level": "INFO"},
    }


def setup_logging(level: str = "INFO", log_file: Path | str = "logs/app.log") -> None:
    """应用日志配置（CLI 入口用）。"""
    logging.config.dictConfig(build_log_config(level, log_file))
