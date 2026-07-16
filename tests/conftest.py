"""pytest 共享夹具。

会话存储测试要点：
- 每个测试用独立临时目录作 session_dir，互不污染；
- 每个测试重置 ``session_store._lock``，避免模块级 ``asyncio.Lock`` 跨事件循环
  绑定报错（pytest-asyncio 默认每个 async 测试一个新 loop）。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from config import settings
from src import session_store


@pytest.fixture(autouse=True)
def _fresh_lock() -> None:
    """每个测试重置会话存储的 asyncio.Lock。"""
    session_store._lock = asyncio.Lock()


@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 settings.session_dir 指向临时目录并创建之。"""
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(settings, "session_dir", d)
    return d
