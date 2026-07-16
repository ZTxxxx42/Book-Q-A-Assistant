"""会话管理 API 路由测试（FastAPI TestClient）。

仅测试 /sessions 四个端点 —— 它们只依赖 session_store，不触图、不连 Neo4j/Qdrant，
故用 TestClient(app) 且不进入 lifespan 上下文（避免启动实例池与外部依赖）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import api


@pytest.fixture
def client(session_dir: Path) -> TestClient:
    # 不使用 `with` → lifespan 不触发，无需 Neo4j/Qdrant/Ollama
    return TestClient(api.app)


def test_create_list_get_delete_lifecycle(client: TestClient) -> None:
    r = client.post("/sessions", json={"user_id": "u1", "book": "alice.txt", "mode": "hybrid"})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    assert r.json()["messages"] == []

    # 列表
    r = client.get("/sessions", params={"user_id": "u1"})
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["id"] == sid

    # 详情
    r = client.get(f"/sessions/{sid}", params={"user_id": "u1"})
    assert r.status_code == 200
    assert r.json()["book"] == "alice.txt"

    # 删除
    r = client.delete(f"/sessions/{sid}", params={"user_id": "u1"})
    assert r.status_code == 204

    # 删除后查询 → 404
    assert client.get(f"/sessions/{sid}", params={"user_id": "u1"}).status_code == 404


def test_create_with_default_mode(client: TestClient) -> None:
    r = client.post("/sessions", json={"user_id": "u1", "book": "alice.txt"})
    assert r.status_code == 201
    assert r.json()["mode"] == "hybrid"


def test_cross_user_isolation(client: TestClient) -> None:
    r = client.post("/sessions", json={"user_id": "A", "book": "alice.txt"})
    sid = r.json()["id"]

    # B 看不到 A 的会话
    assert client.get(f"/sessions/{sid}", params={"user_id": "B"}).status_code == 404
    # B 删不掉 A 的会话
    assert client.delete(f"/sessions/{sid}", params={"user_id": "B"}).status_code == 404
    # A 仍可访问
    assert client.get(f"/sessions/{sid}", params={"user_id": "A"}).status_code == 200
    # B 的列表不含该会话
    assert client.get("/sessions", params={"user_id": "B"}).json() == []


def test_list_missing_user_id_rejected(client: TestClient) -> None:
    # user_id 是必填查询参数
    assert client.get("/sessions").status_code == 422


def test_create_invalid_book_rejected(client: TestClient) -> None:
    # book 含路径分隔符 → 校验失败
    r = client.post("/sessions", json={"user_id": "u1", "book": "../evil.txt"})
    assert r.status_code == 422


def test_create_missing_user_id_rejected(client: TestClient) -> None:
    r = client.post("/sessions", json={"book": "alice.txt"})
    assert r.status_code == 422


def test_get_and_delete_nonexistent_returns_404(client: TestClient) -> None:
    assert client.get("/sessions/nope", params={"user_id": "u1"}).status_code == 404
    assert client.delete("/sessions/nope", params={"user_id": "u1"}).status_code == 404


def test_create_with_explicit_title(client: TestClient) -> None:
    r = client.post(
        "/sessions",
        json={"user_id": "u1", "book": "alice.txt", "title": "我的会话"},
    )
    assert r.status_code == 201
    assert r.json()["title"] == "我的会话"
