"""session_store 单元测试：JSON 文件会话存储的核心行为。

覆盖：创建、列表过滤与排序、归属隔离、消息追加、标题生成、索引一致性、删除、
索引损坏容错。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import session_store


async def test_create_session_returns_empty_messages(session_dir: Path) -> None:
    s = await session_store.create_session("userA", "alice.txt", "hybrid")
    assert s["id"]
    assert s["user_id"] == "userA"
    assert s["book"] == "alice.txt"
    assert s["mode"] == "hybrid"
    assert s["messages"] == []
    assert s["title"] == ""
    # 会话文件与索引文件均已落盘
    assert (session_dir / f"{s['id']}.json").exists()
    assert (session_dir / "index.json").exists()


async def test_list_sessions_filters_by_user_and_sorts_desc(session_dir: Path) -> None:
    a1 = await session_store.create_session("userA", "a.txt", "hybrid")
    a2 = await session_store.create_session("userA", "b.txt", "hybrid")
    await session_store.create_session("userB", "c.txt", "hybrid")

    items = await session_store.list_sessions("userA")
    assert [it["id"] for it in items] == [a2["id"], a1["id"]]  # 按 updated_at 降序
    assert all(it["user_id"] == "userA" for it in items)


async def test_get_session_cross_user_returns_none(session_dir: Path) -> None:
    s = await session_store.create_session("userA", "alice.txt", "hybrid")
    assert await session_store.get_session(s["id"], "userA") is not None
    assert await session_store.get_session(s["id"], "userB") is None  # 越权
    assert await session_store.get_session("nonexistent", "userA") is None  # 不存在


async def test_append_message_sets_title_from_first_user_message(session_dir: Path) -> None:
    s = await session_store.create_session("userA", "alice.txt", "hybrid")
    long_question = "请帮我详细分析一下爱丽丝梦游仙境这本书的主要角色和故事情节"
    await session_store.append_message(s["id"], "userA", "user", long_question)
    await session_store.append_message(s["id"], "userA", "assistant", "爱丽丝是主角…")

    got = await session_store.get_session(s["id"], "userA")
    assert got is not None
    assert len(got["messages"]) == 2
    # 标题 = 首条 user 消息前 30 字
    assert got["title"] == long_question[:30]
    # assistant 消息带 refs 字段
    assert got["messages"][1]["role"] == "assistant"
    assert got["messages"][1]["refs"] == []


async def test_append_message_with_refs_and_index_sync(session_dir: Path) -> None:
    s = await session_store.create_session("userA", "alice.txt", "hybrid")
    refs = [{"file_path": "alice.txt", "reference_id": "c1"}]
    await session_store.append_message(s["id"], "userA", "user", "问题")
    await session_store.append_message(
        s["id"], "userA", "assistant", "答复", refs=refs,
    )

    # 索引项的 title/updated_at 已同步
    items = await session_store.list_sessions("userA")
    assert items[0]["title"] == "问题"
    got = await session_store.get_session(s["id"], "userA")
    assert got["messages"][1]["refs"] == refs


async def test_append_message_wrong_user_returns_none(session_dir: Path) -> None:
    s = await session_store.create_session("userA", "alice.txt", "hybrid")
    result = await session_store.append_message(s["id"], "userB", "user", "hi")
    assert result is None
    got = await session_store.get_session(s["id"], "userA")
    assert got["messages"] == []  # 未写入


async def test_delete_session_owner_only(session_dir: Path) -> None:
    s = await session_store.create_session("userA", "alice.txt", "hybrid")
    # 越权删除失败
    assert await session_store.delete_session(s["id"], "userB") is False
    assert (session_dir / f"{s['id']}.json").exists()  # 文件仍在
    # 属主删除成功
    assert await session_store.delete_session(s["id"], "userA") is True
    assert not (session_dir / f"{s['id']}.json").exists()
    # 索引中已移除
    assert await session_store.list_sessions("userA") == []


async def test_delete_nonexistent_returns_false(session_dir: Path) -> None:
    assert await session_store.delete_session("nope", "userA") is False


async def test_index_corruption_tolerated(session_dir: Path) -> None:
    """index.json 损坏时回退空索引，不阻断读取。"""
    await session_store.create_session("userA", "alice.txt", "hybrid")
    (session_dir / "index.json").write_text("not json{", encoding="utf-8")
    assert await session_store.list_sessions("userA") == []
    # 写操作仍可恢复（读索引失败→空，再写回）
    s2 = await session_store.create_session("userA", "b.txt", "hybrid")
    assert len(await session_store.list_sessions("userA")) == 1


async def test_atomic_index_write_no_tmp_leftover(session_dir: Path) -> None:
    """索引原子写：正常路径下不留 .tmp 残留。"""
    await session_store.create_session("userA", "alice.txt", "hybrid")
    await session_store.append_message(
        (await session_store.create_session("userA", "b.txt", "hybrid"))["id"],
        "userA", "user", "q",
    )
    assert not list(session_dir.glob("*.tmp"))
