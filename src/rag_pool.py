"""LightRAG 实例池：按 workspace 缓存，LRU 淘汰，消除每请求新建实例的连接泄漏。

每本书一个 workspace；查询类只读请求从池取（命中则复用已 initialize_storages 的实例），
用完 release 归还。容量满时淘汰最久未用且 in-flight=0 的实例并 finalize_storages（关闭
Neo4j driver / Qdrant client）。重写操作（ingest/delete）不入池，独占 build+init+finalize。
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from config import settings


class _Entry:
    __slots__ = ("rag", "in_flight", "lock")

    def __init__(self, rag: Any) -> None:
        self.rag = rag
        self.in_flight = 0
        # per-workspace 初始化锁：防止同 workspace 首次并发请求双 initialize
        self.lock = asyncio.Lock()


class RagPool:
    """按 workspace 的 LRU LightRAG 实例池。"""

    def __init__(self, max_size: int | None = None) -> None:
        self._max_size = max_size or settings.rag_pool_size
        self._pool: "OrderedDict[str, _Entry]" = OrderedDict()
        self._global_lock = asyncio.Lock()

    async def acquire(self, workspace: str) -> Any:
        """取一个已初始化的 LightRAG 实例（命中复用，未命中构建）。"""
        async with self._global_lock:
            entry = self._pool.get(workspace)
            if entry is None:
                entry = _Entry(None)  # placeholder，先占位
                self._pool[workspace] = entry
            else:
                self._pool.move_to_end(workspace)
            entry.in_flight += 1

        # 若实例未就绪，在 per-workspace lock 内初始化（防同 workspace 并发双 init）
        if entry.rag is None:
            async with entry.lock:
                if entry.rag is None:
                    from src.graph_builder import _build_rag

                    rag = _build_rag(workspace=workspace)
                    await rag.initialize_storages()
                    entry.rag = rag
                    async with self._global_lock:
                        await self._evict_if_needed()
        return entry.rag

    async def release(self, workspace: str) -> None:
        """归还一个实例（递减 in-flight）。"""
        async with self._global_lock:
            entry = self._pool.get(workspace)
            if entry is not None:
                entry.in_flight = max(0, entry.in_flight - 1)
                self._pool.move_to_end(workspace)

    async def _evict_if_needed(self) -> None:
        """容量超限时淘汰最久未用且 in-flight=0 的实例。须持有 _global_lock。"""
        while len(self._pool) > self._max_size:
            evicted_ws: str | None = None
            evicted_entry: _Entry | None = None
            # OrderedDict 按插入序（最近 used 在尾），从头找第一个 in_flight=0
            for ws, entry in self._pool.items():
                if entry.in_flight == 0:
                    evicted_ws = ws
                    evicted_entry = entry
                    break
            if evicted_ws is None:
                break  # 全在用，本轮不淘汰
            del self._pool[evicted_ws]
            if evicted_entry.rag is not None:
                try:
                    await evicted_entry.rag.finalize_storages()
                except Exception:
                    pass

    async def shutdown(self) -> None:
        """lifespan 退出时关闭全池实例。"""
        async with self._global_lock:
            for ws, entry in list(self._pool.items()):
                if entry.rag is not None:
                    try:
                        await entry.rag.finalize_storages()
                    except Exception:
                        pass
            self._pool.clear()

    async def invalidate(self, workspace: str) -> None:
        """使池中该 workspace 的实例失效并关闭（重写操作后调用，避免复用过期实例）。

        in_flight>0 的实例也会被移除（重写已发生，旧实例不可信），关闭由 finalize 尽力执行。
        """
        async with self._global_lock:
            entry = self._pool.pop(workspace, None)
        if entry is not None and entry.rag is not None:
            try:
                await entry.rag.finalize_storages()
            except Exception:
                pass


# 进程级单例（由 api.py lifespan 管理；CLI 入口用 get_pool 按需创建）
_pool_singleton: RagPool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> RagPool:
    global _pool_singleton
    if _pool_singleton is None:
        async with _pool_lock:
            if _pool_singleton is None:
                _pool_singleton = RagPool()
    return _pool_singleton


async def shutdown_pool() -> None:
    global _pool_singleton
    if _pool_singleton is not None:
        await _pool_singleton.shutdown()
        _pool_singleton = None
