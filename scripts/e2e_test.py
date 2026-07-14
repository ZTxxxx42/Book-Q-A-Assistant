"""端到端测试：用新管线（Milvus + 本地 bge + 智谱 LLM）导入 alice 节选并流式问答。

用法：python scripts/e2e_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


async def main() -> None:
    from src.graph_builder import ingest_book
    from src.query import ask_stream

    book = "data/books/alice_en.txt"
    print(f"=== 导入 {book}（max_chunks=15）===")
    info = await ingest_book(book, max_chunks=15)
    print(info)

    print("\n=== 流式问答（hybrid + rerank）===")
    question = "Who are the main characters in this story?"
    print(f"Q: {question}")
    print("A: ", end="", flush=True)
    async for chunk in ask_stream(question, mode="hybrid"):
        print(chunk, end="", flush=True)
    print("\n\n=== 流式问答完成 ===")


if __name__ == "__main__":
    asyncio.run(main())
