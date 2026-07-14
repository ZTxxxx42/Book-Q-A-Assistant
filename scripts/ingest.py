"""批量导入示例：扫描 data/books 下所有支持的文件。"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.graph_builder import ingest_book

EXTS = {".pdf", ".txt", ".md", ".markdown", ".epub"}


async def main() -> None:
    settings.ensure_dirs()
    books = sorted(p for p in settings.DATA_DIR.iterdir() if p.suffix.lower() in EXTS)
    if not books:
        print(f"未在 {settings.DATA_DIR} 找到书籍（支持 {EXTS}）")
        return

    for book in books:
        print(f"\n=== 导入 {book.name} ===")
        info = await ingest_book(str(book))
        print(info)


if __name__ == "__main__":
    asyncio.run(main())
