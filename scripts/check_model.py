"""快速验证 .env 中配置的 LLM 模型是否可用。"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from openai import AsyncOpenAI


async def main() -> None:
    print(f"base_url : {settings.llm_base_url}")
    print(f"model    : {settings.llm_model}")
    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    try:
        r = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "回复两个字：可用"}],
        )
        print(f"响应     : {r.choices[0].message.content}")
        print("✅ 模型可用")
    except Exception as e:
        print(f"❌ 调用失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
