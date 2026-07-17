"""快速验证 .env 中配置的 LLM 模型是否可用。

默认验证抽取 LLM（GLM）；加 --role query 验证答复 LLM（本地 Ollama Qwen）。
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from openai import AsyncOpenAI


def _pick(role: str):
    """按角色返回 (api_key, base_url, model) 三元组。"""
    if role == "query":
        return (
            settings.query_llm_api_key or "ollama",
            settings.query_llm_base_url,
            settings.query_llm_model,
        )
    if role == "extract":
        return settings.llm_api_key, settings.llm_base_url, settings.llm_model
    raise ValueError(f"未知 role: {role}（支持 extract / query）")


async def main() -> None:
    parser = argparse.ArgumentParser(description="验证 .env 配置的 LLM 端点可达。")
    parser.add_argument(
        "--role",
        choices=["extract", "query"],
        default="extract",
        help="extract=抽取 LLM（GLM-4.7，默认）；query=答复 LLM（本地 Ollama Qwen）",
    )
    args = parser.parse_args()

    api_key, base_url, model = _pick(args.role)
    print(f"role     : {args.role}")
    print(f"base_url : {base_url}")
    print(f"model    : {model}")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "回复两个字：可用"}],
        )
        print(f"响应     : {r.choices[0].message.content}")
        print("✅ 模型可用")
    except Exception as e:
        print(f"❌ 调用失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
