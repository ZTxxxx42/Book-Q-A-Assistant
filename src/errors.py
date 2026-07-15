"""应用层自定义异常。"""
from __future__ import annotations


class EmptyLLMResponseError(Exception):
    """LLM 返回空响应（content 为 None/空字符串）时抛出，避免静默成功。"""


class LLMTimeoutError(Exception):
    """LLM 调用超时（含流式首 token 超时）。"""
