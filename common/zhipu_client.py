# -*- coding: utf-8 -*-
"""智谱 GLM 接口的手写 HTTP 封装（不依赖 LangChain）。

只做三件事：
1. 统一的 POST：超时、重试、把各种失败情况转成带上下文的异常；
2. Embedding：单条查询 + 批量文档向量化（按 index 回填，保证顺序不错位）；
3. Chat：把 prompt / messages 发给 GLM，返回纯文本回答。

设计取舍：失败时抛 ZhipuAPIError，而不是返回 None。
这样调用方能明确知道"卡在哪一步"，也能据此降级，而不是静默拿到 None。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

from common import config


class ZhipuAPIError(RuntimeError):
    """智谱接口调用失败：缺 Key、鉴权失败、HTTP 错误、返回结构异常、网络超时。"""


def post_json(path: str, payload: Dict[str, Any], tag: str = "API",
              timeout: Optional[int] = None) -> Dict[str, Any]:
    """统一 POST 封装：超时、重试、状态码与返回结构校验。"""
    url = path if path.startswith("http") else f"{config.API_BASE}/{path.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {config.require_api_key()}",
        "Content-Type": "application/json",
    }
    timeout = timeout or config.REQUEST_TIMEOUT
    last_error: Optional[Exception] = None

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < config.MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))  # 超时退避重试
                continue
            raise ZhipuAPIError(f"[{tag}] 请求超时（已重试 {config.MAX_RETRIES} 次）") from e
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < config.MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))  # 连接抖动/被限流时退避重试
                continue
            raise ZhipuAPIError(f"[{tag}] 网络请求失败：{e}") from e

        if resp.status_code == 401:
            raise ZhipuAPIError(f"[{tag}] 鉴权失败(401)：API Key 无效或已过期")
        if resp.status_code == 429:
            last_error = ZhipuAPIError(f"[{tag}] 触发限流(429)")
            if attempt < config.MAX_RETRIES:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise ZhipuAPIError(f"[{tag}] 触发限流(429)，请降低频率或稍后重试")
        if resp.status_code >= 400:
            raise ZhipuAPIError(f"[{tag}] HTTP {resp.status_code}：{resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError as e:
            raise ZhipuAPIError(f"[{tag}] 返回不是合法 JSON：{resp.text[:200]}") from e
        if not isinstance(data, dict):
            raise ZhipuAPIError(f"[{tag}] 返回结构异常：{str(data)[:200]}")
        return data

    raise ZhipuAPIError(f"[{tag}] 请求失败：{last_error}")


def embed_texts(texts: Sequence[str], batch_size: Optional[int] = None,
                model: Optional[str] = None) -> List[List[float]]:
    """批量向量化。接口单次有条数上限，这里按 batch_size 分组并按 index 回填。"""
    if not texts:
        return []
    batch_size = batch_size or config.EMBED_BATCH_SIZE
    model = model or config.EMBEDDING_MODEL
    vectors: List[List[float]] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = list(texts[start:start + batch_size])
        data = post_json("embeddings", {"model": model, "input": batch}, tag="Embedding")
        items = data.get("data")
        if not items:
            raise ZhipuAPIError(f"[Embedding] 返回缺少 data 字段：{str(data)[:200]}")
        # 接口返回带 index 字段，按 index 排序后回填，避免顺序错位
        items = sorted(items, key=lambda item: item.get("index", 0))
        vectors.extend(item["embedding"] for item in items)
        print(f"[Embedding] 进度 {len(vectors)}/{total}")

    if len(vectors) != total:
        raise ZhipuAPIError(f"[Embedding] 返回条数不匹配：期望 {total}，实际 {len(vectors)}")
    return vectors


def embed_query(text: str, model: Optional[str] = None) -> List[float]:
    return embed_texts([text], model=model)[0]


def chat(messages: Union[str, List[Dict[str, Any]]], model: Optional[str] = None,
         temperature: float = 0.3, timeout: Optional[int] = None) -> str:
    """调用对话模型。messages 传字符串时按单轮 user 消息处理。"""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    payload = {
        "model": model or config.CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,  # 低温度减少幻觉，回答更贴资料
    }
    data = post_json("chat/completions", payload, tag="LLM", timeout=timeout)
    choices = data.get("choices")
    if not choices:
        raise ZhipuAPIError(f"[LLM] 返回缺少 choices 字段：{str(data)[:200]}")
    return choices[0]["message"]["content"]
