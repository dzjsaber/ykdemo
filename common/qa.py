# -*- coding: utf-8 -*-
"""RAG 问答编排：检索 → 拼接 Prompt → 生成回答（含拒答分支）。

手写版命令行和 Qwen-Agent 工具都用这一个函数，避免"prompt 写了两份、改漏一份"。
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from common import config
from common.retrieval import VectorIndex
from common.zhipu_client import ZhipuAPIError, chat

# Prompt 只有这一份：手写版拼字符串、LangChain 版塞进 ChatPromptTemplate，
# 用的都是下面两个常量，避免改了一处漏了另一处。
SYSTEM_INSTRUCTION = (
    "你是一个严谨的文档问答助手。请严格根据用户提供的资料回答问题：\n"
    "1. 只使用资料中出现的信息，不要编造，也不要补充资料以外的常识；\n"
    "2. 资料中没有答案时，只回答“{reject}”，不要解释原因。"
)
HUMAN_TEMPLATE = "资料：\n{context}\n\n问题：{question}"
PROMPT_TEMPLATE = SYSTEM_INSTRUCTION + "\n\n" + HUMAN_TEMPLATE


def build_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question,
                                  reject=config.REJECT_MSG)


def format_hits(hits: List[Tuple[str, float]]) -> str:
    return "\n".join(f"  Top{i + 1} 相似度 {score:.4f}：{chunk[:40]}..."
                     for i, (chunk, score) in enumerate(hits))


def answer_question(index: VectorIndex, question: str, *,
                    top_k: Optional[int] = None,
                    min_score: Optional[float] = None,
                    llm: Callable[..., str] = chat,
                    query_embedder=None,
                    verbose: bool = True) -> dict:
    """完整问答链路，返回结构化结果，方便命令行和测试复用。

    返回 dict：
        answer     最终回答（拒答或模型输出）
        hits       [(chunk, 相似度)]
        rejected   是否走了拒答分支
        error      出错时的错误信息（None 表示正常）
    """
    hits = index.search(question, top_k=top_k, min_score=min_score,
                        embedder=query_embedder)
    if verbose:
        if hits:
            print("[检索] 命中：")
            print(format_hits(hits))
        else:
            print(f"[检索] 最高相似度低于阈值 {config.MIN_SCORE if min_score is None else min_score}，"
                  "按拒答处理")

    if not hits:
        return {"answer": config.REJECT_MSG, "hits": [], "rejected": True, "error": None}

    context = "\n".join(chunk for chunk, _ in hits)
    try:
        answer = llm(build_prompt(context, question))
    except ZhipuAPIError as e:
        return {"answer": "", "hits": hits, "rejected": False, "error": str(e)}
    return {"answer": answer, "hits": hits, "rejected": False, "error": None}
