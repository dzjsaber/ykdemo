# -*- coding: utf-8 -*-
"""Qwen-Agent 版：一个能自己决定"要不要查知识库"的 Agent。

Agent = glm 模型（负责推理决策） + RagSearchTool（负责知识检索）。
system_message 里写清了三条路由规则，这是让弱模型稳定调用工具的关键：

1. 英科医疗相关问题 → 必须先调 rag_search，并把工具返回的内容当最终答案，不要自己编；
2. 无关问题（写代码、数学、闲聊）→ 直接回答，不要调工具；
3. 工具返回拒答话术时 → 原样输出，不要用模型自己的知识补答案。

另外加了"同一次提问最多调用 N 次工具"的硬闸门：即使模型反复调工具也不会无限循环，
超过上限就直接把最后一次工具结果当作最终答案（对应项目里说的"工具调用死循环"问题）。

用法：
    python -m qwen_agent_rag.agent_demo "英科医疗2025年上半年营收是多少？"
    python -m qwen_agent_rag.agent_demo "帮我写一个Python快速排序"
    python -m qwen_agent_rag.agent_demo                      # 交互模式
    python -m qwen_agent_rag.agent_demo --model glm-4-air "英科医疗总部在哪？"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config
from qwen_agent_rag.rag_tool import RagSearchTool

SYSTEM_MESSAGE = (
    "你是英科医疗知识库问答助手，可以调用 rag_search 工具查询本地知识库。规则如下：\n"
    "1. 只要问题和英科医疗有关（公司基本信息、主营业务与产品、技术与资质、财务数据等），"
    "你必须先调用 rag_search 工具，然后直接把工具返回的内容作为最终答案输出，"
    "不要用你自己的知识改写或补充，也不要再次调用 rag_search。\n"
    "2. 与英科医疗无关的问题（写代码、数学计算、闲聊等）直接回答，不要调用任何工具。\n"
    "3. 如果工具返回的内容是拒答话术，就原样输出该话术。"
)


def build_bot(model: Optional[str] = None, tool_cfg: Optional[dict] = None):
    from qwen_agent.agents import Assistant

    llm_cfg = {
        "model": model or config.CHAT_MODEL,
        "model_type": "oai",  # 智谱提供 OpenAI 兼容接口
        "model_server": config.API_BASE,
        "api_key": config.require_api_key(),
        "generate_cfg": {"max_retries": 2},
    }
    return Assistant(
        llm=llm_cfg,
        function_list=[RagSearchTool(tool_cfg)],
        system_message=SYSTEM_MESSAGE,
    )


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(getattr(item, "text", "") or (item.get("text", "") if isinstance(item, dict) else "")
                       for item in content)
    return ""


def run_agent(bot, question: str, max_tool_calls: int = 3, verbose: bool = True) -> dict:
    """跑一轮对话，返回结构化结果（供测试与命令行共用）。

    返回 dict：
        final             最终回答
        tool_calls        rag_search 被调用的次数
        guard_triggered   是否因为超过 max_tool_calls 被强制中断
        messages          全部消息
        error             异常信息（None 表示正常）
    """
    messages = [{"role": "user", "content": question}]
    seen = 0
    tool_calls = 0
    guard_triggered = False
    final = ""
    error = None

    try:
        for rsp in bot.run(messages):
            for msg in rsp[seen:]:
                seen += 1
                role = msg.get("role")
                content = _content_to_text(msg.get("content"))
                if role == "function" and msg.get("name") == "rag_search":
                    tool_calls += 1
                    if verbose:
                        print(f"\n--- rag_search 返回（第 {tool_calls} 次）---\n{content[:300]}")
                elif role == "assistant" and content:
                    final = content
                    if verbose:
                        print(f"\n--- 模型输出 ---\n{content[:300]}")
                elif verbose and role == "assistant":
                    print("\n--- 模型请求调用工具（无文本）---")
            if tool_calls >= max_tool_calls:
                guard_triggered = True
                if verbose:
                    print(f"\n[保护] rag_search 已调用 {tool_calls} 次，达到上限，强制结束本轮")
                break
    except Exception as e:  # noqa: BLE001 - Agent 内部异常（鉴权/网络/模型名错误）统一往上抛给用户
        error = f"{type(e).__name__}: {e}"

    return {"final": final, "tool_calls": tool_calls, "guard_triggered": guard_triggered,
            "messages": list(messages), "error": error}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Agent + RAG 工具 演示")
    parser.add_argument("question", nargs="*", help="要问的问题；不传则进入交互模式")
    parser.add_argument("--model", default=config.CHAT_MODEL,
                        help="对话模型，如 glm-4-flash / glm-4-air")
    parser.add_argument("--max-tool-calls", type=int, default=3, help="单轮最多调用工具次数")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        bot = build_bot(args.model)
    except RuntimeError as e:
        print(f"[配置] {e}")
        return 1

    questions: List[str] = [" ".join(args.question)] if args.question else []
    if not questions:
        print(f"当前模型：{args.model}。输入问题开始问答，输入 q 退出。")
        while True:
            try:
                question = input("\n请输入问题：").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in {"q", "quit", "exit"}:
                break
            questions.append(question)
            result = run_agent(bot, question, max_tool_calls=args.max_tool_calls)
            print(f"\n最终回答：{result['final'] or '（模型没有输出文本）'}")
        return 0

    for question in questions:
        result = run_agent(bot, question, max_tool_calls=args.max_tool_calls)
        print(f"\n问题：{question}")
        if result["error"]:
            print(f"出错：{result['error']}")
        else:
            print(f"最终回答：{result['final'] or '（模型没有输出文本）'}")
        print(f"（本轮 rag_search 调用次数：{result['tool_calls']}，"
              f"是否触发循环保护：{'是' if result['guard_triggered'] else '否'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
