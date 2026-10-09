# -*- coding: utf-8 -*-
"""联网用例：三个工具路由测试（对应项目成果里说的三个测试用例）。

1. 知识库问题 → 必须调用 rag_search，并且答案来自知识库；
2. 非知识库问题（写代码/闲聊）→ 不调用工具，直接用模型能力回答；
3. 超范围问题（问知识库里没有的信息）→ 返回拒答话术，不编造。

需要有效的 ZHIPU_API_KEY，没配 Key 时整组用例自动跳过（不会假装通过）。
运行：
    python -m pytest tests/test_tool_routing.py -v
    python -m unittest tests.test_tool_routing -v
"""
from __future__ import annotations

import unittest

from common import config


@unittest.skipUnless(config.API_KEY, "未配置 ZHIPU_API_KEY，跳过联网工具路由用例")
class ToolRoutingTest(unittest.TestCase):
    def setUp(self):
        from qwen_agent_rag.agent_demo import build_bot

        self.bot = build_bot()  # 每个用例一个干净的 Agent，避免上下文互相污染

    def _run(self, question: str) -> dict:
        from qwen_agent_rag.agent_demo import run_agent

        result = run_agent(self.bot, question, verbose=False)
        print(f"\n[{question}] → 工具调用 {result['tool_calls']} 次 | "
              f"文本工具调用兜底：{'是' if result.get('recovered_text_call') else '否'} | "
              f"回答：{(result['final'] or '')[:120]}")
        return result

    def test_1_knowledge_question_calls_tool(self):
        """知识库问题：必须调用 rag_search，且不能拒答。"""
        result = self._run("英科医疗2025年上半年的营收和净利润是多少？")
        self.assertIsNone(result["error"], result["error"])
        self.assertGreaterEqual(result["tool_calls"], 1, "知识库问题没有调用 rag_search")
        self.assertNotIn(config.REJECT_MSG, result["final"] or "")
        self.assertNotIn("rag_search", result["final"] or "",
                         "工具调用不应以文字形式透给用户")

    def test_2_general_question_skips_tool(self):
        """非知识库问题：不应该调用工具。"""
        result = self._run("帮我写一个Python快速排序")
        self.assertIsNone(result["error"], result["error"])
        self.assertEqual(result["tool_calls"], 0, "闲聊/写代码类问题不该调用 rag_search")
        self.assertTrue((result["final"] or "").strip(), "模型没有给出回答")

    def test_3_out_of_scope_question_refuses(self):
        """超范围问题：知识库里没有，应按拒答话术回答。"""
        result = self._run("英科医疗今天的股价是多少？")
        self.assertIsNone(result["error"], result["error"])
        self.assertIn(config.REJECT_MSG, result["final"] or "",
                      "知识库没有的信息应该拒答，而不是编造")


if __name__ == "__main__":
    unittest.main(verbosity=2)
