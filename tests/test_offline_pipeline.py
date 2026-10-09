# -*- coding: utf-8 -*-
"""离线用例：不需要 API Key，用确定性假 Embedding 验证切分/检索/拒答逻辑。

真正调用大模型的三个路由用例在 tests/test_tool_routing.py（需要有效 Key）。
"""
from __future__ import annotations

import unittest

from common import config
from common.qa import answer_question
from common.retrieval import VectorIndex, split_text
from common.zhipu_client import ZhipuAPIError
from tests import fake_embeddings as fe

RELATED_QUESTION = "英科医疗2025年上半年的营收和净利润是多少"
UNRELATED_QUESTION = "帮我写一个Python快速排序"


class SplitTextTest(unittest.TestCase):
    def test_overlap_and_coverage(self):
        text = "".join(str(i % 10) for i in range(500))
        chunks = split_text(text, 200, 30)
        self.assertEqual(len(chunks), 3)          # 步长 170，起点 0/170/340
        self.assertEqual(chunks[0], text[:200])
        self.assertEqual(chunks[1], text[170:370])  # 相邻块有 30 字重叠

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(split_text("", 200, 30), [])

    def test_overlap_larger_than_chunk_raises(self):
        with self.assertRaises(ValueError):
            split_text("abc", 100, 100)


class RetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = VectorIndex.build(use_cache=False, embedder=fe.embed_texts, verbose=False)

    def test_related_question_hits_the_right_chunk(self):
        hits = self.index.search(RELATED_QUESTION, top_k=3, min_score=0.0,
                                 embedder=fe.embed_query)
        self.assertIn("49.13亿元", hits[0][0])

    def test_related_scores_beat_unrelated(self):
        related = self.index.search(RELATED_QUESTION, top_k=1, min_score=0.0,
                                    embedder=fe.embed_query)[0][1]
        unrelated = self.index.search(UNRELATED_QUESTION, top_k=1, min_score=0.0,
                                      embedder=fe.embed_query)[0][1]
        self.assertGreater(related, unrelated)

    def test_threshold_filters_unrelated_question(self):
        hits = self.index.search(UNRELATED_QUESTION, top_k=3, min_score=0.15,
                                 embedder=fe.embed_query)
        self.assertEqual(hits, [])


class QaPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = VectorIndex.build(use_cache=False, embedder=fe.embed_texts, verbose=False)

    def test_refuses_below_threshold_without_calling_llm(self):
        calls = []

        def should_not_be_called(prompt):  # pragma: no cover - 被调用即失败
            calls.append(prompt)
            raise AssertionError("低于阈值时不应该调用大模型")

        result = answer_question(self.index, UNRELATED_QUESTION, min_score=0.15,
                                 llm=should_not_be_called, query_embedder=fe.embed_query,
                                 verbose=False)
        self.assertTrue(result["rejected"])
        self.assertEqual(result["answer"], config.REJECT_MSG)
        self.assertEqual(calls, [])

    def test_prompt_carries_context_and_refusal_rule(self):
        captured = {}

        def fake_llm(prompt):
            captured["prompt"] = prompt
            return "2025年上半年总营收49.13亿元。"

        result = answer_question(self.index, RELATED_QUESTION, min_score=0.15,
                                 llm=fake_llm, query_embedder=fe.embed_query, verbose=False)
        self.assertFalse(result["error"])
        self.assertIn("49.13亿元", captured["prompt"])       # 检索到的资料进了 Prompt
        self.assertIn(config.REJECT_MSG, captured["prompt"])  # 拒答约束进了 Prompt
        self.assertIn(RELATED_QUESTION, captured["prompt"])

    def test_api_error_is_returned_not_raised(self):
        def failing_llm(prompt):
            raise ZhipuAPIError("[LLM] 鉴权失败(401)：API Key 无效或已过期")

        result = answer_question(self.index, RELATED_QUESTION, min_score=0.15,
                                 llm=failing_llm, query_embedder=fe.embed_query, verbose=False)
        self.assertFalse(result["rejected"])
        self.assertIn("401", result["error"])


class RagToolParsingTest(unittest.TestCase):
    """工具的参数解析容错：模型给的可能是 JSON 字符串、dict，甚至带前后噪声。"""

    def setUp(self):
        from qwen_agent_rag.rag_tool import RagSearchTool
        self.parse = RagSearchTool._parse_query

    def test_parses_json_string(self):
        self.assertEqual(self.parse('{"query": "英科医疗总部在哪"}'), "英科医疗总部在哪")

    def test_parses_dict(self):
        self.assertEqual(self.parse({"query": "年化产能"}), "年化产能")

    def test_tolerates_noise_around_json(self):
        self.assertEqual(self.parse('调用工具如下 {"query": "营业收入"} 谢谢'), "营业收入")

    def test_falls_back_to_raw_text(self):
        self.assertEqual(self.parse("英科医疗总部在哪"), "英科医疗总部在哪")

    def test_tool_builds_index_lazily(self):
        from qwen_agent_rag.rag_tool import RagSearchTool
        tool = RagSearchTool({})
        self.assertIsNone(tool._index)  # 实例化时不碰网络，首次调用才建索引


class RagToolCallTest(unittest.TestCase):
    """工具调用链路：把检索问答环节 mock 掉，验证工具本身的输入输出契约。"""

    def _tool_with_mocked_pipeline(self, **return_value):
        from unittest import mock

        from qwen_agent_rag import rag_tool

        # index 属性会被求值（用于取检索结果），必须一起 mock 掉，否则会真的去调 Embedding 接口
        self.enterContext(mock.patch.object(rag_tool.RagSearchTool, "index",
                                            new_callable=mock.PropertyMock))
        mocked = self.enterContext(
            mock.patch.object(rag_tool, "answer_question", return_value=return_value))
        return rag_tool.RagSearchTool({}), mocked

    def test_returns_answer_and_passes_query_through(self):
        tool, mocked = self._tool_with_mocked_pipeline(
            answer="2025年上半年总营收49.13亿元。", hits=[], rejected=False, error=None)
        result = tool.call('{"query": "2025年上半年营收"}')
        self.assertEqual(result, "2025年上半年总营收49.13亿元。")
        self.assertEqual(mocked.call_args.args[1], "2025年上半年营收")

    def test_returns_refusal_when_no_hits(self):
        tool, _ = self._tool_with_mocked_pipeline(
            answer=config.REJECT_MSG, hits=[], rejected=True, error=None)
        self.assertEqual(tool.call({"query": "英科医疗今天的股价"}), config.REJECT_MSG)

    def test_reports_error_instead_of_crashing(self):
        tool, _ = self._tool_with_mocked_pipeline(
            answer="", hits=[], rejected=False, error="[LLM] HTTP 500")
        self.assertIn("HTTP 500", tool.call({"query": "营收"}))

    def test_missing_query_returns_hint(self):
        from qwen_agent_rag.rag_tool import RagSearchTool

        tool = RagSearchTool({})
        self.assertIn("query", tool.call('{"foo": "bar"}'))

    def test_tool_is_registered_for_agent(self):
        from qwen_agent.tools.base import TOOL_REGISTRY
        import qwen_agent_rag.rag_tool  # noqa: F401 - 触发 @register_tool
        self.assertIn("rag_search", TOOL_REGISTRY)


class _FakeTool:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def call(self, params, **kwargs):
        self.calls.append(params)
        return self.reply


class _FakeBot:
    """模拟 qwen-agent Assistant：run() 每次 yield 一份累积的消息快照（含流式更新）。"""

    def __init__(self, snapshots, tool_reply=config.REJECT_MSG):
        self._snapshots = snapshots
        self.tool = _FakeTool(tool_reply)
        self.function_map = {"rag_search": self.tool}

    def run(self, messages):
        for extra in self._snapshots:
            yield [dict(m) for m in messages] + [dict(m) for m in extra]


class ToolCallFallbackTest(unittest.TestCase):
    """针对两种"弱模型绕过工具"的兜底，以及拒答话术不被改写的规则。"""

    def setUp(self):
        from qwen_agent_rag.agent_demo import run_agent
        self.run_agent = run_agent

    def test_native_tool_call_is_counted_and_used(self):
        bot = _FakeBot([
            [{"role": "assistant", "content": ""},
             {"role": "function", "name": "rag_search", "content": "总营收49.13亿元。"},
             {"role": "assistant", "content": "总营收49.13亿元。"}],
        ])
        result = self.run_agent(bot, "英科医疗2025年上半年的营收是多少？", verbose=False)
        self.assertEqual(result["tool_calls"], 1)
        self.assertFalse(result["recovered_text_call"])
        self.assertFalse(result["forced_tool_call"])
        self.assertEqual(result["final"], "总营收49.13亿元。")

    def test_recovers_text_style_tool_call(self):
        # glm-4-flash 实测行为：把工具调用当正文写出来
        bot = _FakeBot([
            [{"role": "assistant", "content": 'rag_search\n{"query": "2025年上半年营收"}'}],
        ], tool_reply="总营收49.13亿元。")
        result = self.run_agent(bot, "英科医疗2025年上半年的营收是多少？", verbose=False)
        self.assertTrue(result["recovered_text_call"])
        self.assertEqual(bot.tool.calls, [{"query": "2025年上半年营收"}])
        self.assertEqual(result["final"], "总营收49.13亿元。")

    def test_forced_retrieval_when_model_answers_from_its_own_knowledge(self):
        # glm-4-flash 实测行为：不调工具，直接编造"XX亿元"
        bot = _FakeBot([
            [{"role": "assistant", "content": "2025年上半年营收为XX亿元。"}],
        ], tool_reply="总营收49.13亿元。")
        result = self.run_agent(bot, "英科医疗的营收是多少？", verbose=False)
        self.assertTrue(result["forced_tool_call"])
        self.assertEqual(bot.tool.calls, [{"query": "英科医疗的营收是多少？"}])
        self.assertEqual(result["final"], "总营收49.13亿元。")

    def test_no_fallback_for_unrelated_question(self):
        bot = _FakeBot([
            [{"role": "assistant", "content": "快速排序的实现如下..."}],
        ])
        result = self.run_agent(bot, "帮我写一个Python快速排序", verbose=False)
        self.assertEqual(result["tool_calls"], 0)
        self.assertFalse(result["forced_tool_call"])
        self.assertFalse(result["recovered_text_call"])
        self.assertEqual(bot.tool.calls, [])

    def test_refusal_is_never_rephrased_by_model(self):
        # 模型拿到拒答话术后又想补一句"建议你去财经网站查询"，最终答案应该固定为拒答话术
        bot = _FakeBot([
            [{"role": "assistant", "content": ""},
             {"role": "function", "name": "rag_search", "content": config.REJECT_MSG + "。"},
             {"role": "assistant", "content": "很抱歉，我无法提供实时股价，建议查询财经网站。"}],
        ])
        result = self.run_agent(bot, "英科医疗今天的股价是多少？", verbose=False)
        self.assertTrue(result["refusal_from_tool"])
        self.assertEqual(result["final"], config.REJECT_MSG)


if __name__ == "__main__":
    unittest.main(verbosity=2)
