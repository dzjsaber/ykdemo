# -*- coding: utf-8 -*-
"""RagSearchTool：把 RAG 检索问答封装成 Qwen-Agent 的标准工具。

继承 `BaseTool` + `@register_tool('rag_search')`，模型通过 Function Calling
决定是否调用它，工具内部完成「向量检索 → 拼接资料 → 生成答案」。

工程上要处理的三个坑（都是实测踩出来的）：
1. 参数可能是 dict，也可能是模型吐出来的 JSON 字符串（甚至带前后噪声）；
   → `_parse_query` 先 json.loads，失败再用 raw_decode 兜底，最后退化成"整段当问题"。
2. 工具初始化就去调 Embedding 接口，会让 import / 建 Agent 变慢甚至直接报错；
   → 改成懒加载：第一次真正调用工具时才建索引。
3. 向量库检索不到内容时必须返回明确的拒答话术，不能让模型自由发挥。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Union

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_agent.tools.base import BaseTool, register_tool

from common import config
from common.qa import answer_question
from common.retrieval import VectorIndex
from common.zhipu_client import ZhipuAPIError


@register_tool("rag_search")
class RagSearchTool(BaseTool):
    """本地知识库检索工具（英科医疗文档）。"""

    description = (
        "本地知识库检索工具。当用户询问英科医疗相关的问题"
        "（公司基本信息、主营业务与产品、技术与资质、财务数据等）时调用本工具，"
        "参数 query 直接填用户的原问题。与英科医疗无关的问题不要调用。"
    )
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "要检索的用户问题原文",
            "required": True,
        }
    ]

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.doc_path = self.cfg.get("doc_path", str(config.DOC_PATH))
        self.top_k = int(self.cfg.get("top_k", config.TOP_K))
        self.min_score = float(self.cfg.get("min_score", config.MIN_SCORE))
        self.model = self.cfg.get("model", config.CHAT_MODEL)
        self._index: Optional[VectorIndex] = None

    # ---------- 懒加载向量库 ----------
    @property
    def index(self) -> VectorIndex:
        if self._index is None:
            print("[RAG工具] 首次调用，正在构建向量索引 ...")
            self._index = VectorIndex.build(self.doc_path, rebuild=self.cfg.get("rebuild", False))
        return self._index

    # ---------- 参数解析 ----------
    @staticmethod
    def _parse_query(params: Union[str, dict]) -> str:
        """容错解析模型给的参数：dict / JSON 字符串 / 带噪声的 JSON / 裸问题文本。"""
        if isinstance(params, dict):
            return str(params.get("query", "")).strip()
        if not isinstance(params, str):
            return str(params).strip()

        text = params.strip()
        if not text:
            return ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 容忍 "好的，我来查询 {"query": "..."} 谢谢" 这类前后噪声：
            # 从第一个 '{' 开始尝试解析出 JSON 对象。
            start = text.find("{")
            data = None
            if start >= 0:
                try:
                    data, _ = json.JSONDecoder().raw_decode(text[start:])
                except json.JSONDecodeError:
                    data = None
            if data is None:
                return text  # 模型直接给了问题原文，兜底当 query 用
        if isinstance(data, dict):
            return str(data.get("query", "")).strip()
        return str(data).strip()

    # ---------- 工具入口 ----------
    def call(self, params: Union[str, dict], **kwargs) -> str:
        query = self._parse_query(params)
        if not query:
            return ('调用参数缺少 query 字段，请用 {"query": "用户的问题"} 重新调用 rag_search。')

        try:
            result = answer_question(self.index, query, top_k=self.top_k,
                                     min_score=self.min_score, verbose=False)
        except ZhipuAPIError as e:
            return f"知识库检索失败：{e}"
        except RuntimeError as e:  # 例如没配 API Key
            return f"知识库不可用：{e}"

        if result["error"]:
            return f"检索到资料但生成答案失败：{result['error']}"
        return result["answer"]
