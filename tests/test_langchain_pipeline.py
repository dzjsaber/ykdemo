# -*- coding: utf-8 -*-
"""离线用例：验证 LangChain 版与手写版"同一个相似度口径"。

重点不是"能跑起来"，而是两个实现的分数能不能对上：
手写版是 numpy 手算余弦，LangChain 版是 FAISS(normalize_L2 + 内积)。
如果口径不一致，同一个阈值在两边会给出完全不同的召回结果，
"对比两套实现"这件事就失去意义了。
"""
from __future__ import annotations

import unittest

from common import config
from common.retrieval import VectorIndex, cosine_similarity, load_document, split_text
from tests import fake_embeddings as fe
from tests.test_offline_pipeline import RELATED_QUESTION, UNRELATED_QUESTION

MODEL_CALLS = []


def _make_faiss(chunks):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.documents import Document
    from langchain_rag.main import NormalizedEmbeddings

    docs = [Document(page_content=chunk, metadata={"source": "yingke.txt"}) for chunk in chunks]
    return FAISS.from_documents(
        docs, NormalizedEmbeddings(fe.as_langchain_embeddings()),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,  # 归一化后的内积 == 余弦
    )


class FaissCosineEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = load_document()
        cls.chunks = split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        cls.db = _make_faiss(cls.chunks)
        cls.index = VectorIndex(chunks=cls.chunks, vectors=fe.embed_texts(cls.chunks))

    def test_faiss_score_equals_manual_cosine(self):
        scored = self.db.similarity_search_with_score(RELATED_QUESTION, k=3)
        manual = cosine_similarity(fe.embed_query(RELATED_QUESTION), self.index.vectors)
        self.assertAlmostEqual(scored[0][1], float(manual.max()), places=6)

    def test_same_threshold_gives_same_hits(self):
        threshold = 0.15
        faiss_hits = [doc.page_content for doc, score
                      in self.db.similarity_search_with_score(RELATED_QUESTION, k=3)
                      if score >= threshold]
        manual_hits = [chunk for chunk, _ in self.index.search(
            RELATED_QUESTION, top_k=3, min_score=threshold, embedder=fe.embed_query)]
        self.assertEqual(faiss_hits, manual_hits)

    def test_both_implementations_filter_unrelated_question(self):
        threshold = 0.15
        faiss_hits = [doc for doc, score
                      in self.db.similarity_search_with_score(UNRELATED_QUESTION, k=3)
                      if score >= threshold]
        manual_hits = self.index.search(UNRELATED_QUESTION, top_k=3, min_score=threshold,
                                        embedder=fe.embed_query)
        self.assertEqual(faiss_hits, [])
        self.assertEqual(manual_hits, [])


class LangChainChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = split_text(load_document(), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        cls.db = _make_faiss(cls.chunks)

    def test_chain_answers_related_question(self):
        from langchain_core.language_models.chat_models import SimpleChatModel

        class RecordingModel(SimpleChatModel):
            @property
            def _llm_type(self) -> str:
                return "recording"

            def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
                MODEL_CALLS.append(messages)
                return "占位回答"

        from langchain_rag.main import answer, build_chain

        MODEL_CALLS.clear()
        chain = build_chain(llm=RecordingModel())
        reply = answer(self.db, chain, RELATED_QUESTION, top_k=3, min_score=0.15)
        self.assertEqual(reply, "占位回答")
        self.assertEqual(len(MODEL_CALLS), 1)
        # Prompt 里应该带上检索到的资料和拒答约束
        prompt_text = "\n".join(str(m.content) for m in MODEL_CALLS[0])
        self.assertIn("49.13亿元", prompt_text)
        self.assertIn(config.REJECT_MSG, prompt_text)

    def test_chain_skips_llm_when_below_threshold(self):
        from langchain_rag.main import answer, build_chain

        MODEL_CALLS.clear()
        chain = build_chain(llm=_never_called_model())
        reply = answer(self.db, chain, UNRELATED_QUESTION, top_k=3, min_score=0.15)
        self.assertEqual(reply, config.REJECT_MSG)
        self.assertEqual(MODEL_CALLS, [])


def _never_called_model():
    from langchain_core.language_models.chat_models import SimpleChatModel

    class NeverCalledModel(SimpleChatModel):
        @property
        def _llm_type(self) -> str:
            return "never-called"

        def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
            MODEL_CALLS.append(messages)
            raise AssertionError("低于阈值时不应该调用大模型")

    return NeverCalledModel()


if __name__ == "__main__":
    unittest.main(verbosity=2)
