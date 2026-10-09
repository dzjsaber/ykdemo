# -*- coding: utf-8 -*-
"""离线测试用的确定性 Embedding：字符二元组哈希 + 词频计数。

不联网、不消耗 API 额度，但保留了"字面越接近、向量越接近"的性质，
足够验证切分、相似度排序、阈值过滤这些检索逻辑。
注意这里刻意**不做**归一化：真实接口返回的也是原始向量，
归一化要由各自的实现负责（手写版在算余弦时归一化，LangChain 版在 Embeddings 外面包一层）。
"""
from __future__ import annotations

import zlib
from typing import List, Sequence

import numpy as np

DIM = 512


def _tokens(text: str) -> List[str]:
    clean = "".join(ch for ch in text if not ch.isspace())
    if len(clean) < 2:
        return [clean] if clean else []
    return [clean[i:i + 2] for i in range(len(clean) - 1)]


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for text in texts:
        vec = np.zeros(DIM, dtype=np.float64)
        for tok in _tokens(text):
            vec[zlib.crc32(tok.encode("utf-8")) % DIM] += 1.0
        vectors.append(vec.tolist())
    return vectors


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]


def as_langchain_embeddings():
    """包成 LangChain 的 Embeddings 接口，供 FAISS 使用。"""
    from langchain_core.embeddings import Embeddings

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
            return embed_texts(texts)

        def embed_query(self, text: str) -> List[float]:
            return embed_texts([text])[0]

    return FakeEmbeddings()
