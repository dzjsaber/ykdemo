# -*- coding: utf-8 -*-
"""检索核心：切分 → 向量化 → 余弦检索 → 阈值过滤。

手写版与 Qwen-Agent 工具共用这里的实现，LangChain 版用框架组件做同一件事。
两边保持同一套语义：相似度都是「query 向量 · chunk 向量」的余弦相似度，
因此同一套阈值、同一套拒答规则可以直接横向对比。
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from common import config
from common.zhipu_client import embed_query, embed_texts

EmbedFn = Callable[[Sequence[str]], List[List[float]]]
QueryEmbedFn = Callable[[str], List[float]]


def load_document(path: Optional[Union[str, Path]] = None) -> str:
    doc_path = Path(path) if path else config.DOC_PATH
    if not doc_path.exists():
        raise FileNotFoundError(f"未找到文档：{doc_path}")
    return doc_path.read_text(encoding="utf-8")


def split_text(text: str, chunk_size: Optional[int] = None,
               overlap: Optional[int] = None) -> List[str]:
    """固定窗口切分 + 重叠，避免答案正好被切在两个块的边界上。"""
    size = chunk_size or config.CHUNK_SIZE
    step = size - (overlap if overlap is not None else config.CHUNK_OVERLAP)
    if step <= 0:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def cosine_similarity(query_vec: Sequence[float],
                      matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """一次算完 query 与所有 chunk 的余弦相似度（矩阵化，避免 Python 循环）。"""
    q = np.asarray(query_vec, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    return (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-8)  # 1e-8 防除零


@dataclass
class VectorIndex:
    """chunk 列表 + 对应向量，负责检索与阈值过滤。"""

    chunks: List[str]
    vectors: np.ndarray
    embedding_model: str = config.EMBEDDING_MODEL
    metadata: dict = field(default_factory=dict)

    # ---------- 构建 ----------
    @staticmethod
    def cache_key(text: str, chunk_size: int, overlap: int, model: str) -> str:
        raw = f"{text}|{chunk_size}|{overlap}|{model}".encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    @classmethod
    def build(cls, doc_path: Optional[Union[str, Path]] = None, *,
              chunk_size: Optional[int] = None, overlap: Optional[int] = None,
              use_cache: bool = True, rebuild: bool = False,
              embedder: Optional[EmbedFn] = None, verbose: bool = True) -> "VectorIndex":
        text = load_document(doc_path)
        size = chunk_size or config.CHUNK_SIZE
        over = overlap if overlap is not None else config.CHUNK_OVERLAP
        chunks = split_text(text, size, over)
        if verbose:
            print(f"[切分] 文档 {len(text)} 字 → {len(chunks)} 块"
                  f"（chunk_size={size}, overlap={over}）")

        key = cls.cache_key(text, size, over, config.EMBEDDING_MODEL)
        cache_file = config.CACHE_DIR / "vectors.pkl"

        # 缓存只在"真实调用接口"时启用；测试注入的假 embedder 不读缓存
        if use_cache and not rebuild and embedder is None and cache_file.exists():
            try:
                cached = pickle.loads(cache_file.read_bytes())
                if cached.get("key") == key and len(cached.get("chunks", [])) == len(chunks):
                    if verbose:
                        print("[缓存] 命中向量缓存，跳过 Embedding 计算")
                    return cls(chunks=cached["chunks"],
                               vectors=np.asarray(cached["vectors"], dtype=np.float64),
                               metadata={"cached": True, "key": key})
            except Exception as e:  # noqa: BLE001 - 缓存坏了就重建，不影响主流程
                if verbose:
                    print(f"[缓存] 读取失败，将重新计算：{e}")

        if verbose:
            print("[Embedding] 正在向量化文档块 ...")
        vecs = (embedder or embed_texts)(chunks)
        matrix = np.asarray(vecs, dtype=np.float64)

        if use_cache and embedder is None:
            try:
                config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(pickle.dumps({"key": key, "chunks": chunks,
                                                     "vectors": matrix}))
                if verbose:
                    print(f"[缓存] 向量已写入 {cache_file}")
            except Exception as e:  # noqa: BLE001 - 缓存写失败不影响使用
                if verbose:
                    print(f"[缓存] 写入失败（不影响使用）：{e}")

        return cls(chunks=chunks, vectors=matrix, metadata={"cached": False, "key": key})

    # ---------- 检索 ----------
    def search(self, question: str, top_k: Optional[int] = None,
               min_score: Optional[float] = None,
               embedder: Optional[QueryEmbedFn] = None) -> List[Tuple[str, float]]:
        """返回 [(chunk, 余弦相似度)]，已按相似度降序并按阈值过滤。"""
        top_k = top_k or config.TOP_K
        min_score = config.MIN_SCORE if min_score is None else min_score

        scores = cosine_similarity((embedder or embed_query)(question), self.vectors)
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]

    def retrieve_context(self, question: str, top_k: Optional[int] = None,
                         min_score: Optional[float] = None,
                         embedder: Optional[QueryEmbedFn] = None
                         ) -> Tuple[str, List[Tuple[str, float]]]:
        hits = self.search(question, top_k=top_k, min_score=min_score, embedder=embedder)
        return "\n".join(chunk for chunk, _ in hits), hits
