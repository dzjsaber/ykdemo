# -*- coding: utf-8 -*-
"""LangChain 版 RAG 检索问答（命令行入口）。

和手写版做同样的事，但全部换成框架组件：
    TextLoader → RecursiveCharacterTextSplitter → ZhipuAIEmbeddings
    → FAISS → ChatZhipuAI(LCEL: prompt | llm | parser)

两个关键工程细节（不是照抄教程就能对的地方）：
1. 相似度口径要和手写版一致：FAISS 默认用 L2 距离，它的 relevance score 和余弦不是一回事。
   这里把 Embeddings 包一层 L2 归一化 + 用内积索引(MAX_INNER_PRODUCT)，让
   similarity_search_with_score 返回的就是余弦相似度，两边共用同一个阈值。
2. 中文切分要自定义分隔符（段落 → 句号/问号/感叹号 → 逗号），否则会按空格切，句子被拦腰截断。

用法：
    python -m langchain_rag.main
    python langchain_rag/main.py
    python -m langchain_rag.main --question "英科医疗的主营业务是什么？"
    python -m langchain_rag.main --diagnose "英科医疗2025年上半年营收是多少"
    python -m langchain_rag.main --rebuild
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config
from common.qa import HUMAN_TEMPLATE, SYSTEM_INSTRUCTION

try:  # 没装 langchain 时也能 import 本模块（真正用到时会给出明确的缺包报错）
    from langchain_core.embeddings import Embeddings as _Embeddings
except ImportError:  # pragma: no cover
    _Embeddings = object


def _l2_normalize(vectors):
    """把向量按行做 L2 归一化（内积索引下，只有归一化后内积才等于余弦相似度）。"""
    import numpy as np

    if not vectors:
        return vectors
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / (norms + 1e-8)).tolist()


class NormalizedEmbeddings(_Embeddings):
    """在任意 Embeddings 外面包一层 L2 归一化。

    智谱返回的是原始向量（未归一化），而 FAISS 默认按 L2 距离算相似度，
    它的"relevance score"和余弦不是一回事。这里统一成：
        内积索引(MAX_INNER_PRODUCT) + 向量归一化  ⇒  相似度 == 余弦相似度
    这样 LangChain 版和手写版共用同一个阈值，召回结果可以直接对比。
    """

    def __init__(self, base):
        self.base = base

    def embed_documents(self, texts):
        return _l2_normalize(self.base.embed_documents(texts))

    def embed_query(self, text):
        return _l2_normalize([self.base.embed_query(text)])[0]


def load_docs(doc_path: str):
    from langchain_community.document_loaders import TextLoader

    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到文档：{path}")
    return TextLoader(str(path), encoding="utf-8").load()


def split_docs(docs, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        # 中文场景：优先按段落切，其次句子，最后逗号兜底，避免句子被硬切两半
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"[切分] 文档 {len(docs[0].page_content)} 字 → {len(chunks)} 块"
          f"（chunk_size={chunk_size}, overlap={overlap}）")
    return chunks


def build_embeddings():
    from langchain_community.embeddings import ZhipuAIEmbeddings

    base = ZhipuAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.require_api_key())
    return NormalizedEmbeddings(base)


def _index_dir(doc_text: str, chunk_size: int, overlap: int) -> Path:
    key = hashlib.md5(f"{doc_text}|{chunk_size}|{overlap}|{config.EMBEDDING_MODEL}"
                      .encode("utf-8")).hexdigest()[:12]
    return config.CACHE_DIR / f"faiss_index_{key}"


def build_vectorstore(chunks, embeddings=None, rebuild: bool = False):
    """向量化 + 本地持久化。文档/切分参数/Embedding 模型任一变化，缓存目录名就变，自动失效。"""
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    embeddings = embeddings or build_embeddings()
    index_dir = _index_dir(chunks[0].metadata.get("source", "") +
                           "".join(doc.page_content for doc in chunks),
                           config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    if index_dir.exists() and not rebuild:
        try:
            db = FAISS.load_local(str(index_dir), embeddings,
                                  allow_dangerous_deserialization=True)  # 加载的是自己刚存的索引
            print(f"[缓存] 已加载本地向量库 {index_dir.name}，跳过 Embedding 计算")
            return db
        except Exception as e:  # noqa: BLE001 - 缓存坏了就重建
            print(f"[缓存] 加载失败，将重建索引：{e}")

    print("[Embedding] 正在向量化并构建 FAISS 索引 ...")
    db = FAISS.from_documents(
        chunks, embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,  # 内积（向量已归一化 == 余弦）
    )
    try:
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        db.save_local(str(index_dir))
        print(f"[缓存] 向量库已保存至 {index_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"[缓存] 保存失败（不影响使用）：{e}")
    return db


def retrieve(db, question: str, top_k: int, min_score: float
             ) -> Tuple[str, List[Tuple[object, float]]]:
    """检索 + 阈值过滤。归一化后的内积就是余弦相似度，和手写版口径一致。"""
    scored = db.similarity_search_with_score(question, k=top_k)
    hits = [(doc, float(score)) for doc, score in scored if score >= min_score]
    if not hits:
        print(f"[检索] 最高相似度低于阈值 {min_score}，按拒答处理")
        return "", []
    print("[检索] 命中：")
    for i, (doc, score) in enumerate(hits, start=1):
        print(f"  Top{i} 相似度 {score:.4f}：{doc.page_content[:40]}...")
    return "\n".join(doc.page_content for doc, _ in hits), hits


def build_chain(llm=None):
    """LCEL 管道：Prompt → ChatModel → 解析成字符串。"""
    from langchain_community.chat_models import ChatZhipuAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = llm or ChatZhipuAI(api_key=config.require_api_key(),
                             model=config.CHAT_MODEL,
                             temperature=0.3)  # 低温度减少幻觉
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTION.format(reject=config.REJECT_MSG)),
        ("human", HUMAN_TEMPLATE),
    ])
    return prompt | llm | StrOutputParser()


def answer(db, chain, question: str, top_k: int, min_score: float) -> str:
    context, _ = retrieve(db, question, top_k, min_score)
    if not context:
        return config.REJECT_MSG
    return chain.invoke({"context": context, "question": question})


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain 版 RAG 检索问答")
    parser.add_argument("--question", "-q", help="问一个问题后退出；不传则进入交互模式")
    parser.add_argument("--diagnose", metavar="QUESTION",
                        help="只检索不生成，打印相似度分数（用于校准 RAG_MIN_SCORE）")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--min-score", type=float, default=config.MIN_SCORE)
    parser.add_argument("--doc", default=str(config.DOC_PATH))
    parser.add_argument("--rebuild", action="store_true", help="忽略本地索引，重新向量化")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        docs = load_docs(args.doc)
        chunks = split_docs(docs)
        db = build_vectorstore(chunks, rebuild=args.rebuild)
    except FileNotFoundError as e:
        print(f"[文档] {e}")
        return 1
    except Exception as e:  # noqa: BLE001 - 依赖缺失、鉴权失败都从这里给出可读信息
        print(f"[初始化失败] {type(e).__name__}: {e}")
        return 1

    if args.diagnose:
        retrieve(db, args.diagnose, args.top_k, args.min_score)
        return 0

    try:
        chain = build_chain()
    except Exception as e:  # noqa: BLE001
        print(f"[初始化失败] {type(e).__name__}: {e}")
        return 1

    if args.question:
        print(f"回答：{answer(db, chain, args.question, args.top_k, args.min_score)}")
        return 0

    print("\n输入问题开始问答，输入 q 退出。")
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
        try:
            print(f"回答：{answer(db, chain, question, args.top_k, args.min_score)}")
        except Exception as e:  # noqa: BLE001 - 单次调用失败不该让整个会话退出
            print(f"回答：模型调用失败（{type(e).__name__}: {e}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
