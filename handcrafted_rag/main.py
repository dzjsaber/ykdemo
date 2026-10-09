# -*- coding: utf-8 -*-
"""手写版 RAG 检索问答（命令行入口）。

这一版刻意不用 LangChain：接口用 requests 手写，相似度用 numpy 手算，
方便和 LangChain 版逐行对照，看清 RAG 每一步到底发生了什么。

用法：
    python -m handcrafted_rag.main                     # 交互式问答
    python handcrafted_rag/main.py                     # 直接运行（PyCharm 点 Run 也可以）
    python -m handcrafted_rag.main --question "英科医疗2025年上半年营收是多少？"
    python -m handcrafted_rag.main --diagnose "帮我写一个快速排序"   # 只打印相似度，用于校准阈值
    python -m handcrafted_rag.main --rebuild           # 忽略向量缓存，重新计算
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 直接运行脚本时（__package__ 为空），把项目根目录加进 sys.path，
# 这样 `python handcrafted_rag/main.py` 和 PyCharm 的 Run 按钮都能正常工作。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config
from common.qa import answer_question
from common.retrieval import VectorIndex
from common.zhipu_client import ZhipuAPIError


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手写版 RAG 检索问答")
    parser.add_argument("--question", "-q", help="问一个问题后退出；不传则进入交互模式")
    parser.add_argument("--diagnose", metavar="QUESTION",
                        help="只检索不生成，打印各块的相似度分数（用于校准 RAG_MIN_SCORE）")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="召回条数")
    parser.add_argument("--min-score", type=float, default=config.MIN_SCORE,
                        help="余弦相似度阈值，低于此值拒答")
    parser.add_argument("--doc", default=str(config.DOC_PATH), help="知识库文档路径")
    parser.add_argument("--rebuild", action="store_true", help="忽略向量缓存，重新计算")
    return parser.parse_args(argv)


def diagnose(index: VectorIndex, question: str, top_k: int) -> None:
    """打印相似度分数分布：一眼看出"相关问题"和"无关问题"的分数差多远。"""
    from common.retrieval import cosine_similarity
    from common.zhipu_client import embed_query

    scores = cosine_similarity(embed_query(question), index.vectors)
    order = scores.argsort()[::-1][:top_k]
    print(f"\n[诊断] 问题：{question}")
    for rank, i in enumerate(order, start=1):
        print(f"  Top{rank} 相似度 {scores[i]:.4f}（阈值 {config.MIN_SCORE}）｜ "
              f"{index.chunks[i][:50].replace(chr(10), ' ')}")


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        index = VectorIndex.build(args.doc, rebuild=args.rebuild)
    except FileNotFoundError as e:
        print(f"[文档] {e}")
        return 1
    except (ZhipuAPIError, RuntimeError) as e:  # RuntimeError: 没配 API Key
        print(f"[Embedding] {e}")
        return 1

    if args.diagnose:
        diagnose(index, args.diagnose, args.top_k)
        return 0

    if args.question:
        result = answer_question(index, args.question, top_k=args.top_k,
                                 min_score=args.min_score)
        if result["error"]:
            print(f"回答：模型调用失败（{result['error']}）")
            return 1
        print(f"回答：{result['answer']}")
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

        result = answer_question(index, question, top_k=args.top_k,
                                 min_score=args.min_score)
        if result["error"]:
            print(f"回答：模型调用失败（{result['error']}）")
        else:
            print(f"回答：{result['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
