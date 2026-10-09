# -*- coding: utf-8 -*-
"""模型选型脚本：对比不同 GLM 模型在工具路由上的稳定性。

对每个模型跑同一组用例，统计：
    - 工具调用次数（该调的有没有调、不该调的有没有乱调）
    - 是否出现"把工具调用写成文字"（需要兜底恢复），这是弱模型最典型的失败模式
    - 是否触发死循环保护
    - 拒答是否正确
    - 单轮耗时

用法（需要有效 ZHIPU_API_KEY，会产生 API 调用费用/额度）：
    python -m tools.compare_models --models glm-4-flash glm-4-air
    python -m tools.compare_models --models glm-4-flash glm-4-air --repeat 3
    python -m tools.compare_models --out docs/model_comparison.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config

# (用例名, 问题, 期望是否调用工具, 期望是否拒答)
CASES = [
    ("知识库问题", "英科医疗2025年上半年的营收和净利润是多少？", True, False),
    ("非知识库问题", "帮我写一个Python快速排序", False, False),
    ("超范围问题", "英科医疗今天的股价是多少？", True, True),
]


def run_one(model: str, question: str) -> dict:
    from qwen_agent_rag.agent_demo import build_bot, run_agent

    bot = build_bot(model)
    start = time.time()
    result = run_agent(bot, question, verbose=False)
    result["elapsed"] = time.time() - start
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="对比 GLM 模型的 Function Calling 稳定性")
    parser.add_argument("--models", nargs="+", default=["glm-4-flash", "glm-4-air"])
    parser.add_argument("--repeat", type=int, default=1, help="每个用例重复次数，取最差表现")
    parser.add_argument("--out", help="把结果写成 markdown 表格")
    args = parser.parse_args(argv)

    try:
        config.require_api_key()
    except RuntimeError as e:
        print(f"[配置] {e}")
        return 1

    rows = []
    for model in args.models:
        print(f"\n===== 模型 {model} =====")
        passed = 0
        total = 0
        for name, question, want_tool, want_reject in CASES:
            for i in range(args.repeat):
                result = run_one(model, question)
                called = result["tool_calls"] >= 1
                rejected = config.REJECT_MSG in (result["final"] or "")
                ok = (called == want_tool) and (rejected == want_reject) and result["error"] is None
                total += 1
                passed += int(ok)
                print(f"  [{name}] 第{i + 1}次 → 工具调用 {result['tool_calls']} 次，"
                      f"拒答 {'是' if rejected else '否'}，耗时 {result['elapsed']:.1f}s，"
                      f"{'通过' if ok else '不符合预期'}")
                if result.get("recovered_text_call"):
                    print("      注意：模型把工具调用写成了文字，走了兜底恢复")
                if result["error"]:
                    print(f"      错误：{result['error']}")
                rows.append({
                    "model": model, "case": name, "round": i + 1,
                    "tool_calls": result["tool_calls"],
                    "guard": result["guard_triggered"],
                    "recovered": result.get("recovered_text_call", False),
                    "rejected": rejected, "elapsed": result["elapsed"],
                    "ok": ok, "error": result["error"],
                })
        print(f"  —— {model}: {passed}/{total} 通过")

    if args.out:
        lines = ["| 模型 | 用例 | 工具调用次数 | 文本工具调用(走兜底) | 触发循环保护 | 拒答 | 耗时(s) | 是否符合预期 |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            lines.append(f"| {row['model']} | {row['case']}#{row['round']} | {row['tool_calls']} | "
                         f"{'是' if row['recovered'] else '否'} | "
                         f"{'是' if row['guard'] else '否'} | {'是' if row['rejected'] else '否'} | "
                         f"{row['elapsed']:.1f} | {'✅' if row['ok'] else '❌'} |")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
