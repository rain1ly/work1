"""M19 评估脚本。

先用内置指标跑通（零依赖、秒出结果），装了 RAGAS 再补生成质量指标。

内置指标（答辩时直接给这几个数就够）：
  - 拒答准确率  : 该拒的拒了、该答的答了
  - 引用覆盖率  : 作答时是否都带了出处
  - 关键词命中率: 回答是否覆盖 ground_truth 中的关键实体（召回质量的粗代理）

用法：
    python eval/run_ragas.py               # 内置指标
    python eval/run_ragas.py --ragas       # 追加 RAGAS 指标（需 pip install ragas datasets）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.indexer import build_index  # noqa: E402
from src.pipeline import answer  # noqa: E402
from src.text_utils import content_bigrams  # noqa: E402

TESTSET = Path(__file__).parent / "testset.jsonl"


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for line in TESTSET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def keyword_hit(answer_text: str, ground_truth: str) -> float:
    """用 ground_truth 的实词 bigram 衡量回答覆盖了多少关键信息。"""
    gold = set(content_bigrams(ground_truth))
    if not gold:
        return 0.0
    got = set(content_bigrams(answer_text))
    return len(gold & got) / len(gold)


def run_builtin(cases: list[dict]) -> dict:
    rows: list[dict] = []
    for case in cases:
        result = answer(case["question"], channels=case.get("channels") or None)
        refused = bool(result.get("refused"))
        expect = case.get("expect", "answer")
        ok = (refused and expect == "refuse") or (not refused and expect == "answer")
        rows.append(
            {
                "question": case["question"],
                "expect": expect,
                "refused": refused,
                "correct": ok,
                "confidence": result.get("confidence", 0.0),
                "has_citation": bool(result.get("citations")),
                "has_video": bool(result.get("video")),
                "kw_hit": 0.0
                if refused
                else round(keyword_hit(result.get("answer", ""), case.get("ground_truth", "")), 3),
                "channels": result.get("channels", {}).get("display", ""),
            }
        )

    total = len(rows)
    answered = [r for r in rows if not r["refused"]]
    refusals = [r for r in rows if r["refused"]]
    should_refuse = [r for r in rows if r["expect"] == "refuse"]

    return {
        "rows": rows,
        "metrics": {
            "case_count": total,
            "decision_accuracy": round(sum(r["correct"] for r in rows) / total, 3) if total else 0.0,
            "refusal_accuracy": round(
                sum(r["correct"] for r in should_refuse) / len(should_refuse), 3
            )
            if should_refuse
            else None,
            "citation_rate": round(
                sum(r["has_citation"] for r in answered) / len(answered), 3
            )
            if answered
            else 0.0,
            "avg_keyword_hit": round(
                sum(r["kw_hit"] for r in answered) / len(answered), 3
            )
            if answered
            else 0.0,
            "answered": len(answered),
            "refused": len(refusals),
        },
    }


def run_ragas(cases: list[dict], builtin: dict) -> None:
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import answer_relevancy, faithfulness  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"\n[RAGAS] 未安装或不可用，跳过：{exc}")
        print("        安装方式：pip install ragas datasets")
        return

    samples = []
    for case, row in zip(cases, builtin["rows"]):
        if row["refused"]:
            continue
        result = answer(case["question"], channels=case.get("channels") or None)
        contexts = [c.get("snippet", "") for c in result.get("citations", [])]
        samples.append(
            {
                "question": case["question"],
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "ground_truth": case.get("ground_truth", ""),
            }
        )

    if not samples:
        print("\n[RAGAS] 没有可评估的样本。")
        return

    print(f"\n[RAGAS] 正在评估 {len(samples)} 条样本…")
    dataset = Dataset.from_list(samples)
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(f"[RAGAS] {scores}")


def main() -> int:
    if not TESTSET.exists():
        print(f"测试集不存在：{TESTSET}")
        return 1

    info = build_index()
    print(f"已建库：{info.get('docs')} 篇文档 / {info.get('chunks')} 个切块\n")

    cases = load_cases()
    result = run_builtin(cases)

    print(f"{'问题':<28}{'期望':<8}{'实际':<8}{'置信度':<9}{'引用':<6}{'关键词'}")
    print("-" * 76)
    for row in result["rows"]:
        actual = "拒答" if row["refused"] else "作答"
        expect = "拒答" if row["expect"] == "refuse" else "作答"
        mark = "" if row["correct"] else "  ← 不符"
        print(
            f"{row['question'][:26]:<28}{expect:<8}{actual:<8}"
            f"{row['confidence']:<9.2f}{'有' if row['has_citation'] else '无':<6}"
            f"{row['kw_hit']:.2f}{mark}"
        )

    print("\n汇总指标")
    for key, value in result["metrics"].items():
        print(f"  {key:<20}: {value}")

    if "--ragas" in sys.argv:
        run_ragas(cases, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
