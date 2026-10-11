"""基础回归测试。

跑法：
    python -m pytest tests/ -q     # 装了 pytest
    python tests/test_pipeline.py  # 不装 pytest 也能跑

覆盖四条最关键的行为：
  1. 建库成功且切块带元数据（引用溯源的前提）
  2. 知识库内的问题能作答，且每条答案都带引用
  3. 超出知识库范围的问题必须拒答（安全领域最重要的行为）
  4. 渠道过滤真的生效（选了「事故案例」就不该出「本校文件」的引用）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.indexer import build_index, get_chunks  # noqa: E402
from src.pipeline import answer  # noqa: E402


def test_index_builds_with_metadata() -> None:
    info = build_index()
    assert info["ok"], f"建库失败：{info.get('message')}"
    assert info["chunks"] > 0

    chunks = get_chunks()
    for chunk in chunks:
        assert chunk["chunk_id"]
        assert chunk["channel"], "每个切块都必须带渠道元数据"
        assert chunk["section"], "每个切块都必须带章节元数据（引用溯源依赖它）"


def test_in_domain_question_is_answered_with_citation() -> None:
    result = answer("浓硫酸溅到手上该怎么处理？")
    assert not result["refused"], "知识库内的问题不应被拒答"
    assert result["citations"], "作答时必须给出引用来源"
    assert result["citations"][0]["locator"]


def test_out_of_domain_question_is_refused() -> None:
    for question in ["学校游泳池几点关门？", "附近哪家火锅好吃？"]:
        result = answer(question)
        assert result["refused"], f"超范围问题必须拒答：{question}"
        assert not result["citations"], "拒答时不应给引用"


def test_channel_filter_is_effective() -> None:
    result = answer("废液桶里可以混着倒吗？", channels=["事故案例"])
    assert not result["refused"]
    channels = {c["channel"] for c in result["citations"]}
    assert channels <= {"事故案例"}, f"渠道过滤失效，出现了其他渠道：{channels}"


def test_unknown_channel_does_not_break() -> None:
    result = answer("乙醚怎么存放？", channels=["不存在的渠道"])
    assert result["channels"]["effective"], "渠道全部不可用时应回退到默认渠道"
    assert result["channels"]["warnings"], "回退行为必须给出提示，不能静默处理"


def _main() -> int:
    tests = [
        test_index_builds_with_metadata,
        test_in_domain_question_is_answered_with_citation,
        test_out_of_domain_question_is_refused,
        test_channel_filter_is_effective,
        test_unknown_channel_does_not_break,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
