"""最小可运行演示（不启动服务，直接看效果）。

用法：
    python scripts/demo.py                      # 跑内置的 5 个演示问题
    python scripts/demo.py "乙醚怎么存放"        # 问一个自己的问题
    python scripts/demo.py --rebuild            # 先重建索引再演示

这个脚本能在**零配置、零外部依赖**的情况下跑通完整链路：
查询改写 → 渠道过滤 → 混合检索 → 重排 → 置信度 → 生成 → 引用 → 视频。
第一天就能录演示视频，不用等模型和向量库都装好。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.indexer import build_index, stats  # noqa: E402
from src.pipeline import answer, render  # noqa: E402

DEMO_CASES: list[tuple[str, list[str] | None]] = [
    ("浓硫酸溅到手上，该怎么处理？", None),
    ("乙醚可以放在普通试剂柜里吗？", ["本校文件", "国家规范"]),
    ("气瓶和明火要保持多远？", None),
    ("废液桶里可以混着倒吗？", ["事故案例"]),
    ("学校游泳池几点关门？", None),
]


def main() -> int:
    args = sys.argv[1:]

    if "--rebuild" in args:
        print(">> 重建索引…")
        info = build_index(rebuild=True)
        print(f"   {info.get('docs')} 篇文档 → {info.get('chunks')} 个切块\n")

    info = stats()
    if not info["chunks"]:
        print("索引为空，正在自动建库…")
        build_index()
        info = stats()

    print(f"索引：{info['docs']} 篇文档 / {info['chunks']} 个切块｜向量后端 {info['backend']}")
    print(f"渠道分布：{info['channels']}")
    print()

    custom = [a for a in args if not a.startswith("--")]
    cases = [(custom[0], None)] if custom else DEMO_CASES

    for question, channels in cases:
        result = answer(question, channels=channels)
        print(render(result))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
