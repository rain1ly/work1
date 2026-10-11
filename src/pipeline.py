"""问答主流程：把五层模块串成一条可运行的链路。

调用顺序（也是答辩时讲架构的顺序）：

    图片(可选) → M13 图像理解
      ↓
    M5 查询改写 → M26 渠道路由 → M6 混合检索 → M7 重排
      ↓
    M8 置信度判定 ──拒答──▶ 直接返回「未找到」
      ↓ 通过
    M12 安全分级 → M10 生成 → M11 引用溯源 → M24 视频匹配
      ↓
    返回：答案 + 引用 + 视频 + 警示 + 全过程 trace
"""

from __future__ import annotations

import time
from pathlib import Path

from .config import CONFIG
from .generation import citation as citation_mod
from .generation import llm_client, safety, video_match, vision
from .ingestion.indexer import ensure_index
from .retrieval import channel_router, confidence, hybrid_search, query_rewrite, reranker


def answer(
    question: str,
    channels: list[str] | None = None,
    history: list[dict] | None = None,
    image: str | Path | None = None,
) -> dict:
    """回答一个问题，返回结构化结果。"""
    started = time.time()
    trace: dict = {}

    # 0) 建库兜底：索引不存在时自动构建，避免第一次运行就报错
    ensure_index()

    # 1) 图片入口（可选）：只做识别，不做回答
    vision_result = None
    if image:
        vision_result = vision.recognize(image)
        trace["vision"] = {
            "ok": vision_result.get("ok"),
            "mode": vision_result.get("mode"),
            "message": vision_result.get("message", ""),
        }
        if vision_result.get("ok"):
            question = vision.to_query(vision_result, question)

    # 2) 查询改写（术语扩展 + 多轮指代消解）
    rewrite = query_rewrite.rewrite(question, history)
    trace["rewrite"] = {
        "expansions": rewrite["expansions"],
        "is_followup": rewrite["is_followup"],
    }

    # 3) 渠道路由（在检索阶段生效）
    route = channel_router.route(channels)
    trace["channels"] = route["effective"]

    # 4) 混合检索（召回用扩展后的查询，覆盖率用未扩展的原句）
    hits = hybrid_search.search(
        rewrite["rewritten"],
        channels=route["effective"],
        coverage_text=rewrite["resolved"],
    )
    trace["recall"] = len(hits)

    # 5) 重排（同样用原句，扩展别名会干扰相关性打分）
    top_hits = reranker.rerank(rewrite["resolved"], hits)
    trace["rerank"] = len(top_hits)
    trace["top_score"] = round(float(getattr(top_hits[0], "score", 0.0)), 4) if top_hits else 0.0

    # 6) 置信度判定
    decision = confidence.decide(top_hits)
    source_info = channel_router.analyze_sources(top_hits)

    base = {
        "question": question,
        "original_question": rewrite["original"],
        "rewritten": rewrite["rewritten"],
        "expansions": rewrite["expansions"],
        "channels": {
            "requested": list(channels or []),
            "effective": route["effective"],
            "display": route["display"],
            "warnings": route["warnings"],
            "notices": route.get("notices", []),
            "sources": source_info,
        },
        "vision": vision_result,
        "trace": trace,
    }

    if not decision.answer:
        refusal = confidence.build_refusal(rewrite["original"], decision)
        refusal.update(base)
        refusal["video"] = None
        refusal["safety"] = safety.assess(rewrite["original"], []).to_dict()
        refusal["trace"]["elapsed_ms"] = int((time.time() - started) * 1000)
        return refusal

    # 7) 安全分级
    assessment = safety.assess(rewrite["original"], [h.chunk for h in top_hits])

    # 8) 生成
    contexts = [h.chunk for h in top_hits]
    generated = llm_client.generate(rewrite["original"], contexts, history)

    # 9) 引用溯源
    citations = citation_mod.build(top_hits)

    # 10) 视频匹配
    video = video_match.match(rewrite["original"], contexts)

    text = safety.decorate(generated["text"], assessment)

    result = {
        **base,
        "answer": text,
        "refused": False,
        "confidence": round(decision.confidence, 4),
        "mode": generated["mode"],
        "model_error": generated.get("error"),
        "citations": citations,
        "video": video,
        "safety": assessment.to_dict(),
        "disclaimer": assessment.disclaimer,
    }
    result["trace"]["elapsed_ms"] = int((time.time() - started) * 1000)
    return result


def render(result: dict) -> str:
    """把结果渲染成终端可读文本（供 demo 与调试使用）。"""
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append(f"问：{result.get('original_question', result.get('question', ''))}")
    if result.get("rewritten") and result["rewritten"] != result.get("original_question"):
        lines.append(f"改写：{result['rewritten']}")
    if result.get("expansions"):
        lines.append(f"术语扩展：{'、'.join(result['expansions'])}")
    lines.append(f"渠道：{result['channels'].get('display', '')}")
    lines.append("-" * 68)

    if result.get("refused"):
        lines.append("【拒答】" + result.get("answer", ""))
        lines.append(f"（原因：{result.get('reason', '')}｜置信度 {result.get('confidence', 0):.2f}）")
    else:
        # 警示与免责声明已由 safety.decorate 拼进 answer，这里不要重复输出
        lines.append(result.get("answer", ""))
        lines.append("")
        if result.get("citations"):
            lines.append("依据来源：")
            for c in result["citations"]:
                lines.append(f"  [{c['index']}] 《{c['title']}》 {c['locator']}　渠道：{c['channel']}")
        if result.get("video"):
            v = result["video"]
            lines.append("")
            lines.append(f"示范视频：{v['title']}（{v['duration']}）")
            lines.append(f"  {v.get('jump_label', '')}｜来源：{v.get('source', '')}")
            if v.get("note"):
                lines.append(f"  注意：{v['note']}")

    lines.append("-" * 68)
    for notice in result["channels"].get("notices", []):
        lines.append(f"提示：{notice}")
    src = result["channels"].get("sources", {})
    if src.get("note"):
        lines.append(f"来源提示：{src['note']}")
    lines.append(
        f"[trace] 召回 {result['trace'].get('recall', 0)} → 重排 {result['trace'].get('rerank', 0)}"
        f"｜置信度 {result.get('confidence', 0):.2f}"
        f"｜生成模式 {result.get('mode', '-')}"
        f"｜耗时 {result['trace'].get('elapsed_ms', 0)}ms"
    )
    lines.append("=" * 68)
    return "\n".join(lines)
