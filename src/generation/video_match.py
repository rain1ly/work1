"""M24 视频·场景匹配模块（F12 回答附带操作示范视频）。

匹配逻辑故意做得很简单：**场景标签的字面命中数**。为什么不用向量？
因为视频数量很少（个位数到几十），而且标签是人工标的，字面匹配已经足够准，
更重要的是**可解释**：能说清楚「为什么推这个视频」，答辩时好讲。

三道闸门（来自 M23）：
  1. 必须 approved=True；
  2. 必须有来源机构；
  3. 必须有审核人与审核日期。
  缺任何一条都不出现在答案里 —— 宁可不给，也不推一个没审核过的视频。
"""

from __future__ import annotations

from ..ingestion import video_registry
from ..text_utils import tokenize


def _fmt(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _scene_text(question: str, contexts: list[dict] | None) -> str:
    parts = [question]
    for ctx in (contexts or [])[:3]:
        parts.append(ctx.get("text") or "")
        parts.append(ctx.get("section") or "")
    return " ".join(parts)


def _pick_chapter(video: dict, scene: str) -> dict | None:
    """挑出最该跳转的步骤；挑不出来就用第一章。"""
    chapters = video.get("chapters") or []
    if not chapters:
        return None
    scene_tokens = set(tokenize(scene))
    best = None
    best_score = 0
    for chapter in chapters:
        label = str(chapter.get("label", ""))
        score = len(scene_tokens & set(tokenize(label)))
        if score > best_score:
            best, best_score = chapter, score
    return best or chapters[0]


def match(question: str, contexts: list[dict] | None = None, min_score: int = 1) -> dict | None:
    """返回匹配到的已审核视频；没有合适的就返回 None。

    打分规则：**问题里命中的标签权重 3 分，仅出现在参考资料里的标签 1 分**。
    为什么这样设计：检索回来的片段主题经常跑偏（问乙醚会捎带召回灼伤条款），
    如果两边等权，就会推错视频。以问题为准才符合用户预期。
    """
    q = question or ""
    scene = _scene_text(question, contexts)

    best: dict | None = None
    best_tags: list[str] = []
    best_score = 0

    for video in video_registry.list_videos(approved_only=True):
        if not video_registry.is_publishable(video):
            continue
        tags = [t for t in (video.get("scene_tags") or []) if t]
        q_hits = [t for t in tags if t in q]
        c_hits = [t for t in tags if t in scene and t not in q_hits]
        score = 3 * len(q_hits) + len(c_hits)
        if score > best_score:
            best, best_tags, best_score = video, q_hits + c_hits, score

    if best is None or best_score < min_score:
        return None

    chapter = _pick_chapter(best, scene)
    placeholder = bool(best.get("placeholder"))

    return {
        "video_id": best.get("video_id"),
        "title": best.get("title"),
        "file": best.get("file"),
        "duration": best.get("duration"),
        "source": best.get("source"),
        "reviewer": best.get("reviewer"),
        "review_date": best.get("review_date"),
        "chapters": best.get("chapters") or [],
        "matched_tags": best_tags,
        "jump_to": chapter,
        "jump_label": (
            f"跳到 {_fmt(int(chapter.get('start', 0)))} {chapter.get('label', '')}".strip()
            if chapter
            else ""
        ),
        "placeholder": placeholder,
        "note": "该视频为示例占位素材，正式提交前需替换为真实已审核视频。" if placeholder else "",
    }
