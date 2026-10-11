"""M11 引用溯源模块（F2）。

把检索片段转成前端可直接渲染的引用卡片。
这是全项目最不能砍的模块之一：它决定了答案「可不可核查」，
也是评委判断「这是真 RAG 还是套壳」的第一眼依据。

定位信息说明：
  - Markdown / 文本：定位到「章节 > 小节」
  - PDF：需在 M2 解析时保留页码（当前实现未保留，见 TODO）
"""

from __future__ import annotations

from ..text_utils import snippet

# TODO(第2周)：PDF 解析时按页保留页码，写入 chunk 的 page 字段，
#              这里优先输出「P.xx」形式，更符合用户对「出处」的直觉。
_COLON = " § "


def build(hits: list, limit: int | None = None) -> list[dict]:
    """把重排后的命中转成引用列表。"""
    items = hits if limit is None else hits[:limit]
    citations: list[dict] = []
    for idx, hit in enumerate(items, start=1):
        chunk = hit.chunk
        citations.append(
            {
                "index": idx,
                "chunk_id": chunk.get("chunk_id", ""),
                "doc_id": chunk.get("doc_id", ""),
                "title": chunk.get("title", "未知文件"),
                "channel": chunk.get("channel", "未标注"),
                "version": chunk.get("version", ""),
                "section": chunk.get("section", ""),
                "page": chunk.get("page"),
                "locator": _locator(chunk),
                "snippet": snippet(chunk.get("text", ""), 160),
                "score": round(float(getattr(hit, "score", 0.0)), 4),
                "rerank": round(float(getattr(hit, "rerank", 0.0)), 4),
            }
        )
    return citations


def _locator(chunk: dict) -> str:
    parts: list[str] = []
    if chunk.get("page"):
        parts.append(f"P.{chunk['page']}")
    if chunk.get("section"):
        parts.append(str(chunk["section"]))
    if chunk.get("version"):
        parts.append(f"版本 {chunk['version']}")
    return _COLON.join(parts) if parts else "未标注位置"


def to_markdown(citations: list[dict]) -> str:
    """生成给文档/README 用的 Markdown 来源列表。"""
    if not citations:
        return ""
    lines = ["**依据来源**", ""]
    for c in citations:
        lines.append(
            f"- [{c['index']}] 《{c['title']}》 {c['locator']}　"
            f"（渠道：{c['channel']}）"
        )
    return "\n".join(lines)
