"""M3 语义切分模块（含元数据标注）。

切分策略：
  1. 先按 Markdown 标题层级做「结构感知切分」，保留章节路径；
  2. 相邻短章节合并，避免碎片化；
  3. 超长章节按字数二次切分，并保留重叠（防止一句话被切断）。

每条 chunk 都带完整元数据（来源 / 渠道 / 版本 / 章节 / 序号），
这是引用溯源（M11）和渠道过滤（M26）的基础，缺一不可。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

MIN_SECTION_CHARS = 60  # 小于此长度的章节参与合并


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    channel: str
    version: str
    section: str
    seq: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def split_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题把正文切成 [(章节路径, 正文), ...]。"""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in buf):
            path = " > ".join(title for _, title in stack) or "正文"
            sections.append((path, "\n".join(buf).strip()))

    for raw_line in text.splitlines():
        m = _HEADING_RE.match(raw_line.strip())
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = [(lv, t) for lv, t in stack if lv < level]
            stack.append((level, title))
        else:
            buf.append(raw_line)
    flush()
    return sections


def _windows(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def _pack(
    sections: list[tuple[str, str]], max_chars: int, overlap: int
) -> list[tuple[str, str]]:
    """把章节打包成合适大小的块：短的合并，长的切窗。"""
    packed: list[tuple[str, str]] = []
    cur_section = ""
    cur_parts: list[str] = []
    cur_len = 0

    def flush_current() -> None:
        nonlocal cur_section, cur_parts, cur_len
        if cur_parts:
            packed.append((cur_section, "\n\n".join(cur_parts)))
        cur_section, cur_parts, cur_len = "", [], 0

    for section, content in sections:
        if len(content) > max_chars:
            flush_current()
            for piece in _windows(content, max_chars, overlap):
                packed.append((section, piece))
            continue

        # 章节路径一变就断开：宁可块小一点，也要保证引用能精确到小节，
        # 否则「依据来源」会指向错误的章节，引用溯源就废了。
        if cur_parts and (cur_section != section or cur_len + len(content) > max_chars):
            flush_current()
        if not cur_parts:
            cur_section = section
        cur_parts.append(content)
        cur_len += len(content)

    flush_current()
    return packed


def chunk_doc(doc, max_chars: int = 600, overlap: int = 80) -> list[Chunk]:
    """把一份 RawDoc 切成若干 Chunk。"""
    sections = split_sections(doc.body)
    packed = _pack(sections, max_chars, overlap)

    chunks: list[Chunk] = []
    for seq, (section, text) in enumerate(packed, start=1):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{seq:03d}",
                doc_id=doc.doc_id,
                title=doc.title,
                channel=doc.channel,
                version=doc.version,
                section=section,
                seq=seq,
                text=text,
            )
        )
    return chunks


def chunk_all(docs: Iterable, max_chars: int = 600, overlap: int = 80) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_doc(doc, max_chars=max_chars, overlap=overlap))
    return out
