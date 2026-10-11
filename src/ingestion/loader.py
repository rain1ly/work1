"""M1 文档接入模块。

职责：扫描 data/raw、读取元信息、检测变更。

元信息优先级：
  1) md / txt 顶部的 front-matter（推荐，最省事）
  2) 同名 sidecar 文件 <文件名>.meta.json（适合 PDF / Word 这类不方便写头部的格式）
  3) 兜底默认值：channel = 本校文件

channel 字段是渠道过滤（M26）的基础，务必保证准确。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import RAW_DIR
from . import parser as parser_mod

SUPPORTED_EXT = {".md", ".txt", ".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg"}

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)

DEFAULT_CHANNEL = "本校文件"


@dataclass
class RawDoc:
    doc_id: str
    path: str
    filename: str
    title: str
    channel: str
    version: str
    body: str
    digest: str
    warnings: list[str] = field(default_factory=list)

    def to_meta(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "title": self.title,
            "channel": self.channel,
            "version": self.version,
            "digest": self.digest,
        }


def parse_front_matter(text: str) -> tuple[dict, str]:
    """解析 `---` 包裹的简单 key: value 头部。"""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, text[m.end() :]


def _sidecar_meta(path: Path) -> dict:
    sidecar = path.with_suffix(".meta.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_doc(path: Path) -> RawDoc:
    body, warnings = parser_mod.parse(path)
    meta: dict[str, str] = {}

    if path.suffix.lower() in (".md", ".txt"):
        fm, body = parse_front_matter(body)
        meta.update(fm)
    meta.update(_sidecar_meta(path))

    doc_id = meta.get("doc_id") or path.stem
    title = meta.get("title") or path.stem
    channel = meta.get("channel") or DEFAULT_CHANNEL
    version = meta.get("version") or ""
    digest = hashlib.md5(body.encode("utf-8")).hexdigest()

    return RawDoc(
        doc_id=doc_id,
        path=str(path),
        filename=path.name,
        title=title,
        channel=channel,
        version=version,
        body=body,
        digest=digest,
        warnings=list(warnings),
    )


def discover(raw_dir: Path | None = None) -> list[Path]:
    base = Path(raw_dir or RAW_DIR)
    if not base.exists():
        return []
    files = [
        p
        for p in sorted(base.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXT
        and not p.name.endswith(".meta.json")
        and not p.name.startswith(".")
    ]
    return files


def load_all(raw_dir: Path | None = None) -> tuple[list[RawDoc], list[str]]:
    """返回 (文档列表, 全局警告列表)。"""
    docs: list[RawDoc] = []
    warnings: list[str] = []
    for path in discover(raw_dir):
        try:
            doc = load_doc(path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"[{path.name}] 读取失败：{exc}")
            continue
        if not doc.body.strip():
            warnings.append(f"[{path.name}] 解析结果为空，已跳过（扫描件请先接入 OCR）")
            continue
        warnings.extend(doc.warnings)
        docs.append(doc)
    return docs, warnings


def describe_changes(docs: list[RawDoc], previous: dict[str, str]) -> dict[str, list[str]]:
    """对比上次索引，给出新增 / 修改 / 删除清单（M1 增量检测）。"""
    current = {d.doc_id: d.digest for d in docs}
    added = [k for k in current if k not in previous]
    modified = [k for k in current if k in previous and previous[k] != current[k]]
    removed = [k for k in previous if k not in current]
    return {"added": added, "modified": modified, "removed": removed}
