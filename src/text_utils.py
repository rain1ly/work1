"""共享文本工具（非编号模块）。

被数据层与检索层共用：中文分词、字符 bigram、稳定哈希。
优先使用 jieba；未安装时退化为「中文二字组合 + 拉丁/数字串」，保证零依赖也能跑通。
"""

from __future__ import annotations

import hashlib
import re

# 轻量停用词：只保留高频虚词，避免过度过滤掉「不得」「严禁」这类关键否定词
_STOP = {
    "的", "了", "和", "与", "及", "或", "在", "是", "为", "对", "把", "被",
    "将", "给", "到", "从", "等", "这", "那", "有", "可以", "需要", "应该",
    "如果", "若", "则", "且", "并", "而", "于", "以及", "其中", "上下", "内",
    "外", "地", "得", "着", "啊", "吧", "呢", "吗", "之", "其", "也", "都",
    "很", "非常", "更加", "一些", "一个", "进行", "使用", "通过", "时候",
    "问题", "情况", "相关", "我们", "你们", "他们", "请问", "怎么", "如何",
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RE = re.compile(r"[a-z0-9]+")

# 虚词字符集：含这些字的 bigram 一律不当作实词
_FUNC_CHARS = set(
    "的了和与及或在是有要几多什么怎么呢吗吧啊把被将给到从就都也很更还只又再等则以且而且"
    "于其中上下内外这那它我你他她它们地得着对为并且如果若呀嘛哦嗯个之其所能可"
)


def content_bigrams(text: str) -> list[str]:
    """只保留「两个字都是实词」的 bigram。

    为什么必须过滤：中文 bigram 会把「瓶和」「和明」「要保」这种跨虚词的组合也切出来，
    这些噪声 Token 会摊薄覆盖率分母，把正常问题误判成「超出知识范围」。
    过滤后「气瓶和明火要保持多远」→ 气瓶 / 明火 / 保持 / 多远，覆盖率立刻恢复正常。
    """
    low = (text or "").lower()
    out: list[str] = []
    for run in _CJK_RE.findall(low):
        for i in range(len(run) - 1):
            a, b = run[i], run[i + 1]
            if a in _FUNC_CHARS or b in _FUNC_CHARS:
                continue
            out.append(a + b)
    for word in _LATIN_RE.findall(low):
        if len(word) >= 2:
            out.append(word)
    return out

_jieba = None
_jieba_tried = False


def _get_jieba():
    global _jieba, _jieba_tried
    if not _jieba_tried:
        _jieba_tried = True
        try:
            import jieba  # type: ignore

            _jieba = jieba
        except Exception:
            _jieba = None
    return _jieba


def tokenize(text: str) -> list[str]:
    """切词。优先 jieba；未安装时用「中文二字组合 + 拉丁数字串」。"""
    if not text:
        return []
    low = text.strip().lower()
    tokens: list[str] = []

    jb = _get_jieba()
    if jb is not None:
        for t in jb.lcut(low):
            t = t.strip()
            if len(t) >= 2 and t not in _STOP and not t.isdigit():
                tokens.append(t)
        return tokens

    for word in content_bigrams(low):
        if word not in _STOP:
            tokens.append(word)
    return tokens


def stable_hash(text: str) -> int:
    """跨进程稳定的哈希（Python 内置 hash 带随机种子，不能用于持久化索引）。"""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def normalize(text: str) -> str:
    """去掉多余空白，便于比对与展示。"""
    return re.sub(r"\s+", " ", (text or "").strip())


def snippet(text: str, limit: int = 120) -> str:
    """生成短摘要，用于列表展示。"""
    text = normalize(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"
