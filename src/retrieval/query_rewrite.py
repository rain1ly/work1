"""M5 查询改写模块。

做三件事：
  1. **术语同义扩展**（F7）：用户说「酒精」，库里写的是「乙醇」，不做扩展就检索不到；
  2. **多轮指代消解**（F5）：把「那溅到眼睛呢」这类追问补全成可检索的完整问题；
  3. **口语词清洗**：去掉「请问」「怎么办啊」这类不影响语义的词。

安全领域的同义词特别多（化学品有俗名、学名、商品名），这张表要持续补充，
它是「花小钱办大事」的典型 —— 加几行字典，召回率就能上一个台阶。
"""

from __future__ import annotations

import re

# 领域同义词表：key 为规范名，value 为常见别名。会做双向扩展。
TERM_ALIASES: dict[str, list[str]] = {
    "乙醇": ["酒精"],
    "氢氧化钠": ["烧碱", "火碱", "苛性钠"],
    "盐酸": ["氢氯酸"],
    "硫酸": ["浓硫酸"],
    "乙酸": ["醋酸"],
    "碳酸氢钠": ["小苏打", "苏打"],
    "乙醚": ["麻醉醚", "乙氧基乙烷"],
    "丙酮": ["二甲基酮"],
    "甲醇": ["木醇"],
    "苯": ["纯苯"],
    "MSDS": ["SDS", "安全技术说明书", "化学品安全说明书"],
    "洗眼器": ["紧急洗眼器", "洗眼装置"],
    "通风柜": ["通风橱", "排风柜"],
    "气瓶": ["钢瓶", "气罐"],
    "废液": ["废液桶", "实验废水"],
    "个体防护装备": ["PPE", "防护装备", "劳保用品"],
    "灼伤": ["烧伤", "烫伤", "腐蚀伤"],
    "泄漏": ["泄露", "渗漏"],
    "明火": ["火源", "点火源"],
}

# 指代词：出现这些时尝试拼接上一轮问题（多轮追问）
_ANAPHORA_RE = re.compile(r"^(那|那么|那要|呢|还有|它|这个|那个|这种情况|这种|如果是)")
_ANAPHORA_PAT = re.compile(r"(呢|那呢|的话)")
_ORAL_RE = re.compile(r"(请问|帮我|我想知道|能不能|可以吗|一下|啊|呢|吧|谢谢|你好|怎么弄|咋办)")


def _build_lookup() -> dict[str, set[str]]:
    """为每个词（含别名）建立「应一起检索」的词集合。"""
    lookup: dict[str, set[str]] = {}
    for canonical, aliases in TERM_ALIASES.items():
        group = {canonical, *aliases}
        for word in group:
            lookup.setdefault(word, set()).update(group - {word})
    return lookup


_LOOKUP = _build_lookup()


def expand(question: str) -> tuple[str, list[str]]:
    """术语同义扩展，返回 (扩展后的检索文本, 命中的扩展词)。"""
    hits: list[str] = []
    extra: list[str] = []
    for word, related in _LOOKUP.items():
        if word in question:
            hits.append(word)
            for r in related:
                if r not in question:
                    extra.append(r)
    if not extra:
        return question, []
    deduped = list(dict.fromkeys(extra))
    return f"{question} {' '.join(deduped)}", deduped


def clean(question: str) -> str:
    """去掉口语成分，保留检索信息。"""
    return _ORAL_RE.sub(" ", question).strip()


def is_followup(question: str) -> bool:
    q = question.strip()
    if len(q) <= 6:
        return True
    return bool(_ANAPHORA_RE.match(q) or _ANAPHORA_PAT.search(q))


def resolve_followup(question: str, history: list[dict] | None) -> str:
    """多轮指代消解：把追问补全成完整问题。"""
    if not history or not is_followup(question):
        return question
    last_user = ""
    for turn in reversed(history):
        if turn.get("role") == "user" and turn.get("content"):
            last_user = str(turn["content"])
            break
    if not last_user:
        return question
    return f"{last_user}；追问：{question}"


def rewrite(question: str, history: list[dict] | None = None) -> dict:
    """完整改写流程。"""
    resolved = resolve_followup(question, history)
    cleaned = clean(resolved)
    expanded, terms = expand(cleaned)

    # 多轮合并后可能与上一轮重复，这里做去重
    if expanded.count("；追问：") == 1 and resolved != question:
        head, _, tail = expanded.partition("；追问：")
        if head.strip() and head.strip() in tail:
            expanded = tail.strip()

    return {
        "original": question,
        "resolved": resolved,
        "rewritten": expanded,
        "expansions": terms,
        "is_followup": resolved != question,
    }
