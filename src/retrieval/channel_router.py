"""M26 检索渠道路由模块（对应功能 F10 搜索渠道自主选择）。

职责：
  1. 把用户选择的渠道解析成实际生效的渠道列表（含联网开关校验）；
  2. **校验并明确提示**：选了但没有内容的渠道要说出来，不静默回退；
  3. 多源命中时标注「从严处理」—— 本校规定通常严于国家规定，取严是安全领域的合理默认。

关键设计：渠道过滤作用在**检索阶段**，不是界面上筛结果。
所以本模块只负责「决定检哪些渠道」，过滤动作由 M6 在召回时执行。
"""

from __future__ import annotations

import re

from ..config import CONFIG
from ..ingestion.indexer import stats as index_stats

# 严格表述标记：命中说明该条依据本身是「禁止性/强制性」表述
_STRICT_RE = re.compile(r"(严禁|禁止|不得|不许|不得超过|必须|应当立即|严禁使用|切不可)")

ALL_LABEL = "全部来源"


def channel_status() -> list[dict]:
    """列出所有渠道及其可用状态（供前端渲染选择器）。"""
    per_channel = index_stats().get("channels", {})
    out: list[dict] = []
    for name in CONFIG.channels.available:
        if name == "联网搜索" and not CONFIG.channels.online_enabled:
            out.append({"name": name, "enabled": False, "chunks": 0, "note": "外部内容未经审核，默认关闭"})
            continue
        out.append(
            {
                "name": name,
                "enabled": True,
                "chunks": int(per_channel.get(name, 0)),
                "note": "",
            }
        )
    return out


def resolve(selected: list[str] | None) -> dict:
    """解析用户选择，返回生效渠道与提示信息。"""
    available = {c["name"] for c in channel_status() if c["enabled"]}
    requested = list(selected or [])

    warnings: list[str] = []
    dropped: list[str] = []

    if not requested or ALL_LABEL in requested:
        effective = [c for c in CONFIG.channels.default_enabled if c in available]
    else:
        effective = []
        for name in requested:
            if name in available:
                effective.append(name)
            else:
                dropped.append(name)
                if name == "联网搜索":
                    warnings.append("联网搜索渠道未开启（安全领域默认关闭外部未审核内容）")
                else:
                    warnings.append(f"渠道「{name}」当前不可用")
        if not effective:
            effective = [c for c in CONFIG.channels.default_enabled if c in available]
            warnings.append("所选渠道均不可用，已回退到默认渠道（请确认）")

    empty = [c for c in effective if int(index_stats().get("channels", {}).get(c, 0)) == 0]
    if empty:
        warnings.append(f"渠道「{'、'.join(empty)}」暂无可检索内容，本次结果可能不完整")

    # 渠道被限定制时给一个温和提示：结论可能不完整，但不说「一定不对」，
    # 因为用户很可能就是有意只看某一类文件（比如只想看本校规定）。
    notices: list[str] = []
    default_set = set(CONFIG.channels.default_enabled)
    if set(effective) < default_set:
        notices.append(
            f"本次仅检索「{'、'.join(effective)}」；若结论不完整，可放开渠道重试。"
        )

    return {
        "effective": effective,
        "dropped": dropped,
        "warnings": warnings,
        "notices": notices,
        "all_label": ALL_LABEL,
    }


def filter_hits(hits: list, channels: list[str]) -> list:
    """兜底过滤（正常情况 M6 已按渠道召回，这里再校验一次）。"""
    if not channels:
        return hits
    allowed = set(channels)
    return [h for h in hits if h.chunk.get("channel") in allowed]


def analyze_sources(hits: list) -> dict:
    """分析命中来源，判断是否需要「从严」提示。"""
    if not hits:
        return {"channels": [], "multi_source": False, "strict": False, "note": ""}

    channels: list[str] = []
    for hit in hits:
        ch = hit.chunk.get("channel", "")
        if ch and ch not in channels:
            channels.append(ch)

    strict = any(_STRICT_RE.search(hit.chunk.get("text", "") or "") for hit in hits)
    multi = len(channels) > 1

    note = ""
    if multi:
        note = "多来源命中：若结论不一致，按更严格的规定执行，并同时展示各来源口径。"
    if strict:
        note = (note + " " if note else "") + "命中的依据为禁止性/强制性表述，请优先遵守。"

    return {
        "channels": channels,
        "multi_source": multi,
        "strict": strict,
        "note": note.strip(),
    }


def route(selected: list[str] | None) -> dict:
    """完整路由：解析渠道 + 给出前端需要的状态。"""
    info = resolve(selected)
    return {
        "effective": info["effective"],
        "dropped": info["dropped"],
        "warnings": info["warnings"],
        "notices": info["notices"],
        "channels": channel_status(),
        "display": " + ".join(info["effective"]) or "（无）",
    }
