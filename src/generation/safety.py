"""M12 安全分级模块（F6）。

三件事：
  1. **高危操作警示** —— 涉及强酸强碱、易燃易爆、气瓶、高压等操作时，答案顶部加警示；
  2. **免责与口径说明** —— 明示「以现行规范条款与现场指导教师要求为准」；
  3. **内容边界** —— 涉及人员受伤时只给规程处置流程，不做医学诊断。

这一条是安全类题目独有的差异化，别的题目抄不走：
它来自「这个领域答错代价高」的特质，而不是来自技术。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HIGH_RISK_PATTERNS = [
    (r"(强酸|浓硫酸|浓硝酸|盐酸|硫酸|硝酸)", "腐蚀性化学品"),
    (r"(强碱|氢氧化钠|烧碱|火碱|氨水)", "腐蚀性化学品"),
    (r"(易燃|乙醚|丙酮|甲醇|乙醇|苯|可燃)", "易燃化学品"),
    (r"(爆炸|过氧化物|高氯酸|硝化)", "爆炸性物质"),
    (r"(气瓶|钢瓶|高压|减压阀|可燃气体)", "压力容器"),
    (r"(剧毒|氰化|砷|汞|重金属)", "剧毒物质"),
    (r"(明火|点火|加热|回流|蒸馏)", "明火与热源操作"),
    (r"(废液|危废|混合|混放)", "危险废物处置"),
]

INJURY_RE = re.compile(r"(受伤|灼伤|烧伤|烫伤|溅到|溅入|中毒|吸入|误食|割伤|刺伤|出血)")

WARNING_TEMPLATE = "⚠ 涉及{topic}相关操作，务必在指导教师在场、佩戴相应防护装备的情况下进行，严禁独自操作。"

DISCLAIMER = "以上内容依据知识库文件整理，具体操作请以现行规范条款与现场指导教师的要求为准。"

MEDICAL_BOUNDARY = "涉及人员受伤时，请立即按规程处置并送医，本助手不提供医学诊断或用药建议。"


@dataclass
class Assessment:
    level: str          # "high" | "normal"
    topics: list[str]
    warning: str | None
    disclaimer: str
    medical_boundary: str | None

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "topics": self.topics,
            "warning": self.warning,
            "disclaimer": self.disclaimer,
            "medical_boundary": self.medical_boundary,
        }


def assess(question: str, contexts: list[dict] | None = None) -> Assessment:
    """危险等级判定。

    主题以**问题本身**为准；只有问题里看不出风险时，才回退到参考资料。
    原因：检索回来的片段常常横跨多个主题（问乙醚会捎带召回灼伤条款），
    用它们判定会把警示内容搞乱，反而降低可信度。
    """
    q = question or ""

    topics: list[str] = []
    for pattern, label in HIGH_RISK_PATTERNS:
        if re.search(pattern, q) and label not in topics:
            topics.append(label)

    if not topics and contexts:
        blob = " ".join((c.get("text") or "") for c in contexts)
        for pattern, label in HIGH_RISK_PATTERNS:
            if re.search(pattern, blob) and label not in topics:
                topics.append(label)

    # 医疗边界只由问题触发：问的是「怎么处置」，不是「怎么了」
    injury = bool(INJURY_RE.search(q))

    if topics:
        warning = WARNING_TEMPLATE.format(topic="、".join(topics[:2]))
        level = "high"
    else:
        warning = None
        level = "normal"

    return Assessment(
        level=level,
        topics=topics,
        warning=warning,
        disclaimer=DISCLAIMER,
        medical_boundary=MEDICAL_BOUNDARY if injury else None,
    )


def decorate(answer: str, assessment: Assessment) -> str:
    """把警示与说明拼接到答案前后。"""
    parts: list[str] = []
    if assessment.warning:
        parts.append(assessment.warning)
    if assessment.medical_boundary:
        parts.append(assessment.medical_boundary)
    parts.append(answer.strip())
    parts.append(f"—— {assessment.disclaimer}")
    return "\n\n".join(p for p in parts if p)
