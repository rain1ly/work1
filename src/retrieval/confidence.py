"""M8 置信度与拒答模块（F3）。

安全领域最重要的功能之一：**宁可拒答，也不能猜**。
一个自信地说错的助手，比一个老实说「我不知道」的助手危险得多。

判定逻辑很朴素：取重排后的最高分作为置信度，低于阈值就拒答。
阈值放在 config（默认 0.30），需要按语料调 —— 调保守一点没关系，
多拒答几次的代价，远小于答错一次。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CONFIG

REFUSE_TEMPLATE = (
    "知识库中未找到与「{question}」相关的可靠依据，因此无法作答。\n"
    "可尝试：① 换一种说法或补充具体化学品/设备名称；"
    "② 检查检索渠道是否选对了；③ 直接联系实验室管理处或指导教师。"
)


@dataclass
class Decision:
    answer: bool
    confidence: float
    top_score: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "top_score": round(self.top_score, 4),
            "reason": self.reason,
        }


def decide(hits: list) -> Decision:
    """根据重排结果判断是否作答。两道闸门，任一不过就拒答。"""
    threshold = CONFIG.retrieval.min_score
    min_coverage = CONFIG.retrieval.min_coverage

    if not hits:
        return Decision(False, 0.0, 0.0, "未召回任何候选片段")

    top = hits[0]
    top_score = float(getattr(top, "score", 0.0))
    rerank_score = float(getattr(top, "rerank", 0.0))
    coverage = float(getattr(top, "coverage", 0.0))

    # 综合分：粗排分数与重排分数取较大者，避免单一指标误杀
    confidence = max(top_score, rerank_score)

    # 闸门一：查询实词覆盖率。问「游泳池几点关门」时覆盖率会是 0，
    # 这条能挡住「用一个漂亮分数硬答不相关问题」——安全领域最怕的就是这个。
    if coverage < min_coverage:
        return Decision(
            False,
            confidence,
            top_score,
            f"关键术语覆盖率 {coverage:.2f} 低于门槛 {min_coverage:.2f}，问题可能超出知识库范围",
        )

    # 闸门二：置信度阈值
    if confidence < threshold:
        return Decision(
            False,
            confidence,
            top_score,
            f"最高置信度 {confidence:.2f} 低于阈值 {threshold:.2f}",
        )

    return Decision(True, confidence, top_score, "置信度达标")


def refuse_message(question: str) -> str:
    return REFUSE_TEMPLATE.format(question=question)


def build_refusal(question: str, decision: Decision) -> dict:
    """构造拒答响应（与正常响应同构，方便前端统一处理）。"""
    return {
        "answer": refuse_message(question),
        "refused": True,
        "confidence": round(decision.confidence, 4),
        "reason": decision.reason,
        "citations": [],
        "video": None,
        "warning": None,
    }
