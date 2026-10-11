"""M9 提示模板模块。

安全领域的系统提示词有四条硬约束，缺一条都会出问题：
  1. **只依据给定材料回答** —— 不许用模型的内部知识补全；
  2. **必须标注来源编号** —— 每个结论后面跟 [1][2]，对应引用卡片；
  3. **材料里没有就直说** —— 宁可说「未找到」，编造是重大缺陷；
  4. **不做医疗诊断** —— 涉及伤口只给规程里的处置流程与求助渠道。

这套提示词建议在答辩时直接讲出来，评委很吃这一套。
"""

from __future__ import annotations

SYSTEM_PROMPT = """你是「高校实验室安全规程问答助手」，服务于高校实验室的师生。

【回答规则】（必须严格遵守）
1. 只能依据【参考资料】中的内容作答。资料中没有的信息，不要凭常识补充。
2. 每一条结论后面必须标注来源编号，格式为 [1]、[2]，编号对应参考资料序号。
3. 如果参考资料不足以回答问题，直接回答「知识库中未找到相关依据」，并说明可以怎么补充提问。严禁编造条文、编号、数值或文件名称。
4. 涉及危险操作的，在回答开头用一句话给出安全警示。
5. 涉及人员受伤的，只输出规程中的应急处置流程与就医建议，不做医学诊断、不给用药建议。
6. 若不同来源的结论不一致，明确指出差异，并说明应按更严格的规定执行。

【输出格式】
- 先给结论，再给步骤，必要时用编号列表。
- 语言简洁、口语化，避免照搬规范原文的长句。
- 不要输出「根据参考资料」这类套话，直接给内容。"""


def format_contexts(contexts: list[dict]) -> str:
    """把检索到的片段格式化成参考资料区块。"""
    if not contexts:
        return "（无参考资料）"
    blocks: list[str] = []
    for idx, ctx in enumerate(contexts, start=1):
        locator = ctx.get("section") or ""
        header = f"[{idx}] 来源：{ctx.get('title', '未知文件')}"
        if ctx.get("version"):
            header += f"（版本 {ctx['version']}）"
        header += f"｜渠道：{ctx.get('channel', '未标注')}"
        if locator:
            header += f"｜位置：{locator}"
        blocks.append(f"{header}\n{ctx.get('text', '').strip()}")
    return "\n\n".join(blocks)


def build_messages(question: str, contexts: list[dict], history: list[dict] | None = None) -> list[dict]:
    """构造发给大模型的 messages。"""
    user_content = (
        f"【参考资料】\n{format_contexts(contexts)}\n\n"
        f"【用户问题】\n{question}\n\n"
        f"请按规则作答，并标注来源编号。"
    )
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": user_content})
    return messages
