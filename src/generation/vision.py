"""M13 图像理解模块（F11 图片识别问答）。

核心原则（写在代码最显眼的位置）：
    **多模态模型只做「看图 → 输出结构化信息」，绝不直接回答安全问题。**

原因很简单：如果把图片丢给模型让它凭内部知识作答，它一定会编。
正确链路是「图 → 识别出物质/标志 → 生成结构化查询 → 走原有 RAG 检索 → 带引用回答」，
这样复用 90% 的既有模块，且答案仍然可溯源。

未配置多模态模型时，明确返回「未启用」，不做任何猜测。
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from ..config import CONFIG, ROOT
from .llm_client import _post_json

VISION_PROMPT = """你是实验室场景识别助手。你的任务**只是识别**，不要回答任何安全操作问题。

请识别图片内容，严格输出 JSON（不要输出 JSON 以外的任何文字）：

{
  "objects": [
    {"name": "识别到的物体名称", "type": "化学品|警示标志|设备|废液容器|其他", "evidence": "图中可见的依据"}
  ],
  "text": "图中可读的文字，原样摘录，没有就留空",
  "uncertain": true 或 false,
  "query_hint": "建议用于检索的关键词，例如「硫酸 存放 危害」"
}

要求：
1. 只描述图上真实存在的内容，**严禁编造**。
2. 无法确认时把 uncertain 设为 true，并说明不确定的原因，不要硬猜。
3. query_hint 要写成适合检索规范文件的短词组合。"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _encode_image(path: Path) -> str:
    mime = _MIME.get(path.suffix.lower(), "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _extract_json(text: str) -> dict | None:
    candidate = text.strip()
    m = _FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None


def recognize(image_path: str | Path) -> dict:
    """识别图片，返回结构化信息（不含安全结论）。"""
    if not CONFIG.model.vl_ready:
        return {
            "ok": False,
            "mode": "disabled",
            "message": (
                "图片识别未启用：请在 .env 中配置 VL_MODEL（及 VL_API_KEY / VL_API_BASE）。"
                "当前仍可使用文字提问。"
            ),
        }

    path = Path(image_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {"ok": False, "mode": "error", "message": f"图片不存在：{path}"}

    base = (CONFIG.model.vl_api_base or CONFIG.model.llm_api_base).rstrip("/")
    key = CONFIG.model.vl_api_key or CONFIG.model.llm_api_key
    payload = {
        "model": CONFIG.model.vl_model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": _encode_image(path)}},
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    try:
        result = _post_json(f"{base}/chat/completions", payload, headers, timeout=90)
        raw = result["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "mode": "error", "message": f"图像识别调用失败：{exc}"}

    parsed = _extract_json(raw)
    if not parsed:
        return {"ok": False, "mode": "error", "message": "识别结果无法解析为 JSON", "raw": raw}

    objects = parsed.get("objects") or []
    query = parsed.get("query_hint") or ""
    if not query:
        names = [str(o.get("name", "")) for o in objects if isinstance(o, dict)]
        query = " ".join(n for n in names if n)

    return {
        "ok": True,
        "mode": "vlm",
        "objects": objects,
        "text": parsed.get("text", ""),
        "uncertain": bool(parsed.get("uncertain")),
        "query": query.strip(),
        "raw": raw,
        "notice": "识别结果仅供参考，请以实物标签为准。" if parsed.get("uncertain") else "",
    }


def to_query(result: dict, question: str = "") -> str:
    """把识别结果拼成检索用的查询串。"""
    parts: list[str] = []
    if question:
        parts.append(question)
    for obj in result.get("objects") or []:
        if isinstance(obj, dict) and obj.get("name"):
            parts.append(str(obj["name"]))
    if result.get("text"):
        parts.append(str(result["text"])[:80])
    if result.get("query"):
        parts.append(str(result["query"]))
    return " ".join(parts).strip()
