"""M10 模型接入模块。

用标准库 urllib 直接调 OpenAI 兼容接口，不依赖任何 SDK ——
DeepSeek / 通义 / 智谱 / 本地 vLLM 都兼容这一套，换供应商只改 .env。

未配置模型时自动降级为「摘录模式」：直接返回最相关的片段原文。
这样保证零配置也能跑通「检索 → 引用 → 拒答」全链路，方便第一天就出演示效果。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..config import CONFIG
from . import prompts

EXTRACTIVE_BANNER = "（当前未配置大模型，以下为知识库中检索到的原文摘录）"


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(messages: list[dict], temperature: float = 0.2, timeout: int = 60) -> str:
    """调用兼容 OpenAI 协议的 chat completions。"""
    base = CONFIG.model.llm_api_base.rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": CONFIG.model.llm_model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CONFIG.model.llm_api_key}",
    }
    result = _post_json(url, payload, headers, timeout=timeout)
    return result["choices"][0]["message"]["content"].strip()


def _extractive(contexts: list[dict]) -> str:
    if not contexts:
        return ""
    top = contexts[0]
    lines = [EXTRACTIVE_BANNER, "", f"{top.get('text', '').strip()} [1]"]
    if len(contexts) > 1:
        lines.append("")
        lines.append("其他相关片段：")
        for idx, ctx in enumerate(contexts[1:3], start=2):
            text = (ctx.get("text") or "").strip()
            lines.append(f"- {text[:120]}… [{idx}]")
    return "\n".join(lines)


def generate(question: str, contexts: list[dict], history: list[dict] | None = None) -> dict:
    """生成回答。返回 {text, mode, error}。"""
    if not contexts:
        return {"text": "", "mode": "none", "error": None}

    if not CONFIG.model.llm_ready:
        return {"text": _extractive(contexts), "mode": "extractive", "error": None}

    messages = prompts.build_messages(question, contexts, history)
    try:
        text = chat(messages)
        return {"text": text, "mode": "llm", "error": None}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:200]
        return {
            "text": _extractive(contexts),
            "mode": "extractive",
            "error": f"模型接口返回 {exc.code}：{detail}",
        }
    except Exception as exc:  # noqa: BLE001
        return {"text": _extractive(contexts), "mode": "extractive", "error": f"模型调用失败：{exc}"}


def health() -> dict:
    return {
        "llm_ready": CONFIG.model.llm_ready,
        "vl_ready": CONFIG.model.vl_ready,
        "llm_model": CONFIG.model.llm_model or "（未配置）",
        "vl_model": CONFIG.model.vl_model or "（未配置）",
        "mode": "llm" if CONFIG.model.llm_ready else "extractive",
    }
