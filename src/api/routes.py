"""API 路由层。

对应模块：
  M15 问答界面 / M16 知识库后台 / M18 图片上传 —— 前端调用的接口

接口一览：
  GET  /api/health     服务与模型配置状态
  GET  /api/channels   检索渠道列表（含可用状态，供渠道选择器渲染）
  GET  /api/stats      索引概况
  POST /api/ask        提问（支持图片、渠道选择、多轮历史）
  POST /api/ingest     重建索引
  GET  /api/videos     视频资源概况
"""

from __future__ import annotations

import base64
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..generation.llm_client import health
from ..ingestion.indexer import build_index, stats
from ..ingestion.video_registry import list_videos
from ..ingestion.video_registry import summary as video_summary
from ..pipeline import answer
from ..retrieval.channel_router import channel_status

router = APIRouter(prefix="/api", tags=["labqa"])


class AskRequest(BaseModel):
    question: str = ""
    channels: list[str] = Field(default_factory=list)
    image: str | None = None          # data URL，形如 data:image/png;base64,xxx
    history: list[dict] = Field(default_factory=list)


class IngestRequest(BaseModel):
    rebuild: bool = False


def _save_data_url(data_url: str) -> Path:
    header, _, payload = data_url.partition(",")
    suffix = ".png" if "png" in header else ".jpg"
    raw = base64.b64decode(payload)
    tmp = Path(tempfile.gettempdir()) / f"labqa_{int(time.time() * 1000)}{suffix}"
    tmp.write_bytes(raw)
    return tmp


@router.get("/health")
def api_health() -> dict:
    return {"ok": True, "model": health(), "index": stats()}


@router.get("/channels")
def api_channels() -> dict:
    return {"channels": channel_status()}


@router.get("/stats")
def api_stats() -> dict:
    return stats()


@router.get("/videos")
def api_videos() -> dict:
    return {"summary": video_summary(), "videos": list_videos()}


@router.post("/ingest")
def api_ingest(payload: IngestRequest) -> dict:
    result = build_index(rebuild=payload.rebuild)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.post("/ask")
def api_ask(payload: AskRequest) -> dict:
    question = (payload.question or "").strip()
    if not question and not payload.image:
        raise HTTPException(status_code=400, detail="请提供问题文本或上传图片")

    image_path = None
    if payload.image:
        try:
            image_path = _save_data_url(payload.image)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"图片解析失败：{exc}") from exc

    result = answer(
        question,
        channels=payload.channels or None,
        history=payload.history or None,
        image=image_path,
    )

    # 图片识别未启用时，明确告诉前端，避免用户以为是自己传错了
    vision_info = result.get("vision")
    if vision_info and not vision_info.get("ok"):
        result.setdefault("notices", []).append(vision_info.get("message", ""))

    return result
