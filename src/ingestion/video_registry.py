"""M23 视频资源管理模块。

铁律（写在代码里，也写在文档里）：
  1. 本模块只管理**经过人工审核**的视频素材，禁止任何 AI 生成视频进入登记表；
  2. 每条视频必须登记来源机构、审核人、审核日期、适用场景；
  3. 审核信息不全的条目一律不得出库（`is_publishable()` 会拦住）。

注意：当前 registry.json 中的条目均为 `placeholder=true` 的示例，
仅用于跑通流程。正式提交前必须替换为真实已审核素材，或删除。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import VIDEO_REGISTRY


def load_registry(path: Path | None = None) -> dict:
    target = Path(path or VIDEO_REGISTRY)
    if not target.exists():
        return {"version": "1.0", "videos": []}
    return json.loads(target.read_text(encoding="utf-8"))


def save_registry(data: dict, path: Path | None = None) -> None:
    target = Path(path or VIDEO_REGISTRY)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def is_publishable(video: dict) -> bool:
    """是否允许出现在答案中。三道闸门缺一不可。"""
    return bool(
        video.get("approved") is True
        and str(video.get("source", "")).strip()
        and str(video.get("reviewer", "")).strip()
        and str(video.get("review_date", "")).strip()
    )


def list_videos(approved_only: bool = True, include_placeholder: bool = True) -> list[dict]:
    videos = load_registry().get("videos", [])
    if approved_only:
        videos = [v for v in videos if v.get("approved") is True]
    if not include_placeholder:
        videos = [v for v in videos if not v.get("placeholder")]
    return videos


def get(video_id: str) -> dict | None:
    for video in load_registry().get("videos", []):
        if video.get("video_id") == video_id:
            return video
    return None


def register(video: dict) -> None:
    """新增或更新一条视频登记。"""
    data = load_registry()
    videos = data.setdefault("videos", [])
    for idx, existing in enumerate(videos):
        if existing.get("video_id") == video["video_id"]:
            videos[idx] = {**existing, **video}
            break
    else:
        videos.append(video)
    save_registry(data)


def approve(video_id: str, reviewer: str, review_date: str) -> bool:
    """审核通过：补齐审核信息后置 approved=True。"""
    video = get(video_id)
    if not video:
        return False
    video.update({"reviewer": reviewer, "review_date": review_date, "approved": True, "placeholder": False})
    register(video)
    return True


def summary() -> dict:
    videos = load_registry().get("videos", [])
    return {
        "total": len(videos),
        "approved": sum(1 for v in videos if v.get("approved")),
        "publishable": sum(1 for v in videos if is_publishable(v)),
        "placeholder": sum(1 for v in videos if v.get("placeholder")),
    }


if __name__ == "__main__":
    info = summary()
    print("视频资源概况")
    print(f"  条目总数   : {info['total']}")
    print(f"  已审核     : {info['approved']}")
    print(f"  可出库     : {info['publishable']}  （需来源/审核人/审核日期齐全）")
    print(f"  示例占位   : {info['placeholder']}  ← 正式提交前必须替换或删除")
