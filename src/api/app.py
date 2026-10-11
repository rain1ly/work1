"""FastAPI 应用入口。

启动：
    uvicorn src.api.app:app --reload
    # 或
    python -m src.api.app

然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..config import WEB_DIR
from ..ingestion.indexer import ensure_index
from ..generation.llm_client import health
from ..ingestion.video_registry import summary as video_summary
from .routes import router

app = FastAPI(
    title="高校实验室安全规程问答",
    description="第九届传智杯 · AI 创新应用挑战赛参赛作品",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def _startup() -> None:
    """启动时确保索引存在，避免第一次提问才报错。"""
    ensure_index()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        return FileResponse(__file__)  # 理论上不会走到这里
    return FileResponse(page)


@app.get("/status", include_in_schema=False)
def status() -> dict:
    return {"model": health(), "videos": video_summary()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=False)
