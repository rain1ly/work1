"""M4 向量索引模块 + 建库命令行入口。

存储采用本地 JSON（`data/index/store.json`），零依赖、便于比赛期快速重建。
生产环境可平滑换成 Qdrant：数据结构已按「向量 + payload 元数据」组织，迁移只改这一层。

用法：
    python -m src.ingestion.indexer             # 建库
    python -m src.ingestion.indexer --stats     # 查看索引概况
    python -m src.ingestion.indexer --rebuild   # 强制全量重建
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ..config import CHUNK_PATH, CONFIG, STORE_PATH
from . import chunker as chunker_mod
from . import embedder as embedder_mod
from . import loader


def build_index(rebuild: bool = False, raw_dir: Path | None = None) -> dict:
    docs, warnings = loader.load_all(raw_dir)
    if not docs:
        return {
            "ok": False,
            "message": "data/raw 下没有可用文档，请先放入资料",
            "warnings": warnings,
        }

    chunks = chunker_mod.chunk_all(
        docs, max_chars=CONFIG.chunk.max_chars, overlap=CONFIG.chunk.overlap_chars
    )
    embedder = embedder_mod.get_embedder(CONFIG.model.embed_model)
    vectors = embedder.encode([c.text for c in chunks])

    CHUNK_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CHUNK_PATH.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    channel_stats: dict[str, int] = {}
    for chunk in chunks:
        channel_stats[chunk.channel] = channel_stats.get(chunk.channel, 0) + 1

    store = {
        "version": 1,
        "backend": embedder.backend_tag,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "docs": {
            d.doc_id: {
                "digest": d.digest,
                "title": d.title,
                "channel": d.channel,
                "filename": d.filename,
            }
            for d in docs
        },
        "channel_stats": channel_stats,
        "chunks": [{**c.to_dict(), "vector": v} for c, v in zip(chunks, vectors)],
    }
    STORE_PATH.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")

    return {
        "ok": True,
        "docs": len(docs),
        "chunks": len(chunks),
        "channels": channel_stats,
        "backend": embedder.backend_tag,
        "warnings": warnings,
    }


def load_index() -> dict:
    if not STORE_PATH.exists():
        return {"version": 0, "chunks": [], "docs": {}, "backend": ""}
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))


def get_chunks() -> list[dict]:
    return load_index().get("chunks", [])


def ensure_index() -> dict:
    """索引不存在时自动建一次，方便第一次运行。"""
    store = load_index()
    if store.get("chunks"):
        return store
    build_index()
    return load_index()


def stats() -> dict:
    store = load_index()
    return {
        "backend": store.get("backend", ""),
        "built_at": store.get("built_at", ""),
        "docs": len(store.get("docs", {})),
        "chunks": len(store.get("chunks", [])),
        "channels": store.get("channel_stats", {}),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--stats" in argv:
        info = stats()
        if not info["chunks"]:
            print("索引为空，请先运行：python -m src.ingestion.indexer")
            return 1
        print("索引概况")
        print(f"  向量后端   : {info['backend']}")
        print(f"  构建时间   : {info['built_at']}")
        print(f"  文档数     : {info['docs']}")
        print(f"  切块数     : {info['chunks']}")
        print("  渠道分布   :")
        for channel, count in info["channels"].items():
            print(f"    - {channel}: {count} 块")
        return 0

    result = build_index(rebuild="--rebuild" in argv)
    if not result["ok"]:
        print(f"建库失败：{result['message']}")
        for w in result["warnings"]:
            print(f"  ! {w}")
        return 1

    print("建库完成")
    print(f"  文档 {result['docs']} 篇 → 切块 {result['chunks']} 个")
    print(f"  向量后端：{result['backend']}")
    print("  渠道分布：")
    for channel, count in result["channels"].items():
        print(f"    - {channel}: {count} 块")
    if result["warnings"]:
        print("  警告：")
        for w in result["warnings"]:
            print(f"    ! {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
