"""集中配置（对应模块 M21 配置管理）。

比赛期间最常调的就是这里的参数：chunk 大小、topK、拒答阈值、渠道定义。
原则：所有可调项集中在此，不要在代码各处写死。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNK_PATH = DATA_DIR / "chunks" / "chunks.jsonl"
INDEX_DIR = DATA_DIR / "index"
STORE_PATH = INDEX_DIR / "store.json"
VIDEO_DIR = DATA_DIR / "videos"
VIDEO_REGISTRY = VIDEO_DIR / "registry.json"
WEB_DIR = ROOT / "web"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


@dataclass
class ChunkConfig:
    """M3 语义切分参数。中文安全手册建议 400~800 字一块。"""

    max_chars: int = 600
    overlap_chars: int = 80


@dataclass
class RetrievalConfig:
    """M6/M7/M8 参数。"""

    top_k: int = 20           # 粗排召回条数
    top_n: int = 5            # 重排后保留条数
    bm25_weight: float = 0.6  # 混合检索中 BM25 权重
    vector_weight: float = 0.4
    bm25_saturation: float = 3.0
    # ↑ BM25 绝对分数饱和系数：score = raw / (raw + k)

    min_score: float = 0.30   # M8：低于此分数触发拒答
    min_coverage: float = 0.20
    # ↑ M8：查询实词覆盖率低于此值直接拒答（防止「问不相关的事也硬答」）


@dataclass
class ChannelConfig:
    """M26 检索渠道定义。

    online_enabled 默认 False —— 安全领域外部内容未经审核，不应作为依据。
    """

    available: list[str] = field(
        default_factory=lambda: ["本校文件", "国家规范", "事故案例", "联网搜索"]
    )
    default_enabled: list[str] = field(
        default_factory=lambda: ["本校文件", "国家规范", "事故案例"]
    )
    online_enabled: bool = False

    def resolve(self, selected: list[str] | None) -> list[str]:
        """把用户选择解析为实际生效的渠道列表。"""
        if not selected or "全部来源" in selected:
            chosen = list(self.default_enabled)
        else:
            chosen = [c for c in selected if c in self.available]
        if not self.online_enabled:
            chosen = [c for c in chosen if c != "联网搜索"]
        return chosen or list(self.default_enabled)


@dataclass
class ModelConfig:
    """M10 / M13 模型接入配置。"""

    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    vl_api_base: str = ""
    vl_api_key: str = ""
    vl_model: str = ""
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_api_base and self.llm_api_key and self.llm_model)

    @property
    def vl_ready(self) -> bool:
        base = self.vl_api_base or self.llm_api_base
        key = self.vl_api_key or self.llm_api_key
        return bool(base and key and self.vl_model)


@dataclass
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config() -> Config:
    cfg = Config()
    cfg.chunk.max_chars = _env_int("CHUNK_MAX_CHARS", cfg.chunk.max_chars)
    cfg.chunk.overlap_chars = _env_int("CHUNK_OVERLAP_CHARS", cfg.chunk.overlap_chars)

    cfg.retrieval.top_k = _env_int("RETRIEVAL_TOP_K", cfg.retrieval.top_k)
    cfg.retrieval.top_n = _env_int("RETRIEVAL_TOP_N", cfg.retrieval.top_n)
    cfg.retrieval.min_score = _env_float("RETRIEVAL_MIN_SCORE", cfg.retrieval.min_score)
    cfg.retrieval.min_coverage = _env_float("RETRIEVAL_MIN_COVERAGE", cfg.retrieval.min_coverage)
    cfg.retrieval.bm25_saturation = _env_float("RETRIEVAL_BM25_SATURATION", cfg.retrieval.bm25_saturation)

    cfg.channels.online_enabled = _env("ONLINE_SEARCH_ENABLED") == "1"

    cfg.model.llm_api_base = _env("LLM_API_BASE")
    cfg.model.llm_api_key = _env("LLM_API_KEY")
    cfg.model.llm_model = _env("LLM_MODEL")
    cfg.model.vl_api_base = _env("VL_API_BASE")
    cfg.model.vl_api_key = _env("VL_API_KEY")
    cfg.model.vl_model = _env("VL_MODEL")
    return cfg


CONFIG = load_config()
