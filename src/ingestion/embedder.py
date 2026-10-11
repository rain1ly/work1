"""M4 向量化模块。

对外只暴露 `get_embedder()`，调用方不需要关心底层实现。

两种后端：
  - `bge`  ：BGE-M3（sentence-transformers），语义检索效果最好，需下载模型（GB 级）
  - `hash` ：哈希 bigram 向量（纯标准库），零依赖、秒级启动，用于快速跑通链路

通过环境变量 `EMBED_BACKEND` 切换，默认 `hash`。
建库与检索必须使用同一后端，否则向量不在同一空间、结果无意义 ——
索引文件里存了 backend 名称，加载时会做一致性校验。
"""

from __future__ import annotations

import math
import os

from ..text_utils import stable_hash, tokenize

DEFAULT_DIM = 512


class HashingEmbedder:
    """零依赖降级实现：把 bigram 哈希到固定维度并归一化。

    效果介于「关键词匹配」与「真正的语义向量」之间，
    对中文专有名词（化学品名、设备名）表现尚可，适合开发期跑通链路。
    """

    name = "hash"

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    @property
    def backend_tag(self) -> str:
        return f"hash-{self.dim}"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            idx = stable_hash(token) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class BGEEmbedder:
    """BGE-M3 后端。首次调用会下载模型。"""

    name = "bge"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.model_name = model_name

    @property
    def backend_tag(self) -> str:
        return f"bge-{self.dim}"

    def encode(self, texts: list[str]) -> list[list[float]]:
        arr = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, row)) for row in arr]


_EMBEDDER = None


def get_embedder(model_name: str = "BAAI/bge-m3"):
    """按环境变量返回向量化器；默认走零依赖的 hash 后端。"""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    backend = os.environ.get("EMBED_BACKEND", "hash").strip().lower()
    if backend == "bge":
        try:
            _EMBEDDER = BGEEmbedder(model_name)
            return _EMBEDDER
        except Exception as exc:  # noqa: BLE001
            print(f"[embedder] BGE-M3 加载失败，降级为 hash 后端：{exc}")
    _EMBEDDER = HashingEmbedder()
    return _EMBEDDER


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
