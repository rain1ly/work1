"""M6 混合检索模块。

BM25（关键词） + 向量（语义）双路召回，加权融合。
只做纯向量检索是新手最容易犯的错 —— 化学品名、标准号这类专有名词，
关键词匹配往往比语义相似更准；反过来，口语化提问又只有向量能救。
两路都要，缺一不可。

渠道过滤（M26）在这一层生效：先按 channel 元数据缩小候选集，再做排序。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import CONFIG
from ..ingestion import embedder as embedder_mod
from ..ingestion.indexer import ensure_index
from ..text_utils import content_bigrams, tokenize

_CORPUS_CACHE: dict[str, object] = {}


@dataclass
class Hit:
    chunk: dict
    bm25: float = 0.0
    vector: float = 0.0
    score: float = 0.0
    rerank: float = 0.0
    coverage: float = 0.0
    # ↑ 查询实词在该片段中的覆盖率，M8 用它做「超范围提问」的硬门槛

    @property
    def channel(self) -> str:
        return self.chunk.get("channel", "")

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.get("chunk_id"),
            "doc_id": self.chunk.get("doc_id"),
            "title": self.chunk.get("title"),
            "channel": self.chunk.get("channel"),
            "section": self.chunk.get("section"),
            "text": self.chunk.get("text"),
            "bm25": round(self.bm25, 4),
            "vector": round(self.vector, 4),
            "coverage": round(self.coverage, 4),
            "score": round(self.score, 4),
        }


def _bm25_scores(
    query_tokens: list[str], doc_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    n_docs = len(doc_tokens)
    if n_docs == 0 or not query_tokens:
        return [0.0] * n_docs

    lengths = [len(t) for t in doc_tokens]
    avgdl = sum(lengths) / n_docs or 1.0

    df: dict[str, int] = {}
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    scores = [0.0] * n_docs
    for term in set(query_tokens):
        if term not in df:
            continue
        idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
        for i, tokens in enumerate(doc_tokens):
            freq = tokens.count(term)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * lengths[i] / avgdl)
            scores[i] += idf * freq * (k1 + 1) / denom
    return scores


def _load_corpus() -> tuple[dict, list[dict], list[list[str]]]:
    """加载索引与分词缓存；索引重建后自动失效。"""
    store = ensure_index()
    key = f"{store.get('built_at', '')}-{store.get('backend', '')}"
    cached = _CORPUS_CACHE.get(key)
    if cached is not None:
        return store, store["chunks"], cached  # type: ignore[return-value]
    doc_tokens = [tokenize(c.get("text", "")) for c in store.get("chunks", [])]
    _CORPUS_CACHE.clear()
    _CORPUS_CACHE[key] = doc_tokens
    return store, store.get("chunks", []), doc_tokens


def search(
    question: str,
    channels: list[str] | None = None,
    top_k: int | None = None,
    coverage_text: str | None = None,
) -> list[Hit]:
    """渠道过滤 → BM25 + 向量双路召回 → 加权融合。

    `coverage_text` 用于计算覆盖率，应当传**未经同义扩展**的原始问句，
    否则扩展进来的别名（如「气罐」「钢瓶」）会拉低覆盖率，把正常问题误拒。
    """
    store, docs, doc_tokens = _load_corpus()
    if not docs:
        return []

    top_k = top_k or CONFIG.retrieval.top_k
    channels = channels or CONFIG.channels.default_enabled

    # 1) 渠道过滤：这一步必须在排序前做，否则召回本身就是错的
    allowed = [i for i, d in enumerate(docs) if d.get("channel") in channels]
    if not allowed:
        return []

    filtered_docs = [docs[i] for i in allowed]
    filtered_tokens = [doc_tokens[i] for i in allowed]

    # 2) BM25 —— 关键：绝对分数做「饱和变换」，不要按本次结果集归一化。
    #    归一化会让垃圾结果也拿到满分（第一名永远是 1.0），拒答判据直接失效。
    query_tokens = tokenize(question)
    bm25_raw = _bm25_scores(query_tokens, filtered_tokens)
    k = max(CONFIG.retrieval.bm25_saturation, 1e-6)
    bm25_squash = [s / (s + k) if s > 0 else 0.0 for s in bm25_raw]

    # 3) 向量
    embedder = embedder_mod.get_embedder(CONFIG.model.embed_model)
    query_vec = embedder.encode([question])[0]
    vectors = [embedder_mod.cosine(query_vec, d.get("vector") or []) for d in filtered_docs]

    # 4) 覆盖率：用户问题里的实词有多少真的在这个片段里出现过。
    #    统一用 content_bigrams（不依赖 jieba），保证换分词后端时阈值仍然稳定。
    q_set = set(content_bigrams(coverage_text or question))
    coverages = [
        (len(q_set & set(content_bigrams(d.get("text", "")))) / len(q_set)) if q_set else 0.0
        for d in filtered_docs
    ]

    # 5) 加权融合
    w_bm25 = CONFIG.retrieval.bm25_weight
    w_vec = CONFIG.retrieval.vector_weight

    hits: list[Hit] = []
    for idx, doc in enumerate(filtered_docs):
        score = w_bm25 * bm25_squash[idx] + w_vec * vectors[idx]
        hits.append(
            Hit(
                chunk=doc,
                bm25=bm25_squash[idx],
                vector=vectors[idx],
                score=score,
                coverage=coverages[idx],
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
