"""M7 重排模块。

粗排（top-20）召回的是「可能相关」，重排负责挑出「真的相关」。
这一步是提升准确率性价比最高的一步，但也是最常被跳过的一步。

两种实现：
  - `bge`  ：BGE-Reranker-v2（cross-encoder），效果最好，需下载模型；
  - 启发式 ：查询词覆盖率 + 章节标题命中加成（零依赖，默认）。

通过环境变量 `USE_RERANKER=1` 启用 BGE 重排。
"""

from __future__ import annotations

import os

from ..config import CONFIG
from ..text_utils import tokenize

_RERANKER = None
_RERANKER_TRIED = False


def _get_reranker():
    global _RERANKER, _RERANKER_TRIED
    if not _RERANKER_TRIED:
        _RERANKER_TRIED = True
        if os.environ.get("USE_RERANKER", "0") == "1":
            try:
                from sentence_transformers import CrossEncoder

                _RERANKER = CrossEncoder(CONFIG.model.rerank_model)
            except Exception as exc:  # noqa: BLE001
                print(f"[reranker] BGE 重排加载失败，改用启发式：{exc}")
                _RERANKER = None
    return _RERANKER


def _coverage(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def _heuristic(question: str, hits: list) -> None:
    q_tokens = set(tokenize(question))
    for hit in hits:
        chunk_tokens = set(tokenize(hit.chunk.get("text", "")))
        cov = _coverage(q_tokens, chunk_tokens)

        section = hit.chunk.get("section", "") or ""
        title_hit = 1.0 if (q_tokens & set(tokenize(section))) else 0.0

        # 覆盖率为主，标题命中加成，再混入少量粗排分数兜底
        hit.rerank = round(0.6 * cov + 0.2 * title_hit + 0.2 * hit.score, 4)


def _bge(question: str, hits: list) -> None:
    model = _get_reranker()
    if model is None:
        _heuristic(question, hits)
        return
    pairs = [(question, hit.chunk.get("text", "")) for hit in hits]
    scores = model.predict(pairs)
    lo, hi = min(scores), max(scores)
    for hit, raw in zip(hits, scores):
        hit.rerank = round(float((raw - lo) / (hi - lo)) if hi > lo else 0.0, 4)


def rerank(question: str, hits: list, top_n: int | None = None) -> list:
    if not hits:
        return []
    top_n = top_n or CONFIG.retrieval.top_n
    _bge(question, hits)
    ordered = sorted(hits, key=lambda h: h.rerank, reverse=True)
    return ordered[:top_n]
