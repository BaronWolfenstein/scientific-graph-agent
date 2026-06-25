"""
Hybrid reranking: dense (SPECTER) + cross-encoder (BGE) + relative drop-off filter.

Models are lazy-loaded and cached on first use. Expect ~30s on first call while
sentence-transformers downloads the weights (~430MB SPECTER, ~100MB BGE reranker).
"""
import logging
import math
import warnings
from functools import lru_cache

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"
DENSE_MODEL = "allenai-specter"
CANDIDATE_FETCH = 30      # how many BM25 candidates to pull from ArXiv before reranking
MAX_K = 10                # hard ceiling on papers returned after drop-off
MIN_K = 2                 # minimum papers to keep regardless of score gap
DROP_RATIO = 0.50         # relative score drop that triggers cutoff
MIN_SCORE = 0.20          # absolute combined-score floor (0-1); papers below this are always dropped


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder
    logger.info(f"Loading cross-encoder {CROSS_ENCODER_MODEL} (one-time download)...")
    return CrossEncoder(CROSS_ENCODER_MODEL)


@lru_cache(maxsize=1)
def _get_dense_encoder():
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading dense encoder {DENSE_MODEL} (one-time download)...")
    return SentenceTransformer(DENSE_MODEL)


def _normalize(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    span = mx - mn
    if span == 0:
        return [1.0] * len(values)
    return [(v - mn) / span for v in values]


def score_papers(query: str, papers: list[dict]) -> list[dict]:
    """
    Add dense_score and ce_score (both 0-1) to each paper dict in-place.
    Does not reorder — call apply_dropoff after this.
    """
    if not papers:
        return papers

    texts = [f"{p.get('title', '')} {p.get('summary', '')[:400]}" for p in papers]

    # Dense cosine similarity via SPECTER
    dense_enc = _get_dense_encoder()
    q_emb = dense_enc.encode([query], normalize_embeddings=True)[0]
    p_embs = dense_enc.encode(texts, normalize_embeddings=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dense_raw = (p_embs @ q_emb).tolist()
    dense_raw = [0.0 if (math.isnan(x) or math.isinf(x)) else x for x in dense_raw]
    dense_scores = _normalize(dense_raw)

    # Cross-encoder scores (logit scale → normalize)
    ce = _get_cross_encoder()
    pairs = [[query, t] for t in texts]
    ce_raw = ce.predict(pairs).tolist()
    ce_scores = _normalize(ce_raw)

    for paper, ds, cs in zip(papers, dense_scores, ce_scores):
        paper["dense_score"] = round(float(ds), 4)
        paper["ce_score"] = round(float(cs), 4)

    return papers


def _pre_llm_combined(paper: dict) -> float:
    """Combined score using dense + cross-encoder only (before LLM scoring)."""
    return 0.55 * paper.get("ce_score", 0.5) + 0.45 * paper.get("dense_score", 0.5)


def apply_dropoff(
    papers: list[dict],
    max_k: int = MAX_K,
    min_k: int = MIN_K,
    drop_ratio: float = DROP_RATIO,
    min_score: float = MIN_SCORE,
) -> list[dict]:
    """
    Sort by pre-LLM combined score, apply an absolute score floor, then cut where
    relative drop exceeds drop_ratio. Hard ceiling at max_k; min_k bypasses the
    ratio check so at least that many above-floor papers are kept.
    """
    if not papers:
        return papers

    ranked = sorted(papers, key=_pre_llm_combined, reverse=True)

    # Absolute floor: drop papers below min_score; always keep at least the top paper
    above_floor = [p for p in ranked if _pre_llm_combined(p) >= min_score] or [ranked[0]]

    kept = [above_floor[0]]
    for i in range(1, min(len(above_floor), max_k)):
        prev = _pre_llm_combined(above_floor[i - 1])
        curr = _pre_llm_combined(above_floor[i])
        if i >= min_k and prev > 0 and (prev - curr) / prev > drop_ratio:
            break
        kept.append(above_floor[i])

    return kept


def final_relevance_score(paper: dict) -> int:
    """
    Combine cross-encoder, dense, and LLM scores into a single 1-100 relevance_score.
    Weights: CE 45%, dense 35%, LLM 20%.
    """
    ce = paper.get("ce_score", 0.5)
    dense = paper.get("dense_score", 0.5)
    llm_norm = (paper.get("relevance_score", 50) - 1) / 99
    combined = 0.45 * ce + 0.35 * dense + 0.20 * llm_norm
    return max(1, min(100, round(combined * 100)))
