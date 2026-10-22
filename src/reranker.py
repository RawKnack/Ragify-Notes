import gc
import os
import tempfile
from config.settings import RERANK_MODEL_NAME

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

_ranker = None


def get_reranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        cache_dir = os.environ.get("FLASHRANK_CACHE_DIR", tempfile.gettempdir())
        _ranker = Ranker(model_name=RERANK_MODEL_NAME, cache_dir=cache_dir)
    return _ranker


@traceable(name="FlashRank Reranker", run_type="retriever")
def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []

    try:
        ranker_instance = get_reranker()
    except Exception as e:
        print(f"Reranker unavailable: {e}, using hybrid results")
        return chunks[:top_k]

    passages = []
    for i, chunk in enumerate(chunks):
        passages.append({
            "id": chunk.get("chunk_id", str(i)),
            "text": chunk.get("text", ""),
            "original_score": chunk.get("score", 0),
            "source": chunk.get("source", "unknown"),
            "dense_rank": chunk.get("dense_rank"),
            "bm25_rank": chunk.get("bm25_rank"),
            "metadata": chunk.get("metadata", {}),
        })

    try:
        from flashrank import RerankRequest
        req = RerankRequest(query=query, passages=passages)
        reranked = ranker_instance.rerank(req)
    except Exception as e:
        print(f"Reranking error: {e}, using hybrid results")
        return chunks[:top_k]
    finally:
        gc.collect()

    results = []
    for item in reranked[:top_k]:
        relevance_score = item.get("score")
        results.append({
            "chunk_id": item.get("id", ""),
            "text": item.get("text", ""),
            "score": item.get("original_score", 0),
            "rerank_score": round(float(relevance_score), 4) if relevance_score is not None else 0.0,
            "metadata": item.get("metadata", {}),
            "source": item.get("source", "unknown"),
            "dense_rank": item.get("dense_rank"),
            "bm25_rank": item.get("bm25_rank"),
        })

    return results
