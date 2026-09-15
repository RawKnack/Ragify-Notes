from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.documents import Document
from config.settings import RERANK_MODEL_NAME

hf_cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL_NAME)
cross_encoder_reranker = CrossEncoderReranker(model=hf_cross_encoder, top_n=5)


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []

    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk.get("chunk_id", ""),
                "original_score": chunk.get("score", 0),
                "source": chunk.get("source", "unknown"),
                "dense_rank": chunk.get("dense_rank"),
                "bm25_rank": chunk.get("bm25_rank"),
                **chunk.get("metadata", {}),
            }
        )
        docs.append(doc)

    cross_encoder_reranker.top_n = top_k
    reranked_docs = cross_encoder_reranker.compress_documents(docs, query)

    results = []
    for doc in reranked_docs:
        meta = doc.metadata.copy()

        chunk_id = meta.pop("chunk_id", "")
        original_score = meta.pop("original_score", 0)
        source = meta.pop("source", "unknown")
        dense_rank = meta.pop("dense_rank", None)
        bm25_rank = meta.pop("bm25_rank", None)
        relevance_score = meta.pop("relevance_score", None)

        results.append({
            "chunk_id": chunk_id,
            "text": doc.page_content,
            "score": original_score,
            "rerank_score": round(float(relevance_score), 4) if relevance_score is not None else 0.0,
            "metadata": meta,
            "source": source,
            "dense_rank": dense_rank,
            "bm25_rank": bm25_rank,
        })

    return results
