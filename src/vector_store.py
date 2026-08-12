import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.embedder import embeddings
from config.settings import FAISS_INDEX_PATH, EMBED_DIMENSION


def create_and_store(chunks: list[dict], recreate: bool = False) -> FAISS:
    if os.path.exists(FAISS_INDEX_PATH) and not recreate:
        return load_store()

    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk.get("chunk_id", ""),
                **chunk.get("metadata", {}),
            }
        )
        docs.append(doc)

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    return db


def load_store() -> FAISS:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"No FAISS index found at '{FAISS_INDEX_PATH}'.")

    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search(
    query_vector: list[float],
    top_k: int = 5,
    filter_confidence: str = None,
    filter_section: str = None,
) -> list[dict]:
    db = load_store()
    raw_results = db.similarity_search_with_score_by_vector(
        query_vector, k=top_k * 3 if (filter_confidence or filter_section) else top_k
    )

    results = []
    for doc, score in raw_results:
        metadata = doc.metadata.copy()
        chunk_id = metadata.pop("chunk_id", "")

        if filter_confidence and metadata.get("confidence") != filter_confidence:
            continue
        if filter_section and metadata.get("section") != filter_section:
            continue

        similarity = max(0.0, 1.0 - float(score) / 10.0)

        results.append({
            "chunk_id": chunk_id,
            "text": doc.page_content,
            "score": round(similarity, 4),
            "raw_distance": round(float(score), 4),
            "metadata": metadata,
        })

        if len(results) >= top_k:
            break

    return results


def search_by_text(
    query_text: str,
    top_k: int = 5,
    filter_confidence: str = None,
    filter_section: str = None,
) -> list[dict]:
    db = load_store()
    raw_results = db.similarity_search_with_score(
        query_text, k=top_k * 3 if (filter_confidence or filter_section) else top_k
    )

    results = []
    for doc, score in raw_results:
        metadata = doc.metadata.copy()
        chunk_id = metadata.pop("chunk_id", "")

        if filter_confidence and metadata.get("confidence") != filter_confidence:
            continue
        if filter_section and metadata.get("section") != filter_section:
            continue

        similarity = max(0.0, 1.0 - float(score) / 10.0)

        results.append({
            "chunk_id": chunk_id,
            "text": doc.page_content,
            "score": round(similarity, 4),
            "raw_distance": round(float(score), 4),
            "metadata": metadata,
        })

        if len(results) >= top_k:
            break

    return results


def get_all_chunks() -> list[dict]:
    try:
        db = load_store()
    except FileNotFoundError:
        return []

    all_chunks = []
    for doc_id in db.docstore._dict:
        doc = db.docstore._dict[doc_id]
        metadata = doc.metadata.copy()
        chunk_id = metadata.pop("chunk_id", "")

        all_chunks.append({
            "chunk_id": chunk_id,
            "text": doc.page_content,
            "metadata": metadata,
        })

    return all_chunks


def get_collection_info() -> dict:
    try:
        db = load_store()
        total = db.index.ntotal
        dimension = db.index.d
    except FileNotFoundError:
        total = 0
        dimension = EMBED_DIMENSION

    return {
        "name": "faiss_index",
        "total_documents": total,
        "total_points": total,
        "dimension": dimension,
        "vector_size": dimension,
        "distance": "cosine",
        "index_path": FAISS_INDEX_PATH,
    }