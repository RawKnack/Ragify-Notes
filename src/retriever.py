"""
src/retriever.py — Hybrid RAG query engine.

WHAT RAG DOES:
  Retrieval Augmented Generation — instead of asking an LLM a question
  from its training data (which may be outdated or wrong), you:

  1. RETRIEVE relevant chunks from YOUR notes (the vector database)
  2. AUGMENT the LLM prompt with those chunks as context
  3. GENERATE an answer grounded in your actual notes

  The LLM can only answer using what you retrieved — it can't make up
  content from training data because you explicitly tell it to use only
  the provided context.

HYBRID RETRIEVAL:
  Instead of relying on dense (vector) search alone, we combine:
  - DENSE search: semantic similarity via embeddings (catches meaning)
  - SPARSE search: BM25 keyword matching (catches exact terms, equations)

  Results are fused using Reciprocal Rank Fusion (RRF), which merges
  ranked lists by giving higher scores to items ranked highly in
  either or both retrieval methods.

FLOW:
  User query: "what is the ROC for a causal sequence?"
       ↓
  Embed query → vector
       ↓
  Dense search (Qdrant) → top chunks by semantic similarity
  BM25 search (in-memory) → top chunks by keyword relevance
       ↓
  Reciprocal Rank Fusion → merged top-K
       ↓
  Build prompt: [your chunks] + "Answer this: what is ROC..."
       ↓
  LLM generates answer grounded in your notes
       ↓
  Return answer + source chunks (for citation)
"""

import re
from openai import OpenAI
from rank_bm25 import BM25Okapi
from src.embedder import embed_text
from src.vector_store import search, get_all_chunks
from config.settings import OPENROUTER_API_KEY


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Model for answer generation — can be different from extraction model
# GPT-4o-mini is fine, Claude Haiku gives better structured answers
ANSWER_MODEL = "openai/gpt-4o-mini"

RAG_SYSTEM_PROMPT = """
You are a helpful study assistant that answers questions about engineering and mathematics.

You are given excerpts from a student's handwritten notes below as CONTEXT.
Your job is to answer the QUESTION using ONLY the provided context.

RULES:
- Answer only from the context provided — do not use outside knowledge
- If the context doesn't contain enough information, say "The notes don't cover this clearly"
- Preserve all mathematical expressions exactly as they appear (LaTeX format)
- Keep answers concise and structured
- If multiple chunks are relevant, synthesize them into one clear answer
- At the end, cite which pages the answer came from
"""


# ── BM25 Index (built lazily on first query) ────────────────────
_bm25_index = None
_bm25_chunks = None


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove short tokens."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if len(t) > 1]


def _build_bm25_index():
    """
    Builds the BM25 index from all chunks stored in Qdrant.
    Called once on first query, then cached in memory.
    """
    global _bm25_index, _bm25_chunks

    print("   📚 Building BM25 index from stored chunks...")
    _bm25_chunks = get_all_chunks()

    if not _bm25_chunks:
        print("   ⚠️  No chunks found in Qdrant — BM25 index is empty")
        _bm25_index = None
        return

    corpus = [_tokenize(chunk["text"]) for chunk in _bm25_chunks]
    _bm25_index = BM25Okapi(corpus)
    print(f"   ✅ BM25 index built with {len(_bm25_chunks)} chunks")


def refresh_bm25_index():
    """Force rebuild the BM25 index (call after re-ingestion)."""
    global _bm25_index, _bm25_chunks
    _bm25_index = None
    _bm25_chunks = None
    _build_bm25_index()


def _bm25_search(query_text: str, top_k: int = 10) -> list[dict]:
    """
    Sparse keyword search using BM25.
    Returns chunks ranked by BM25 score with normalized scores.
    """
    global _bm25_index, _bm25_chunks

    if _bm25_index is None:
        _build_bm25_index()

    if _bm25_index is None or not _bm25_chunks:
        return []

    query_tokens = _tokenize(query_text)
    scores = _bm25_index.get_scores(query_tokens)

    # Pair chunks with scores and sort descending
    scored = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:top_k]

    # Normalize scores to 0–1 range
    max_score = scored[0][1] if scored else 1
    results = []
    for idx, score in scored:
        chunk = _bm25_chunks[idx]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": round(score / max_score, 4),
            "metadata": chunk["metadata"],
            "source": "bm25",
        })

    return results


def hybrid_search(
    query_text: str,
    query_vector: list[float],
    top_k: int = 5,
    alpha: float = 0.6,
    filter_confidence: str = None,
    filter_section: str = None,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Hybrid search combining dense (vector) and sparse (BM25) retrieval
    using Reciprocal Rank Fusion (RRF).

    Args:
        query_text:        raw query string (for BM25)
        query_vector:      embedded query vector (for dense search)
        top_k:             final number of results to return
        alpha:             weight balance (0 = pure BM25, 1 = pure dense, 0.6 = recommended)
        filter_confidence: Qdrant payload filter on confidence level
        filter_section:    Qdrant payload filter on section name
        rrf_k:             RRF smoothing constant (default 60)

    Returns:
        Fused list of top_k chunks, each with combined RRF score.
    """
    # Fetch more candidates than needed for better fusion
    fetch_k = top_k * 3

    # 1. Dense vector search via Qdrant
    dense_results = search(
        query_vector=query_vector,
        top_k=fetch_k,
        filter_confidence=filter_confidence,
        filter_section=filter_section,
    )
    for r in dense_results:
        r["source"] = "dense"

    # 2. Sparse BM25 search
    sparse_results = _bm25_search(query_text, top_k=fetch_k)

    # 3. Reciprocal Rank Fusion
    rrf_scores = {}  # chunk_id → { rrf_score, chunk_data }

    for rank, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        rrf_score = alpha * (1.0 / (rrf_k + rank + 1))
        if cid not in rrf_scores:
            rrf_scores[cid] = {"chunk": chunk, "rrf_score": 0, "dense_rank": rank + 1, "bm25_rank": None}
        rrf_scores[cid]["rrf_score"] += rrf_score
        rrf_scores[cid]["dense_rank"] = rank + 1

    for rank, chunk in enumerate(sparse_results):
        cid = chunk["chunk_id"]
        rrf_score = (1 - alpha) * (1.0 / (rrf_k + rank + 1))
        if cid not in rrf_scores:
            rrf_scores[cid] = {"chunk": chunk, "rrf_score": 0, "dense_rank": None, "bm25_rank": rank + 1}
        rrf_scores[cid]["rrf_score"] += rrf_score
        if rrf_scores[cid]["bm25_rank"] is None:
            rrf_scores[cid]["bm25_rank"] = rank + 1

    # Sort by combined RRF score
    fused = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    fused = fused[:top_k]

    # Build clean result list
    results = []
    for entry in fused:
        chunk = entry["chunk"]
        chunk["score"] = round(entry["rrf_score"], 4)
        chunk["dense_rank"] = entry["dense_rank"]
        chunk["bm25_rank"] = entry["bm25_rank"]
        # Mark the fusion source
        if entry["dense_rank"] and entry["bm25_rank"]:
            chunk["source"] = "both"
        elif entry["dense_rank"]:
            chunk["source"] = "dense"
        else:
            chunk["source"] = "bm25"
        results.append(chunk)

    return results


def query(
    question: str,
    top_k: int = 5,
    filter_confidence: str = "high",
    filter_section: str = None,
    alpha: float = 0.6,
    verbose: bool = True,
) -> dict:
    """
    Full RAG pipeline: embed query → hybrid retrieve chunks → generate answer.

    Args:
        question:          the user's question in plain English
        top_k:             how many chunks to retrieve (more = more context)
        filter_confidence: only use chunks with this confidence level
                           set None to use all chunks regardless of confidence
        filter_section:    scope search to a specific section of notes
        alpha:             hybrid weight (0 = pure BM25, 1 = pure dense, 0.6 = default)
        verbose:           print retrieved chunks to terminal

    Returns:
        {
            "question": "...",
            "answer": "...",
            "sources": [ { "chunk_id", "text", "score", "metadata", "source" } ]
        }
    """

    # ── Step 1: Embed the query ──────────────────────────────────
    if verbose:
        print(f"\n🔍 Query: {question}")
        print("   Embedding query...")

    query_vector = embed_text(question)

    if not query_vector:
        return {"question": question, "answer": "Embedding failed.", "sources": []}

    # ── Step 2: Hybrid retrieve relevant chunks ─────────────────
    if verbose:
        print(f"   Hybrid searching (α={alpha}) for top {top_k} chunks...")

    retrieved = hybrid_search(
        query_text=question,
        query_vector=query_vector,
        top_k=top_k,
        alpha=alpha,
        filter_confidence=filter_confidence,
        filter_section=filter_section,
    )

    if not retrieved:
        return {
            "question": question,
            "answer": "No relevant content found in your notes.",
            "sources": [],
        }

    if verbose:
        print(f"   Found {len(retrieved)} chunks:")
        for r in retrieved:
            src = r.get("source", "?")
            dr = r.get("dense_rank", "-")
            br = r.get("bm25_rank", "-")
            print(f"     [rrf={r['score']}] Page {r['metadata'].get('page_no')} — {r['metadata'].get('section')} (via {src}, dense#{dr}, bm25#{br})")

    # ── Step 3: Build context from retrieved chunks ──────────────
    context_parts = []
    for i, chunk in enumerate(retrieved):
        page = chunk["metadata"].get("page_no", "?")
        section = chunk["metadata"].get("section", "Unknown")
        context_parts.append(
            f"[Chunk {i+1} | Page {page} | Section: {section}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # ── Step 4: Generate answer ──────────────────────────────────
    if verbose:
        print("   Generating answer...")

    user_message = f"""CONTEXT FROM NOTES:
{context}

---

QUESTION: {question}

Answer based only on the context above."""

    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"LLM error: {e}"

    return {
        "question": question,
        "answer":   answer,
        "sources":  retrieved,
    }


def pretty_print(result: dict):
    """Prints a RAG result cleanly to the terminal."""
    print("\n" + "="*60)
    print(f"❓ Question: {result['question']}")
    print("="*60)
    print(f"\n💡 Answer:\n{result['answer']}")
    print("\n📚 Sources:")
    for s in result["sources"]:
        src = s.get("source", "?")
        print(f"   • Page {s['metadata'].get('page_no')} | {s['metadata'].get('section')} | score: {s['score']} | via: {src}")
    print("="*60)
