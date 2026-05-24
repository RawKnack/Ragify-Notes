"""
src/retriever.py — RAG query engine.

WHAT RAG DOES:
  Retrieval Augmented Generation — instead of asking an LLM a question
  from its training data (which may be outdated or wrong), you:

  1. RETRIEVE relevant chunks from YOUR notes (the vector database)
  2. AUGMENT the LLM prompt with those chunks as context
  3. GENERATE an answer grounded in your actual notes

  The LLM can only answer using what you retrieved — it can't make up
  content from training data because you explicitly tell it to use only
  the provided context.

FLOW:
  User query: "what is the ROC for a causal sequence?"
       ↓
  Embed query → vector
       ↓
  Search Qdrant → top 5 most similar chunks from your notes
       ↓
  Build prompt: [your chunks] + "Answer this: what is ROC..."
       ↓
  LLM generates answer grounded in your notes
       ↓
  Return answer + source chunks (for citation)
"""

from openai import OpenAI
from src.embedder import embed_text
from src.vector_store import search
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


def query(
    question: str,
    top_k: int = 5,
    filter_confidence: str = "high",
    filter_section: str = None,
    verbose: bool = True,
) -> dict:
    """
    Full RAG pipeline: embed query → retrieve chunks → generate answer.

    Args:
        question:          the user's question in plain English
        top_k:             how many chunks to retrieve (more = more context)
        filter_confidence: only use chunks with this confidence level
                           set None to use all chunks regardless of confidence
        filter_section:    scope search to a specific section of notes
        verbose:           print retrieved chunks to terminal

    Returns:
        {
            "question": "...",
            "answer": "...",
            "sources": [ { "chunk_id", "text", "score", "metadata" } ]
        }
    """

    # ── Step 1: Embed the query ──────────────────────────────────
    if verbose:
        print(f"\n🔍 Query: {question}")
        print("   Embedding query...")

    query_vector = embed_text(question)

    if not query_vector:
        return {"question": question, "answer": "Embedding failed.", "sources": []}

    # ── Step 2: Retrieve relevant chunks ────────────────────────
    if verbose:
        print(f"   Searching top {top_k} chunks...")

    retrieved = search(
        query_vector=query_vector,
        top_k=top_k,
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
            print(f"     [{r['score']}] Page {r['metadata'].get('page_no')} — {r['metadata'].get('section')}")

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
        print(f"   • Page {s['metadata'].get('page_no')} | {s['metadata'].get('section')} | score: {s['score']}")
    print("="*60)
