import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from src.vector_store import load_store, get_all_chunks
from src.reranker import rerank
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, DEFAULT_ALPHA, DEFAULT_MODE

warnings.filterwarnings("ignore", category=UserWarning)

answer_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    max_output_tokens=2048,
    temperature=0.2,
    timeout=45,
    max_retries=1,
)


RAG_SYSTEM_PROMPT = """
You are an expert academic study assistant that helps engineering students understand concepts thoroughly from their lecture notes.
Answer the user's QUESTION using the provided CONTEXT from their study notes as your primary source.

STRUCTURE & STRICT ACCURACY GUIDELINES:
1. **Strict Factual Grounding**:
   - Rely strictly and exclusively on the information, definitions, and equations explicitly stated in the CONTEXT.
   - Do NOT assume, extrapolate, speculate, or introduce external concepts, history, or theories from outside the notes.
   - Every factual statement you make must be directly verifiable from the provided context.
2. **Mathematical Formulation**:
   - If equations, derivations, or variables are present in the context, format them cleanly using LaTeX ($...$ for inline, $$...$$ for block equations).
   - If mathematical formulas for this concept are NOT mentioned in the notes, do NOT invent or import outside equations.
3. **Structured & Clear Organization**:
   - Organize the answer into clear, readable sections using markdown headings (e.g., `### Definition & Overview`, `### Mathematical Formulation` (only when in context), `### Key Properties & Insights`).
   - Use **bold text** for key terminology and bullet points for enumerating properties, conditions, or rules.
4. **Context Absence**:
   - If the provided context contains absolutely no relevant information to answer the question, state: "The provided notes do not contain information about this topic."
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", """CONTEXT FROM NOTES:
{context}

---

QUESTION: {question}

Provide a structured, accurate explanation based STRICTLY on the context above. Do not include external facts or unmentioned formulas."""),
])

rag_chain = rag_prompt | answer_llm | StrOutputParser()

_bm25_retriever = None
_bm25_docs = None


def _get_faiss_retriever(top_k: int = 15):
    db = load_store()
    return db.as_retriever(search_kwargs={"k": top_k})


def _build_bm25_retriever(top_k: int = 15):
    global _bm25_retriever, _bm25_docs

    all_chunks = get_all_chunks()
    if not all_chunks:
        _bm25_retriever = None
        return None

    _bm25_docs = []
    for chunk in all_chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                **chunk["metadata"],
            }
        )
        _bm25_docs.append(doc)

    _bm25_retriever = BM25Retriever.from_documents(_bm25_docs, k=top_k)
    return _bm25_retriever


def _get_bm25_retriever(top_k: int = 15):
    global _bm25_retriever
    if _bm25_retriever is None:
        return _build_bm25_retriever(top_k)
    _bm25_retriever.k = top_k
    return _bm25_retriever


def refresh_bm25_index():
    global _bm25_retriever, _bm25_docs
    _bm25_retriever = None
    _bm25_docs = None
    _build_bm25_retriever()


def _docs_to_chunks(docs: list[Document], source_tag: str = "dense") -> list[dict]:
    chunks = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata.copy()
        chunk_id = metadata.pop("chunk_id", "")

        chunks.append({
            "chunk_id": chunk_id,
            "text": doc.page_content,
            "score": round(1.0 / (i + 1), 4),
            "metadata": metadata,
            "source": source_tag,
            "dense_rank": None,
            "bm25_rank": None,
        })

    return chunks


def _generate_answer(question: str, context: str) -> str:
    try:
        return rag_chain.invoke({"question": question, "context": context})
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            return (
                "⚠️ **Gemini API Daily Quota Limit Reached**\n\n"
                "The current API key has exhausted its daily free-tier requests on the Gemini API.\n\n"
                "**To restore instant answering:**\n"
                "1. Go to [Google AI Studio](https://aistudio.google.com/apikey) (Free tier).\n"
                "2. Click **Create API Key**.\n"
                "3. Paste your new key into your `.env` file (and in Render Environment Variables).\n\n"
                "*(Your matching lecture chunks from notes have still been retrieved and ranked below!)*"
            )
        if "503" in err_msg or "UNAVAILABLE" in err_msg:
            return (
                "⚠️ **Gemini Service Temporarily Busy (503)**\n\n"
                "Google's free-tier servers are temporarily experiencing high demand. Please re-run your search in a few seconds."
            )
        return f"⚠️ Unable to generate answer: {err_msg}"
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator


@traceable(name="RagifyNotes Query Pipeline", run_type="chain")
def query(
    question: str,
    top_k: int = 5,
    filter_confidence: str = None,
    filter_section: str = None,
    alpha: float = None,
    mode: str = DEFAULT_MODE,
    verbose: bool = True,
) -> dict:
    if alpha is None:
        alpha = DEFAULT_ALPHA

    if verbose:
        print(f"\n[Retriever] Query: {question}")
        print(f"   Mode: {mode}")

    try:
        if mode == "naive":
            faiss_retriever = _get_faiss_retriever(top_k=top_k)
            docs = faiss_retriever.invoke(question)
            retrieved = _docs_to_chunks(docs, source_tag="dense")

        elif mode in ("hybrid", "hybrid_rerank"):
            fetch_k = top_k * 3 if mode == "hybrid_rerank" else top_k
            faiss_retriever = _get_faiss_retriever(top_k=fetch_k)
            bm25_retriever = _get_bm25_retriever(top_k=fetch_k)

            if bm25_retriever:
                # Inline weighted fusion (replaces EnsembleRetriever removed in LangChain 1.4+)
                faiss_docs = faiss_retriever.invoke(question)
                bm25_docs = bm25_retriever.invoke(question)
                # Weighted interleave: score = alpha * (1/faiss_rank) + (1-alpha) * (1/bm25_rank)
                seen, fused = set(), []
                scores = {}
                for rank, doc in enumerate(faiss_docs, 1):
                    key = doc.page_content
                    scores[key] = scores.get(key, 0) + alpha * (1.0 / rank)
                    if key not in seen:
                        seen.add(key)
                        fused.append(doc)
                for rank, doc in enumerate(bm25_docs, 1):
                    key = doc.page_content
                    scores[key] = scores.get(key, 0) + (1 - alpha) * (1.0 / rank)
                    if key not in seen:
                        seen.add(key)
                        fused.append(doc)
                docs = sorted(fused, key=lambda d: scores.get(d.page_content, 0), reverse=True)
            else:
                docs = faiss_retriever.invoke(question)

            retrieved = _docs_to_chunks(docs, source_tag="both")

        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

    except FileNotFoundError:
        return {
            "question": question,
            "answer": "No notes have been indexed yet. Please upload a PDF to create the knowledge base.",
            "sources": [],
            "mode": mode,
        }

    if not retrieved:
        return {
            "question": question,
            "answer": "No relevant content found in your notes.",
            "sources": [],
            "mode": mode,
        }

    if mode == "hybrid_rerank":
        retrieved = rerank(question, retrieved, top_k=top_k)
    else:
        retrieved = retrieved[:top_k]

    context_parts = []
    for i, chunk in enumerate(retrieved):
        page = chunk["metadata"].get("page_no", "?")
        section = chunk["metadata"].get("section", "Unknown")
        context_parts.append(
            f"[Chunk {i+1} | Page {page} | Section: {section}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    answer = _generate_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "sources": retrieved,
        "mode": mode,
    }


def pretty_print(result: dict):
    print("\n" + "=" * 60)
    print(f"Question: {result['question']}")
    print(f"Mode: {result.get('mode', 'unknown')}")
    print("=" * 60)
    print(f"\nAnswer:\n{result['answer']}")
    print("\nSources:")
    for s in result["sources"]:
        src = s.get("source", "?")
        rr = s.get("rerank_score", "")
        rr_str = f" | rerank: {rr}" if rr != "" else ""
        print(f"   • Page {s['metadata'].get('page_no')} | {s['metadata'].get('section')} "
              f"| score: {s['score']} | via: {src}{rr_str}")
    print("=" * 60)
