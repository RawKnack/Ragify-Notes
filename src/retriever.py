import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.vector_store import load_store, get_all_chunks
from src.reranker import rerank
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, DEFAULT_ALPHA

warnings.filterwarnings("ignore", category=UserWarning)

answer_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    max_output_tokens=2048,
)

RAG_SYSTEM_PROMPT = """
You are a helpful academic study assistant that explains concepts thoroughly.
Answer the user's QUESTION using the provided CONTEXT from their study notes as the primary source.

RULES:
1. **Elaborate and explain**: Don't just extract text — synthesize the context into a comprehensive, well-structured explanation. If the context mentions a concept (e.g., "principal components"), use your general knowledge to provide the full explanation (e.g., full form, definition, working) while staying grounded in what the notes cover.
2. **Structure your answer clearly**:
   - Start with a direct answer to the question.
   - Use **bold text** for key terms and definitions.
   - Use bullet points or numbered lists to break down complex topics.
   - Use LaTeX ($...$) for mathematical equations and formulas.
   - Use headings (## or ###) to organize multi-part answers.
3. **Connect concepts**: If the notes mention related ideas, connect them to give the student a complete picture.
4. **Be thorough**: Provide enough detail that a student can understand the concept fully from your answer alone.
5. **Stay grounded**: Your answer must be primarily based on the context. You may use general knowledge to fill in obvious gaps (like full forms of abbreviations, standard definitions) but do NOT fabricate information that contradicts the notes.
6. Do NOT include prompt meta-headers, meta-questions, or preambles.
7. If the context contains absolutely no relevant information, state: "The provided notes do not contain information about this topic."
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", """CONTEXT FROM NOTES:
{context}

---

QUESTION: {question}

Answer based on the context above."""),
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
        return f"LLM error: {e}"


def query(
    question: str,
    top_k: int = 5,
    filter_confidence: str = None,
    filter_section: str = None,
    alpha: float = None,
    mode: str = "hybrid_rerank",
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
                ensemble = EnsembleRetriever(
                    retrievers=[faiss_retriever, bm25_retriever],
                    weights=[alpha, 1 - alpha],
                )
                docs = ensemble.invoke(question)
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
