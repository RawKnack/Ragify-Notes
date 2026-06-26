"""
app.py — FastAPI web application for the Handwritten Notes RAG pipeline.

Provides a single-page UI for querying your ingested handwritten notes
using hybrid retrieval (dense + BM25) and LLM-generated answers.

Run:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.retriever import query, refresh_bm25_index

app = FastAPI(title="Handwritten Notes RAG")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    alpha: float = 0.6


@app.get("/", response_class=HTMLResponse)
def read_index():
    """Serves the single-page query UI."""
    template_path = os.path.join("templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="index.html not found in templates directory.")
    
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/api/query")
def api_query(req: QueryRequest):
    """
    Accepts a JSON body matching QueryRequest.
    Returns: { "question": "...", "answer": "...", "sources": [...] }
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = query(
            question=question,
            top_k=req.top_k,
            filter_confidence="high",
            alpha=req.alpha,
            verbose=True,
        )

        # Clean sources for JSON serialization
        clean_sources = []
        for s in result.get("sources", []):
            clean_sources.append({
                "chunk_id": s.get("chunk_id"),
                "text": s.get("text", ""),
                "score": s.get("score", 0),
                "source": s.get("source", "unknown"),
                "dense_rank": s.get("dense_rank"),
                "bm25_rank": s.get("bm25_rank"),
                "page_no": s.get("metadata", {}).get("page_no", "?"),
                "section": s.get("metadata", {}).get("section", "Unknown"),
                "confidence": s.get("metadata", {}).get("confidence", "unknown"),
            })

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": clean_sources,
        }

    except Exception as e:
        print(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refresh")
def api_refresh():
    """Force-refresh the BM25 index (call after re-ingestion)."""
    try:
        refresh_bm25_index()
        return {"status": "BM25 index refreshed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Handwritten Notes RAG -- FastAPI Web Interface")
    print("   Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=5000)
