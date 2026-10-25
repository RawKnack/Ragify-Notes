"""
app.py — FastAPI web application for the RagifyNotes v2 pipeline.

Provides a web UI for:
  - Querying ingested handwritten notes
  - Uploading and ingesting new PDFs
  - Switching between retrieval modes (naive / hybrid / hybrid_rerank)
  - Viewing pipeline stats

Run:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
import shutil
import glob
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.retriever import query, refresh_bm25_index
from config.settings import DEFAULT_MODE
from src.vector_store import get_collection_info
from config.settings import INPUT_PDF_DIR


app = FastAPI(
    title="RagifyNotes v2",
    description="AI-powered handwritten notes search with hybrid retrieval + cross-encoder re-ranking",
    version="2.0.0",
)

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
    mode: str = DEFAULT_MODE  # "naive", "hybrid", or "hybrid_rerank"


@app.get("/", response_class=HTMLResponse)
def read_index():
    """Serves the single-page query UI."""
    template_path = os.path.join("templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="index.html not found in templates directory.")

    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.head("/")
def head_index():
    """Health check responder for Render and load balancers."""
    return HTMLResponse(content="")


@app.post("/api/query")
def api_query(req: QueryRequest):
    """
    Accepts a JSON body matching QueryRequest.
    Returns: { "question": "...", "answer": "...", "sources": [...], "mode": "..." }
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if req.mode not in ("naive", "hybrid", "hybrid_rerank"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {req.mode}. Use 'naive', 'hybrid', or 'hybrid_rerank'.")

    try:
        result = query(
            question=question,
            top_k=req.top_k,
            filter_confidence=None,
            alpha=req.alpha,
            mode=req.mode,
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
                "rerank_score": s.get("rerank_score"),
                "page_no": s.get("metadata", {}).get("page_no", "?"),
                "section": s.get("metadata", {}).get("section", "Unknown"),
                "confidence": s.get("metadata", {}).get("confidence", "unknown"),
                "source_pdf": s.get("metadata", {}).get("source_pdf", "unknown"),
            })

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": clean_sources,
            "mode": result.get("mode", req.mode),
        }

    except Exception as e:
        print(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """
    Upload a PDF file for ingestion.
    Saves to the input_pdfs directory.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    os.makedirs(INPUT_PDF_DIR, exist_ok=True)
    dest_path = os.path.join(INPUT_PDF_DIR, file.filename)

    try:
        with open(dest_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {
            "status": "uploaded",
            "filename": file.filename,
            "path": dest_path,
            "message": f"PDF saved. Run 'python main.py' to ingest, or use the /api/ingest endpoint.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refresh")
def api_refresh():
    """Force-refresh the BM25 index (call after re-ingestion)."""
    try:
        refresh_bm25_index()
        return {"status": "BM25 index refreshed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
def api_stats():
    """Get pipeline statistics."""
    try:
        info = get_collection_info()

        # Count PDFs in input directory
        pdf_count = len(glob.glob(os.path.join(INPUT_PDF_DIR, "*.pdf")))

        return {
            "vector_store": info,
            "pdf_count": pdf_count,
            "available_modes": ["naive", "hybrid", "hybrid_rerank"],
        }
    except Exception as e:
        return {
            "vector_store": {"total_points": 0, "status": "not initialized"},
            "pdf_count": 0,
            "available_modes": ["naive", "hybrid", "hybrid_rerank"],
        }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n[RagifyNotes] Starting server on 0.0.0.0:{port}...", flush=True)
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
