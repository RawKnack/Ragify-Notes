import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Local embedding model (runs on CPU, no API calls)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension

# Local cross-encoder reranker (runs on CPU, no API calls)
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── FAISS Configuration ────────────────────────────────────────
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")

# ── Chunking Configuration ──────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))        # Target characters per chunk (~80 words)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))    # Overlap characters (~15-20 words)

# ── Retrieval Configuration ────────────────────────────────────
RRF_K = int(os.getenv("RRF_K", "60"))           # RRF smoothing constant
DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.6"))  # Dense vs Sparse weight
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# ── Data Paths ──────────────────────────────────────────────────
OUTPUT_JSON = os.getenv("OUTPUT_JSON", "Data/output_json/output.json")
CHUNKS_JSON = os.getenv("CHUNKS_JSON", "Data/output_json/chunks.json")
INPUT_PDF_DIR = os.getenv("INPUT_PDF_DIR", "Data/input_pdfs")

# ── MLflow Configuration ───────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "RagifyNotes-RAG-Benchmark")

# ── Warnings ────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not set — needed for OCR extraction and answer generation")
    print("   Get one free at: https://aistudio.google.com/apikey")

if not HUGGINGFACE_API_KEY:
    print("⚠️  HUGGINGFACE_API_KEY not set — needed for embedding via HuggingFace Inference API")
    print("   Get one free at: https://huggingface.co/settings/tokens")
