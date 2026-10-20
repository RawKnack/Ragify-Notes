import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIMENSION = 384

RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "ms-marco-TinyBERT-L-2-v2")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

RRF_K = int(os.getenv("RRF_K", "60"))
DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.6"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

OUTPUT_JSON = os.getenv("OUTPUT_JSON", "Data/output_json/output.json")
CHUNKS_JSON = os.getenv("CHUNKS_JSON", "Data/output_json/chunks.json")
INPUT_PDF_DIR = os.getenv("INPUT_PDF_DIR", "Data/input_pdfs")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "RagifyNotes-RAG-Benchmark")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set")
