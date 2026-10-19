<div align="center">

# 📝 RagifyNotes v2

### *Transform Handwritten Engineering Notes into an AI-Powered, Grounded Knowledge Base*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Integrated-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-3.8%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)

<br />

[Features](#-key-features) •
[Architecture](#-pipeline-architecture) •
[Retrieval Modes](#-retrieval-modes) •
[Benchmarking](#-evaluation--benchmarking) •
[Quickstart](#-quickstart-guide) •
[Docker & Deployment](#-docker--cloud-deployment) •
[API Reference](#-api-reference)

---

</div>

## 📌 Overview

Engineering students generate hundreds of pages of complex handwritten notes filled with mathematical derivations, schematics, and specialized technical terminology. Standard OCR systems fail miserably on handwritten cursive, non-linear mathematical equations, and unstructured margin notes.

**RagifyNotes v2** is an end-to-end Generative AI and Retrieval-Augmented Generation (RAG) platform that ingests raw handwritten PDF notes, transcribes them into structured Markdown and LaTeX using **Google Gemini 3.8 Flash Vision OCR**, indexes them into a high-performance **FAISS vector database**, and enables natural language question-answering with strict source attribution and zero hallucinations.

Featuring a **3-stage hybrid retrieval pipeline** (Dense FAISS + Lexical BM25 + Reciprocal Rank Fusion + Cross-Encoder Re-Ranking), RagifyNotes surfaces the exact handwritten lecture context required to answer even the most nuanced technical questions.

---

## 🚀 Key Features

* 👁️ **Multimodal Vision OCR**: Uses Gemini 3.8 Flash at 150 DPI to accurately transcribe messy handwriting, mathematical proofs, circuit diagrams, and Greek variables into structured LaTeX/Markdown.
* 🔗 **Full LangChain Framework Integration**: Standardized architecture leveraging `HuggingFaceEndpointEmbeddings`, `BM25Retriever`, and `CrossEncoderReranker`.
* ⚡ **Ultra-Lightweight FAISS Vector Store**: High-speed, local file-based vector storage (`faiss-cpu`) replacing bulky external database containers.
* 🌐 **Zero-Footprint Embeddings**: Uses Hugging Face Remote Inference API (`sentence-transformers/all-MiniLM-L6-v2`) for instant 384-dimensional dense representations without local GPU overhead.
* 🔀 **3-Stage Hybrid Retrieval**:
  1. **Dense Semantic Search**: Captures high-level conceptual intent via cosine similarity in FAISS.
  2. **Sparse Lexical Search**: Captures exact mathematical variables and formula terms using BM25.
  3. **Reciprocal Rank Fusion (RRF, $k=60$)**: Combines sparse and dense candidate ranks harmoniously.
* 🎯 **Cross-Encoder Re-Ranking**: Employs `cross-encoder/ms-marco-MiniLM-L-6-v2` to compute deep query-document interaction scores, filtering out false positives before LLM generation.
* 📊 **RAGAS Evaluation & MLflow Tracking**: Built-in benchmark suite measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall with incremental disk caching and MLflow experiment logging.
* 🖥️ **Modern Web Application**: Single-page FastAPI interface featuring interactive chat, MathJax LaTeX equation rendering, real-time retrieval mode switching, and drag-and-drop multi-PDF uploads.
* 🐳 **Containerized & Cloud-Ready**: Fully containerized with Docker & Docker Compose, dynamic `$PORT` handling, and isolated MLflow tracking services.

---

## 🏛️ Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                              │
└────────────────────────────────────────────────────────────────────────┘
        Handwritten Course Notes (PDFs in Data/input_pdfs/)
                               │
                               ▼
        PyMuPDF (fitz) Rendering Engine (150 DPI Page Images)
                               │
                               ▼
        Gemini 3.8 Flash Vision OCR (Text, Formulas & LaTeX)
                               │
                               ▼
       Text Cleaning, Header Extraction & Document Structuring
                               │
                               ▼
    Overlapping Recursive Chunking (Target ~500 chars | ~100 overlap)
                               │
                               ▼
    LangChain HuggingFace Inference API (all-MiniLM-L6-v2, 384-dim)
                               │
                               ▼
           FAISS Vector Store (persisted to ./faiss_index)

──────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────┐
│                         QUERY & RAG PIPELINE                           │
└────────────────────────────────────────────────────────────────────────┘
                          Student Question
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   Dense Semantic Retrieval           Sparse Lexical Retrieval
   (FAISS Vector Similarity)          (BM25 Inverted Index)
              │                                 │
              └────────────────┬────────────────┘
                               ▼
               Reciprocal Rank Fusion (RRF, k=60)
                     (Top 15-20 Candidates)
                               │
                               ▼
               Cross-Encoder Re-Ranking Engine
                 (ms-marco-MiniLM-L-6-v2)
                               │
                               ▼
                 Filtered Context (Top 5 Chunks)
                               │
                               ▼
             Grounded Prompt Engineering Template
                               │
                               ▼
             Gemini 3.8 Flash Response Generation
                               │
                               ▼
         Accurate Answer with Source Chunks & MathJax LaTeX
```

---

## 🔍 Retrieval Modes

RagifyNotes provides three selectable retrieval modes, allowing users to benchmark the precision-latency trade-offs in real time:

| Retrieval Mode | Dense (FAISS) | Sparse (BM25) | RRF Fusion | Cross-Encoder Re-Ranking | Best Used For |
|:---|:---:|:---:|:---:|:---:|:---|
| **`naive`** | ✅ | ❌ | ❌ | ❌ | Conceptual queries where keywords are flexible. |
| **`hybrid`** | ✅ | ✅ | ✅ | ❌ | Queries containing specific equations, variable names, or acronyms. |
| **`hybrid_rerank`** *(Default)* | ✅ | ✅ | ✅ | ✅ | Complex multi-concept exam questions requiring maximum precision. |

### Why Two-Stage Hybrid + Re-Ranking Matters
1. **The Bi-Encoder Limitation**: Bi-encoders (`embedder.py`) independently project queries and documents into a shared vector space: $\text{Score} = \cos(\mathbf{q}, \mathbf{d})$. While extremely fast, they miss intricate token-level interactions.
2. **The Cross-Encoder Advantage**: Cross-encoders pass query and candidate text simultaneously through a BERT transformer encoder, modeling every word-to-word cross-attention relation. Running this over an entire database is computationally prohibitive, but re-ranking the **top 15-20 candidates** takes $<1$ second on CPU while dramatically reducing hallucination.

---

## 📈 Evaluation & Benchmarking

RagifyNotes v2 incorporates industry-standard **RAGAS** (Retrieval Augmented Generation Assessment) evaluation tracked via **MLflow**.

### Benchmark Metrics

* **Faithfulness**: Quantifies whether every factual claim in the generated answer is directly grounded in the retrieved note excerpts.
* **Answer Relevancy**: Measures how directly the generated response addresses the user's inquiry.
* **Context Precision**: Computes whether the truly relevant chunks are ranked at the top of the context window.
* **Context Recall**: Evaluates whether all requisite reference facts were successfully retrieved.

### Observed Quantitative Gains

Benchmarked against hand-curated engineering ground-truth questions (e.g., *Z-Transform ROC derivation*, *Conjoint Analysis Part-Worth Utilities*):

| Metric | Naive RAG | Hybrid RAG | Hybrid + Re-Ranking |
|:---|:---:|:---:|:---:|
| **Context Recall** | `0.70` | `0.85` | **`0.95+`** |
| **Context Precision** | `0.52` | `0.68` | **`0.88+`** |
| **Faithfulness** | `0.76` | `0.84` | **`0.93+`** |
| **Answer Relevancy** | `0.91` | `0.94` | **`0.97+`** |

> **Incremental Caching**: The evaluation runner (`evaluation/benchmark.py`) caches individual metric scores per question to disk (`evaluation/results/eval_cache.json`). If rate limits or network hiccups occur, the runner immediately resumes without repeating completed evaluations.

---

## 🛠️ Tech Stack

| Category | Technologies / Libraries |
|:---|:---|
| **Core Framework** | [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-google-genai`) |
| **LLM & Vision OCR** | [Google Gemini 3.8 Flash](https://aistudio.google.com/) via Google GenAI SDK |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face Inference API |
| **Vector Database** | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| **Sparse Retrieval** | [Rank-BM25](https://github.com/dorianbrown/rank_bm25) |
| **Re-Ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Web Server & UI** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), Vanilla HTML5/CSS3/JS, MathJax v3 |
| **Document Processing** | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/), Pillow |
| **Benchmarking & MLOps**| [RAGAS](https://github.com/explodinggradients/ragas), [MLflow](https://mlflow.org/), Hugging Face `datasets` |
| **Containerization** | Docker, Docker Compose |

---

## 📂 Project Structure

```text
RagifyNotes/
├── app.py                      # FastAPI web server & REST endpoints
├── main.py                     # CLI ingestion & interactive query tool
├── Dockerfile                  # Container definition with dynamic PORT support
├── docker-compose.yml          # Dual-service config (FastAPI app + MLflow)
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment configuration
├── .dockerignore               # Build optimization exclusions
│
├── config/
│   └── settings.py             # Centralized pipeline configuration & thresholds
│
├── src/
│   ├── preprocessing.py        # PDF extraction & DPI image rendering
│   ├── extract.py              # Gemini 3.8 Flash Vision OCR engine
│   ├── structure.py            # Text structuring, cleaning & metadata tagging
│   ├── chunker.py              # Recursive overlapping chunker
│   ├── embedder.py             # LangChain Hugging Face remote embeddings
│   ├── vector_store.py         # FAISS vector store indexing & similarity search
│   ├── retriever.py            # Hybrid RRF retriever & pipeline coordinator
│   └── reranker.py             # LangChain Cross-Encoder re-ranker compressor
│
├── evaluation/
│   ├── benchmark.py            # RAGAS evaluation runner with incremental caching
│   ├── test_questions.json     # Curated engineering test QA dataset
│   └── results/                # Cached evaluation scores and JSON results
│
├── templates/
│   └── index.html              # Responsive single-page web UI with MathJax
│
└── Data/
    └── input_pdfs/             # Input directory for raw handwritten lecture PDFs
```

---

## ⚡ Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/RawKnack/Ragify-Notes.git
cd Ragify-Notes
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Open `.env` and insert your API keys:
* **`GEMINI_API_KEY`**: Obtain from [Google AI Studio](https://aistudio.google.com/apikey) (Free tier available).
* **`HUGGINGFACE_API_KEY`**: Obtain from [Hugging Face Settings](https://huggingface.co/settings/tokens) (Free tier available).

```env
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_token_here
```

### 5. Ingest Notes

Place your handwritten PDF notes in `Data/input_pdfs/` (e.g. `Z-Transform.pdf`):

```bash
# Ingest all PDFs in Data/input_pdfs/:
python main.py

# Or ingest a specific PDF:
python main.py --pdf "Data/input_pdfs/Z-Transform.pdf"
```

### 6. Run the Web Application

```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 💻 CLI Usage

You can also interact with the pipeline directly from your terminal:

```bash
# Query using default Hybrid + Re-Ranking mode:
python main.py --query "What is the Z-Transform?"

# Query using Hybrid mode (Dense + BM25):
python main.py --query "Explain the Region of Convergence" --mode hybrid

# Query using Naive mode (Vector search only):
python main.py --query "What is Parseval's Theorem?" --mode naive --top-k 3
```

---

## 📊 Running Benchmarks

Evaluate pipeline performance with RAGAS and view results in MLflow:

```bash
# Run evaluation across all three retrieval modes:
python -m evaluation.benchmark

# Run evaluation for specific modes:
python -m evaluation.benchmark --modes naive hybrid_rerank

# Launch MLflow UI to compare experiment runs:
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Open **`http://localhost:5001`** to explore interactive comparison dashboards.

---

## 🐳 Docker & Cloud Deployment

### Local Docker Compose

Run both the **RagifyNotes web app** and the **MLflow tracking server** in isolated containers:

```bash
docker compose up --build
```
* **RagifyNotes Web Interface**: `http://localhost:5000`
* **MLflow Tracking Dashboard**: `http://localhost:5001`

### Cloud Deployment (Render, Railway, Hugging Face Spaces, Cloud Run)

The application Dockerfile includes automatic `$PORT` binding and lightweight remote embeddings:

1. **Push your code to GitHub** (make sure `.env` is ignored by `.gitignore`).
2. **Connect your repository** to [Render](https://render.com), [Railway](https://railway.app), or [Hugging Face Spaces](https://huggingface.co/spaces) (Docker SDK).
3. **Set Environment Variables** in your platform settings dashboard:
   - `GEMINI_API_KEY`
   - `HUGGINGFACE_API_KEY`
4. The service will automatically build the container and launch FastAPI on the assigned port.

---

## 📡 API Reference

| Method | Endpoint | Description | Payload / Parameters |
|:---|:---|:---|:---|
| `GET` | `/` | Serves the single-page application UI | None |
| `POST` | `/api/query` | Submits a question to the RAG pipeline | `{"question": "...", "top_k": 5, "alpha": 0.6, "mode": "hybrid_rerank"}` |
| `POST` | `/api/upload` | Uploads and processes a new PDF note file | Multipart Form (`file: .pdf`) |
| `GET` | `/api/stats` | Returns vector store index stats and PDF count | None |
| `POST` | `/api/refresh`| Rebuilds the BM25 sparse index | None |

---

## 🛡️ Best Practices & Architecture Decisions

1. **Why FAISS over hosted vector databases?**
   - For study notes and course materials, FAISS provides microsecond similarity search with zero network hops, negligible memory footprint, and seamless disk persistence.
2. **Why Remote Hugging Face Inference?**
   - Eliminates downloading heavy PyTorch/Transformer model weights in memory-constrained cloud environments, keeping container builds under 500MB and boot times under 15 seconds.
3. **Formula & Diagram Preservation**:
   - Gemini 3.8 Flash produces clean LaTeX formulas (`$$ ... $$`) which are rendered seamlessly on the frontend via MathJax.

---

## 👨‍💻 Author

**Raunak Suman**  
*AI / Generative AI Engineer*  
*Specializing in RAG Architectures, Multimodal AI, NLP & Full-Stack AI Applications*

* GitHub: [@RawKnack](https://github.com/RawKnack)
* Repository: [Ragify-Notes](https://github.com/RawKnack/Ragify-Notes)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
