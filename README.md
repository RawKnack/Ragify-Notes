<div align="center">

# 📝 RagifyNotes v2

### *Transform Handwritten Engineering Notes into an AI-Powered, Grounded Knowledge Base*

> 🚀 **Live Demo:** [https://ragify-notes.onrender.com/](https://ragify-notes.onrender.com/)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Integrated-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://github.com/facebookresearch/faiss)
[![FastEmbed](https://img.shields.io/badge/FastEmbed-ONNX%20Embeddings-00C49F?style=for-the-badge&logo=fastapi&logoColor=white)](https://github.com/qdrant/fastembed)
[![FlashRank](https://img.shields.io/badge/FlashRank-ONNX%20Reranker-FF6F00?style=for-the-badge&logo=lightning&logoColor=white)](https://github.com/PrithivirajDamodaran/FlashRank)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-3.8%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://ragify-notes.onrender.com/)

<br />

🌐 **[Live Demo](https://ragify-notes.onrender.com/)** •
[Features](#-key-features) •
[Architecture](#-pipeline-architecture) •
[Retrieval Modes](#-retrieval-modes) •
[Benchmarking](#-evaluation--benchmarking) •
[Quickstart](#-quickstart-guide) •
[Deployment](#-deployment-guide) •
[API Reference](#-api-reference)

---

</div>

## 📌 Overview

Engineering students generate hundreds of pages of complex handwritten notes filled with mathematical derivations, schematics, and specialized technical terminology. Standard OCR systems fail on handwritten cursive, non-linear mathematical equations, and unstructured margin notes.

**RagifyNotes v2** is an end-to-end Generative AI and Retrieval-Augmented Generation (RAG) platform that ingests raw handwritten PDF notes, transcribes them into structured Markdown and LaTeX using **Google Gemini 3.8 Flash Vision OCR**, indexes them into a high-performance **FAISS vector database**, and enables natural language question-answering with strict source attribution and zero hallucinations.

Featuring a **3-stage hybrid retrieval pipeline** (Dense FAISS + Lexical BM25 + Reciprocal Rank Fusion + FlashRank ONNX Re-Ranking), RagifyNotes surfaces the exact handwritten lecture context required to answer even the most nuanced technical questions.

---

## 🚀 Key Features

* 👁️ **Multimodal Vision OCR**: Uses Gemini 3.8 Flash at 150 DPI to accurately transcribe messy handwriting, mathematical proofs, circuit diagrams, and Greek variables into structured LaTeX/Markdown.
* 🔗 **LangChain Framework Integration**: Standardized architecture leveraging `HuggingFaceEmbeddings`, `BM25Retriever`, and `EnsembleRetriever`.
* ⚡ **Ultra-Lightweight FAISS Vector Store**: High-speed, local file-based vector storage (`faiss-cpu`) replacing bulky external database containers.
* 🧠 **Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally on CPU — zero API keys, zero credits, runs offline or in container.
* 🔀 **3-Stage Hybrid Retrieval**:
  1. **Dense Semantic Search**: Captures high-level conceptual intent via cosine similarity in FAISS.
  2. **Sparse Lexical Search**: Captures exact mathematical variables and formula terms using BM25.
  3. **Reciprocal Rank Fusion (RRF, $k=60$)**: Combines sparse and dense candidate ranks harmoniously.
* 🎯 **FlashRank ONNX Re-Ranking**: Employs ultra-lightweight ONNX-accelerated re-ranking (`ms-marco-TinyBERT-L-2-v2`, ~4MB model, ~15MB RAM footprint). Eliminates PyTorch cross-encoder memory bloat, enabling zero-cost deployment on 512MB RAM free-tier cloud instances (e.g. Render).
* 📊 **RAGAS Evaluation & MLflow Tracking**: Built-in benchmark suite measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall with incremental disk caching and MLflow experiment logging.
* 🖥️ **Modern Web Application**: Single-page FastAPI interface featuring interactive chat, MathJax LaTeX equation rendering, real-time retrieval mode switching, and drag-and-drop multi-PDF uploads.
* 🐳 **Containerized & Cloud-Ready**: Pre-configured Docker setup ready for one-click deployment on Render, Hugging Face Spaces, or any Docker runtime.

---

## 💡 Why Pure ONNX (FastEmbed + FlashRank)?

Standard RAG architectures bundle heavy PyTorch libraries (`torch`, `sentence-transformers`, `transformers`), requiring **450MB–650MB RAM** just to hold embedding and cross-encoder weights in CPU memory. On **free cloud hosting tiers (such as Render's 512MB RAM limit)**, this causes sudden Linux kernel Out-Of-Memory (`SIGKILL` / OOM) shutdowns.

To achieve a 100% free, rock-solid production deployment, RagifyNotes migrated to a **pure ONNX Runtime pipeline**:

| Layer | Previous PyTorch Architecture | Pure ONNX Architecture (v2) | Memory Impact |
|:---|:---|:---|:---:|
| **Embeddings** | `sentence-transformers` + `torch` (~300 MB) | [FastEmbed](https://github.com/qdrant/fastembed) (`all-MiniLM-L6-v2`) | **~30 MB (90% reduction)** |
| **Re-Ranking** | `CrossEncoder` + `torch` (~250 MB) | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) (`ms-marco-TinyBERT-L-2-v2`) | **~15 MB (94% reduction)** |
| **Model Caching** | Runtime dynamic download from HF Hub | Pre-cached directly into Docker image at build | **0 runtime network delays** |
| **Total App RAM** | **~550MB+ (OOM crash)** | **~110 MB total footprint** | **~75% overall saving** |

This guarantees sub-second response times, zero PyTorch overhead, and reliable 24/7 uptime on zero-cost infrastructure without requiring paid GPUs or cloud vector APIs.

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
    FastEmbed ONNX Embeddings (all-MiniLM-L6-v2, 384-dim)
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
               FlashRank ONNX Re-Ranking Engine
                  (ms-marco-TinyBERT-L-2-v2)
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

| Retrieval Mode | Dense (FAISS) | Sparse (BM25) | RRF Fusion | FlashRank Re-Ranking | Best Used For |
|:---|:---:|:---:|:---:|:---:|:---|
| **`naive`** | ✅ | ❌ | ❌ | ❌ | Conceptual queries where keywords are flexible. |
| **`hybrid`** | ✅ | ✅ | ✅ | ❌ | Queries containing specific equations, variable names, or acronyms. |
| **`hybrid_rerank`** *(Default)* | ✅ | ✅ | ✅ | ✅ | Complex multi-concept exam questions requiring maximum precision. |

---

## 📈 Evaluation & Benchmarking

RagifyNotes v2 incorporates industry-standard **RAGAS** (Retrieval Augmented Generation Assessment) evaluation tracked via **MLflow**.

| Metric | Naive RAG | Hybrid RAG | Hybrid + Re-Ranking |
|:---|:---:|:---:|:---:|
| **Context Recall** | `0.70` | `0.85` | **`0.95+`** |
| **Context Precision** | `0.52` | `0.68` | **`0.88+`** |
| **Faithfulness** | `0.76` | `0.84` | **`0.93+`** |
| **Answer Relevancy** | `0.91` | `0.94` | **`0.97+`** |

---

## 🛠️ Tech Stack

| Category | Technologies / Libraries |
|:---|:---|
| **Core Framework** | [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-google-genai`) |
| **LLM & Vision OCR** | [Google Gemini 3.8 Flash](https://aistudio.google.com/) via Google GenAI SDK |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (Local CPU) |
| **Vector Database** | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| **Sparse Retrieval** | [Rank-BM25](https://github.com/dorianbrown/rank_bm25) |
| **Re-Ranking** | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) (`ms-marco-TinyBERT-L-2-v2`, ONNX Runtime) |
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
├── Dockerfile                  # Multi-stage container (pure ONNX, pre-cached models)
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
│   ├── embedder.py             # FastEmbed ONNX-accelerated vector embedder
│   ├── vector_store.py         # FAISS vector store indexing & similarity search
│   ├── retriever.py            # Hybrid RRF retriever & pipeline coordinator
│   └── reranker.py             # FlashRank ONNX-accelerated re-ranker compressor
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

Set your Gemini API key in `.env`:
* **`GEMINI_API_KEY`**: Obtain from [Google AI Studio](https://aistudio.google.com/apikey) (Free tier).

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Ingest Notes

Place your handwritten PDF notes in `Data/input_pdfs/` (e.g. `Z-Transform.pdf`):

```bash
# Ingest all PDFs in Data/input_pdfs/:
python main.py
```

### 6. Run the Web Application

```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 💻 CLI Usage

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

```bash
# Run evaluation across all three retrieval modes:
python -m evaluation.benchmark

# Launch MLflow UI to compare experiment runs:
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## 🚀 Deployment Guide

### Deploying to Render (Free Tier, 512MB RAM)

RagifyNotes is fully optimized for **Render's Free Web Service** tier ($0/month, 512MB RAM ceiling) thanks to FlashRank's lightweight ONNX re-ranking and pre-built FAISS vector caching.

1. Fork or push this repository to your GitHub account.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** $\rightarrow$ **Web Service**.
3. Connect your `Ragify-Notes` GitHub repository.
4. Render automatically detects the root `Dockerfile`:
   * **Runtime**: Docker
   * **Instance Type**: Free (0.1 CPU, 512 MB RAM)
5. Under **Environment Variables**, add:
   * **`GEMINI_API_KEY`**: Your Google AI Studio API key.
   * **`PORT`**: `5000` (Render binds to `PORT` automatically).
6. Click **Deploy Web Service**. Once built, the application is live and accessible worldwide.
   * **Live Deployment URL**: [https://ragify-notes.onrender.com/](https://ragify-notes.onrender.com/)

---

### Deploying to Hugging Face Spaces (Docker SDK)

You can also deploy to Hugging Face Spaces:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Docker** $\rightarrow$ **Blank**.
3. In Space **Settings** $\rightarrow$ **Variables and Secrets**, add:
   * **`GEMINI_API_KEY`**: Your Google AI Studio API key.
4. Clone the space repo and push, or link your GitHub repo:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```
5. Hugging Face Spaces will build the Docker container and launch the web UI automatically.

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

## 👨‍💻 Author

**Raunak Suman**  
*AI / Generative AI Engineer*  
*Specializing in RAG Architectures, Multimodal AI, NLP & Full-Stack AI Applications*

* GitHub: [@RawKnack](https://github.com/RawKnack)
* Repository: [Ragify-Notes](https://github.com/RawKnack/Ragify-Notes)
