# Ragify Notes

> Transform handwritten engineering notes into an AI-powered, queryable knowledge base using Retrieval-Augmented Generation (RAG).

## Overview

Ragify Notes is a Generative AI application that enables students to interact with their handwritten engineering notes through natural language. Instead of manually searching through notebooks or PDFs, students can ask questions and receive context-aware answers grounded exclusively in their professors' notes.

The system combines LLM-based OCR, hybrid retrieval, and Retrieval-Augmented Generation (RAG) to create an intelligent study assistant that minimizes hallucinations while keeping responses faithful to the original course material.

---

## Features

* AI-powered question answering over handwritten engineering notes
* LLM-based OCR for handwritten text extraction
* PDF parsing using PyMuPDF (fitz)
* Intelligent overlapping document chunking

  * Maximum chunk size: 200 words
  * Minimum chunk size: 30 words
  * Overlap: 30 words
* Semantic vector search with Qdrant
* BM25 lexical retrieval
* Hybrid Search using Reciprocal Rank Fusion (RRF, k = 60)
* FastAPI backend with a lightweight web interface
* Context-grounded responses with source attribution
* Mathematical equation rendering
* Optimized for engineering coursework and exam preparation

---

## Architecture

```text
Handwritten PDF Notes
          │
          ▼
  PyMuPDF (fitz)
          │
          ▼
   LLM-based OCR
          │
          ▼
Text Cleaning & Preprocessing
          │
          ▼
 Overlapping Chunking
 (200 max | 30 min | 30 overlap)
          │
          ▼
 Embedding Generation
          │
          ▼
       Qdrant
(Vector Database)
          │
          ▼
     User Query
          │
     ┌───────────────┐
     │ Hybrid Search │
     └───────────────┘
      ▲             ▲
   BM25         Semantic
      └─────┬───────┘
            ▼
   Reciprocal Rank Fusion
         (k = 60)
            │
            ▼
 Retrieved Context
            │
            ▼
     OpenRouter LLM
            │
            ▼
 Grounded Response
```

---

## Tech Stack

### AI / Generative AI

* Retrieval-Augmented Generation (RAG)
* Large Language Models (LLMs)
* OpenRouter
* Prompt Engineering

### NLP

* Semantic Search
* BM25
* Hybrid Retrieval
* Embeddings
* Document Chunking
* OCR

### Backend

* Python
* FastAPI

### Vector Database

* Qdrant

### Document Processing

* PyMuPDF (fitz)

---

## Retrieval Pipeline

The retrieval system combines the strengths of lexical and semantic retrieval.

### Semantic Retrieval

* Vector embeddings generated for every document chunk
* Stored inside Qdrant
* Retrieves semantically similar passages

### Lexical Retrieval

* BM25 keyword search
* Captures exact terminology and formulas

### Hybrid Retrieval

Results from both retrieval methods are merged using **Reciprocal Rank Fusion (RRF)** with **k = 60**, producing more relevant and robust rankings than either retrieval strategy alone.

---

## Project Workflow

1. Upload handwritten engineering notes.
2. Extract PDF pages using PyMuPDF.
3. Convert handwritten text using LLM-based OCR.
4. Clean and preprocess extracted text.
5. Create overlapping chunks.
6. Generate embeddings.
7. Store vectors in Qdrant.
8. Receive a natural language query.
9. Perform hybrid retrieval (BM25 + Semantic Search).
10. Merge rankings using RRF.
11. Generate a grounded answer with an LLM.
12. Display retrieved sources alongside the response.

---

## Example Queries

* What is the Z-Transform?
* Explain the Initial Value Theorem.
* Derive the Fourier Transform.
* What are the properties of ROC?
* Compare DTFT and DFT.
* Explain Parseval's theorem.

---

## Why Ragify Notes?

Traditional AI assistants rely on general knowledge, which may not match the material taught in a specific course. Ragify Notes grounds every response in the professor's handwritten notes, ensuring students receive accurate, course-specific explanations that align with classroom instruction and examination content.

---

## Future Improvements

* Multi-document support
* Course-wise knowledge bases
* PDF and image upload through the UI
* Voice-based question answering
* Citation highlighting within notes
* Streaming responses
* User authentication
* Chat history
* Multi-modal RAG
* Agentic workflows

---

## Author

**Raunak Suman**

AI Engineer | Generative AI | RAG | NLP | FastAPI | LangChain | Machine Learning | Python
