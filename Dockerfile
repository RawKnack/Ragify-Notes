FROM python:3.11-slim

WORKDIR /app

# System dependencies for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# No local model downloads needed — embeddings use HuggingFace Inference API
# Cross-encoder reranker model is downloaded on first use (~80MB)

ENV PORT=5000
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.getenv('PORT', '5000'); urllib.request.urlopen(f'http://localhost:{port}/api/stats')"

# Run the FastAPI server with dynamic PORT support
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-5000}"]
