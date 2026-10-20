FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache ONNX models into the Docker image so there are 0 network downloads at runtime
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='/tmp')"

COPY . .

CMD ["python", "app.py"]
