import os
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding
from config.settings import EMBED_MODEL_NAME

_model = None


def get_embed_model():
    global _model
    if _model is None:
        cache_dir = os.environ.get("FASTEMBED_CACHE_DIR")
        _model = TextEmbedding(model_name=EMBED_MODEL_NAME, cache_dir=cache_dir)
    return _model


class _FastEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = get_embed_model()
        return [vec.tolist() for vec in model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        model = get_embed_model()
        return next(model.embed([text])).tolist()

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


embeddings = _FastEmbeddings()


def embed_text(text: str) -> list[float]:
    return embeddings.embed_query(text)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    return chunks