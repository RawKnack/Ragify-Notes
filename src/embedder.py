from langchain_core.embeddings import Embeddings
from config.settings import EMBED_MODEL_NAME

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass

        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


class _LazyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return get_embeddings().embed_query(text)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def __getattr__(self, name):
        return getattr(get_embeddings(), name)


embeddings = _LazyEmbeddings()


def embed_text(text: str) -> list[float]:
    return get_embeddings().embed_query(text)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    vectors = get_embeddings().embed_documents(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    return chunks