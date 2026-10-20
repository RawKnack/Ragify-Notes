from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBED_MODEL_NAME

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    return embeddings.embed_query(text)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    return chunks