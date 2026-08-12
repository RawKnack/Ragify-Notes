from langchain_huggingface import HuggingFaceEndpointEmbeddings
from config.settings import HUGGINGFACE_API_KEY, EMBED_MODEL_NAME, EMBED_DIMENSION

embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBED_MODEL_NAME,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
)


def embed_text(text: str) -> list[float]:
    return embeddings.embed_query(text)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec

    return chunks