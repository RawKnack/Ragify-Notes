from sentence_transformers import SentenceTransformer

# Local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    """
    Embed a single query/text.
    Returns list[float]
    """
    embedding = model.encode(text)
    return embedding.tolist()


def embed_chunks(chunks: list[dict]):
    """
    Batch embed chunks locally.
    Adds embedding to each chunk.
    """

    print(f"\n🔢 Embedding {len(chunks)} chunks locally...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts).tolist()

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    print(f"✅ Embedded {len(chunks)} chunks")

    return chunks