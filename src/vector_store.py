from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

import uuid

# Local Qdrant storage
client = QdrantClient(path="./qdrant_storage")

COLLECTION_NAME = "notes_pipeline"

# all-MiniLM-L6-v2 = 384 dimensions
VECTOR_SIZE = 384


def create_collection(recreate: bool = False):
    """
    Creates the Qdrant collection.
    """

    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
            print(f"🗑️ Deleted existing collection '{COLLECTION_NAME}'")
        else:
            print(f"✅ Collection '{COLLECTION_NAME}' already exists")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(f"✅ Created collection '{COLLECTION_NAME}'")


def upsert_chunks(embedded_chunks: list[dict]):
    """
    Store embedded chunks in Qdrant.
    """

    points = []

    for chunk in embedded_chunks:

        if not chunk.get("embedding"):
            print(f"⚠️ Skipping chunk {chunk['chunk_id']} — no embedding")
            continue

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk["embedding"],
            payload={
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                **chunk["metadata"],
            }
        )

        points.append(point)

    batch_size = 100

    for i in range(0, len(points), batch_size):

        batch = points[i:i + batch_size]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )

        print(f"   ⬆️ Uploaded {i + len(batch)}/{len(points)} points")

    print(f"✅ Stored {len(points)} chunks in Qdrant")


def search(
    query_vector: list[float],
    top_k: int = 5,
    filter_confidence: str = None,
    filter_section: str = None,
) -> list[dict]:

    must_conditions = []

    if filter_confidence:
        must_conditions.append(
            FieldCondition(
                key="confidence",
                match=MatchValue(value=filter_confidence)
            )
        )

    if filter_section:
        must_conditions.append(
            FieldCondition(
                key="section",
                match=MatchValue(value=filter_section)
            )
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    return [
        {
            "chunk_id": r.payload.get("chunk_id"),
            "text": r.payload.get("text"),
            "score": round(r.score, 4),
            "metadata": {
                k: v for k, v in r.payload.items()
                if k not in ("chunk_id", "text")
            }
        }
        for r in results
    ]


def get_collection_info():

    info = client.get_collection(COLLECTION_NAME)

    return {
        "name": COLLECTION_NAME,
        "total_points": info.points_count,
        "vector_size": info.config.params.vectors.size,
        "distance": info.config.params.vectors.distance,
    }


def get_all_chunks() -> list[dict]:
    """
    Retrieve all stored chunks from Qdrant.
    Used to build the BM25 corpus for hybrid retrieval.

    Returns a list of dicts with chunk_id, text, and metadata.
    """
    info = client.get_collection(COLLECTION_NAME)
    total = info.points_count

    if total == 0:
        return []

    # Scroll through all points in the collection
    all_chunks = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in results:
            payload = point.payload
            all_chunks.append({
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text", ""),
                "metadata": {
                    k: v for k, v in payload.items()
                    if k not in ("chunk_id", "text")
                },
            })

        if next_offset is None:
            break
        offset = next_offset

    return all_chunks