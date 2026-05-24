

import json
import os
import argparse

from src.preprocessing import pdf_to_images
from src.extract import extract_page     # swap to extract.py or extract_haiku.py
from src.structure import structure_page
from src.chunker import chunk_all_pages
from src.embedder import embed_chunks
from src.vector_store import create_collection, upsert_chunks, get_collection_info
from src.retriever import query, pretty_print


PDF_PATH    = r"D:\VS Code\HandwrittenNotesPipeline\Data\input_pdfs\Z-Transform.pdf"
OUTPUT_JSON = "Data/output_json/output.json"
CHUNKS_JSON = "Data/output_json/chunks.json"
PAGE_LIMIT  = None   # None = all pages, or set a number for testing e.g. 2


def run_ingest():
    print("\n" + "="*60)
    print("INGEST PIPELINE STARTING")
    print("="*60)

    # STEP 1: PDF to images
    print("\nStep 1: Converting PDF to images...")
    images = pdf_to_images(PDF_PATH)
    total = len(images)
    print(f"   {total} pages found")

    if PAGE_LIMIT:
        images = images[:PAGE_LIMIT]

    # STEP 2: Extract and structure
    print("\nStep 2: Extracting text with LLM...")
    all_pages = []

    for i, img in enumerate(images):
        print(f"\n   Processing page {i+1}/{len(images)}...")
        raw_text   = extract_page(img, i + 1)
        structured = structure_page(raw_text, i + 1)
        all_pages.append(structured)
        print(f"   Page {i+1} done - {structured['metadata']['word_count']} words")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=4, ensure_ascii=False)
    print(f"\nPage JSON saved to {OUTPUT_JSON}")

    # STEP 3: Chunk
    print("\nStep 3: Chunking pages...")
    all_chunks = chunk_all_pages(all_pages)
    print(f"   {len(all_pages)} pages -> {len(all_chunks)} chunks total")

    saveable = [{k: v for k, v in c.items() if k != "embedding"} for c in all_chunks]
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(saveable, f, indent=4, ensure_ascii=False)
    print(f"Chunks JSON saved to {CHUNKS_JSON}")

    # STEP 4: Embed
    print("\nStep 4: Generating embeddings...")
    embedded = embed_chunks(all_chunks)

    # STEP 5: Store in Qdrant
    print("\nStep 5: Storing in Qdrant...")
    create_collection(recreate=True)
    upsert_chunks(embedded)

    info = get_collection_info()
    print(f"\nCollection stats:")
    print(f"   Total points: {info['total_points']}")
    print(f"   Vector size:  {info['vector_size']}")

    print("\nINGEST COMPLETE - your notes are now searchable!")
    print('Run: python main.py --query "what is the Z-Transform?"')


def run_query(question: str, section: str = None):
    print("\nQUERY MODE")
    result = query(
        question=question,
        top_k=5,
        filter_confidence="high",
        filter_section=section,
        verbose=True,
    )
    pretty_print(result)

    result_path = "Data/output_json/last_query.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Query result saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, default=None)
    parser.add_argument("--section", "-s", type=str, default=None)
    args = parser.parse_args()

    if args.query:
        run_query(args.query, section=args.section)
    else:
        run_ingest()