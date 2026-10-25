"""
main.py — CLI entry point for the RagifyNotes ingestion and query pipeline.

Features:
- Incremental Page Caching: Saves extracted pages to disk after every page.
  If the process is stopped or encounters a rate limit error, re-running
  will automatically resume from where it left off without re-extracting completed pages!

INGESTION:
    python main.py                              # Ingest default PDFs
    python main.py --pdf "path/to/notes.pdf"    # Ingest specific PDF
    python main.py --pdf-dir "Data/input_pdfs"  # Ingest all PDFs in a directory

QUERY:
    python main.py --query "what is the Z-Transform?"
    python main.py --query "ROC properties" --mode hybrid_rerank
"""

import json
import os
import glob
import argparse

from src.preprocessing import pdf_to_images
from src.extract import extract_page
from src.structure import structure_page
from src.chunker import chunk_all_pages
from src.embedder import embed_chunks
from src.vector_store import create_and_store, get_collection_info
from src.retriever import query, pretty_print
from config.settings import OUTPUT_JSON, CHUNKS_JSON, INPUT_PDF_DIR, DEFAULT_MODE

PAGE_LIMIT = None  # Set a number for testing e.g. 5, or None for all pages

CACHE_FILE = "Data/output_json/extraction_cache.json"


def load_cache() -> dict:
    """Loads cached page extraction results."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: dict):
    """Saves cached page extraction results to disk."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def run_ingest(pdf_path: str = None, pdf_dir: str = None):
    """
    Run the ingestion pipeline on one or more PDFs with incremental caching and resume capability.
    """
    print("\n" + "=" * 60)
    print("INGEST PIPELINE STARTING")
    print("=" * 60)

    pdf_files = []
    if pdf_path:
        pdf_files = [pdf_path]
    elif pdf_dir:
        pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_dir}")
            return
    else:
        pdf_files = sorted(glob.glob(os.path.join(INPUT_PDF_DIR, "*.pdf")))
        if not pdf_files:
            print(f"❌ No PDF files found in {INPUT_PDF_DIR}")
            print(f"   Place your PDFs in {INPUT_PDF_DIR}/ and try again")
            return

    print(f"\n📂 Found {len(pdf_files)} PDF(s) to process:")
    for f in pdf_files:
        print(f"   • {os.path.basename(f)}")

    cache = load_cache()
    all_pages = []
    all_chunks = []

    for pdf_idx, pdf_file in enumerate(pdf_files):
        pdf_name = os.path.basename(pdf_file)
        print(f"\n{'─'*60}")
        print(f"📄 Processing PDF {pdf_idx+1}/{len(pdf_files)}: {pdf_name}")
        print(f"{'─'*60}")

        print("\n  Step 1: Converting PDF to images...")
        images = pdf_to_images(pdf_file)
        total = len(images)
        print(f"     {total} pages found")

        if PAGE_LIMIT:
            images = images[:PAGE_LIMIT]

        print("\n  Step 2: Extracting text with Gemini 3.8 Flash...")
        pdf_pages = []
        skipped_count = 0

        for i, img in enumerate(images):
            page_key = f"{pdf_name}_page_{i+1}"
            print(f"\n     Processing page {i+1}/{len(images)}...", end="")

            # Check if this page was already extracted in a previous run
            if page_key in cache:
                raw_text = cache[page_key]
                print(" ⚡ [Cached]", end="")
            else:
                raw_text = extract_page(img, i + 1)
                if raw_text:  # Save to disk cache immediately if successfully extracted
                    cache[page_key] = raw_text
                    save_cache(cache)

            if raw_text == "[SKIP]" or not raw_text.strip():
                skipped_count += 1
                continue

            structured = structure_page(raw_text, i + 1)
            structured["metadata"]["source_pdf"] = pdf_name
            pdf_pages.append(structured)
            print(f" ✓ {structured['metadata']['word_count']} words")

        print(f"\n     📊 {len(pdf_pages)} content pages extracted, {skipped_count} low-content pages skipped")
        all_pages.extend(pdf_pages)

        print(f"\n  Step 3: Chunking pages from {pdf_name}...")
        pdf_chunks = chunk_all_pages(pdf_pages)
        print(f"     {len(pdf_pages)} pages → {len(pdf_chunks)} chunks")

        for chunk in pdf_chunks:
            chunk["metadata"]["source_pdf"] = pdf_name

        all_chunks.extend(pdf_chunks)

    # Save final page & chunk JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=4, ensure_ascii=False)
    print(f"\n📝 Page JSON saved to {OUTPUT_JSON}")

    saveable = [{k: v for k, v in c.items() if k != "embedding"} for c in all_chunks]
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(saveable, f, indent=4, ensure_ascii=False)
    print(f"📝 Chunks JSON saved to {CHUNKS_JSON}")

    print(f"\n📊 Total: {len(all_pages)} pages → {len(all_chunks)} chunks across {len(pdf_files)} PDFs")

    print("\nStep 4: Generating embeddings via HuggingFace API...")
    embedded = embed_chunks(all_chunks)

    print("\nStep 5: Storing in FAISS...")
    create_and_store(embedded, recreate=True)

    info = get_collection_info()
    print("\n" + "=" * 60)
    print("INGEST COMPLETE!")
    print(f"  • Total documents in FAISS: {info.get('total_documents', 0)}")
    print(f"  • Vector dimension:         {info.get('dimension', 0)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RagifyNotes RAG Engine")
    parser.add_argument("--pdf", type=str, help="Path to a single PDF to ingest")
    parser.add_argument("--pdf-dir", type=str, help="Path to a directory of PDFs to ingest")
    parser.add_argument("--query", type=str, help="Run a query against the index")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["naive", "hybrid", "hybrid_rerank"],
        help=f"Retrieval mode (default: {DEFAULT_MODE})",
    )
    parser.add_argument("--section", type=str, default=None, help="Filter search by section tag")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    if args.query:
        print(f"\n🔍 Querying: '{args.query}' (mode={args.mode})...\n")
        results = query(
            args.query, mode=args.mode, top_k=args.top_k, section_filter=args.section
        )
        pretty_print(results)
    else:
        run_ingest(pdf_path=args.pdf, pdf_dir=args.pdf-dir if hasattr(args, 'pdf-dir') else getattr(args, 'pdf_dir', None))


if __name__ == "__main__":
    main()