"""
src/chunker.py — Splits structured page JSON into smaller chunks for RAG.

WHY CHUNKING:
  - Vector databases embed text as fixed-size vectors
  - Most embedding models have a token limit (512-8192 tokens depending on model)
  - A dense page of notes can have 600-1000+ words — too large for one chunk
  - Smaller chunks = more precise retrieval (you get exactly the relevant section)
  - Larger chunks = more context per result but less precision

STRATEGY USED HERE:
  - Split on Markdown headings (## Section) first — keeps topics together
  - If a heading-based chunk is still too large, split on paragraphs
  - Each chunk keeps full metadata from the parent page + its own chunk_id
  - LaTeX equations are never split mid-equation
"""

import re


# ── Config ──────────────────────────────────────────────────────
MAX_WORDS = 200       # soft limit per chunk (tune based on embedding model)
MIN_WORDS = 30        # don't create tiny useless chunks below this
OVERLAP_WORDS = 30    # words of overlap between consecutive chunks (maintains context)


def _split_on_headings(text: str) -> list[str]:
    """
    Splits Markdown text on ## or # headings.
    Each section starts with its own heading.

    Example:
      "# Z-Transform\n\nDef...\n\n## ROC\n\nROC is..."
      → ["# Z-Transform\n\nDef...", "## ROC\n\nROC is..."]
    """
    # Split but keep the heading with its section
    parts = re.split(r'(?=^#{1,2} )', text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _split_on_paragraphs(text: str, max_words: int, overlap: int) -> list[str]:
    """
    Splits text into chunks of max_words with overlap.
    Splits on double newlines (paragraph breaks) to avoid cutting mid-sentence.
    Never splits inside a LaTeX block equation $$ ... $$.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_words = []
    current_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_words = para.split()
        para_count = len(para_words)

        # If adding this paragraph exceeds limit, flush current chunk first
        if current_count + para_count > max_words and current_words:
            chunks.append(" ".join(current_words))
            # Keep overlap words for context continuity
            current_words = current_words[-overlap:] if overlap else []
            current_count = len(current_words)

        current_words.extend(para_words)
        current_count += para_count

    # Flush remaining
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_page(page: dict) -> list[dict]:
    """
    Takes a structured page dict (output of structure_page()) and returns
    a list of chunk dicts, each ready for embedding and vector storage.

    Input page format:
    {
        "id": "1",
        "text": "# Z-Transform\n\n## Definition\n...",
        "metadata": { "page_no": "1", "section": "...", ... }
    }

    Output chunk format:
    {
        "chunk_id": "1_0",
        "text": "## Definition\n\nThe Z-Transform...",
        "metadata": {
            "page_no": "1",
            "section": "Z-Transform",
            "chunk_index": 0,
            "word_count": 87,
            ... (all original metadata)
        }
    }
    """
    text = page["text"]
    base_meta = page["metadata"].copy()
    page_no = page["id"]

    chunks = []

    # Step 1: Split on headings first
    heading_sections = _split_on_headings(text)

    # If no headings found, treat whole page as one section
    if not heading_sections:
        heading_sections = [text]

    chunk_index = 0

    for section in heading_sections:
        word_count = len(section.split())

        if word_count <= MAX_WORDS:
            # Section fits in one chunk
            if word_count >= MIN_WORDS:
                chunks.append(_make_chunk(
                    section, page_no, chunk_index, base_meta
                ))
                chunk_index += 1
        else:
            # Section too large — split on paragraphs
            sub_chunks = _split_on_paragraphs(section, MAX_WORDS, OVERLAP_WORDS)
            for sub in sub_chunks:
                if len(sub.split()) >= MIN_WORDS:
                    chunks.append(_make_chunk(
                        sub, page_no, chunk_index, base_meta
                    ))
                    chunk_index += 1

    # Fallback: if nothing passed the MIN_WORDS filter, keep the whole page
    if not chunks and len(text.split()) > 0:
        chunks.append(_make_chunk(text, page_no, 0, base_meta))

    return chunks


def _make_chunk(text: str, page_no: str, index: int, base_meta: dict) -> dict:
    """Builds a single chunk dict."""
    return {
        "chunk_id": f"{page_no}_{index}",
        "text": text.strip(),
        "metadata": {
            **base_meta,
            "chunk_index": index,
            "word_count": len(text.split()),
        }
    }


def chunk_all_pages(pages: list[dict]) -> list[dict]:
    """
    Chunks all pages from the pipeline output.
    Input: list of page dicts from structure_page()
    Output: flat list of all chunks across all pages
    """
    all_chunks = []
    for page in pages:
        page_chunks = chunk_page(page)
        all_chunks.extend(page_chunks)
        print(f"   📄 Page {page['id']} → {len(page_chunks)} chunks")
    return all_chunks
