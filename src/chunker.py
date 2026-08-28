from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]


def chunk_page(page: dict) -> list[dict]:
    text = page.get("text", "").strip()
    if not text:
        return []

    base_meta = page["metadata"].copy()
    page_no = page["id"]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    header_docs = markdown_splitter.split_text(text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    chunk_index = 0

    if header_docs:
        for doc in header_docs:
            sub_docs = char_splitter.split_text(doc.page_content)
            header_str = " > ".join([str(v) for k, v in doc.metadata.items() if k.startswith("header_")])
            
            for sub_text in sub_docs:
                if len(sub_text.strip()) < 30 and len(sub_docs) > 1:
                    continue
                
                chunk_meta = {
                    **base_meta,
                    **doc.metadata,
                    "section": header_str if header_str else base_meta.get("section", "General"),
                    "chunk_index": chunk_index,
                    "word_count": len(sub_text.split()),
                }
                
                chunks.append({
                    "chunk_id": f"{page_no}_{chunk_index}",
                    "text": sub_text.strip(),
                    "metadata": chunk_meta
                })
                chunk_index += 1
    else:
        raw_sub_docs = char_splitter.split_text(text)
        for sub_text in raw_sub_docs:
            chunk_meta = {
                **base_meta,
                "chunk_index": chunk_index,
                "word_count": len(sub_text.split()),
            }
            chunks.append({
                "chunk_id": f"{page_no}_{chunk_index}",
                "text": sub_text.strip(),
                "metadata": chunk_meta
            })
            chunk_index += 1

    return chunks


def chunk_all_pages(pages: list[dict]) -> list[dict]:
    all_chunks = []
    for page in pages:
        page_chunks = chunk_page(page)
        all_chunks.extend(page_chunks)
        print(f"   📄 Page {page['id']} → {len(page_chunks)} chunks")
    return all_chunks
