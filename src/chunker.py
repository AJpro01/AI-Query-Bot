"""
chunker.py
----------
Takes the ParsedPage output from pdf_parser.py and produces a flat list of
BookChunk objects (schema.py) ready for embedding.

TWO-STAGE SPLITTING (this is the key idea worth understanding):

Stage A - Group by heading ("macro" split):
  Walk through every line in the book IN ORDER. Whenever we hit a line
  classified as a chapter/section heading, we update "current chapter" /
  "current section". Every body line gets tagged with whatever heading it's
  currently under. This is how a paragraph on page 114 knows it belongs to
  Chapter 3, Section 3.2 -- it inherits that from the last heading seen
  before it.

Stage B - Split by size ("micro" split):
  A single section might be 5 pages of body text -- too long for one chunk.
  RecursiveCharacterTextSplitter breaks it into ~800-character pieces with
  100-character overlap, so a sentence that spans a chunk boundary isn't
  lost, and no single embedding has to represent too much content at once.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.pdf_parser import parse_pdf
from src.schema import BookChunk, ChunkMetadata


def group_by_heading(parsed_pages, book_title: str):
    """
    Stage A: produces a list of (chapter, section, page_number, body_text)
    tuples, concatenating consecutive body lines that share the same
    chapter/section into a single block of text.
    """
    groups = []
    current_chapter = "Unknown"
    current_section = "Unknown"
    buffer_text = []
    buffer_page = None

    def flush():
        nonlocal buffer_page
        if buffer_text:
            groups.append({
                "chapter": current_chapter,
                "section": current_section,
                "page_number": buffer_page,
                "text": " ".join(buffer_text),
            })
            buffer_text.clear()
        buffer_page = None  # reset so the next group picks up its own starting page

    for page in parsed_pages:
        for line, level in page.lines:
            if level == "chapter":
                flush()
                current_chapter = line.text
                current_section = "Unknown"  # new chapter resets section
            elif level == "section":
                flush()
                current_section = line.text
            else:
                # body text -- accumulate it under the current heading
                if buffer_page is None:
                    buffer_page = page.page_number
                buffer_text.append(line.text)
    flush()
    return groups


def build_chunks(pdf_path: str, book_title: str, page_offset: int = 0) -> list[BookChunk]:
    """
    Stage A + Stage B combined: the full parse -> chunk pipeline for one book.
    """
    parsed_pages = parse_pdf(pdf_path, page_offset=page_offset)
    groups = group_by_heading(parsed_pages, book_title)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    chunks: list[BookChunk] = []
    for group_index, group in enumerate(groups):
        sub_texts = splitter.split_text(group["text"])
        for sub_index, sub_text in enumerate(sub_texts):
            chunk_id = f"{book_title.lower().replace(' ', '_')}_p{group['page_number']}_g{group_index}_c{sub_index}"
            chunks.append(BookChunk(
                chunk_id=chunk_id,
                text=sub_text,
                metadata=ChunkMetadata(
                    book_title=book_title,
                    chapter=group["chapter"],
                    section=group["section"],
                    page_number=group["page_number"],
                    chunk_index_on_page=sub_index,
                ),
            ))
    return chunks


if __name__ == "__main__":
    import sys
    import json
    import os

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test_book.pdf"
    book_title = sys.argv[2] if len(sys.argv) > 2 else "Test Book"
    library_path = "library_chunks.json"

    chunks = build_chunks(pdf_path, book_title)
    print(f"Produced {len(chunks)} chunks from {pdf_path}\n")

    for c in chunks:
        print(f"--- {c.chunk_id} ---")
        print(f"  Chapter: {c.metadata.chapter} | Section: {c.metadata.section} | Page: {c.metadata.page_number}")
        print(f"  Text: {c.text[:100]}...")
        print()

    # Merge into the shared multi-book library, replacing any prior entries
    # for this same book_title so re-running ingestion is idempotent.
    existing = []
    if os.path.exists(library_path):
        with open(library_path) as f:
            existing = json.load(f)
    existing = [c for c in existing if c["metadata"]["book_title"] != book_title]
    existing.extend([c.model_dump() for c in chunks])

    with open(library_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved {len(chunks)} chunks for '{book_title}' into {library_path} "
          f"(library now has {len(existing)} chunks across all books)")
