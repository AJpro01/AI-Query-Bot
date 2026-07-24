"""
pipeline.py
-----------
The full MVP, start to finish:

    PDF file --> parse --> chunk --> embed & store --> retrieve --> answer

Run this once per book to ingest it, then ask questions against it.

Usage:
    python pipeline.py list-chapters path/to/book.pdf
    python pipeline.py ingest path/to/book.pdf "Book Title"
    python pipeline.py ingest-chapter path/to/book.pdf "Book Title" <chapter_index>
    python pipeline.py fix-chapters path/to/book.pdf "Book Title"
    python pipeline.py sample-chunks "Book Title" [n]
    python pipeline.py caption-images path/to/book.pdf "Book Title" <chapter_index>
    python pipeline.py reset-library
    python pipeline.py ask "your question here"
"""

import sys
import json
import os
from src.chunker import build_chunks
from src.schema import BookChunk

LIBRARY_PATH = "library_chunks.json"


def ingest(pdf_path: str, book_title: str, page_offset: int = 0, skip_front_matter: bool = True):
    """
    skip_front_matter: if True (default) and the PDF has chapter info
    available, trims off title page/copyright/dedication/preface/TOC
    pages before parsing -- these pages are unlikely to ever match a real
    question, and just add noise and wasted embedding time. Set to False
    to ingest every page exactly as-is (e.g. if a book's front matter
    genuinely contains content you want searchable).
    """
    actual_pdf_path = pdf_path
    temp_trimmed_path = None

    if skip_front_matter and page_offset == 0:  # don't double-trim a chapter extract that's already scoped
        from src.chapter_utils import get_content_start_page, extract_chapter_pdf
        import fitz
        content_start = get_content_start_page(pdf_path)
        if content_start > 1:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
            temp_trimmed_path = "_front_matter_trimmed.pdf"
            extract_chapter_pdf(pdf_path, content_start, total_pages, temp_trimmed_path)
            actual_pdf_path = temp_trimmed_path
            page_offset = content_start - 1
            print(f"Skipping {content_start - 1} front-matter page(s) (title/copyright/preface/etc.)")

    print(f"Parsing and chunking {actual_pdf_path}...")
    chunks = build_chunks(actual_pdf_path, book_title, page_offset=page_offset)
    print(f"Produced {len(chunks)} chunks")

    if temp_trimmed_path:
        os.remove(temp_trimmed_path)

    # Merge into the shared multi-book library, replacing prior entries for
    # this book so re-ingestion is idempotent and doesn't duplicate data.
    existing = []
    if os.path.exists(LIBRARY_PATH):
        with open(LIBRARY_PATH) as f:
            existing = json.load(f)
    existing = [c for c in existing if c["metadata"]["book_title"] != book_title]
    new_chunk_dicts = [c.model_dump() for c in chunks]
    existing.extend(new_chunk_dicts)
    with open(LIBRARY_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print("Embedding and storing this book (other books in the library are untouched)...")
    from src.embedder import add_or_update_book
    add_or_update_book(new_chunk_dicts, book_title)

    print(f"\nIngestion complete. Library now has {len(existing)} chunks across "
          f"{len(set(c['metadata']['book_title'] for c in existing))} book(s).")
    print('You can now run: python pipeline.py ask "your question"')


def list_chapters_command(pdf_path: str):
    from src.chapter_utils import list_chapters
    chapters = list_chapters(pdf_path)
    print(f"\n{len(chapters)} chapter(s)/section(s) found:\n")
    for i, ch in enumerate(chapters):
        indent = "  " * (ch["level"] - 1)
        print(f"{i}: {indent}{ch['title']}  (pages {ch['start_page']}-{ch['end_page']})")
    print("\nUse the index shown above with: python pipeline.py ingest-chapter <pdf> \"<Book Title>\" <index>")


def ingest_chapter(pdf_path: str, book_title: str, chapter_index: int):
    """
    Extracts ONE chapter's pages into a small standalone PDF, then runs
    the normal ingest pipeline on just that -- much faster than ingesting
    a whole 700-page book, and the right way to test/iterate before
    committing to a full-book ingest.
    """
    from src.chapter_utils import list_chapters, extract_chapter_pdf

    chapters = list_chapters(pdf_path)
    if chapter_index >= len(chapters):
        print(f"Chapter index {chapter_index} out of range (found {len(chapters)} chapters). "
              f"Run `python pipeline.py list-chapters {pdf_path}` to see valid indices.")
        return

    ch = chapters[chapter_index]
    extracted_path = f"_chapter_{chapter_index}_extract.pdf"
    print(f"Extracting '{ch['title']}' (pages {ch['start_page']}-{ch['end_page']}) to {extracted_path}...")
    extract_chapter_pdf(pdf_path, ch["start_page"], ch["end_page"], extracted_path)

    # Tag the book title with the chapter so it doesn't collide with (or
    # silently overwrite) a full-book or other-chapter ingest of the same book.
    chapter_book_title = f"{book_title} — {ch['title']}"
    # page_offset makes citations report the ORIGINAL book's page numbers
    # (e.g. page 6) instead of the extracted chapter file's own page 1 --
    # otherwise every chapter-scoped ingest would cite "page 1, 2, 3..."
    # regardless of where that chapter actually sits in the real book.
    ingest(extracted_path, chapter_book_title, page_offset=ch["start_page"] - 1)

    os.remove(extracted_path)


def sample_chunks_command(book_title: str, n: int = 10):
    """
    Prints N real, spread-out chunks from a book's ingested library --
    chunk_id, chapter/page, and a text snippet -- so you can pick real
    questions to write against real chunk_ids, instead of the eval set
    silently pointing at chunk IDs from the old synthetic test book (the
    #1 cause of a 0% Hit Rate / 0.000 MRR eval run).

    Spreads picks evenly across the book (by page order) rather than
    taking the first N, so your eval set isn't accidentally all from one
    chapter.
    """
    import json
    if not os.path.exists(LIBRARY_PATH):
        print(f"{LIBRARY_PATH} not found -- ingest a book first.")
        return

    with open(LIBRARY_PATH) as f:
        all_chunks = json.load(f)
    book_chunks = [c for c in all_chunks if c["metadata"]["book_title"] == book_title]
    if not book_chunks:
        titles = sorted(set(c["metadata"]["book_title"] for c in all_chunks))
        print(f"No chunks found for '{book_title}'. Available book titles: {titles}")
        return

    book_chunks.sort(key=lambda c: c["metadata"]["page_number"])
    n = min(n, len(book_chunks))
    step = max(1, len(book_chunks) // n)
    sample = book_chunks[::step][:n]

    print(f"{n} sample chunk(s) from '{book_title}' (spread across the book):\n")
    for c in sample:
        m = c["metadata"]
        print(f"chunk_id: {c['chunk_id']}")
        print(f"  {m['chapter']} | p.{m['page_number']}")
        print(f"  {c['text'][:200]}...")
        print()

    print("Copy real chunk_id values above into eval_set.json entries, e.g.:")
    print("""[
  {
    "question": "your question about this chunk's content",
    "ground_truth_answer": "the correct answer, in your own words",
    "ground_truth_chunk_id": "<paste a real chunk_id from above>"
  }
]""")


def ask(question: str, force_all_books: bool = False):
    from src.retriever import HybridRetriever
    from src.answer_synth import generate_answer
    from src.book_router import build_book_profiles, route_query
    from src.chapter_router import route_to_chapter

    with open(LIBRARY_PATH) as f:
        library_chunks = json.load(f)

    book_filter = None
    chapter_filter = None
    if not force_all_books:
        profiles = build_book_profiles(library_chunks, top_n=60)
        book_filter = route_query(question, profiles)
        if book_filter:
            print(f"[router] scoping search to: {book_filter}")
            # Chapter-level routing only makes sense once we've committed to
            # a book -- reuses local embeddings (zero extra API cost), same
            # "don't guess wrong" fallback as book-level routing.
            chapter_filter = route_to_chapter(question, book_filter, LIBRARY_PATH)
            if chapter_filter:
                print(f"[chapter router] further scoping to: {chapter_filter}")
            else:
                print("[chapter router] not confident enough to scope to one chapter -- searching whole book")
        else:
            print("[router] not confident enough to scope to one book -- searching all books")

    retriever = HybridRetriever(LIBRARY_PATH)
    chunks = retriever.search(question, top_k=6, book_filter=book_filter, chapter_filter=chapter_filter)

    print(f"\nQuestion: {question}\n")
    print("Retrieved context:")
    for c in chunks:
        m = c["metadata"]
        print(f"  - [{m['book_title']}] {m['chapter']} | {m['section']} | p.{m['page_number']}  (RRF: {c['rrf_score']})")
    print()

    answer = generate_answer(question, chunks)
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "list-chapters":
        list_chapters_command(sys.argv[2])
    elif command == "sample-chunks":
        book_title = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        sample_chunks_command(book_title, n)
    elif command == "reset-library":
        import shutil
        if os.path.exists(LIBRARY_PATH):
            os.remove(LIBRARY_PATH)
            print(f"Removed {LIBRARY_PATH}")
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")
            print("Removed chroma_db/")
        if os.path.exists("concept_cache.json"):
            os.remove("concept_cache.json")
            print("Removed concept_cache.json")
        print("Library fully reset. Re-ingest your books to start fresh.")
    elif command == "fix-chapters":
        pdf_path = sys.argv[2]
        book_title = sys.argv[3]
        from src.chapter_utils import reassign_chapters
        reassign_chapters(pdf_path, book_title)
    elif command == "ingest":
        pdf_path = sys.argv[2]
        book_title = sys.argv[3] if len(sys.argv) > 3 else "Untitled Book"
        ingest(pdf_path, book_title)
    elif command == "ingest-chapter":
        pdf_path = sys.argv[2]
        book_title = sys.argv[3]
        chapter_index = int(sys.argv[4])
        ingest_chapter(pdf_path, book_title, chapter_index)
    elif command == "ask":
        question = sys.argv[2]
        ask(question)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
