"""
embedder.py
-----------
Takes chunks_output.json (from chunker.py) and:
  1. Converts each chunk's text into a dense vector using a local
     sentence-transformer model (runs on CPU, no API key needed).
  2. Stores those vectors + the chunk metadata in a local Chroma database
     on disk, so retrieval later doesn't need to re-embed anything.

MODEL CHOICE: bge-small-en-v1.5 instead of bge-large.
On CPU (no GPU), bge-large is noticeably slower to embed at book scale
(700+ pages -> thousands of chunks). bge-small trades a small amount of
retrieval quality for a large speed win, which matters when you don't have
a GPU and a week to submit. Swap to bge-large in EMBED_MODEL_NAME below if
you have time to spare and want to squeeze out better retrieval quality.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "book_chunks"

# NOTE ON MODEL CHOICE: switched from bge-small to all-MiniLM-L6-v2.
# On a 700-page book on CPU, bge-small was noticeably slow to embed
# thousands of chunks. all-MiniLM-L6-v2 is roughly 3-5x faster on CPU with
# a modest retrieval-quality tradeoff -- a better fit given your no-GPU
# constraint and deadline. If retrieval quality feels weak once you're
# testing on a real book, bge-small (or bge-base) is the first thing to
# try switching back to, but expect ingestion time to go up accordingly.
# IMPORTANT: if you change this, you must re-ingest every book (embeddings
# from different models aren't compatible with each other in the same
# Chroma collection) -- delete the chroma_db/ folder and re-run ingestion.


def load_chunks(json_path: str) -> list[dict]:
    with open(json_path) as f:
        return json.load(f)


def add_or_update_book(chunks: list[dict], book_title: str):
    """
    Embeds and stores chunks for ONE book, without wiping other books
    already in the collection. Removes any prior entries for this same
    book_title first, so re-ingesting a book you've updated doesn't leave
    stale duplicate chunks behind.
    """
    print(f"Loading embedding model ({EMBED_MODEL_NAME})... this can take a minute on first run.")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Remove any existing chunks for this book so re-ingestion is idempotent
    collection.delete(where={"book_title": book_title})

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks for '{book_title}' (CPU)...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=64)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"Stored {len(chunks)} chunks for '{book_title}' (collection now has {collection.count()} total chunks)")
    return collection


def build_vector_store(chunks_json_path: str = "library_chunks.json"):
    """
    Rebuilds the ENTIRE collection from a full library file. Use this for a
    from-scratch rebuild; use add_or_update_book() for incremental ingestion
    of one new book at a time (this is what pipeline.py calls).
    """
    chunks = load_chunks(chunks_json_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_json_path}")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # group by book so add_or_update_book's per-book delete-then-add logic
    # behaves consistently even on a full rebuild
    by_book: dict[str, list[dict]] = {}
    for c in chunks:
        by_book.setdefault(c["metadata"]["book_title"], []).append(c)

    for book_title, book_chunks in by_book.items():
        add_or_update_book(book_chunks, book_title)


if __name__ == "__main__":
    import sys
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "library_chunks.json"
    build_vector_store(chunks_path)
