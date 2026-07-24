"""
chapter_router.py
--------------------
A "PageIndex-lite" node selector: instead of asking an LLM to reason over
a book's chapter tree at query time (real cost, real quota pressure),
this uses LOCAL embeddings -- the same sentence-transformers model
embedder.py already loads for dense chunk retrieval -- to pick the most
relevant chapter(s) for a query, at ZERO extra API cost.

WHAT THIS GIVES UP vs a true LLM-reasoning tree traversal: no
cross-reference following, no multi-hop reasoning across non-adjacent
sections, and a heavily paraphrased query with no semantic overlap with
any chapter title can fail to route confidently. WHAT IT STILL FIXES: the
exact failure mode that caused the CNN/ToC pollution bug -- a query never
gets scoped to "Table of Contents" or "Acknowledgments" by a real chapter
title's embedding similarity the way it could by raw keyword density.

DESIGN: same "don't guess wrong" philosophy as book_router.py -- if no
chapter clearly wins, return None and let the caller search the whole
book rather than confidently routing to the wrong place.
"""

import os
import json
import hashlib

CACHE_PATH = "chapter_embedding_cache.json"


def _get_model():
    from sentence_transformers import SentenceTransformer
    from src.embedder import EMBED_MODEL_NAME
    return SentenceTransformer(EMBED_MODEL_NAME)


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def _cache_key(book_title: str, chapter_titles: list) -> str:
    joined = book_title + "|" + "|".join(sorted(chapter_titles))
    return hashlib.md5(joined.encode()).hexdigest()


def get_chapters_for_book(book_title: str, library_path: str = "library_chunks.json") -> list:
    """Distinct chapter names for a book, in page order (same logic as app.py's version)."""
    with open(library_path) as f:
        all_chunks = json.load(f)
    book_chunks = [c for c in all_chunks if c["metadata"]["book_title"] == book_title]
    book_chunks.sort(key=lambda c: c["metadata"]["page_number"])
    seen = []
    for c in book_chunks:
        chapter = c["metadata"]["chapter"]
        if chapter not in seen and chapter != "Front/Back Matter":
            seen.append(chapter)
    return seen


def build_chapter_embeddings(book_title: str, library_path: str = "library_chunks.json") -> dict:
    """
    Returns {chapter_title: embedding_vector} for a book's chapters.
    Cached to disk since chapter titles don't change between queries --
    no reason to re-embed the same dozen titles on every question.
    """
    chapters = get_chapters_for_book(book_title, library_path)
    if not chapters:
        return {}

    cache = _load_cache()
    key = _cache_key(book_title, chapters)
    if key in cache:
        return {title: vec for title, vec in zip(chapters, cache[key])}

    model = _get_model()
    embeddings = model.encode(chapters, normalize_embeddings=True)
    cache[key] = embeddings.tolist()
    _save_cache(cache)
    return {title: vec for title, vec in zip(chapters, embeddings.tolist())}


def route_to_chapter(query: str, book_title: str, library_path: str = "library_chunks.json",
                      min_margin: float = 1.15, min_similarity: float = 0.25):
    """
    Returns the chapter title the query most likely belongs to, or None if
    not confident enough -- in which case the caller should search the
    whole book rather than risk scoping to the wrong chapter.
    """
    import numpy as np

    chapter_embeddings = build_chapter_embeddings(book_title, library_path)
    if not chapter_embeddings:
        return None

    model = _get_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    titles = list(chapter_embeddings.keys())
    vectors = np.array([chapter_embeddings[t] for t in titles])
    similarities = vectors @ query_vec  # cosine similarity, since both sides are normalized

    ranked_idx = np.argsort(-similarities)
    top_title = titles[ranked_idx[0]]
    top_score = similarities[ranked_idx[0]]
    second_score = similarities[ranked_idx[1]] if len(titles) > 1 else 0.0

    if top_score < min_similarity:
        return None  # not similar enough to anything -- don't guess
    if second_score > 0 and top_score < second_score * min_margin:
        return None  # too close a call between two chapters -- search both rather than pick wrong

    return top_title


if __name__ == "__main__":
    import sys
    book_title = sys.argv[1] if len(sys.argv) > 1 else None
    query = sys.argv[2] if len(sys.argv) > 2 else "describe the architecture of CNN"

    if not book_title:
        print("Usage: python chapter_router.py \"<Book Title>\" \"<query>\"")
        sys.exit(1)

    chapters = get_chapters_for_book(book_title)
    print(f"Chapters found for '{book_title}': {chapters}")

    result = route_to_chapter(query, book_title)
    print(f"\nQuery: {query!r}")
    print(f"Routed to chapter: {result or '(no confident match -- search whole book)'}")
