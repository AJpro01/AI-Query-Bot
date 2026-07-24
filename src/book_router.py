"""
book_router.py
---------------
Decides which book(s) a query should be searched against, BEFORE running
the expensive hybrid retrieval step. At 2 books this saves little; at
10-20 books (the scale your original proposal targets) this is what stops
every query from searching the entire library.

WHY KEYWORD PROFILES INSTEAD OF AN LLM CALL:
Gemini's plan (and a lot of production RAG systems) uses an LLM or a
dedicated "semantic router" library for this. That works, but it's an
extra network call and extra latency on EVERY query, and it's harder to
test/debug than plain arithmetic. A simpler, fully local alternative:

  1. For each book, find its DISTINCTIVE words -- words that appear much
     more often in this book than in the library as a whole. "gradient"
     is distinctive to a deep learning book; "the" is not distinctive to
     anything.
  2. For a new query, count how many of its words match each book's
     distinctive-word set. Route to whichever book scores highest.

This is essentially a simplified TF-IDF comparison. It's not as smart as
an LLM at handling paraphrased or implicit queries, but it's free, instant,
and fully explainable -- good enough for a router whose only job is
"narrow the search space," not "understand the query" (that's still the
retriever's job).

FALLBACK: if no book scores meaningfully higher than the others (e.g. a
genuinely cross-book comparative question), don't force a choice --
return None and let the caller search across all books instead of guessing
wrong and hiding the right answer.
"""

import json
import re
from collections import Counter, defaultdict

STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "in", "on", "for",
    "that", "this", "it", "as", "by", "with", "or", "be", "at", "from",
    "which", "each", "can", "will", "its", "into", "used", "also",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def build_book_profiles(chunks: list[dict], top_n: int = 60) -> dict:
    """
    Returns {book_title: set_of_distinctive_words}. A word is distinctive
    to a book if its frequency IN that book is much higher than its
    average frequency ACROSS all books.

    NOTE ON top_n: this needs to scale with book size. On a tiny 4-chunk
    test book, top_n=25 misses real topic words simply because the whole
    book only has ~60 unique content words total, and one topic can crowd
    out another in the ranking. On a real 700-page book with thousands of
    unique terms, top_n=60 would be a tiny fraction of vocabulary and is
    too small -- raise this to 150-300 once you're routing across your
    actual Goodfellow/Bishop/Jurafsky-Martin library, and re-check routing
    accuracy on a handful of real questions per book.
    """
    per_book_counts: dict[str, Counter] = defaultdict(Counter)
    for c in chunks:
        book = c["metadata"]["book_title"]
        per_book_counts[book].update(_tokenize(c["text"]))

    global_counts = Counter()
    for counts in per_book_counts.values():
        global_counts.update(counts)

    profiles = {}
    for book, counts in per_book_counts.items():
        book_total = sum(counts.values()) or 1
        scored = []
        for word, count in counts.items():
            global_freq = global_counts[word] or 1
            # How concentrated is this word in THIS book vs the whole library?
            distinctiveness = (count / book_total) / (global_freq / sum(global_counts.values()))
            scored.append((word, distinctiveness * count))  # weight by raw count too, avoid rare-word noise
        scored.sort(key=lambda x: x[1], reverse=True)
        profiles[book] = set(word for word, _ in scored[:top_n])
    return profiles


def route_query(query: str, profiles: dict, min_margin: float = 1.5):
    """
    Returns the book_title the query most likely belongs to, or None if
    the router isn't confident enough (e.g. a cross-book question) --
    in which case the caller should search all books rather than guess.
    """
    query_words = set(_tokenize(query))
    if not query_words:
        return None

    scores = {book: len(query_words & words) for book, words in profiles.items()}
    if all(s == 0 for s in scores.values()):
        return None  # no signal at all -- don't guess

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_book, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # Require the top book to clearly beat the runner-up, not just barely win
    if second_score > 0 and top_score < second_score * min_margin:
        return None
    return top_book


if __name__ == "__main__":
    with open("library_chunks.json") as f:
        chunks = json.load(f)

    profiles = build_book_profiles(chunks)
    print("Distinctive keywords per book:")
    for book, words in profiles.items():
        print(f"  {book}: {sorted(words)[:12]}...")
    print()

    test_queries = [
        "what is a foreign key in a database table",
        "how does backpropagation work in neural networks",
        "what is the difference between primary key and normalization",
        "explain gradient descent optimization",
    ]
    for q in test_queries:
        routed = route_query(q, profiles)
        print(f"  '{q}' -> {routed or 'NO CONFIDENT ROUTE (search all books)'}")
