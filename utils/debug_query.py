"""
debug_query.py
-----------------
Unlike debug_search.py (which checks raw keyword presence in the JSON),
this runs the ACTUAL retrieval pipeline -- router + hybrid search -- for a
real question, and shows exactly what chunks would be sent to the LLM.

Use this when debug_search.py says content EXISTS but the bot still
says "not found" -- that means the problem is in RANKING (the chunk
exists but didn't make the top-K), not in parsing/pollution. This script
shows you the actual ranked results so you can see:
  - Did the router pick the right book?
  - Are the top-K results real chapter content, or still pollution?
  - What RRF score did each result get, and how close was the real
    content to making the cut?

Usage:
    python debug_query.py "describe the architecture of CNN"
    python debug_query.py "describe the architecture of CNN" "Deep Learning by Ian Goodfellow"
"""

import sys
import json

LIBRARY_PATH = "library_chunks.json"


def debug_query(question: str, manual_book: str = None, top_k: int = 10):
    from src.retriever import HybridRetriever
    from src.book_router import build_book_profiles, route_query

    with open(LIBRARY_PATH) as f:
        library_chunks = json.load(f)

    book_filter = manual_book
    if not book_filter:
        profiles = build_book_profiles(library_chunks, top_n=60)
        book_filter = route_query(question, profiles)
        print(f"[router] auto-scoped to: {book_filter or '(no confident match -- searching all books)'}")
    else:
        print(f"[router] manually scoped to: {book_filter}")

    retriever = HybridRetriever(LIBRARY_PATH)
    results = retriever.search(question, top_k=top_k, book_filter=book_filter)

    print(f"\nTop {len(results)} results for: {question!r}\n")
    if not results:
        print("NO RESULTS AT ALL. This means retrieval itself returned nothing -- "
              "check that the book was actually ingested and book_filter matches "
              "an actual book_title in library_chunks.json exactly.")
        return

    pollution_keywords = ["content", "acknowledg", "bibliograph", "index", "appendix", "preface", "notation"]
    for i, r in enumerate(results):
        m = r["metadata"]
        is_suspicious = any(kw in m["chapter"].lower() for kw in pollution_keywords)
        flag = "  <-- SUSPICIOUS (looks like front/back matter, cleanup may not have run)" if is_suspicious else ""
        print(f"{i+1}. [{r['rrf_score']}] {m['book_title']} | {m['chapter']} | p.{m['page_number']}{flag}")
        print(f"   {r['text'][:120]}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    question = sys.argv[1]
    manual_book = sys.argv[2] if len(sys.argv) > 2 else None
    debug_query(question, manual_book)