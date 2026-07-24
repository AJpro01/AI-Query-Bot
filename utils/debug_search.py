"""
debug_search.py
------------------
When the bot says "information not found," the FIRST thing to check is
whether the content ever made it into library_chunks.json at all -- not
whether retrieval or the LLM is doing something wrong. If parsing missed a
chapter (very possible for diagram/figure-heavy sections like a CNN
architecture chapter, where a lot of the "content" is in images and
captions rather than clean body paragraphs), no retrieval algorithm in the
world will find something that was never captured as text in the first
place.

Usage:
    python debug_search.py "convolutional"
    python debug_search.py "CNN" "Deep Learning"   # optionally scope to one book
"""

import sys
import json

LIBRARY_PATH = "library_chunks.json"


def search_raw_text(keyword: str, book_filter: str = None):
    with open(LIBRARY_PATH) as f:
        chunks = json.load(f)

    if book_filter:
        chunks = [c for c in chunks if c["metadata"]["book_title"] == book_filter]

    keyword_lower = keyword.lower()
    matches = [c for c in chunks if keyword_lower in c["text"].lower()]

    print(f"Searched {len(chunks)} chunks" + (f" in '{book_filter}'" if book_filter else " across all books"))
    print(f"Found '{keyword}' in {len(matches)} chunk(s)\n")

    if not matches:
        print("NOT FOUND. This means the word never made it into any chunk's text.")
        print("This points to a PARSING problem, not a retrieval problem -- likely causes:")
        print("  - The relevant chapter is dominated by diagrams/figures with little running text")
        print("  - Font-size heading detection misfired on that chapter, but check the text is still THERE")
        print("    (a mis-tagged chapter/section label is a separate, smaller problem than missing text)")
        print("  - PyMuPDF extraction skipped or garbled a multi-column or image-heavy page")
        print("\nNext step: open the PDF to that chapter and check visually whether it's mostly figures.")
    else:
        print("Found in chunks -- if the bot still said 'not found', this points to a RETRIEVAL problem")
        print("(the chunk exists but didn't rank in the top-K results), not a parsing problem.")
        print("Try increasing top_k, or check if the book router misfired to the wrong book.\n")
        for c in matches[:5]:
            m = c["metadata"]
            print(f"  [{c['chunk_id']}] {m['book_title']} | {m['chapter']} | {m['section']} | p.{m['page_number']}")
            idx = c["text"].lower().find(keyword_lower)
            snippet = c["text"][max(0, idx - 40): idx + 60]
            print(f"      ...{snippet}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_search.py \"keyword\" [\"Book Title\"]")
        sys.exit(1)
    keyword = sys.argv[1]
    book_filter = sys.argv[2] if len(sys.argv) > 2 else None
    search_raw_text(keyword, book_filter)
