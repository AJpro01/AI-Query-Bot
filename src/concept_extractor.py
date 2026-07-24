"""
concept_extractor.py
----------------------
Asks Gemini to read text and pull out:
  1. Core terms/concepts mentioned (with a one-line definition)
  2. Relationships between those terms (e.g. "gradient descent" -> "used by" -> "backpropagation")

TWO CHANGES FROM THE FIRST VERSION, BOTH DIRECTLY BECAUSE OF QUOTA LIMITS
HIT ON A REAL 700-PAGE BOOK:

1. BATCHING: extraction now runs per BATCH of chunks (default 8 at a time),
   not per single chunk. A 700-page book might be ~900 chunks -- at one
   Gemini call per chunk, that's 900 API calls just to build one concept
   map, which blows through free-tier quota fast. Batching 8 chunks per
   call cuts that to ~113 calls, an ~8x reduction.

2. CACHING: every batch's result is saved to disk (concept_cache.json) as
   soon as it's extracted. If you hit a quota error partway through a
   700-page book, rerunning does NOT start over -- it skips every batch
   that's already cached and only calls Gemini for what's missing. This
   also means iterating/debugging the concept map doesn't re-burn quota
   on chunks you've already processed.

RECOMMENDATION STILL HOLDS: for a book this size, extract one chapter at a
time (pass a filtered chunk list) rather than the whole book in one run,
both for quota reasons and because a whole-book graph is too dense to be
readable anyway (see visualize_graph.py's redesign notes).
"""

import os
import json
import re
import time
import hashlib
from src.llm_client import call_llm

CACHE_PATH = "concept_cache.json"
BATCH_SIZE = 15  # increased from 8 -- combined with chapter-scoping in app.py, this keeps a
                 # single chapter's worth of chunks comfortably within a handful of API calls
SECONDS_BETWEEN_CALLS = 3.5  # stays comfortably under typical free-tier rate limits (Gemini's free tier is commonly ~10-15 requests/minute for Flash models)

BATCH_EXTRACTION_PROMPT = """Extract core technical terms and their relationships from the text excerpts below. Each excerpt is separated by "---".

TEXT EXCERPTS:
{text}

Respond with ONLY valid JSON in this exact format, nothing else:
{{
  "terms": [{{"term": "...", "definition": "one short sentence"}}],
  "relations": [{{"source": "...", "relation": "...", "target": "..."}}]
}}

Only include terms that are actually defined or substantively discussed. Only include relations where both terms appear in the text above. Deduplicate terms that mean the same thing. Keep the list focused (15-25 terms max across all excerpts)."""


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _batch_key(chunk_ids: list) -> str:
    """Deterministic cache key for a batch, based on which chunks it contains."""
    joined = "|".join(sorted(chunk_ids))
    return hashlib.md5(joined.encode()).hexdigest()


def extract_from_batch(chunks: list) -> dict:
    combined_text = "\n---\n".join(c["text"] for c in chunks)
    raw = call_llm(BATCH_EXTRACTION_PROMPT.format(text=combined_text))
    raw = raw.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
    return json.loads(raw)


def extract_from_book(chunks: list, batch_size: int = BATCH_SIZE, use_cache: bool = True) -> dict:
    """
    Runs extraction across a book's chunks IN BATCHES, using a local cache
    so a quota error or a rerun doesn't waste already-spent API calls.
    Returns {"terms": [...], "relations": [...]} merged across all batches.
    """
    cache = _load_cache() if use_cache else {}
    all_terms, all_relations = [], []

    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    print(f"Processing {len(chunks)} chunks in {len(batches)} batches of up to {batch_size}...")

    for i, batch in enumerate(batches):
        chunk_ids = [c["chunk_id"] for c in batch]
        key = _batch_key(chunk_ids)

        if key in cache:
            print(f"  Batch {i+1}/{len(batches)}: cached, skipping API call")
            result = cache[key]
        else:
            try:
                print(f"  Batch {i+1}/{len(batches)}: calling LLM ({len(batch)} chunks)...")
                result = extract_from_batch(batch)
                cache[key] = result
                _save_cache(cache)  # persist immediately -- don't lose progress on a later failure
                if i < len(batches) - 1:  # no need to wait after the last batch
                    time.sleep(SECONDS_BETWEEN_CALLS)
            except Exception as e:
                print(f"  Batch {i+1}/{len(batches)}: FAILED ({e}) -- stopping here. "
                      f"Already-completed batches are cached; rerun to resume from this point.")
                break

        all_terms.extend(result.get("terms", []))
        all_relations.extend(result.get("relations", []))

    return {"terms": all_terms, "relations": all_relations}


if __name__ == "__main__":
    import sys
    with open("library_chunks.json") as f:
        chunks = json.load(f)

    book_title = sys.argv[1] if len(sys.argv) > 1 else chunks[0]["metadata"]["book_title"]
    chapter_filter = sys.argv[2] if len(sys.argv) > 2 else None

    book_chunks = [c for c in chunks if c["metadata"]["book_title"] == book_title]
    if chapter_filter:
        book_chunks = [c for c in book_chunks if c["metadata"]["chapter"] == chapter_filter]
        print(f"Filtered to chapter '{chapter_filter}': {len(book_chunks)} chunks")

    print(f"Extracting concepts from {len(book_chunks)} chunks of '{book_title}'...")
    result = extract_from_book(book_chunks)
    print(json.dumps(result, indent=2))

