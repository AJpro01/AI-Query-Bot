"""
retriever.py
------------
Combines two different search strategies and merges their results:

1. DENSE (vector) search via Chroma -- good at understanding MEANING.
   e.g. query "how do neural nets learn from mistakes" can match a chunk
   about "backpropagation" even with zero shared words.

2. LEXICAL (BM25) search -- good at exact KEYWORD matches.
   e.g. query "ReLU" needs to reliably hit the exact chunk mentioning ReLU,
   which dense search can sometimes miss if the surrounding context pulls
   the embedding in a different semantic direction.

Neither alone is reliable for a technical textbook -- users ask both vague
conceptual questions AND precise "what does X mean" questions. So we run
both and merge with Reciprocal Rank Fusion (RRF):

    RRF_score(chunk) = sum over each ranked list of  1 / (k + rank)

A chunk that ranks well in BOTH lists rises to the top. k=60 is a standard
smoothing constant from the original RRF paper -- it just dampens the
impact of any single very-high or very-low rank.
"""

import json
from collections import defaultdict
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer

from src.embedder import EMBED_MODEL_NAME, CHROMA_DB_PATH, COLLECTION_NAME


class HybridRetriever:
    def __init__(self, chunks_json_path: str = "library_chunks.json"):
        with open(chunks_json_path) as f:
            self.chunks = json.load(f)
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}

        # --- lexical index (built once over ALL books; filtered per-query below) ---
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_ids = [c["chunk_id"] for c in self.chunks]

        # --- dense index (Chroma, already built by embedder.py) ---
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)

    def _dense_search(self, query: str, top_k: int, book_filter: str = None, chapter_filter: str = None) -> list:
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        conditions = []
        if book_filter:
            conditions.append({"book_title": book_filter})
        if chapter_filter:
            conditions.append({"chapter": chapter_filter})
        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}
        results = self.collection.query(query_embeddings=[query_vec], n_results=top_k, where=where)
        return results["ids"][0]  # ranked list of chunk_ids

    def _lexical_search(self, query: str, top_k: int, book_filter: str = None, chapter_filter: str = None) -> list:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_ids = [self.bm25_ids[i] for i in ranked]
        if book_filter:
            ranked_ids = [cid for cid in ranked_ids if self.chunk_by_id[cid]["metadata"]["book_title"] == book_filter]
        if chapter_filter:
            ranked_ids = [cid for cid in ranked_ids if self.chunk_by_id[cid]["metadata"]["chapter"] == chapter_filter]
        return ranked_ids[:top_k]

    def search(self, query: str, top_k: int = 5, rrf_k: int = 60, book_filter: str = None,
               chapter_filter: str = None) -> list:
        """
        Returns the top_k chunks (full dicts, with text + metadata) ranked
        by fused RRF score, highest first. book_filter scopes to one book
        (from book_router.route_query); chapter_filter additionally scopes
        to one chapter within that book (from chapter_router.route_to_chapter)
        -- both search streams respect both filters when set.
        """
        dense_ranked = self._dense_search(query, top_k=20, book_filter=book_filter, chapter_filter=chapter_filter)
        lexical_ranked = self._lexical_search(query, top_k=20, book_filter=book_filter, chapter_filter=chapter_filter)

        rrf_scores = defaultdict(float)
        for rank, chunk_id in enumerate(dense_ranked):
            rrf_scores[chunk_id] += 1.0 / (rrf_k + rank)
        for rank, chunk_id in enumerate(lexical_ranked):
            rrf_scores[chunk_id] += 1.0 / (rrf_k + rank)

        fused_ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        top_ids = fused_ranked_ids[:top_k]

        # Defensive: a chunk_id can exist in Chroma but not in chunk_by_id if
        # the two ever fall out of sync (e.g. library_chunks.json was reset
        # or edited without also clearing chroma_db/ -- this happened in
        # practice: a stale book's embeddings remained in Chroma after the
        # JSON was deleted, causing a KeyError here). Skip anything that's
        # gone missing instead of crashing the whole query on a data-sync
        # issue that a full reset (see pipeline.py's reset-library command)
        # should be used to actually fix.
        results = []
        skipped = 0
        for cid in top_ids:
            if cid in self.chunk_by_id:
                results.append({**self.chunk_by_id[cid], "rrf_score": round(rrf_scores[cid], 5)})
            else:
                skipped += 1
        if skipped:
            print(f"Warning: {skipped} search result(s) referenced chunk IDs no longer in "
                  f"library_chunks.json (stale Chroma data) -- run "
                  f"`python pipeline.py reset-library` for a clean sync, then re-ingest.")
        return results


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "how does a neural network learn from errors"
    book_filter = sys.argv[2] if len(sys.argv) > 2 else None

    retriever = HybridRetriever()
    results = retriever.search(query, top_k=3, book_filter=book_filter)

    print(f"Query: {query}" + (f"  (scoped to: {book_filter})" if book_filter else ""))
    print()
    for r in results:
        print(f"  [{r['rrf_score']}] {r['chunk_id']}")
        print(f"      {r['metadata']['chapter']} > {r['metadata']['section']} (p.{r['metadata']['page_number']})")
        print(f"      {r['text'][:100]}...")
        print()
