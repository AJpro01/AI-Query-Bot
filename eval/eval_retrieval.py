"""
eval_retrieval.py
------------------
Measures whether HybridRetriever actually finds the right chunk for each
question in the eval set. These two metrics need no LLM judge -- they're
pure lookups against ground truth, which is exactly why they're the fastest
and most trustworthy metrics to run.

Hit Rate @ K:
    For each question, check if the ground-truth chunk_id appears ANYWHERE
    in the top K retrieved results. Score = fraction of questions where
    this is true.

Mean Reciprocal Rank (MRR):
    For each question, find the RANK POSITION of the ground-truth chunk_id
    in the retrieved results (1st place = 1, 2nd place = 1/2, not found = 0).
    Average across all questions. Rewards ranking the right answer higher,
    not just anywhere in the list.
"""

from eval.eval_dataset import EVAL_SET


def hit_rate_at_k(retriever, eval_set=EVAL_SET, k: int = 3) -> float:
    hits = 0
    for item in eval_set:
        results = retriever.search(item["question"], top_k=k)
        retrieved_ids = [r["chunk_id"] for r in results]
        if item["ground_truth_chunk_id"] in retrieved_ids:
            hits += 1
    return hits / len(eval_set)


def mean_reciprocal_rank(retriever, eval_set=EVAL_SET, k: int = 10) -> float:
    reciprocal_ranks = []
    for item in eval_set:
        results = retriever.search(item["question"], top_k=k)
        retrieved_ids = [r["chunk_id"] for r in results]
        if item["ground_truth_chunk_id"] in retrieved_ids:
            rank = retrieved_ids.index(item["ground_truth_chunk_id"]) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def run_retrieval_eval(retriever, eval_set=EVAL_SET):
    print("Retrieval Evaluation")
    print("=" * 40)
    for k in [1, 3, 5]:
        hr = hit_rate_at_k(retriever, eval_set, k=k)
        print(f"  Hit Rate @ {k}: {hr:.2%}")
    mrr = mean_reciprocal_rank(retriever, eval_set)
    print(f"  MRR: {mrr:.3f}")
    return {"hit_rate@3": hit_rate_at_k(retriever, eval_set, k=3), "mrr": mrr}


if __name__ == "__main__":
    from src.retriever import HybridRetriever
    retriever = HybridRetriever()
    run_retrieval_eval(retriever)
