"""
run_eval.py
------------
Runs both retrieval and generation evaluation, prints results, and writes
a markdown report (eval_report.md) with a results table you can paste
directly into your project report's Evaluation section.

Usage:
    python run_eval.py
"""

import os
from datetime import datetime
from eval.eval_dataset import EVAL_SET
from eval.eval_retrieval import run_retrieval_eval


def main():
    from src.retriever import HybridRetriever
    retriever = HybridRetriever()

    print(f"Running evaluation on {len(EVAL_SET)} hand-written Q&A pairs\n")

    retrieval_results = run_retrieval_eval(retriever)

    print("\nRunning generation evaluation (requires GEMINI_API_KEY)...")
    try:
        from eval.eval_generation import faithfulness, answer_relevance, context_recall
        from src.answer_synth import generate_answer, format_context

        faith_scores, rel_scores, recall_scores = [], [], []
        for item in EVAL_SET:
            chunks = retriever.search(item["question"], top_k=3)
            context_str = format_context(chunks)
            answer = generate_answer(item["question"], chunks)
            faith_scores.append(faithfulness(context_str, answer))
            rel_scores.append(answer_relevance(item["question"], answer))
            recall_scores.append(context_recall(item["ground_truth_answer"], context_str))

        generation_results = {
            "faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else 0.0,
            "answer_relevance": sum(rel_scores) / len(rel_scores) if rel_scores else 0.0,
            "context_recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        }
    except Exception as e:
        print(f"  Skipped generation eval (API error or key missing): {e}")
        generation_results = None

    # --- write report ---
    lines = [
        f"# Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Eval set size: {len(EVAL_SET)} hand-written Q&A pairs",
        "",
        "## Retrieval Metrics",
        "| Metric | Score |",
        "|---|---|",
        f"| Hit Rate @ 3 | {retrieval_results['hit_rate@3']:.2%} |",
        f"| MRR | {retrieval_results['mrr']:.3f} |",
        "",
    ]
    if generation_results:
        lines += [
            "## Generation Metrics",
            "| Metric | Score |",
            "|---|---|",
            f"| Faithfulness | {generation_results['faithfulness']:.2f} |",
            f"| Answer Relevance | {generation_results['answer_relevance']:.2f} |",
            f"| Context Recall | {generation_results['context_recall']:.2f} |",
        ]
    else:
        lines += ["## Generation Metrics", "Skipped -- GEMINI_API_KEY not set when this report was generated."]

    # Save specifically inside the eval/ directory
    report_path = os.path.join(os.path.dirname(__file__), "eval_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
