"""
eval_dataset.py
----------------
Loads the evaluation Q&A set from eval_set.json if it exists (build one
against your REAL ingested library with `python pipeline.py sample-chunks`,
see below) -- otherwise falls back to a small built-in synthetic set that
only works against the tiny test_book.pdf used during early development.

IMPORTANT: if you ran `run_eval.py` and got 0% Hit Rate / 0.000 MRR on your
real book, this is almost certainly why -- the built-in EVAL_SET's
ground_truth_chunk_id values (like "test_book_p1_g0_c0") only exist in the
synthetic test book, not in a real ingested textbook's library_chunks.json.
Build a real eval_set.json instead (see below) before trusting these
numbers in a report.

WHY HAND-WRITTEN AND NOT LLM-GENERATED:
LLM-generated synthetic Q&A pairs are faster to produce at scale, but if you
don't check them, you're measuring "does my system agree with another LLM's
guess" rather than real correctness. Writing 10-15 by hand against real
chunks (see sample-chunks below, which shows you real text + real chunk_ids
to build these from) takes maybe 20-30 minutes and gives you numbers you can
actually defend in a report if asked "how do you know this metric is correct?"

FORMAT for eval_set.json (a plain JSON list):
[
  {
    "question": "...",
    "ground_truth_answer": "...",
    "ground_truth_chunk_id": "<a real chunk_id from your library_chunks.json>"
  },
  ...
]
"""

import os
import json

# Look for eval_set.json inside the eval/ directory relative to this file
EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")

_FALLBACK_EVAL_SET = [
    {
        "question": "What is the perceptron?",
        "ground_truth_answer": "The perceptron is a simple linear classifier used for binary classification that updates its weights when it misclassifies a training example.",
        "ground_truth_chunk_id": "test_book_p1_g0_c0",
    },
    {
        "question": "What does the learning rate control in gradient descent?",
        "ground_truth_answer": "The learning rate controls the size of each update step in gradient descent.",
        "ground_truth_chunk_id": "test_book_p2_g1_c0",
    },
    {
        "question": "How does backpropagation compute gradients?",
        "ground_truth_answer": "Backpropagation computes the gradient of the loss with respect to each weight by applying the chain rule, propagating error backward from the output layer to the input layer.",
        "ground_truth_chunk_id": "test_book_p3_g2_c0",
    },
    {
        "question": "What does the ReLU activation function do?",
        "ground_truth_answer": "ReLU outputs the input directly if it is positive, and zero otherwise, and helps mitigate vanishing gradients.",
        "ground_truth_chunk_id": "test_book_p4_g3_c0",
    },
    {
        "question": "What is a common cause of learning problems in very deep networks?",
        "ground_truth_answer": "Vanishing gradients, where gradients become extremely small as they propagate backward through many layers.",
        "ground_truth_chunk_id": "test_book_p3_g2_c0",
    },
]


def _load_eval_set():
    if os.path.exists(EVAL_SET_PATH):
        with open(EVAL_SET_PATH) as f:
            return json.load(f)
    print(f"NOTE: {EVAL_SET_PATH} not found -- using the built-in synthetic eval set, "
          f"which only matches the tiny test_book.pdf, NOT a real ingested textbook. "
          f"Run `python pipeline.py sample-chunks \"<Book Title>\"` to see real chunks "
          f"and build a real {EVAL_SET_PATH} before trusting these numbers.")
    return _FALLBACK_EVAL_SET


EVAL_SET = _load_eval_set()

if __name__ == "__main__":
    print(f"Eval set contains {len(EVAL_SET)} question/answer pairs (from "
          f"{'eval_set.json' if os.path.exists(EVAL_SET_PATH) else 'built-in fallback'}).")
    for item in EVAL_SET:
        print(f"  - {item['question']}  (ground truth: {item['ground_truth_chunk_id']})")
