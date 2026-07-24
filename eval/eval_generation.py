"""
eval_generation.py
--------------------
Faithfulness, Answer Relevance, and Context Recall -- the three metrics
from your proposal that need an LLM to JUDGE quality rather than just
look up a fact. This replaces the ragas framework (broken dependency chain
in this environment, and version fragility in general) with direct, simple
Gemini calls doing the same underlying job. This is exactly how ragas
implements these metrics internally anyway -- an LLM prompted to score a
specific comparison -- just without the extra framework layer.

FAITHFULNESS: does the generated answer only contain claims traceable to
the retrieved context? (0.0 - 1.0, judged by Gemini)

ANSWER RELEVANCE: does the answer actually address the question asked,
without padding or going off-topic? (0.0 - 1.0, judged by Gemini)

CONTEXT RECALL: did the retrieved chunks contain everything needed to
answer the question, per the ground-truth answer? (0.0 - 1.0, judged by
Gemini comparing ground truth against retrieved context)
"""

import re
from src.llm_client import call_llm

FAITHFULNESS_PROMPT = """You are a strict fact-checker. Given the CONTEXT and a GENERATED ANSWER, determine what fraction of the answer's claims are directly supported by the context.

CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Respond with ONLY a number between 0.0 and 1.0 representing the fraction of claims in the answer that are explicitly supported by the context. 1.0 means every claim is supported. 0.0 means none are. Respond with the number only, nothing else."""

RELEVANCE_PROMPT = """You are evaluating answer quality. Given a QUESTION and a GENERATED ANSWER, rate how directly and completely the answer addresses the question.

QUESTION: {question}

GENERATED ANSWER: {answer}

Respond with ONLY a number between 0.0 and 1.0. 1.0 means the answer directly and fully addresses the question. 0.0 means it is off-topic or evasive. Respond with the number only, nothing else."""

CONTEXT_RECALL_PROMPT = """You are evaluating retrieval quality. Given a GROUND TRUTH ANSWER and the RETRIEVED CONTEXT that was available to generate an answer, determine what fraction of the information in the ground truth answer could be derived from the retrieved context.

GROUND TRUTH ANSWER: {ground_truth}

RETRIEVED CONTEXT:
{context}

Respond with ONLY a number between 0.0 and 1.0. 1.0 means the context contains everything needed. 0.0 means none of it is present. Respond with the number only, nothing else."""


def _call_llm_for_score(prompt: str) -> float:
    raw = call_llm(prompt)
    match = re.search(r"[01](\.\d+)?", raw.strip())
    if not match:
        raise ValueError(f"Could not parse a score from judge response: {raw!r}")
    return float(match.group())


def faithfulness(context: str, answer: str) -> float:
    return _call_llm_for_score(FAITHFULNESS_PROMPT.format(context=context, answer=answer))


def answer_relevance(question: str, answer: str) -> float:
    return _call_llm_for_score(RELEVANCE_PROMPT.format(question=question, answer=answer))


def context_recall(ground_truth: str, context: str) -> float:
    return _call_llm_for_score(CONTEXT_RECALL_PROMPT.format(ground_truth=ground_truth, context=context))


if __name__ == "__main__":
    from eval.eval_dataset import EVAL_SET
    from src.retriever import HybridRetriever
    from src.answer_synth import generate_answer, format_context

    retriever = HybridRetriever()

    faith_scores, rel_scores, recall_scores = [], [], []

    for item in EVAL_SET:
        chunks = retriever.search(item["question"], top_k=3)
        context_str = format_context(chunks)
        answer = generate_answer(item["question"], chunks)

        f = faithfulness(context_str, answer)
        r = answer_relevance(item["question"], answer)
        c = context_recall(item["ground_truth_answer"], context_str)

        faith_scores.append(f)
        rel_scores.append(r)
        recall_scores.append(c)

        print(f"Q: {item['question']}")
        print(f"   Faithfulness: {f:.2f} | Relevance: {r:.2f} | Context Recall: {c:.2f}")

    print("\nAverages")
    print(f"  Faithfulness:   {sum(faith_scores)/len(faith_scores):.2f}")
    print(f"  Answer Relevance: {sum(rel_scores)/len(rel_scores):.2f}")
    print(f"  Context Recall: {sum(recall_scores)/len(recall_scores):.2f}")
