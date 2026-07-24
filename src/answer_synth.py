"""
answer_synth.py
----------------
Takes retrieved chunks (from retriever.py) and a user question, and asks
Gemini to answer USING ONLY those chunks -- with an inline citation format
that points back to chapter/section/page. This is what turns "search
results" into a usable answer, and is where hallucination gets controlled.

Reuses the same Gemini API key setup as your NEET app -- set the
GEMINI_API_KEY environment variable before running.

KEY DESIGN CHOICE: the prompt explicitly instructs the model to say
"Information not found in the provided context" rather than guess. This is
the single most important line in this file for controlling hallucination --
without it, Gemini will happily fill gaps with its own pretrained knowledge,
which defeats the entire point of "grounded" answers.
"""

import os
from src.llm_client import call_llm

SYSTEM_PROMPT = """You are a study assistant answering questions strictly from the textbook excerpts provided below.

RULES:
- Answer ONLY using information contained in the excerpts below.
- If the excerpts do not contain enough information to answer, say exactly: "Information not found in the provided context."
- Do not use any outside knowledge, even if you know the answer.
- After every claim, cite the source using this exact format: (Chapter, Section, p.PAGE)
- Keep the answer concise and directly address the question.
"""


def format_context(chunks: list[dict]) -> str:
    """Turns retrieved chunks into a labeled block the LLM can cite from."""
    blocks = []
    for c in chunks:
        m = c["metadata"]
        label = f"[{m['chapter']} | {m['section']} | p.{m['page_number']}]"
        blocks.append(f"{label}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context = format_context(retrieved_chunks)
    prompt = f"{SYSTEM_PROMPT}\n\nEXCERPTS:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    return call_llm(prompt)


if __name__ == "__main__":
    import sys
    from src.retriever import HybridRetriever

    question = sys.argv[1] if len(sys.argv) > 1 else "What causes vanishing gradients?"

    retriever = HybridRetriever()
    chunks = retriever.search(question, top_k=3)

    print(f"Question: {question}\n")
    print("Retrieved context:")
    for c in chunks:
        m = c["metadata"]
        print(f"  - {m['chapter']} | {m['section']} | p.{m['page_number']}")
    print()

    answer = generate_answer(question, chunks)
    print("Answer:")
    print(answer)
