"""
sentence_streamer.py
-----------------------
Waiting for a full LLM answer to finish generating before starting TTS
adds real, noticeable latency -- if an answer takes 4 seconds to generate
and then another 2 seconds to synthesize audio, that's 6 seconds of
silence before the user hears anything.

This module lets you feed in text piece by piece (as it streams from an
LLM) and get back complete sentences AS SOON AS they're finished, so the
first sentence can start playing while the rest of the answer is still
being generated.

WHY THIS NEEDS ITS OWN LOGIC INSTEAD OF JUST SPLITTING ON PERIODS:
Splitting naively on "." breaks on abbreviations, decimals, and citations
still in the raw text (e.g. "p.3", "e.g.", "3.2"). This tracks a buffer and
only emits a sentence once it sees clear sentence-ending punctuation
followed by whitespace or end-of-stream, and skips known non-terminating
patterns like single-letter-period-digit (page numbers) and common
abbreviations.
"""

import re

# Patterns that look like sentence endings but aren't -- checked right
# before the candidate split point.
NON_TERMINATING_PATTERNS = [
    re.compile(r"\bp\.\s*\d*$", re.IGNORECASE),   # "p.3" or "p." (page citations mid-formation)
    re.compile(r"\be\.g\.$", re.IGNORECASE),      # "e.g."
    re.compile(r"\bi\.e\.$", re.IGNORECASE),      # "i.e."
    re.compile(r"\b\d+\.$"),                       # "3." (a numbered list marker or decimal in progress)
]


def _looks_like_real_sentence_end(buffer: str) -> bool:
    for pattern in NON_TERMINATING_PATTERNS:
        if pattern.search(buffer):
            return False
    return True


def stream_sentences(text_chunks):
    """
    Generator: takes an iterable of text pieces (e.g. tokens or small
    chunks from a streaming LLM response) and yields complete sentences as
    soon as they're detected. Call with a list of strings for a
    non-streaming source, or an actual generator for real streaming.
    """
    buffer = ""
    for piece in text_chunks:
        buffer += piece
        while True:
            match = re.search(r"[.!?](\s|$)", buffer)
            if not match:
                break
            candidate = buffer[: match.end()].strip()
            if _looks_like_real_sentence_end(candidate):
                yield candidate
                buffer = buffer[match.end():]
            else:
                next_match = re.search(r"[.!?](\s|$)", buffer[match.end():])
                if not next_match:
                    break
                buffer_marker = match.end() + next_match.end()
                candidate2 = buffer[:buffer_marker].strip()
                if _looks_like_real_sentence_end(candidate2):
                    yield candidate2
                    buffer = buffer[buffer_marker:]
                else:
                    break

    remainder = buffer.strip()
    if remainder:
        yield remainder


if __name__ == "__main__":
    # Simulate a streaming LLM response arriving in small pieces
    fake_stream = [
        "Backprop", "agation comput", "es gradients using the chain rule",
        " (see p.3", " for details). ", "Vanishing gradients occur ",
        "in very deep networks, e.g. ", "those with 50+ layers. ",
        "This is a known limit", "ation of sigmoid activations."
    ]

    print("Simulated streaming input (arrives in these pieces):")
    print(fake_stream)
    print()
    print("Detected complete sentences:")
    for sentence in stream_sentences(fake_stream):
        print(f"  -> {sentence!r}")
