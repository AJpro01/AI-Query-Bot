"""
llm_client.py
---------------
Single shared entry point for calling an LLM, used by answer_synth.py,
concept_extractor.py, and eval_generation.py. Keeping this centralized
means switching providers/models only requires editing this one file.

REVERTED FROM OPENROUTER BACK TO GEMINI DIRECTLY.

Model is now "gemini-3.5-flash". Two things worth knowing, verified via a
live search since this moves fast and training data goes stale:

1. Gemini 1.5 and Gemini 2.0 are FULLY SHUT DOWN as of mid-2026 -- any
   request to those model names now returns a 404, which is exactly the
   kind of error this project hit earlier with "gemini-1.5-flash". Gemini
   2.5 (Pro/Flash) is still alive, but Gemini 3.5 Flash is the current,
   generally-available, recommended Flash-tier model as of this writing --
   Google's own docs describe it as "near-Pro intelligence at Flash-tier
   cost and speed."
2. Google is pushing a new "Interactions API" (client.interactions.create)
   as the go-forward recommended primitive for new projects, but the
   standard client.models.generate_content() endpoint used here is still
   fully supported per Google's own API reference -- kept for consistency
   with the working pattern this project already used, rather than
   introducing an untested new API shape under deadline pressure.

Set your key via the GEMINI_API_KEY environment variable (same as
before the OpenRouter detour): export GEMINI_API_KEY="your-key-here"
(or $env:GEMINI_API_KEY="..." on Windows PowerShell).

Before your final submission, double-check ai.google.dev/gemini-api/docs/models
for the current model lineup -- this space moves fast enough that even
this comment could be stale by the time you read it.
"""

import os
import time
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

MODEL_NAME = "gemini-3.5-flash"

# Transient errors worth retrying: 503 (server overloaded, e.g. "high demand")
# and 429 (rate limit) both usually resolve themselves within seconds --
# without a retry, a single transient blip kills an entire eval run that
# makes dozens of sequential calls.
RETRYABLE_STATUS_CODES = {429, 503}
MAX_RETRIES = 4
BACKOFF_SECONDS = [5, 10, 20, 40]


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Set it with: export GEMINI_API_KEY='your-key-here' "
            "(or $env:GEMINI_API_KEY=\"...\" on Windows PowerShell)"
        )
    return genai.Client(api_key=api_key)


def _call_with_retry(fn):
    """Retries fn() on transient server errors (503/429) with backoff; re-raises anything else immediately."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except genai_errors.ServerError as e:
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            if status not in RETRYABLE_STATUS_CODES:
                raise
            last_error = e
            wait = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            print(f"  Gemini returned a transient error ({status}) -- retrying in {wait}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
    raise last_error


def call_llm(prompt: str) -> str:
    client = _get_client()
    response = _call_with_retry(lambda: client.models.generate_content(model=MODEL_NAME, contents=prompt))
    return response.text


def call_llm_with_image(prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
    """
    Same model, same key, just with an image attached alongside the text
    prompt -- Gemini is natively multimodal, so this is the SAME LLM as
    call_llm(), not a second provider. Used by image_captioner.py to
    describe figures/diagrams extracted from a book's pages.
    """
    client = _get_client()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    response = _call_with_retry(
        lambda: client.models.generate_content(model=MODEL_NAME, contents=[prompt, image_part])
    )
    return response.text


if __name__ == "__main__":
    # Sanity check: confirms the missing-key error is clear when no key is set
    try:
        call_llm("test prompt")
    except RuntimeError as e:
        print("Correctly caught missing API key:")
        print(" ", e)
