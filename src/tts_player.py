"""
tts_player.py
---------------
Ties normalization + sentence streaming + actual audio synthesis together.

Uses edge-tts: free, no API key, no GPU needed, runs via a network call to
Microsoft's speech service (this sandbox's network can't reach it -- same
limitation as the Gemini API calls earlier -- so the actual synthesis call
below is untested here, but the normalization/sentence-splitting logic and
the file-combining logic below have both been verified independently).

FLOW:
  raw LLM answer -> normalize_for_speech() -> stream_sentences() ->
  synthesize each sentence to a temp mp3 -> concatenate into ONE final
  file -> delete the temp per-sentence files

CHANGED FROM THE FIRST VERSION: previously returned one mp3 file PER
SENTENCE, so a 3-sentence answer produced 3 separate audio files the
frontend had to play back-to-back. Now produces a single combined file
matching the full displayed answer, which is what you actually want for a
"press play once" experience. The tradeoff: this loses the low-latency
streaming benefit (sentence 1 can no longer start playing while sentence 2
is still being synthesized) -- if that streaming behavior matters more to
you than a single file, say so and it can be added back as a toggle.

MP3 concatenation here is simple byte-level concatenation of independent
mp3 streams -- edge-tts's output is a standalone, self-contained mp3 per
call, and concatenating multiple such streams is a well-known trick that
players handle correctly (each stream has its own valid frame headers).
If you ever hear a click/glitch at sentence boundaries, switching to
pydub + ffmpeg for a sample-accurate merge is the more robust (but heavier
dependency) alternative.
"""

import asyncio
import os
import edge_tts

from src.text_normalizer import normalize_for_speech
from src.sentence_streamer import stream_sentences

VOICE = "en-US-AriaNeural"  # natural-sounding, free Microsoft neural voice


async def synthesize_sentence(sentence: str, output_path: str):
    communicate = edge_tts.Communicate(sentence, VOICE)
    await communicate.save(output_path)


def combine_mp3_files(file_paths: list, output_path: str):
    """Concatenates multiple mp3 files' raw bytes into a single output file."""
    with open(output_path, "wb") as out_f:
        for path in file_paths:
            with open(path, "rb") as in_f:
                out_f.write(in_f.read())


async def speak_answer(raw_answer: str, output_dir: str = ".", output_filename: str = "answer_audio.mp3") -> str:
    """
    Takes a raw LLM answer (with citations and possibly LaTeX), normalizes
    it, splits it into sentences, synthesizes each sentence to a temp
    file, then combines them into ONE final audio file matching the whole
    displayed answer. Returns the path to that single combined file.
    """
    normalized = normalize_for_speech(raw_answer)
    sentences = list(stream_sentences([normalized]))

    temp_paths = []
    for i, sentence in enumerate(sentences):
        temp_path = f"{output_dir}/_tts_temp_sentence_{i:03d}.mp3"
        print(f"Synthesizing: {sentence[:60]}...")
        await synthesize_sentence(sentence, temp_path)
        temp_paths.append(temp_path)

    final_path = f"{output_dir}/{output_filename}"
    combine_mp3_files(temp_paths, final_path)

    for p in temp_paths:
        os.remove(p)

    return final_path


if __name__ == "__main__":
    example_answer = (
        "Backpropagation minimizes the error function (Chapter 4, 4.1 Backpropagation, p.3). "
        "The gradient is computed as $\\frac{\\partial L}{\\partial w}$ and used to update weights. "
        "Vanishing gradients occur in very deep networks, e.g. those with many layers "
        "(Chapter 4, 4.1 Backpropagation, p.3)."
    )

    print("Raw answer:")
    print(example_answer)
    print()
    print("Normalized for speech:")
    print(normalize_for_speech(example_answer))
    print()

    try:
        final_path = asyncio.run(speak_answer(example_answer))
        print(f"\nGenerated single combined audio file: {final_path}")
    except Exception as e:
        print(f"\nTTS synthesis failed (expected in a network-restricted sandbox): {e}")
        print("Normalization + sentence splitting above are the parts to trust; "
              "the actual edge-tts network call needs to be tested on your machine.")
