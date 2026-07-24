"""
text_normalizer.py
--------------------
Cleans up an LLM-generated answer before it goes to a text-to-speech
engine. Without this step, a TTS engine will literally read out raw
citation brackets and LaTeX symbols, which sounds unusable:

    "Backpropagation minimizes error (Chapter 4, 4.1 Backpropagation, p.3)"
      -> spoken as -> "...open parenthesis Chapter 4 comma 4.1
         Backpropagation comma p period 3 close parenthesis"

Two things get normalized:
  1. Citations in the exact format answer_synth.py generates:
     (Chapter, Section, p.PAGE) -- stripped out entirely, since a listener
     can't act on a page number anyway; the fact stands on its own without it.
  2. Common LaTeX math notation -- converted to spoken English.

This is a pattern-matching normalizer, not a full LaTeX parser. It covers
the constructs likely to actually appear in Deep Learning / PRML / Speech
and Language Processing style answers (exponents, fractions, sums, Greek
letters, subscripts). It will NOT correctly handle deeply nested or exotic
LaTeX -- if you hit a case that reads oddly out loud, add a pattern here
rather than trying to build a general LaTeX-to-English parser.
"""

import re

GREEK_LETTERS = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "theta": "theta", "lambda": "lambda", "mu": "mu",
    "sigma": "sigma", "pi": "pi", "omega": "omega", "eta": "eta",
}


def strip_citations(text: str) -> str:
    """Removes (Chapter X, Section Y, p.Z) style citations entirely."""
    return re.sub(r"\s*\([^()]*p\.\d+\)", "", text)


def latex_to_spoken(text: str) -> str:
    """
    Converts common LaTeX constructs to spoken English. Order matters --
    more specific patterns (fractions, sums) must run before generic
    superscript/subscript handling, or they'll partially match and mangle
    the output.
    """
    # \frac{a}{b} -> "a over b " (trailing space guards against the next
    # LaTeX construct butting up directly against this one with no space,
    # e.g. \frac{1}{n}\sum{...} would otherwise read as "n_the sum" with no
    # word boundary)
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1 over \2 ", text)

    # \sum_{i=1}^{n} -> "the sum from i equals 1 to n of"
    text = re.sub(
        r"\\sum_\{([^{}]*)\}\^\{([^{}]*)\}",
        lambda m: f"the sum from {m.group(1).replace('=', ' equals ')} to {m.group(2)} of",
        text,
    )

    # Greek letters written as \alpha, \beta, etc.
    for name, spoken in GREEK_LETTERS.items():
        text = re.sub(rf"\\{name}\b", spoken, text)

    # x^2 or x^{2} -> "x squared"; x^3 -> "x cubed"; x^n -> "x to the power of n"
    def _exponent(match):
        base, exp = match.group(1), match.group(2).strip("{}")
        if exp == "2":
            return f"{base} squared"
        if exp == "3":
            return f"{base} cubed"
        return f"{base} to the power of {exp}"

    text = re.sub(r"(\w+)\^(\{[^{}]*\}|\w+)", _exponent, text)

    # x_i or x_{i} -> "x sub i"
    text = re.sub(r"(\w+)_(\{[^{}]*\}|\w+)", lambda m: f"{m.group(1)} sub {m.group(2).strip('{}')}", text)

    # Common symbols
    text = text.replace("\\times", "times").replace("\\cdot", "times")
    text = text.replace("\\leq", "less than or equal to").replace("\\geq", "greater than or equal to")
    text = text.replace("\\neq", "not equal to").replace("=", " equals ")

    # Strip any leftover $ or $$ delimiters and stray backslashes
    text = text.replace("$$", "").replace("$", "")
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)  # any remaining \command -> command

    return text


def normalize_for_speech(text: str) -> str:
    text = strip_citations(text)
    text = latex_to_spoken(text)
    text = re.sub(r"\s+", " ", text).strip()  # collapse extra whitespace left behind
    return text


if __name__ == "__main__":
    test_cases = [
        "Backpropagation minimizes the error function (Chapter 4, 4.1 Backpropagation, p.3).",
        "The equation is $E = mc^2$ according to the text.",
        "The margin is defined using $\\frac{1}{n}\\sum_{i=1}^{n} x_i$ (Chapter 3, 3.2 The Perceptron, p.1).",
        "Given $\\alpha \\leq \\beta$, the gradient $\\nabla_\\theta L$ converges.",
    ]
    for original in test_cases:
        print("BEFORE:", original)
        print("AFTER: ", normalize_for_speech(original))
        print()
