"""
pdf_parser.py
-------------
Extracts text from a digital-native PDF (Goodfellow / Bishop / Jurafsky-Martin
are all this type -- no OCR needed) while detecting heading structure.

WHY NOT just pdf.get_text()?
Plain text extraction gives you a wall of text with no idea which lines are
chapter titles vs body paragraphs. We need that structure to build the
[Book -> Chapter -> Section -> Page] lineage the whole project depends on.

HOW HEADING DETECTION WORKS:
PyMuPDF can return text as a "dict" of spans, where each span carries its
FONT SIZE. Body text in a textbook is usually one consistent size (e.g. 10pt).
Headings are visibly larger (e.g. 16pt for chapter titles, 13pt for sections).
So: scan the whole document once to find the most common font size (= body
text), then classify any noticeably larger text as a heading, and rank
heading sizes to decide "is this a chapter or a sub-section".

This is a heuristic, not perfect OCR-grade layout understanding -- but it's
free, fast, runs entirely on CPU, and is good enough for well-structured
textbook PDFs like your three source books.
"""

import fitz  # this is PyMuPDF's import name
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Line:
    text: str
    size: float
    page_number: int  # 1-indexed, human-friendly


@dataclass
class ParsedPage:
    page_number: int
    lines: list = field(default_factory=list)  # list[Line]


def extract_lines(pdf_path: str) -> list[Line]:
    """
    Step 1: Walk every page and pull out every line of text along with
    its font size. This is the raw material for heading detection.
    """
    doc = fitz.open(pdf_path)
    all_lines: list[Line] = []

    for page_index, page in enumerate(doc):
        page_number = page_index + 1  # humans count pages from 1
        # "dict" mode gives us structural info (blocks -> lines -> spans),
        # not just a flat string. Each span knows its own font size.
        raw = page.get_text("dict")

        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                # A line can technically mix font sizes (rare in textbooks);
                # we take the first span's size as representative.
                text = "".join(span["text"] for span in spans).strip()
                if not text:
                    continue
                size = round(spans[0]["size"], 1)
                all_lines.append(Line(text=text, size=size, page_number=page_number))

    doc.close()
    return all_lines


def find_body_text_size(lines: list[Line]) -> float:
    """
    Step 2: Find the most frequent font size across the whole book.
    That's almost always the body paragraph text, since body text
    massively outnumbers headings in a 700-page book.
    """
    size_counts = Counter(line.size for line in lines)
    body_size, _count = size_counts.most_common(1)[0]
    return body_size


def classify_heading_level(size: float, body_size: float) -> Optional[str]:
    """
    Step 3: Decide if a given font size represents a Chapter or a Section
    heading, relative to body text size. Tune these ratios if your book's
    headings don't fit -- print out size_counts (see main.py) to check.
    """
    ratio = size / body_size
    if ratio >= 1.5:
        return "chapter"
    elif ratio >= 1.2:
        return "section"
    else:
        return None  # this is body text, not a heading


def parse_pdf(pdf_path: str, page_offset: int = 0) -> list[ParsedPage]:
    """
    Ties steps 1-3 together: extract lines, figure out what's a heading,
    tag every line, and group lines back into pages.

    page_offset: added to every page number. Needed when parsing an
    EXTRACTED CHAPTER PDF (see chapter_utils.py) -- that file's own page 1
    is really page N of the original book, so citations need the offset
    to stay accurate to the source book rather than reporting "page 1"
    for content that's actually on page 6.
    """
    lines = extract_lines(pdf_path)
    for line in lines:
        line.page_number += page_offset
    body_size = find_body_text_size(lines)

    pages: dict[int, ParsedPage] = {}
    for line in lines:
        level = classify_heading_level(line.size, body_size)
        page = pages.setdefault(line.page_number, ParsedPage(page_number=line.page_number))
        page.lines.append((line, level))  # keep the classification alongside the line

    return [pages[k] for k in sorted(pages.keys())]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py path/to/book.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    lines = extract_lines(pdf_path)
    body_size = find_body_text_size(lines)
    print(f"Detected body text size: {body_size}pt")
    print(f"Total lines extracted: {len(lines)}")

    # Show the first few detected headings so you can sanity-check the heuristic
    print("\nFirst 10 detected headings:")
    shown = 0
    for line in lines:
        level = classify_heading_level(line.size, body_size)
        if level:
            print(f"  [{level.upper():7}] p.{line.page_number}  size={line.size}  {line.text[:70]}")
            shown += 1
        if shown >= 10:
            break
