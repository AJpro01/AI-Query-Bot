"""
chapter_utils.py
-------------------
Lets you process ONE chapter of a book instead of all 700 pages -- for
faster iteration while testing, and to scope concept-map generation
(which burns API calls per chunk) to something reasonable.

TWO WAYS TO FIND CHAPTER BOUNDARIES:

1. EMBEDDED TOC (preferred): most real textbook PDFs -- including
   Goodfellow's -- ship with an embedded table of contents/bookmark
   structure. PyMuPDF reads this directly via doc.get_toc(), giving exact
   page numbers with zero guessing. This is faster and far more reliable
   than our font-size heuristic in pdf_parser.py.

2. HEURISTIC FALLBACK: if a PDF has no embedded TOC (get_toc() returns
   empty -- happens with some scanned or DRM-stripped copies), fall back
   to the same font-size heading detection pdf_parser.py already uses, and
   derive page ranges from where each detected chapter heading first
   appears.

Once you know a chapter's page range, extract_chapter_pdf() pulls just
those pages into a small standalone PDF -- THAT smaller file is what you
then run through chunker.py / pipeline.py ingest, instead of the whole book.
"""

import os
import json
import fitz


FRONT_MATTER_KEYWORDS = [
    "preface", "acknowledgment", "acknowledgement", "dedication", "foreword",
    "contents", "table of contents", "about the author", "copyright",
    "list of figures", "list of tables", "notation", "how to use this book",
]


def _is_front_matter(title: str) -> bool:
    lowered = title.strip().lower()
    return any(keyword in lowered for keyword in FRONT_MATTER_KEYWORDS)


def get_content_start_page(pdf_path: str) -> int:
    """
    Returns the page number where the book's REAL content starts -- i.e.
    skips title page, copyright, dedication, and (importantly) entries
    that ARE in the TOC but are still front matter, like "Preface" or
    "Acknowledgments", which many books list as formal TOC/chapter-level
    entries before "Chapter 1" actually begins.

    Falls back to page 1 if no chapters are found at all (nothing to skip).
    """
    chapters = list_chapters(pdf_path)
    top_level = [c for c in chapters if c["level"] == 1] or chapters
    for ch in top_level:
        if not _is_front_matter(ch["title"]):
            return ch["start_page"]
    return 1  # everything looked like front matter, or no chapters found -- don't skip anything


def get_content_start_page(pdf_path: str) -> int:
    """
    Returns the page number where the first REAL chapter begins, per the
    embedded TOC -- i.e. everything before this (title page, dedication,
    acknowledgments, preface, table of contents itself) is front matter
    that never got its own TOC/bookmark entry in the first place.

    Returns 1 (meaning "don't skip anything") if the PDF has no embedded
    TOC at all -- without a TOC we have no reliable signal for where real
    content starts, and guessing wrong risks silently dropping real
    chapter 1 content, which is worse than including a few pages of
    front matter.
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    doc.close()

    if not toc:
        return 1

    top_level = min(level for level, _, _ in toc)
    start_pages = [page for level, _, page in toc if level == top_level]
    return min(start_pages)


NON_CHAPTER_KEYWORDS = [
    "acknowledg", "dedicat", "preface", "notation", "table of contents",
    "bibliograph", "reference", "index", "appendix", "foreword",
    "about the author", "glossary", "copyright", "cover", "title page",
    "how to use this book", "list of figures", "list of tables",
]


def _looks_like_real_chapter(title: str) -> bool:
    lower = title.lower().strip()
    return not any(kw in lower for kw in NON_CHAPTER_KEYWORDS)


def get_content_chapters(pdf_path: str) -> list[dict]:
    """
    Returns only the REAL chapters -- excludes front matter (title page,
    dedication, acknowledgments, preface, notation, table of contents) and
    back matter (bibliography, references, index, appendix, glossary),
    and excludes stray non-chapter entries a heuristic might have picked up
    (like isolated math symbols mistaken for headings due to large font size).

    Prefers the embedded TOC's TOP-LEVEL entries only (subsections like
    "3.1", "3.2" are absorbed into their parent chapter's page range
    automatically, since list_chapters() already computes ranges that way).

    If a PDF has no embedded TOC, falls back to grouping the heuristically-
    detected headings by their LEADING NUMBER (e.g. "3.1 Convolutions" and
    "3.2 Pooling" both become part of "Chapter 3"), which is what most
    textbooks without embedded bookmarks still use as their numbering
    convention even without a real bookmark structure.
    """
    all_chapters = list_chapters(pdf_path)
    if not all_chapters:
        return []

    has_real_toc = any("level" in ch for ch in all_chapters) and _has_embedded_toc(pdf_path)

    if has_real_toc:
        top_level = min(ch["level"] for ch in all_chapters)
        candidates = [ch for ch in all_chapters if ch["level"] == top_level]
        return [ch for ch in candidates if _looks_like_real_chapter(ch["title"])]

    # No embedded TOC -- group heuristic headings by leading chapter number
    return _group_by_leading_number(all_chapters)


def _has_embedded_toc(pdf_path: str) -> bool:
    doc = fitz.open(pdf_path)
    has_toc = bool(doc.get_toc())
    doc.close()
    return has_toc


def _group_by_leading_number(chapters: list[dict]) -> list[dict]:
    """
    Groups headings like "3.1 Convolutions", "3.2 Pooling", "7.1 RNNs" by
    their leading top-level number ("3", "7"), merging each group's page
    range and using the FIRST heading in each group as a representative
    title (or synthesizing "Chapter N" if no clean title is available).
    Also drops anything matching NON_CHAPTER_KEYWORDS and anything with no
    leading number at all (isolated symbols, stray large-font artifacts).
    """
    import re

    groups: dict[str, dict] = {}
    order: list[str] = []

    for ch in chapters:
        if not _looks_like_real_chapter(ch["title"]):
            continue
        match = re.match(r"^(\d+)(?:\.\d+)*\s", ch["title"] + " ")
        if not match:
            continue  # no leading number at all -- likely a stray heuristic artifact, skip it
        number = match.group(1)

        if number not in groups:
            groups[number] = {
                "title": f"Chapter {number}",
                "start_page": ch["start_page"],
                "end_page": ch["end_page"],
                "level": 1,
            }
            order.append(number)
        else:
            groups[number]["start_page"] = min(groups[number]["start_page"], ch["start_page"])
            groups[number]["end_page"] = max(groups[number]["end_page"], ch["end_page"])

    return [groups[n] for n in order]


def reassign_chapters(pdf_path: str, book_title: str, library_path: str = "library_chunks.json"):
    """
    Cleans up an ALREADY-INGESTED book two ways, without needing to
    re-embed the content that IS kept:

    1. RELABELS chunks that fall within a real chapter's page range to
       that chapter's clean top-level title (e.g. a chunk heuristically
       tagged "9.1 The Convolution Operation" becomes "Chapter 9:
       Convolutional Networks").
    2. DELETES chunks that fall OUTSIDE every real chapter's page range --
       front matter (title page, dedication, acknowledgments, notation,
       table of contents) and back matter (bibliography, index, appendix).

    Deletion (not just relabeling) matters because the Table of Contents
    and Acknowledgments pages are a genuine RETRIEVAL POLLUTION problem,
    not just a cosmetic citation issue: a page that lists "9.1 The
    Convolution Operation... 9.10 The Neuroscientific Basis..." repeats
    chapter-title keywords densely in a tiny amount of text, which can
    out-rank real chapter content for exactly the kind of broad
    conceptual query ("describe the architecture of CNN") those keywords
    match on. Relabeling alone doesn't stop that pollution; removal does.

    Deletes from BOTH library_chunks.json AND the Chroma vector store, so
    the two stay in sync -- deleting from only one would leave dense
    search still finding chunk IDs that no longer exist in the citation
    metadata, causing a KeyError at query time.
    """
    content_chapters = get_content_chapters(pdf_path)
    if not content_chapters:
        print("No usable chapter structure found -- nothing to clean up.")
        return {"relabeled": 0, "removed": 0}

    with open(library_path) as f:
        all_chunks = json.load(f)

    kept_chunks = []
    removed_ids = []
    relabeled = 0

    for chunk in all_chunks:
        if chunk["metadata"]["book_title"] != book_title:
            kept_chunks.append(chunk)
            continue

        page = chunk["metadata"]["page_number"]
        matched_title = None
        for ch in content_chapters:
            if ch["start_page"] <= page <= ch["end_page"]:
                matched_title = ch["title"]
                break

        if matched_title is None:
            removed_ids.append(chunk["chunk_id"])
            continue  # dropped -- front/back matter

        if chunk["metadata"]["chapter"] != matched_title:
            chunk["metadata"]["chapter"] = matched_title
            relabeled += 1
        kept_chunks.append(chunk)

    with open(library_path, "w") as f:
        json.dump(kept_chunks, f, indent=2)

    if removed_ids:
        try:
            import chromadb
            from src.embedder import CHROMA_DB_PATH, COLLECTION_NAME
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            collection = client.get_collection(COLLECTION_NAME)
            collection.delete(ids=removed_ids)
        except Exception as e:
            print(f"Warning: removed {len(removed_ids)} chunks from library_chunks.json, "
                  f"but could not remove them from the Chroma vector store ({e}). "
                  f"They may still surface in dense search results until you re-embed.")

    print(f"Relabeled {relabeled} chunk(s) to clean chapter titles. "
          f"Removed {len(removed_ids)} front/back-matter chunk(s) entirely "
          f"(table of contents, acknowledgments, bibliography, etc.) "
          f"using {len(content_chapters)} real chapter(s): {[c['title'] for c in content_chapters]}")
    return {"relabeled": relabeled, "removed": len(removed_ids)}


def list_chapters(pdf_path: str) -> list[dict]:
    """
    Returns [{"title": ..., "start_page": ..., "end_page": ..., "level": ...}, ...]
    using 1-indexed, human-friendly page numbers (matching the rest of
    this project's convention). Tries the embedded TOC first; falls back
    to font-size heuristic detection if the PDF has no embedded TOC.
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # list of [level, title, page], 1-indexed already
    total_pages = doc.page_count
    doc.close()

    if toc:
        chapters = []
        for i, (level, title, start_page) in enumerate(toc):
            # end_page is the page right before the next entry AT THE SAME
            # OR SHALLOWER level starts (so a chapter's range absorbs its
            # own subsections, not just up to the very next TOC line)
            end_page = total_pages
            for level2, title2, start_page2 in toc[i + 1:]:
                if level2 <= level:
                    end_page = start_page2 - 1
                    break
            chapters.append({
                "title": title, "start_page": start_page,
                "end_page": max(end_page, start_page), "level": level,
            })
        return chapters

    print("No embedded TOC found -- falling back to font-size heading detection. "
          "This is less precise; check the results before trusting them.")
    return _heuristic_chapters(pdf_path)


def _heuristic_chapters(pdf_path: str) -> list[dict]:
    """Fallback using the same heading detection pdf_parser.py already does."""
    from src.pdf_parser import parse_pdf

    parsed_pages = parse_pdf(pdf_path)
    headings = []
    for page in parsed_pages:
        for line, level in page.lines:
            if level == "chapter":
                headings.append({"title": line.text, "start_page": page.page_number, "level": 1})

    if not headings:
        return []

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()

    for i, h in enumerate(headings):
        h["end_page"] = headings[i + 1]["start_page"] - 1 if i + 1 < len(headings) else total_pages
    return headings


def extract_chapter_pdf(pdf_path: str, start_page: int, end_page: int, output_path: str):
    """
    Pulls pages [start_page, end_page] (1-indexed, inclusive) out of
    pdf_path into a new standalone PDF at output_path. This smaller file
    is what you then run through the normal ingest/chunk/embed pipeline,
    or through concept_extractor.py -- both run much faster on, say, a
    30-page chapter than a 700-page book.
    """
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()
    # PyMuPDF page indices are 0-based; our chapter list is 1-based
    new_doc.insert_pdf(doc, from_page=start_page - 1, to_page=end_page - 1)
    new_doc.save(output_path)
    new_doc.close()
    doc.close()


if __name__ == "__main__":
    import sys
    import json

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "toc_test.pdf"

    chapters = list_chapters(pdf_path)
    print(f"Found {len(chapters)} chapter(s) in {pdf_path}:\n")
    for i, ch in enumerate(chapters):
        indent = "  " * (ch["level"] - 1)
        print(f"{i}: {indent}{ch['title']}  (pages {ch['start_page']}-{ch['end_page']})")

    if len(sys.argv) > 2:
        chapter_index = int(sys.argv[2])
        ch = chapters[chapter_index]
        output_path = f"chapter_{chapter_index}.pdf"
        extract_chapter_pdf(pdf_path, ch["start_page"], ch["end_page"], output_path)
        print(f"\nExtracted '{ch['title']}' (pages {ch['start_page']}-{ch['end_page']}) to {output_path}")
