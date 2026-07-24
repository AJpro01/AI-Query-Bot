"""
app.py
-------
Run with: streamlit run app.py

FIXES FROM THE FIRST VERSION (all were real bugs, not user error):
1. Enter key now submits the question -- the question input + Ask button
   are wrapped in st.form(), which is what wires Enter to submission in
   Streamlit. A bare st.text_input + separate st.button does NOT do this.
2. The answer no longer disappears when you click "Listen" -- the query
   result is now stored in st.session_state. Previously the Listen button
   was nested inside `if st.button("Ask"):`, so clicking Listen triggered
   a rerun, the Ask button's clicked-state reset to False on that rerun,
   and the entire answer block (including Listen itself) stopped
   rendering. This is a common Streamlit gotcha: any widget interaction
   reruns the whole script from top to bottom, so anything you want to
   persist across that rerun has to live in session_state, not a local
   variable guarded by a button's if-check.
3. Concept map is now an actual tab in the app (previously only existed
   as a separate script, never wired into the frontend).
4. Added a file uploader so you can ingest a new book from the browser
   instead of running `python pipeline.py ingest` in the terminal.
"""

import streamlit as st
import json
import os
import asyncio
import tempfile

LIBRARY_PATH = "library_chunks.json"

st.set_page_config(page_title="AI Query Bot for Large Books", layout="wide")


def apply_background():
    """
    Applies the global background gradient and targets all unique tab container keys
    using a wildcard attribute selector, preventing duplicate element errors.
    """
    st.markdown("""
    <style>
    /* 1. Global App Background Gradient */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: linear-gradient(135deg, #44b8b8 0%, #0F172A 50%, #44b8b8 100%) !important;
    }
    
    /* 2. Target any container whose class contains 'st-key-tab-content-' */
    div[class*="st-key-tab-content-"] {
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15) !important;
        padding: 24px !important;
        margin-top: 16px !important;
        margin-bottom: 24px !important;
    }
    
    /* 3. Ensure native forms inside the glass container merge invisibly */
    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    
    /* 4. Secondary glassmorphism layer for nested content blocks like sources */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 8px !important;
        background: rgba(200, 200, 200, 0.2) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }
    
    /* 5. Readability fix for Alert Boxes (Success, Error, Warning, Info) */
    .stAlertContainer {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        border-radius: 8px !important;
        backdrop-filter: none !important;
    }
    
    /* Force text inside warning/error/info boxes to use our highly visible dark slate color */
    .stAlertContainer p, 
    .stAlertContainer span, 
    .stAlertContainer div,
    .stAlertContainer code {
        color: #ff1414 !important;
    }
    
    /* 6. Micro-transitions for responsive user actions */
    * {
        transition: background-color 0.2s ease-in-out !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_retriever():
    from src.retriever import HybridRetriever
    return HybridRetriever(LIBRARY_PATH)


@st.cache_resource
def get_book_profiles():
    from src.book_router import build_book_profiles
    with open(LIBRARY_PATH) as f:
        chunks = json.load(f)
    return build_book_profiles(chunks, top_n=60)


def get_available_books():
    if not os.path.exists(LIBRARY_PATH):
        return []
    with open(LIBRARY_PATH) as f:
        chunks = json.load(f)
    return sorted(set(c["metadata"]["book_title"] for c in chunks))


def run_query(question: str, manual_book: str = "Auto-detect"):
    from src.book_router import route_query
    from src.answer_synth import generate_answer
    from src.chapter_router import route_to_chapter

    router_note = ""
    if manual_book != "Auto-detect":
        book_filter = manual_book
        router_note = f"Manually scoped to: {manual_book}"
    else:
        profiles = get_book_profiles()
        book_filter = route_query(question, profiles)
        router_note = f"Router scoped to: {book_filter}" if book_filter else "Router unsure — searching all books"

    chapter_filter = None
    if book_filter:
        chapter_filter = route_to_chapter(question, book_filter, LIBRARY_PATH)
        if chapter_filter:
            router_note += f" → {chapter_filter}"

    retriever = get_retriever()
    chunks = retriever.search(question, top_k=6, book_filter=book_filter, chapter_filter=chapter_filter)
    answer = generate_answer(question, chunks)

    return {"router_note": router_note, "chunks": chunks, "answer": answer}


def ingest_uploaded_pdf(uploaded_file, book_title: str):
    """
    Saves an uploaded PDF to a temp file, then runs it through the same
    parse -> chunk -> merge-into-library -> embed pipeline as
    `pipeline.py ingest` does from the command line, PLUS automatic
    chapter cleanup (see chapter_utils.reassign_chapters) -- this removes
    table-of-contents/acknowledgments/bibliography pollution and relabels
    real content to clean chapter titles, all while the temp PDF still
    exists on disk. This is what makes `pipeline.py fix-chapters` work
    automatically for browser-uploaded books, since there's no persistent
    PDF file path for that CLI command to reference otherwise -- the temp
    file this function creates only exists for the duration of this call.
    """
    from src.chunker import build_chunks
    from src.embedder import add_or_update_book
    from src.chapter_utils import reassign_chapters

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    chunks = build_chunks(tmp_path, book_title)

    existing = []
    if os.path.exists(LIBRARY_PATH):
        with open(LIBRARY_PATH) as f:
            existing = json.load(f)
    existing = [c for c in existing if c["metadata"]["book_title"] != book_title]
    new_chunk_dicts = [c.model_dump() for c in chunks]
    existing.extend(new_chunk_dicts)
    with open(LIBRARY_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    add_or_update_book(new_chunk_dicts, book_title)

    # Clean up chapter labels + remove front/back matter -- still while
    # tmp_path exists, since this needs the actual PDF to read its TOC.
    cleanup_result = {"relabeled": 0, "removed": 0}
    try:
        cleanup_result = reassign_chapters(tmp_path, book_title, library_path=LIBRARY_PATH)
    except Exception as e:
        print(f"Chapter cleanup failed for '{book_title}': {e}. "
              f"Chunks were still ingested successfully, just with raw heuristic chapter labels.")

    os.unlink(tmp_path)

    # Cached retriever/router profiles are now stale -- force them to
    # reload on the next call, or the newly ingested book won't show up
    # (or will show up with stale pre-cleanup chapter/chunk data).
    get_retriever.clear()
    get_book_profiles.clear()

    final_count = len(chunks) - cleanup_result["removed"]
    return final_count, cleanup_result["removed"]


def get_chapters_for_book(book_title: str) -> list[str]:
    """
    Returns the distinct chapter names already present in this book's
    ingested chunks, in the order they first appear (by page number).
    Excludes "Front/Back Matter" (see chapter_utils.reassign_chapters) --
    if a book hasn't had its chapters cleaned up yet, run
    `python pipeline.py fix-chapters <pdf> "<Book Title>"` first, or this
    will just show whatever raw heuristic labels were set at ingest time.
    """
    with open(LIBRARY_PATH) as f:
        all_chunks = json.load(f)
    book_chunks = [c for c in all_chunks if c["metadata"]["book_title"] == book_title]
    book_chunks.sort(key=lambda c: c["metadata"]["page_number"])

    seen = []
    for c in book_chunks:
        chapter = c["metadata"]["chapter"]
        if chapter not in seen and chapter != "Front/Back Matter":
            seen.append(chapter)
    return seen


def build_and_render_concept_map(book_title: str, chapter_filter: str = None):
    from src.concept_extractor import extract_from_book
    from src.concept_graph import build_graph
    from src.visualize_graph import render

    with open(LIBRARY_PATH) as f:
        all_chunks = json.load(f)
    book_chunks = [c for c in all_chunks if c["metadata"]["book_title"] == book_title]
    if chapter_filter:
        book_chunks = [c for c in book_chunks if c["metadata"]["chapter"] == chapter_filter]

    extraction = extract_from_book(book_chunks)
    graph_title = f"{book_title} — {chapter_filter}" if chapter_filter else book_title
    graph = build_graph({graph_title: extraction})

    safe_name = (book_title + (chapter_filter or "")).replace(" ", "_")
    output_path = f"concept_map_{safe_name}.html"
    render(graph, output_path=output_path, title=graph_title)
    return output_path


def ask_tab(books):
    with st.form("query_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input("Ask a question about your library:")
        with col2:
            manual_book = st.selectbox("Scope to book", ["Auto-detect"] + books)
        submitted = st.form_submit_button("Ask")  # Enter key submits this form

    if submitted and question:
        with st.spinner("Searching and generating answer..."):
            st.session_state["last_result"] = run_query(question, manual_book)
            st.session_state.pop("audio_file", None)  # clear stale audio from a previous question

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.caption(result["router_note"])
        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for c in result["chunks"]:
            m = c["metadata"]
            with st.expander(f"{m['book_title']} — {m['chapter']} | {m['section']} (p.{m['page_number']})"):
                st.write(c["text"])

        if st.button("🔊 Listen to answer"):
            with st.spinner("Synthesizing audio (needs internet access to Microsoft's TTS service)..."):
                from src.tts_player import speak_answer
                st.session_state["audio_file"] = asyncio.run(speak_answer(result["answer"], output_dir="."))

        if "audio_file" in st.session_state:
            st.audio(st.session_state["audio_file"])


def upload_tab():
    st.write("Ingest a new book directly from the browser (equivalent to running `pipeline.py ingest` in the terminal).")

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    # Show a success message left over from BEFORE the rerun this function
    # triggers after ingesting -- st.success() called right before
    # st.rerun() gets wiped out instantly by the rerun and is never seen,
    # so the message has to be stashed in session_state and shown on the
    # NEXT run instead, then cleared so it doesn't stick around forever.
    if "upload_success_message" in st.session_state:
        st.success(st.session_state.pop("upload_success_message"))

    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key=f"uploader_{st.session_state['uploader_key']}")
        book_title = st.text_input("Book title (used for citations and routing)")
        submitted = st.form_submit_button("Ingest this book")  # Enter key submits this form

    if submitted and uploaded_file and book_title:
        with st.spinner(f"Parsing, chunking, and embedding '{book_title}'... this can take a while on first run (downloads the embedding model)."):
            num_chunks, num_removed = ingest_uploaded_pdf(uploaded_file, book_title)
        # clear_on_submit resets the text input, but file_uploader specifically
        # needs its widget key changed to fully drop the staged file in some
        # Streamlit versions -- bumping the key forces a fresh widget instance.
        st.session_state["uploader_key"] += 1
        cleanup_note = f" ({num_removed} front/back-matter chunk(s) automatically excluded.)" if num_removed else ""
        st.session_state["upload_success_message"] = f"Ingested {num_chunks} chunks for '{book_title}'.{cleanup_note} Switch to the Ask tab to query it."
        st.rerun()
    elif submitted:
        st.warning("Please provide both a PDF file and a book title before submitting.")


def concept_map_tab(books):
    selected_book = st.selectbox("Book", books, key="concept_map_book_select")

    chapters = get_chapters_for_book(selected_book)
    chapter_choice = st.selectbox("Chapter", ["All chapters (not recommended for a full book)"] + chapters,
                                    key="concept_map_chapter_select")
    chapter_filter = None if chapter_choice.startswith("All chapters") else chapter_choice

    if chapter_filter:
        num_chunks = sum(1 for c in _load_book_chunks(selected_book) if c["metadata"]["chapter"] == chapter_filter)
        st.caption(f"'{chapter_filter}' has {num_chunks} chunks — roughly {(num_chunks + 7) // 8} batched LLM calls.")
    else:
        num_chunks = len(_load_book_chunks(selected_book))
        st.caption(f"⚠️ Whole book: {num_chunks} chunks — roughly {(num_chunks + 7) // 8} batched LLM calls. "
                   f"This is very likely to exhaust a free-tier quota partway through. Pick a chapter instead.")

    if st.button("Generate concept map"):
        with st.spinner(f"Extracting concepts for '{chapter_filter or selected_book}'..."):
            output_path = build_and_render_concept_map(selected_book, chapter_filter=chapter_filter)
            st.session_state["concept_map_path"] = output_path

    if "concept_map_path" in st.session_state and os.path.exists(st.session_state["concept_map_path"]):
        st.iframe(st.session_state["concept_map_path"], height=700)


def _load_book_chunks(book_title: str) -> list:
    with open(LIBRARY_PATH) as f:
        all_chunks = json.load(f)
    return [c for c in all_chunks if c["metadata"]["book_title"] == book_title]


def main():
    apply_background()
    st.title("📚 AI Query Bot for Large Books")

    books = get_available_books()

    tab_ask, tab_upload, tab_map = st.tabs(["Ask", "Upload Book", "Concept Map"])

    with tab_upload:
        # Unique Key 1
        with st.container(key="tab-content-upload"):
            upload_tab()

    with tab_ask:
        # Unique Key 2
        with st.container(key="tab-content-ask"):
            if not books:
                st.warning("No books ingested yet. Use the 'Upload Book' tab, or run "
                            "`python pipeline.py ingest <pdf> \"<Book Title>\"` in the terminal.")
            else:
                ask_tab(books)

    with tab_map:
        # Unique Key 3
        with st.container(key="tab-content-map"):
            if not books:
                st.warning("No books ingested yet.")
            else:
                concept_map_tab(books)


if __name__ == "__main__":
    main()
