# AI Query Bot for Large Books

A modular Retrieval-Augmented Generation (RAG) system for querying large technical textbooks with page-cited answers, interactive concept maps, and text-to-speech narration.

**Course:** Beyond Coding: Generative AI, AI Agents and the new Job Landscape — AI-Shala Technologies Pvt. Ltd.  
**Team:** Aman Jain | 
**Mentor:** Dr. Anil Sharma  
**Institute:** Maharaja Surajmal Institute of Technology | AI-Shala Technologies Pvt. Ltd.

---

## 🌟 Key Features

* **Structure-Aware Parsing:** Automatic chapter-boundary detection using embedded Tables of Contents (ToC) or font-size + bold-weight heuristics.
* **Front/Back-Matter Exclusion:** Filters out low-value pages (acknowledgments, ToC, indices, bibliographies) to prevent vector space pollution.
* **Two-Stage Query Routing:**
  * **Book-Level:** Keyword distinctiveness routing (zero LLM cost).
  * **Chapter-Level:** Cosine similarity over local chapter embeddings.
* **Hybrid Retrieval:** Dense vector search (`all-MiniLM-L6-v2`) fused with BM25 lexical search using Reciprocal Rank Fusion (RRF, $k=60$).
* **Grounded Answer Synthesis:** Powered by Gemini (Flash tier) with strict inline chapter/section/page citations and explicit refusal paths for unsupported queries.
* **Interactive Concept Mapping:** Multi-book network graph rendered via `NetworkX` and `vis-network` with offline support.
* **Text-to-Speech Narration:** TTS engine via `edge-tts` with text normalization for LaTeX math and citation brackets.

---

## 📁 Repository Structure

```text
.
├── .gitignore               # Ignores venvs, local vector DBs, and raw PDFs
├── README.md                # Project documentation and setup guide
├── requirements.txt         # Version-pinned dependency manifest
│
├── app.py                   # Streamlit web interface entrypoint
├── pipeline.py              # Centralized CLI entrypoint
│
├── src/                     # Core backend source modules
│   ├── __init__.py
│   ├── answer_synth.py      # Grounded LLM answer generation
│   ├── book_router.py       # Keyword-distinctiveness book router
│   ├── chapter_router.py    # Local vector chapter router
│   ├── chapter_utils.py     # Table of Contents parsing & heuristic detection
│   ├── chunker.py           # Hierarchical chunking engine
│   ├── concept_extractor.py # Concept term and relation extractor
│   ├── concept_graph.py     # NetworkX graph builder
│   ├── embedder.py          # Vector embedding & ChromaDB interface
│   ├── image_captioner.py   # Multimodal figure/diagram captioner
│   ├── llm_client.py        # Centralized Gemini API client
│   ├── pdf_parser.py        # PyMuPDF document extraction parser
│   ├── retriever.py         # Dense + BM25 RRF hybrid retriever
│   ├── schema.py            # Pydantic models (BookChunk, etc.)
│   ├── sentence_streamer.py # Streaming sentence boundary detector
│   ├── text_normalizer.py   # Speech & LaTeX normalization module
│   ├── tts_player.py        # Edge-TTS audio synthesis player
│   └── visualize_graph.py   # Interactive HTML concept map generator
│
├── eval/                    # Benchmark & evaluation suite
│   ├── eval_generation.py   # Faithfulness, Answer Relevance & Context Recall
│   ├── eval_dataset.py
│   ├── eval_retrieval.py    # Hit Rate@K & MRR evaluation suite
│   └── run_eval.py          # End-to-end evaluation orchestrator
│
├── lib/                     # Offline web dependencies
│   └── vis-network.min.js   # Standalone vis-network graphing library
│
└── utils/                   # Debugging & diagnostic scripts
    ├── debug_query.py       # Query execution inspector
    └── debug_search.py      # Lexical index diagnostic tool
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed on your system.

```bash
# Clone repository
git clone [https://github.com/your-username/ai-query-bot-large-books.git](https://github.com/your-username/ai-query-bot-large-books.git)
cd ai-query-bot-large-books

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
# Windows PowerShell: $env:GEMINI_API_KEY="your-gemini-api-key"
```

---

## 💻 Usage

### Launch the Streamlit Web Application
```bash
streamlit run app.py
```

### Command Line Execution (CLI)
```bash
# Ingest a textbook PDF
python pipeline.py ingest deep_learning.pdf "Deep Learning by Ian Goodfellow"

# Query the ingested library
python pipeline.py ask "Describe the architecture of CNN"

# Reset the local vector database & caches
python pipeline.py reset-library
```

---

## 📊 Evaluation & Benchmarking

To run retrieval metrics (**Hit Rate@K**, **MRR**) and generation quality evaluations (**Faithfulness**, **Answer Relevance**, **Context Recall**):

```bash
python -m eval.generate_real_eval.py

python -m eval.run_eval.py
```

* **Retrieval Metrics:** Executed deterministically against hand-written Q&A evaluation pairs (0 API cost).
* **Generation Metrics:** Executed via direct Gemini judge evaluation.

---

## ⚠️ Known Scope & System Limitations

1. **Multimodal Figures:** Diagram and figure captioning are chapter-scoped by design to optimize API token usage.
2. **Conservative Query Routing:** Chapter routing falls back to full-book search whenever top similarity scores fall below the confidence margin, prioritizing search recall over over-narrowing.
3. **Hardware Considerations:** Dense embeddings are generated locally via `all-MiniLM-L6-v2` on CPU for memory efficiency during large textbook processing.

## 🏗️ Architectural Stages (Stage-Wise Pipeline)

The system processes ingested textbooks and resolves user queries through an end-to-end, multi-stage pipeline:

1. **PDF Structure Parsing & Filtering:**
   * Extracts layout, font-size, and font-weight metadata per line using PyMuPDF.
   * Identifies chapter boundaries using embedded Tables of Contents or font-size/bold-weight heuristics.
   * Detects and excludes non-content pages (acknowledgments, indices, bibliographies, ToC) using the chapter tree.
2. **Hierarchical Chunking & Validation:**
   * Groups body text under heading nodes and applies overlapping character splitting (~800 chars, 100 overlap).
   * Enforces strict Pydantic schema validation (`BookChunk`) to ensure metadata integrity (book, chapter, section, page).
3. **Hybrid Indexing (Zero-API Ingestion):**
   * Generates dense vectors locally via `all-MiniLM-L6-v2` and persists them to ChromaDB.
   * Builds an in-memory BM25 lexical index over chunk tokens for exact keyword matches.
4. **Two-Stage Query Routing:**
   * **Stage 1 (Book Router):** Computes keyword distinctiveness across books to scope queries without LLM overhead.
   * **Stage 2 (Chapter Router):** Embeds chapter titles locally and performs cosine similarity matching to restrict search space.
5. **Hybrid Retrieval & Rank Fusion:**
   * Executes parallel dense vector search and BM25 keyword search.
   * Fuses top candidate lists using Reciprocal Rank Fusion (RRF, $k=60$) to generate a final top-$K$ context pool.
6. **Grounded Synthesis & Multi-Modal Output:**
   * Synthesizes answers strictly bounded by retrieved context using Gemini Flash, appending inline citations.
   * Converts text to normalized speech (handling LaTeX math and citation brackets) via `edge-tts`.
   * Extracts entity relationships to construct interactive cross-book concept graphs.

---

## 🗂️ Detailed File Registry

* **`app.py`**: Streamlit frontend application providing multi-tab user interfaces for Q&A, graph visualization, and ingestion.
* **`pipeline.py`**: Centralized CLI entrypoint handling commands like `ingest`, `ask`, and `reset-library`.
* **`src/pdf_parser.py`**: Handles PyMuPDF document loading and structural layout extraction.
* **`src/chapter_utils.py`**: Manages ToC detection, font heuristics, and front/back-matter exclusion logic.
* **`src/chunker.py`**: Implements hierarchical splitting and Pydantic schema validation.
* **`src/embedder.py`**: Interfaces with `sentence-transformers` and local ChromaDB persistence.
* **`src/retriever.py`**: Implements BM25 lexical indexing, dense vector querying, and RRF rank fusion.
* **`src/book_router.py`**: Implements keyword distinctiveness scoring for book selection.
* **`src/chapter_router.py`**: Computes local embedding similarity for chapter selection.
* **`src/answer_synth.py`**: Formulates grounded prompt templates and manages Gemini API generation.
* **`src/concept_extractor.py`**: Batches chunk text to extract entity terms and relationships.
* **`src/concept_graph.py`**: Builds NetworkX graph representations from extracted terms.
* **`src/visualize_graph.py`**: Renders NetworkX objects into interactive HTML canvases using `vis-network`.
* **`src/image_captioner.py`**: Handles multimodal figure and diagram image captioning.
* **`src/text_normalizer.py`**: Cleans LaTeX formulas and citation markers into natural spoken text.
* **`src/tts_player.py`**: Synthesizes and plays audio streams using `edge-tts`.
* **`src/llm_client.py`**: Centralized client wrapper for Gemini API calls.
* **`eval/eval_retrieval.py`**: Computes deterministic retrieval metrics (Hit Rate@K, MRR).
* **`eval/eval_generation.py`**: Computes LLM-as-a-judge metrics (Faithfulness, Answer Relevance, Context Recall).
* **`eval/run_eval.py`**: Orchestrates complete evaluation benchmark runs.

---

## 🛠️ Maintenance & Diagnostic Commands

```bash
# Inspect lexical index and keyword search outputs directly
python utils/debug_search.py --query "convolutional layer"

# Trace query routing decisions and retrieved context chunks
python utils/debug_query.py --query "What is attention mechanism?"

---

## 📈 Evaluation & Benchmark Strategy

The system separates evaluation into two distinct decoupled stages to isolate retrieval performance from generation quality:

### 1. Deterministic Retrieval Evaluation (`eval/eval_retrieval.py`)
* Runs locally with **0 API cost** against hand-written Q&A ground-truth chunks.
* **Hit Rate @ K ($K=1, 3, 5$):** Evaluates if the ground-truth context chunk appears within top-$K$ candidates.
* **Mean Reciprocal Rank (MRR):** Quantifies ranking precision based on the reciprocal rank of the first relevant chunk.

## Retrieval Metrics
| Metric | Score |
|---|---|
| Hit Rate @ 3 | 70.00% |
| MRR | 0.529 |

### 2. LLM-as-a-Judge Generation Evaluation (`eval/eval_generation.py`)
* Leverages direct Gemini calls to evaluate synthesized answers against retrieved contexts.
* **Faithfulness:** Verifies that all assertions in the answer are supported strictly by retrieved chunks.
* **Answer Relevance:** Checks if the generated response directly addresses the user query without fluff.
* **Context Recall:** Assesses whether the retrieved chunks contained all necessary information to answer fully.

---

## 💡 Interactive Features & Edge-Case Handling

* **Interactive NetworkX Graph (`vis-network`):** Generates standalone HTML concept maps with clickable modals showing definitions, source books, and cross-book links.
* **Offline UI Assets:** Bundles local `vis-network.min.js` assets in `lib/` to prevent CDN dependency failures during offline deployment.
* **Orphan Embedding Guard:** Automatically detects and purges orphaned vector store entries missing JSON metadata to prevent runtime `KeyError` crashes.
* **Progressive Batch Caching:** Caches extraction progress to `concept_cache.json` during concept map building, allowing interrupted runs to resume seamlessly.
* **Token Cap Safeguards:** Enforces explicit output limits (`max_tokens=2048`) to avoid token reservation errors during generation.

---

## 🚀 Production Readiness & Next Steps

* [x] Enforce schema validation on all PDF chunk inputs.
* [x] Implement zero-cost two-stage routing to minimize API latency.
* [x] Normalize mathematical LaTeX notation for clean TTS speech output.
* [ ] **Next Steps:** Extend image/figure understanding across entire textbooks beyond chapter-scoped limits.
* [ ] **Next Steps:** Support multi-hop query routing across multiple volumes simultaneously.

Features accessible in the web app:
* **Query Tab:** Ask questions, view grounded answers with expandable page citations, and listen to TTS audio narration.
* **Concept Map Tab:** Explore interactive 2D graph visualizations of extracted terms and cross-book relationships.
* **Library Management:** View currently ingested textbooks, upload new PDFs, or scope searches to specific chapters.

---

## ⚡ Performance Optimization & Cost Management

* **Zero-Cost Routing:** Book routing relies on offline keyword distinctiveness tables, and chapter routing utilizes local `all-MiniLM-L6-v2` embeddings. Neither router makes remote LLM API calls.
* **Local Dense Vectors:** Sentence embeddings are computed entirely on the local CPU to avoid commercial embedding API fees during large book ingestion.
* **Batched Concept Extraction:** Entity and relationship extraction requests are grouped into batches of 15 chunks per LLM call, reducing total extraction API requests by ~93%.
* **Strict Refusal Grounding:** If the top retrieved chunks do not contain sufficient context to answer a query, the model explicitly states that information is missing, preventing hallucinated token consumption.

---

## 📜 Citation & Attribution

If you use this project in your research or academic coursework, please cite it as follows:

```bibtex
@misc{jain2026aiquerybot,
  author       = {Aman Jain},
  title        = {AI Query Bot for Large Books: A Modular Hybrid RAG Architecture},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{[https://github.com/AJpro01/AI-Query-Bot](https://github.com/AJpro01/AI-Query-Bot)}},
  note         = {Guided by Dr. Anil Sharma | Maharaja Surajmal Institute of Technology \& AI-Shala Technologies Pvt. Ltd.}
}
```