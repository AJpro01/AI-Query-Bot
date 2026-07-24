"""
concept_graph.py
------------------
Builds a networkx graph from the terms/relations extracted per book, then
adds cross-book links -- but ONLY via exact term-string matching, never by
asking an LLM to guess a connection between books. This directly follows
the anti-hallucination design from the original project analysis: an LLM
inferring "these two things from different books are related" is exactly
the kind of speculative leap that erodes trust in the concept map as new
books get added.

EDGE TYPES:
  INTERNAL_LINK    -- relation extracted from text within the SAME book.
                       These come straight from concept_extractor.py.
  CROSS_REFERENCE   -- added here, only when the exact same term (case-
                       insensitive, whitespace-normalized) appears in more
                       than one book's term list. No LLM involved in this
                       decision -- it's a deterministic string match.
"""

import networkx as nx


def _normalize(term: str) -> str:
    return term.strip().lower()


def build_graph(extractions_by_book: dict) -> nx.DiGraph:
    """
    extractions_by_book: {book_title: {"terms": [...], "relations": [...]}}
    as produced by concept_extractor.extract_from_book(), one entry per book.
    """
    graph = nx.DiGraph()

    # Step 1: add every term as a node, tagged with which book(s) define it
    term_to_books = {}  # normalized term -> set of book titles
    for book_title, extraction in extractions_by_book.items():
        for term_entry in extraction["terms"]:
            norm = _normalize(term_entry["term"])
            if not graph.has_node(norm):
                graph.add_node(norm, label=term_entry["term"], definition=term_entry["definition"], books=set())
            graph.nodes[norm]["books"].add(book_title)
            term_to_books.setdefault(norm, set()).add(book_title)

    # Step 2: add INTERNAL_LINK edges from each book's extracted relations
    for book_title, extraction in extractions_by_book.items():
        for rel in extraction["relations"]:
            src, tgt = _normalize(rel["source"]), _normalize(rel["target"])
            if graph.has_node(src) and graph.has_node(tgt):
                graph.add_edge(src, tgt, relation=rel["relation"], edge_type="INTERNAL_LINK", book=book_title)

    # Step 3: add CROSS_REFERENCE edges -- deterministic exact-match only,
    # connecting the SAME term when it's defined in more than one book.
    # This does not claim any semantic relationship beyond "same term".
    for norm_term, books in term_to_books.items():
        if len(books) > 1:
            # Nothing to connect to another node here since it's one node
            # shared across books -- record this fact on the node itself
            # rather than inventing a self-loop edge.
            graph.nodes[norm_term]["cross_referenced_in"] = sorted(books)

    return graph


def graph_summary(graph: nx.DiGraph) -> str:
    internal_edges = sum(1 for _, _, d in graph.edges(data=True) if d.get("edge_type") == "INTERNAL_LINK")
    cross_ref_terms = sum(1 for _, d in graph.nodes(data=True) if d.get("cross_referenced_in"))
    lines = [
        f"Nodes (terms): {graph.number_of_nodes()}",
        f"INTERNAL_LINK edges: {internal_edges}",
        f"Terms shared across multiple books (cross-referenced): {cross_ref_terms}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Stand-in extraction data so this can be tested without a live Gemini
    # call. Structurally identical to what concept_extractor.py returns.
    fake_extractions = {
        "Test Book": {
            "terms": [
                {"term": "Perceptron", "definition": "A simple linear classifier for binary classification."},
                {"term": "Gradient Descent", "definition": "Optimization algorithm minimizing a loss function."},
                {"term": "Backpropagation", "definition": "Computes gradients via the chain rule."},
                {"term": "Vanishing Gradients", "definition": "Gradients become too small in deep networks."},
            ],
            "relations": [
                {"source": "Backpropagation", "relation": "computes gradients using", "target": "Gradient Descent"},
                {"source": "Vanishing Gradients", "relation": "is a failure mode of", "target": "Backpropagation"},
            ],
        },
        "Databases Book": {
            "terms": [
                {"term": "Primary Key", "definition": "Uniquely identifies each row in a table."},
                {"term": "Foreign Key", "definition": "References the primary key of another table."},
                {"term": "Normalization", "definition": "Reduces redundancy in database design."},
                {"term": "Gradient Descent", "definition": "Mentioned in passing, unrelated context (test of cross-ref)."},
            ],
            "relations": [
                {"source": "Foreign Key", "relation": "references", "target": "Primary Key"},
            ],
        },
    }

    graph = build_graph(fake_extractions)
    print(graph_summary(graph))
    print()
    print("Nodes cross-referenced across books:")
    for node, data in graph.nodes(data=True):
        if data.get("cross_referenced_in"):
            print(f"  '{data['label']}' appears in: {data['cross_referenced_in']}")
