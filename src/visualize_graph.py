"""
visualize_graph.py
---------------------
Renders the concept graph as a self-contained HTML file, styled to match
the reference design: dark indigo/purple gradient background, rounded
card-style nodes, dotted connector lines, and a click-to-open modal panel
showing a term's full definition and related concepts -- rather than only
a hover tooltip (what the previous pyvis-based version did).

DESIGN TOKENS:
  Background:     linear-gradient, #14101f -> #241a3d -> #1a1330
  Card fill:       rgba(255,255,255,0.05), border rgba(168,139,250,0.28)
  Hub card fill:   rgba(139,92,246,0.30), border #a78bfa  (most-connected node per book)
  Cross-ref fill:  rgba(245,196,81,0.20), border #f5c451  (term shared across books)
  Text primary:    #f4f0ff      Text muted: #9c93b8
  Edge line:       dotted rgba(168,139,250,0.35)
  Accent gradient: linear-gradient(90deg, #8b5cf6, #38bdf8)  -- modal CTA + badges
  Font:            Inter / system sans stack

This uses vis-network directly (via CDN) instead of pyvis, because pyvis's
high-level API doesn't expose enough control over node shape/HTML labels
or click-driven custom UI to hit this design -- hand-writing the vis-network
config gives full control over both.
"""

import json
import os
import html as html_module
import networkx as nx

BOOK_ACCENTS = ["#38bdf8", "#f97316", "#34d399", "#f472b6", "#facc15"]
CROSS_REF_COLOR = "#f5c451"
HUB_COLOR = "#a78bfa"

_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
_VIS_NETWORK_PATH = os.path.join(_LIB_DIR, "vis-network.min.js")


def _load_vis_network_js() -> str:
    """
    Reads the vis-network library from lib/vis-network.min.js and returns
    its contents, so it can be embedded DIRECTLY in the generated HTML
    (inline <script>...</script>, not <script src="...">). This makes each
    generated concept map a single, fully self-contained, offline-capable
    file -- no CDN dependency, and no risk of the map breaking if a CDN is
    unreachable (this was switched from a CDN <script src> after finding
    that this project's own development sandbox couldn't reach unpkg.com;
    better to remove the dependency for anyone else's environment too).
    """
    if not os.path.exists(_VIS_NETWORK_PATH):
        raise FileNotFoundError(
            f"vis-network.min.js not found at {_VIS_NETWORK_PATH}. "
            "Make sure the lib/ folder (shipped alongside this script) is present."
        )
    with open(_VIS_NETWORK_PATH, encoding="utf-8") as f:
        return f.read()


def _esc(text: str) -> str:
    return html_module.escape(text or "", quote=True)


def _build_node_payload(graph: nx.DiGraph):
    degree = dict(graph.degree())
    max_degree = max(degree.values()) if degree else 1

    book_colors = {}
    nodes = []
    for node_id, data in graph.nodes(data=True):
        books = sorted(data.get("books", []))
        primary_book = books[0] if books else "Unknown"
        if primary_book not in book_colors:
            book_colors[primary_book] = BOOK_ACCENTS[len(book_colors) % len(BOOK_ACCENTS)]

        is_cross_ref = bool(data.get("cross_referenced_in"))
        is_hub = degree.get(node_id, 0) >= max(2, max_degree)

        if is_cross_ref:
            fill, border = "rgba(245,196,81,0.20)", CROSS_REF_COLOR
        elif is_hub:
            fill, border = "rgba(139,92,246,0.30)", HUB_COLOR
        else:
            fill, border = "rgba(255,255,255,0.05)", book_colors[primary_book]

        related = []
        for _, target, edata in graph.out_edges(node_id, data=True):
            related.append({"term": graph.nodes[target].get("label", target), "relation": edata.get("relation", "")})
        for source, _, edata in graph.in_edges(node_id, data=True):
            related.append({"term": graph.nodes[source].get("label", source), "relation": edata.get("relation", "")})

        label = data.get("label", node_id)
        subtitle = f"{len(related)} connection{'s' if len(related) != 1 else ''}"

        nodes.append({
            "id": node_id,
            "label": f"<b>{_esc(label)}</b>\n<i>{_esc(subtitle)}</i>",
            "shape": "box",
            "shapeProperties": {"borderRadius": 10},
            "margin": 14,
            "widthConstraint": {"minimum": 140, "maximum": 200},
            "color": {"background": fill, "border": border, "highlight": {"background": fill, "border": border}},
            "font": {"multi": "html", "color": "#f4f0ff", "size": 15, "face": "Inter, Segoe UI, sans-serif",
                     "vadjust": -2, "bold": {"color": "#f4f0ff"}, "ital": {"color": "#9c93b8", "size": 12}},
            "borderWidth": 1.5,
            "term": label,
            "definition": data.get("definition", ""),
            "books": books,
            "isCrossRef": is_cross_ref,
            "related": related[:8],
        })

    edges = []
    for source, target, edata in graph.edges(data=True):
        edges.append({
            "from": source, "to": target,
            "label": edata.get("relation", ""),
            "arrows": "to",
            "dashes": True,
            "color": {"color": "rgba(168,139,250,0.35)", "highlight": "#a78bfa"},
            "font": {"color": "#9c93b8", "size": 11, "strokeWidth": 0, "align": "middle"},
            "smooth": {"type": "continuous"},
        })

    return nodes, edges, book_colors


PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>{vis_network_js}</script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0; height: 100vh; overflow: hidden;
    font-family: Inter, "Segoe UI", sans-serif;
    background: radial-gradient(ellipse at 20% 15%, rgba(139,92,246,0.18), transparent 55%),
                radial-gradient(ellipse at 80% 85%, rgba(56,189,248,0.12), transparent 55%),
                linear-gradient(160deg, #14101f 0%, #241a3d 50%, #1a1330 100%);
  }}
  #network {{ width: 100%; height: 100vh; }}

  #legend {{
    position: absolute; top: 16px; left: 16px; z-index: 5;
    background: rgba(20,16,31,0.75); border: 1px solid rgba(168,139,250,0.25);
    border-radius: 10px; padding: 10px 14px; color: #f4f0ff; font-size: 12px;
    backdrop-filter: blur(6px);
  }}
  #legend .row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  #legend .swatch {{ width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }}

  #modal-overlay {{
    display: none; position: fixed; inset: 0; z-index: 10;
    background: rgba(10,8,16,0.55); backdrop-filter: blur(2px);
    align-items: center; justify-content: center;
  }}
  #modal {{
    width: 420px; max-width: 90vw; max-height: 78vh; overflow-y: auto;
    background: linear-gradient(165deg, #241a3d 0%, #1a1330 100%);
    border: 1px solid rgba(168,139,250,0.3); border-radius: 16px;
    padding: 24px; color: #f4f0ff; box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }}
  #modal .eyebrow {{
    font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
    color: #a78bfa; font-weight: 600; margin-bottom: 6px;
  }}
  #modal .title {{ font-size: 22px; font-weight: 700; margin-bottom: 14px; line-height: 1.3; }}
  #modal .definition {{ font-size: 14px; color: #d8d2ea; line-height: 1.55; margin-bottom: 18px; }}
  #modal .section-label {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: #9c93b8; margin: 16px 0 8px;
  }}
  #modal .related-row {{
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px; padding: 9px 12px; margin-bottom: 6px; font-size: 13px;
  }}
  #modal .related-row .rel {{ color: #9c93b8; font-size: 11px; }}
  #modal .cross-ref-note {{
    font-size: 12px; color: #f5c451; background: rgba(245,196,81,0.12);
    border: 1px solid rgba(245,196,81,0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 14px;
  }}
  #modal .close-btn {{
    position: absolute; top: 18px; right: 18px; width: 30px; height: 30px;
    border-radius: 8px; border: 1px solid rgba(255,255,255,0.15); background: rgba(255,255,255,0.06);
    color: #f4f0ff; cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center;
  }}
  #modal .close-btn:hover {{ background: rgba(255,255,255,0.14); }}
  #modal-inner {{ position: relative; }}
  #modal .cta {{
    width: 100%; margin-top: 8px; padding: 12px; border: none; border-radius: 10px;
    background: linear-gradient(90deg, #8b5cf6, #38bdf8); color: white; font-weight: 600;
    font-size: 14px; cursor: pointer;
  }}
</style>
</head>
<body>

<div id="legend">
  {legend_rows}
  <div class="row"><span class="swatch" style="background:{hub_color}"></span>Most-connected term</div>
  <div class="row"><span class="swatch" style="background:{cross_ref_color}"></span>Shared across books</div>
</div>

<div id="network"></div>

<div id="modal-overlay">
  <div id="modal">
    <div id="modal-inner">
      <button class="close-btn" onclick="closeModal()">✕</button>
      <div class="eyebrow" id="modal-book"></div>
      <div class="title" id="modal-title"></div>
      <div id="modal-cross-ref-note"></div>
      <div class="definition" id="modal-definition"></div>
      <div class="section-label" id="modal-related-label"></div>
      <div id="modal-related"></div>
      <button class="cta" onclick="closeModal()">Close</button>
    </div>
  </div>
</div>

<script>
  const nodesData = {nodes_json};
  const edgesData = {edges_json};

  const nodes = new vis.DataSet(nodesData);
  const edges = new vis.DataSet(edgesData);

  const container = document.getElementById('network');
  const data = {{ nodes: nodes, edges: edges }};
  const options = {{
    physics: {{
      stabilization: {{ iterations: 300 }},
      barnesHut: {{ gravitationalConstant: -20000, springLength: 280, springConstant: 0.02, avoidOverlap: 1 }},
    }},
    interaction: {{ hover: true, tooltipDelay: 150 }},
    edges: {{ width: 1.5, font: {{ strokeWidth: 4, strokeColor: '#14101f' }} }},
    layout: {{ improvedLayout: true }},
  }};
  const network = new vis.Network(container, data, options);

  function closeModal() {{
    document.getElementById('modal-overlay').style.display = 'none';
  }}

  network.on('click', function(params) {{
    if (params.nodes.length === 0) {{ closeModal(); return; }}
    const node = nodes.get(params.nodes[0]);

    document.getElementById('modal-book').textContent = node.books.join(' · ') || 'Concept';
    document.getElementById('modal-title').textContent = node.term;
    document.getElementById('modal-definition').textContent = node.definition || 'No definition extracted.';

    const noteEl = document.getElementById('modal-cross-ref-note');
    if (node.isCrossRef) {{
      noteEl.style.display = 'block';
      noteEl.textContent = '⚡ This term appears in multiple books: ' + node.books.join(', ');
    }} else {{
      noteEl.style.display = 'none';
    }}

    const relLabel = document.getElementById('modal-related-label');
    const relContainer = document.getElementById('modal-related');
    relContainer.innerHTML = '';
    if (node.related.length === 0) {{
      relLabel.textContent = '';
    }} else {{
      relLabel.textContent = 'Related concepts';
      node.related.forEach(r => {{
        const row = document.createElement('div');
        row.className = 'related-row';
        row.innerHTML = '<span>' + r.term + '</span><span class="rel">' + r.relation + '</span>';
        relContainer.appendChild(row);
      }});
    }}

    document.getElementById('modal-overlay').style.display = 'flex';
  }});

  document.getElementById('modal-overlay').addEventListener('click', function(e) {{
    if (e.target === this) closeModal();
  }});
</script>
</body>
</html>
"""


def render(graph: nx.DiGraph, output_path: str = "concept_map.html", title: str = "Concept Map"):
    nodes, edges, book_colors = _build_node_payload(graph)

    legend_rows = "\n".join(
        f'<div class="row"><span class="swatch" style="background:{color}"></span>{_esc(book)}</div>'
        for book, color in book_colors.items()
    )

    html_out = PAGE_TEMPLATE.format(
        title=_esc(title),
        legend_rows=legend_rows,
        hub_color=HUB_COLOR,
        cross_ref_color=CROSS_REF_COLOR,
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges),
        vis_network_js=_load_vis_network_js(),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"Saved concept map to {output_path}")
    print(f"Books in legend: {list(book_colors.keys())}")


if __name__ == "__main__":
    from src.concept_graph import build_graph

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
    render(graph, title="Deep Learning + Databases")
