"""
Visualisasi Knowledge Graph hasil Auto-HKG.
Menghasilkan:
  - output/graph/graph_interactive.html  (pyvis, bisa dibuka di browser)
  - output/graph/graph_static.png        (matplotlib, untuk laporan)
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# Warna per tipe node
NODE_COLORS = {
    "topic_coarse": "#4C51BF",   # indigo
    "topic_fine":   "#0F6E56",   # teal
    "concept":      "#B45309",   # amber
    "question":     "#6B7280",   # gray
    "method":       "#9D174D",   # pink
}

EDGE_COLORS = {
    "has_subtopic":     "#818CF8",
    "has_concept":      "#34D399",
    "has_question":     "#D1D5DB",
    "prerequisite":     "#F59E0B",
    "successor":        "#3B82F6",
    "peer":             "#E5E7EB",
    "requires_method":  "#F9A8D4",
}


def load_graph(graph_path: str = "output/graph/knowledge_graph.json") -> nx.DiGraph:
    with open(graph_path, encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, directed=True)


def plot_static(G: nx.DiGraph, out_path: str = "output/graph/graph_static.png",
                max_nodes: int = 80):
    """Plot static — hanya tampilkan topic + concept (skip question agar tidak penuh)."""
    sub_nodes = [n for n, d in G.nodes(data=True)
                 if d.get("type") in ("topic_coarse", "topic_fine", "concept")][:max_nodes]
    H = G.subgraph(sub_nodes)

    fig, ax = plt.subplots(figsize=(18, 12))
    pos = nx.spring_layout(H, k=1.8, seed=42)

    for ntype, color in NODE_COLORS.items():
        nodes = [n for n, d in H.nodes(data=True) if d.get("type") == ntype]
        if not nodes:
            continue
        size = {"topic_coarse": 1800, "topic_fine": 900, "concept": 400}.get(ntype, 200)
        nx.draw_networkx_nodes(H, pos, nodelist=nodes, node_color=color,
                               node_size=size, ax=ax, alpha=0.9)

    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#CBD5E0",
                           arrows=True, arrowsize=10,
                           width=0.6, alpha=0.5,
                           connectionstyle="arc3,rad=0.1")

    labels = {n: d.get("label", n)[:20] for n, d in H.nodes(data=True)
              if d.get("type") in ("topic_coarse", "topic_fine")}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=7, ax=ax)

    legend = [mpatches.Patch(color=c, label=t) for t, c in NODE_COLORS.items()
              if t != "question"]
    ax.legend(handles=legend, loc="upper left", fontsize=8)
    ax.set_title("Auto-HKG — Hierarchical Knowledge Graph", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Static plot tersimpan → {out_path}")


def plot_interactive(G: nx.DiGraph, out_path: str = "output/graph/graph_interactive.html",
                     max_nodes: int = 200):
    """Plot interaktif dengan pyvis — bisa zoom & drag di browser."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis tidak terinstall. Jalankan: pip install pyvis")
        return

    sub_nodes = [n for n, d in G.nodes(data=True)
                 if d.get("type") != "question"][:max_nodes]
    H = G.subgraph(sub_nodes)

    net = Network(height="750px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#1a1a1a")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

    for nid, data in H.nodes(data=True):
        ntype = data.get("type", "concept")
        color = NODE_COLORS.get(ntype, "#999")
        size = {"topic_coarse": 35, "topic_fine": 22, "concept": 14}.get(ntype, 10)
        label = data.get("label", nid)[:30]
        title = f"<b>{label}</b><br>Type: {ntype}"
        if "bloom_level" in data:
            title += f"<br>Bloom: {data['bloom_level']}"
        if "frequency" in data:
            title += f"<br>Frequency: {data['frequency']}"
        net.add_node(nid, label=label, color=color, size=size, title=title)

    for src, dst, edata in H.edges(data=True):
        rel = edata.get("relation", "")
        color = EDGE_COLORS.get(rel, "#ccc")
        net.add_edge(src, dst, color=color, title=rel, width=1.2)

    net.show_buttons(filter_=["physics"])
    net.save_graph(out_path)
    print(f"Interactive plot tersimpan → {out_path}")


def print_stats(G: nx.DiGraph):
    print("\n── Graph Statistics ─────────────────────────")
    for ntype in ("topic_coarse", "topic_fine", "concept", "question", "method"):
        count = sum(1 for _, d in G.nodes(data=True) if d.get("type") == ntype)
        print(f"  {ntype:<15}: {count} nodes")
    print(f"  {'edges':<15}: {G.number_of_edges()}")

    top_concepts = sorted(
        [(d.get("label", n), d.get("frequency", 0))
         for n, d in G.nodes(data=True) if d.get("type") == "concept"],
        key=lambda x: x[1], reverse=True
    )[:10]
    print("\n── Top 10 Concepts by Frequency ─────────────")
    for label, freq in top_concepts:
        print(f"  {freq:>4}×  {label}")
    print()


if __name__ == "__main__":
    G = load_graph()
    print_stats(G)
    plot_static(G)
    plot_interactive(G)
