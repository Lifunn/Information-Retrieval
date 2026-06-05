"""
data_loader.py
==============
Loads the question corpus (CSV) and Knowledge Graph (JSON) into formats
expected by the CG-IR pipeline.

Assumes CSV schema (adjust COLUMN_MAP if yours differs):
  - topic        : broad subject area
  - concept      : specific concept within topic
  - question     : question text
  - context      : theory/background text (used by LLM judge)
  - bloom_level  : C1–C6 or integer 1–6
"""

import json
import re
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple

# ── Column mapping — update these to match your actual CSV headers ──────────
COLUMN_MAP = {
    "topic":       ["topic", "topik", "subject", "mata_pelajaran"],
    "concept":     ["concept", "konsep", "subtopic", "subtopik"],
    "question":    ["question", "soal", "pertanyaan", "question_text"],
    "context":     ["context", "konteks", "materi", "teori", "background"],
    "bloom_level": ["bloom_level", "bloom", "level", "cognitive_level", "tingkat"],
}

BLOOM_NORMALIZE = {
    "c1": 1, "c2": 2, "c3": 3, "c4": 4, "c5": 5, "c6": 6,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "remember": 1, "understand": 2, "apply": 3,
    "analyze": 4, "evaluate": 5, "create": 6,
    "ingatan": 1, "pemahaman": 2, "aplikasi": 3,
    "analisis": 4, "evaluasi": 5, "kreasi": 6,
}


def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return the first matching column name (case-insensitive)."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in df_cols_lower:
            return df_cols_lower[cand.lower()]
    return None


def _normalize_bloom(value) -> int:
    """Convert any Bloom representation to integer 1–6."""
    if isinstance(value, int) and 1 <= value <= 6:
        return value
    normalized = BLOOM_NORMALIZE.get(str(value).strip().lower(), None)
    return normalized if normalized else 1


def load_corpus(csv_path: str) -> List[Dict]:
    """
    Load CSV dataset into a list of document dicts for BM25 indexing.

    Returns:
        List of dicts with keys: id, topic, concept, question, context, bloom_level
    """
    df = pd.read_csv(csv_path).fillna("")

    col = {key: _resolve_column(df, cands) for key, cands in COLUMN_MAP.items()}
    missing = [k for k, v in col.items() if v is None and k != "context"]
    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing}. "
            f"Kolom yang tersedia: {list(df.columns)}. "
            f"Edit COLUMN_MAP di data_loader.py."
        )

    documents = []
    for idx, row in df.iterrows():
        doc = {
            "id": str(idx),
            "topic":       str(row[col["topic"]]).strip(),
            "concept":     str(row[col["concept"]]).strip(),
            "question":    str(row[col["question"]]).strip(),
            "context":     str(row[col["context"]]).strip() if col["context"] else "",
            "bloom_level": _normalize_bloom(row[col["bloom_level"]]),
            "_raw_row":    idx,   # keep reference back to original CSV
        }
        documents.append(doc)

    print(f"✅ Loaded {len(documents)} documents from '{csv_path}'")
    _print_corpus_stats(documents)
    return documents


def _print_corpus_stats(docs: List[Dict]):
    from collections import Counter
    bloom_dist = Counter(d["bloom_level"] for d in docs)
    topic_dist = Counter(d["topic"] for d in docs)
    print(f"   Bloom distribution : {dict(sorted(bloom_dist.items()))}")
    print(f"   Unique topics      : {len(topic_dist)}")
    print(f"   Top topics         : {dict(list(topic_dist.most_common(5)))}")


def load_graph(graph_json_path: str) -> nx.DiGraph:
    """
    Load auto-HKG JSON output into a NetworkX DiGraph.

    Expected JSON format:
        {
          "nodes": [{"id": "n1", "label": "Pancasila", "type": "topic"}, ...],
          "edges": [{"source": "n1", "target": "n2", "relation": "prerequisite"}, ...]
        }
    """
    with open(graph_json_path) as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data.get("nodes", []):
        G.add_node(node["id"], **node)
    for edge in data.get("edges", []):
        G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""))

    print(f"✅ Loaded KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    rel_counts = {}
    for _, _, d in G.edges(data=True):
        r = d.get("relation", "unknown")
        rel_counts[r] = rel_counts.get(r, 0) + 1
    print(f"   Edge relations: {rel_counts}")
    return G
