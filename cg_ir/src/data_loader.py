import re
import json
import pandas as pd
import networkx as nx
from typing import List, Dict, Optional

COLUMN_MAP = {
    "id":          ["id"],
    "question":    ["Pertanyaan"],
    "context":     ["Konteks"],
    "bloom_level": ["Level Kognitif"],
    "topic":       [],
    "concept":     [],
}


def parse_bloom(value) -> int:
    if isinstance(value, int) and 1 <= value <= 6:
        return value
    s = str(value).strip().lower()
    m = re.match(r"c(\d)", s)
    if m:
        return min(max(int(m.group(1)), 1), 6)
    keywords = {
        "mengingat": 1, "memahami": 2, "mengaplikasikan": 3,
        "menerapkan": 3, "menganalisis": 4, "mengevaluasi": 5,
        "mencipta": 6, "membuat": 6,
    }
    for kw, lvl in keywords.items():
        if kw in s:
            return lvl
    return 1


def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    df_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in df_lower:
            return df_lower[cand.lower()]
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _print_corpus_stats(docs: List[Dict]):
    from collections import Counter
    bloom_dist = Counter(d["bloom_level"] for d in docs)
    topic_dist = Counter(d["topic"] for d in docs if d["topic"])
    print(f"   Bloom dist : { {f'C{k}': v for k, v in sorted(bloom_dist.items())} }")
    print(f"   Topics     : {len(topic_dist)} unique, top-5: {dict(topic_dist.most_common(5))}")


def load_corpus(csv_path: str) -> List[Dict]:
    try:
        df = pd.read_csv(csv_path, sep=";").fillna("")
        if len(df.columns) < 2:
            df = pd.read_csv(csv_path, sep=",").fillna("")
    except Exception:
        df = pd.read_csv(csv_path, sep=",").fillna("")

    col = {key: _resolve_column(df, cands) for key, cands in COLUMN_MAP.items()}

    missing = [
        k for k, v in col.items()
        if v is None and k not in ("topic", "concept")
    ]
    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing}\n"
            f"Kolom tersedia: {list(df.columns)}\n"
            f"Sesuaikan COLUMN_MAP di data_loader.py."
        )

    documents = []
    for idx, row in df.iterrows():
        doc_id = str(row[col["id"]]).strip() if col["id"] else str(idx)
        documents.append({
            "id":          doc_id,
            "question":    str(row[col["question"]]).strip(),
            "context":     str(row[col["context"]]).strip() if col["context"] else "",
            "bloom_level": parse_bloom(row[col["bloom_level"]]),
            "topic":       str(row[col["topic"]]).strip() if col["topic"] else "",
            "concept":     str(row[col["concept"]]).strip() if col["concept"] else "",
        })

    print(f"Loaded {len(documents)} documents from '{csv_path}'")
    _print_corpus_stats(documents)
    return documents


def load_graph(graph_json_path: str) -> nx.DiGraph:
    with open(graph_json_path) as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data.get("nodes", []):
        attrs = dict(node)
        if "bloom_level" in attrs:
            attrs["bloom_level"] = parse_bloom(attrs["bloom_level"])
        G.add_node(node["id"], **attrs)

    raw_edges = data.get("edges") or data.get("links") or []
    for edge in raw_edges:
        G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""))

    from collections import Counter
    rel_counts  = Counter(d["relation"] for _, _, d in G.edges(data=True))
    type_counts = Counter(attrs.get("type", "?") for _, attrs in G.nodes(data=True))
    print(f"Loaded KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Node types : {dict(type_counts)}")
    print(f"   Edge types : {dict(rel_counts)}")
    return G


def link_corpus_to_graph(documents: List[Dict], graph: nx.DiGraph) -> Dict[str, str]:
    """
    Build mapping: doc_id -> graph question node id.
    Tries direct ID match first, falls back to normalized text match.
    """
    q_nodes = {
        nid: attrs
        for nid, attrs in graph.nodes(data=True)
        if attrs.get("type") == "question"
    }
    q_by_label = {
        _normalize_text(attrs["label"]): nid
        for nid, attrs in q_nodes.items()
    }

    mapping     = {}
    direct_hits = 0
    text_hits   = 0

    for doc in documents:
        doc_id = doc["id"]
        if doc_id in q_nodes:
            mapping[doc_id] = doc_id
            direct_hits += 1
            continue
        norm_q = _normalize_text(doc["question"])
        if norm_q in q_by_label:
            mapping[doc_id] = q_by_label[norm_q]
            text_hits += 1

    unlinked = len(documents) - direct_hits - text_hits
    print(f"Corpus-Graph linking: {direct_hits} direct match, "
          f"{text_hits} text match, {unlinked} unlinked")
    return mapping


def enrich_with_graph_topics(
    documents: List[Dict],
    graph: nx.DiGraph,
    doc_graph_map: Dict[str, str],
) -> List[Dict]:
    """
    Fill empty topic & concept fields from graph parent nodes
    using has_question edges.
    """
    q_to_parent: Dict[str, Dict] = {}
    for src, tgt, data in graph.edges(data=True):
        if data.get("relation") != "has_question":
            continue
        src_attrs = graph.nodes[src]
        src_type  = src_attrs.get("type", "")
        src_label = src_attrs.get("label", "")
        if tgt not in q_to_parent:
            q_to_parent[tgt] = {"topic": "", "concept": ""}
        if src_type == "topic_coarse":
            q_to_parent[tgt]["topic"] = src_label
        elif src_type in ("topic_fine", "concept"):
            q_to_parent[tgt]["concept"] = src_label

    enriched = 0
    for doc in documents:
        if doc.get("topic") and doc.get("concept"):
            continue
        graph_id = doc_graph_map.get(doc["id"])
        if not graph_id or graph_id not in q_to_parent:
            continue
        parent = q_to_parent[graph_id]
        if not doc.get("topic"):
            doc["topic"]   = parent["topic"]
        if not doc.get("concept"):
            doc["concept"] = parent["concept"]
        enriched += 1

    print(f"Enriched {enriched} documents with topic/concept from graph")
    return documents
