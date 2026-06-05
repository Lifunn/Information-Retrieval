"""
retrieval/graph_navigator.py
=============================
Menggunakan Knowledge Graph (auto-HKG output) untuk memformulasikan
query yang lebih kaya konteks berdasarkan keyword user + mastery learner.

Logic:
  1. Cari node KG yang paling relevan dengan keyword (fuzzy match)
  2. Tentukan target Bloom level dari mastery score
  3. Traverse edge prerequisite/peer/successor sesuai mastery
  4. Kumpulkan label node yang ditemukan → enriched query terms
"""

import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# Bloom level thresholds (mastery → target Bloom)
# Contoh: mastery < 0.30 → C1, dst.
MASTERY_BLOOM_THRESHOLDS = [0.30, 0.50, 0.65, 0.80, 0.90]


@dataclass
class QueryContext:
    """
    Konteks yang dihasilkan graph navigator.
    Dipass langsung ke BM25 + RM3 pipeline.
    """
    keyword: str
    query_terms: List[str]          = field(default_factory=list)
    target_bloom: int               = 1
    bloom_filter: List[int]         = field(default_factory=list)
    candidate_topics: List[str]     = field(default_factory=list)
    candidate_concepts: List[str]   = field(default_factory=list)
    traversal_path: List[str]       = field(default_factory=list)
    graph_used: bool                = False   # False jika keyword tidak ditemukan di graph

    @property
    def expanded_query(self) -> str:
        return " ".join(dict.fromkeys(self.query_terms))  # deduplicate, preserve order


def mastery_to_bloom(mastery: float) -> int:
    """Map learner mastery [0,1] → target Bloom level [1,6]."""
    for level, threshold in enumerate(MASTERY_BLOOM_THRESHOLDS, start=1):
        if mastery < threshold:
            return level
    return 6


class GraphNavigator:
    """
    Knowledge Graph navigator untuk CG-IR query formulation.

    Args:
        graph: NetworkX DiGraph dari data_loader.load_graph()
    """

    def __init__(self, graph: nx.DiGraph, bloom_window: int = 1):
        self.graph       = graph
        self.bloom_window = bloom_window  # Bloom filter = target ± window

    # ── Node lookup ────────────────────────────────────────────────────────

    def _match_nodes(self, keyword: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Cari nodes yang relevan dengan keyword.
        Score: 2.0 = exact match, 1.0 = substring, 0.5 = partial word match.

        Returns list of (node_id, relevance_score) sorted descending.
        """
        kw = keyword.lower().strip()
        kw_parts = set(kw.split())
        results = []

        for node_id, attrs in self.graph.nodes(data=True):
            label = attrs.get("label", "").lower()
            if not label:
                continue

            score = 0.0
            if label == kw:
                score = 2.0
            elif kw in label or label in kw:
                score = 1.0
            elif kw_parts & set(label.split()):
                score = 0.5

            if score > 0:
                # Boost topic/concept nodes over question nodes
                type_boost = {"topic": 0.3, "concept": 0.2, "question": 0.0}
                score += type_boost.get(attrs.get("type", ""), 0.0)
                results.append((node_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── Graph traversal ────────────────────────────────────────────────────

    def _traverse(self, start_nodes: List[str], mastery: float) -> List[str]:
        """
        BFS-style traversal dari start nodes menggunakan edge relations.

        Traversal policy by mastery:
          mastery < 0.4  → telusuri prerequisite (mundur, kuasai dasar dulu)
          mastery < 0.7  → telusuri peer (soal setara)
          mastery >= 0.7 → telusuri successor (maju ke level berikutnya)
        """
        visited = set(start_nodes)
        frontier = list(start_nodes)
        path = [f"start:{nid}" for nid in start_nodes]

        for node_id in list(frontier):  # copy to avoid mutation mid-loop
            for _, neighbor, data in self.graph.out_edges(node_id, data=True):
                relation = data.get("relation", "")

                if neighbor in visited:
                    continue

                include = False
                if relation == "peer":
                    include = True
                elif relation == "successor" and mastery >= 0.5:
                    include = True
                elif relation == "prerequisite" and mastery < 0.5:
                    include = True

                if include:
                    visited.add(neighbor)
                    frontier.append(neighbor)
                    path.append(f"{relation}:{neighbor}")

        return list(visited), path

    # ── Main entry point ───────────────────────────────────────────────────

    def formulate(self, keyword: str, mastery: float = 0.0) -> QueryContext:
        """
        Formulasikan query context dari keyword + mastery.

        Args:
            keyword : kata kunci dari user (misal "pancasila")
            mastery : skor mastery learner untuk topik ini [0, 1]

        Returns:
            QueryContext berisi enriched query terms + bloom target
        """
        target_bloom = mastery_to_bloom(mastery)
        bloom_filter = list(range(
            max(1, target_bloom - self.bloom_window),
            min(6, target_bloom + self.bloom_window) + 1
        ))

        ctx = QueryContext(
            keyword=keyword,
            query_terms=[keyword],
            target_bloom=target_bloom,
            bloom_filter=bloom_filter,
        )

        # 1. Find matching nodes
        matched = self._match_nodes(keyword)
        if not matched:
            return ctx  # graph_used = False, fallback to plain keyword

        ctx.graph_used = True
        start_ids = [nid for nid, _ in matched]

        # 2. Traverse graph
        all_nodes, traversal_path = self._traverse(start_ids, mastery)
        ctx.traversal_path = traversal_path

        # 3. Collect labels as enriched query terms
        for node_id in all_nodes:
            attrs = self.graph.nodes[node_id]
            label = attrs.get("label", "")
            ntype = attrs.get("type", "")

            if label:
                ctx.query_terms.append(label)

            if ntype == "topic":
                ctx.candidate_topics.append(label)
            elif ntype == "concept":
                ctx.candidate_concepts.append(label)

        # Deduplicate preserving order
        ctx.query_terms = list(dict.fromkeys(ctx.query_terms))[:20]

        return ctx
