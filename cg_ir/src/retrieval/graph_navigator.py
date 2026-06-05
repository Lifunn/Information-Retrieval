import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

MASTERY_BLOOM_THRESHOLDS = [0.30, 0.50, 0.65, 0.80, 0.90]

def mastery_to_bloom(mastery: float) -> int:
    for level, threshold in enumerate(MASTERY_BLOOM_THRESHOLDS, start=1):
        if mastery < threshold:
            return level
    return 6


@dataclass
class QueryContext:
    keyword: str
    query_terms: List[str]              = field(default_factory=list)
    target_bloom: int                   = 1
    bloom_filter: List[int]             = field(default_factory=list)
    candidate_topics: List[str]         = field(default_factory=list)
    candidate_concepts: List[str]       = field(default_factory=list)
    candidate_question_ids: List[str]   = field(default_factory=list)
    traversal_path: List[str]           = field(default_factory=list)
    graph_used: bool                    = False

    @property
    def expanded_query(self) -> str:
        return " ".join(dict.fromkeys(self.query_terms))


class GraphNavigator:
    def __init__(self, graph: nx.DiGraph, bloom_window: int = 1):
        self.graph        = graph
        self.bloom_window = bloom_window
        self._by_type: Dict[str, List[str]] = {}
        for nid, attrs in graph.nodes(data=True):
            t = attrs.get("type", "")
            self._by_type.setdefault(t, []).append(nid)

    def _match_nodes(self, keyword: str, top_k: int = 5) -> List[Tuple[str, float]]:
        kw       = keyword.lower().strip()
        kw_parts = set(kw.split())
        type_boost = {
            "topic_coarse": 0.4,
            "topic_fine":   0.3,
            "concept":      0.2,
            "question":     0.1,
            "method":       0.0,
        }
        results = []
        for nid, attrs in self.graph.nodes(data=True):
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
                score += type_boost.get(attrs.get("type", ""), 0.0)
                results.append((nid, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _traverse(self, start_ids: List[str], mastery: float):
        visited      = set(start_ids)
        question_ids: Set[str] = set()
        frontier     = list(start_ids)
        path         = [f"start:{nid}" for nid in start_ids]

        CONTEXT_EDGES = {"has_subtopic", "has_concept", "peer"}

        for node_id in list(frontier):
            for _, neighbor, data in self.graph.out_edges(node_id, data=True):
                relation = data.get("relation", "")

                if relation == "has_question":
                    question_ids.add(neighbor)
                    path.append(f"has_question:{neighbor}")
                    continue

                if neighbor in visited:
                    continue

                include = False
                if relation in CONTEXT_EDGES:
                    include = True
                elif relation == "successor" and mastery >= 0.5:
                    include = True
                elif relation == "prerequisite" and mastery < 0.5:
                    include = True

                if include:
                    visited.add(neighbor)
                    frontier.append(neighbor)
                    path.append(f"{relation}:{neighbor}")
                    for _, q_neighbor, q_data in self.graph.out_edges(neighbor, data=True):
                        if q_data.get("relation") == "has_question":
                            question_ids.add(q_neighbor)

        return visited, question_ids, path

    def _filter_questions_by_bloom(
        self, question_ids: Set[str], bloom_filter: List[int]
    ) -> List[str]:
        bloom_set = set(bloom_filter)
        result = [
            qid for qid in question_ids
            if self.graph.nodes.get(qid, {}).get("bloom_level") in bloom_set
            or self.graph.nodes.get(qid, {}).get("bloom_level") is None
        ]
        return result if result else list(question_ids)

    def formulate(self, keyword: str, mastery: float = 0.0) -> QueryContext:
        target_bloom = mastery_to_bloom(mastery)
        bloom_filter = list(range(
            max(1, target_bloom - self.bloom_window),
            min(6, target_bloom + self.bloom_window) + 1,
        ))

        ctx = QueryContext(
            keyword=keyword,
            query_terms=[keyword],
            target_bloom=target_bloom,
            bloom_filter=bloom_filter,
        )

        matched = self._match_nodes(keyword)
        if not matched:
            return ctx

        ctx.graph_used = True
        start_ids = [nid for nid, _ in matched]

        all_visited, question_ids, path = self._traverse(start_ids, mastery)
        ctx.traversal_path = path[:20]

        for nid in all_visited:
            attrs = self.graph.nodes[nid]
            label = attrs.get("label", "")
            ntype = attrs.get("type", "")
            if label:
                ctx.query_terms.append(label)
            if ntype in ("topic_coarse", "topic_fine"):
                ctx.candidate_topics.append(label)
            elif ntype == "concept":
                ctx.candidate_concepts.append(label)

        ctx.query_terms = list(dict.fromkeys(ctx.query_terms))[:20]
        ctx.candidate_question_ids = self._filter_questions_by_bloom(
            question_ids, bloom_filter
        )
        return ctx
