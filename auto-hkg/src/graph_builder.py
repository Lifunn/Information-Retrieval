"""
Graph Builder — assembles extracted LLM output into a directed knowledge graph.

This module receives validated extraction dicts from the Auto-HKG pipeline and
incrementally builds a NetworkX DiGraph representing the educational knowledge
structure of the dataset.

Node types and their roles in the hierarchy:
    topic_coarse : Broad subject category (e.g., "Lokasi Geografis").
                   One per unique coarse topic label.
    topic_fine   : Specific sub-topic under a coarse topic (e.g., "Lokasi Absolut dan Relatif").
                   One per unique fine topic label.
    concept      : Individual knowledge concept tested by one or more questions.
                   Enriched with Bloom level, difficulty, and frequency of occurrence.
    question     : A single exam question node, linked to its associated concepts.
                   Stores the question text (truncated), Bloom level, and difficulty.
    method       : Problem-solving method or cognitive strategy linked to a question.

Edge types and their semantic meaning:
    has_subtopic     : topic_coarse --> topic_fine
    has_concept      : topic_fine  --> concept
    has_question     : concept     --> question
    requires_method  : question    --> method
    prerequisite     : concept A   --> concept B  (A must be known before B)
    successor        : concept A   --> concept B  (B naturally follows A)
    peer             : concept A   <-> concept B  (co-occur in the same question)

Node IDs are normalized (lowercase, collapsed whitespace) to enable deduplication
across questions. When the same concept appears in multiple questions, its node is
reused and its 'frequency' counter is incremented.
"""

import re
import networkx as nx
from collections import defaultdict


def _normalize(text: str) -> str:
    """
    Normalize a text string to a consistent node ID.
    Converts to lowercase and collapses all internal whitespace to single spaces.
    This ensures that semantically identical concepts with minor formatting
    differences (e.g., "Lokasi Absolut" vs "lokasi  absolut") map to the same node.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


class GraphBuilder:
    """
    Incrementally constructs a hierarchical educational knowledge graph.

    Each call to add_extraction() processes one validated extraction dict
    and updates the internal DiGraph with new nodes and edges. Duplicate nodes
    (same normalized text) are reused and their attributes are updated in place.

    The final graph is retrieved via get_graph(), which also computes
    the occurrence frequency for all concept nodes before returning.
    """

    def __init__(self):
        self.G = nx.DiGraph()
        # Track how many questions reference each concept node
        self._concept_occurrences: dict[str, int] = defaultdict(int)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def add_extraction(self, data: dict):
        """
        Integrate one validated extraction result into the knowledge graph.

        Processing order:
            1. Add topic_coarse and topic_fine nodes; connect with 'has_subtopic'.
            2. Add concept nodes; connect each to topic_fine via 'has_concept'.
            3. Add question node; connect each concept to it via 'has_question'.
            4. Add method nodes; connect question to each via 'requires_method'.
            5. Add prerequisite concept nodes; connect them to this question's
               concepts via 'prerequisite' edges.
            6. Add successor concept nodes; connect this question's concepts to
               them via 'successor' edges.
            7. Add 'peer' edges between all concept pairs that co-occur in this
               question (bidirectional).

        Args:
            data: Validated extraction dict containing topic_coarse, topic_fine,
                  concepts, methods, bloom_level, prerequisites, successors,
                  difficulty, _question, and _row_id.
        """
        # 1. Topic hierarchy
        tc_id = self._add_node(
            _normalize(data["topic_coarse"]),
            "topic_coarse",
            label=data["topic_coarse"]
        )
        tf_id = self._add_node(
            _normalize(data["topic_fine"]),
            "topic_fine",
            label=data["topic_fine"]
        )
        self._add_edge(tc_id, tf_id, "has_subtopic")

        # 2. Concept nodes
        concept_ids = []
        for c in data["concepts"]:
            if not c.strip():
                continue
            cid = self._add_node(
                _normalize(c),
                "concept",
                label=c,
                bloom_level=data["bloom_level"],
                difficulty=data["difficulty"]
            )
            self._add_edge(tf_id, cid, "has_concept")
            concept_ids.append(cid)
            self._concept_occurrences[cid] += 1

        # 3. Question node
        q_id = f"q_{data['_row_id']}"
        self._add_node(
            q_id,
            "question",
            label=data["_question"][:120],
            bloom_level=data["bloom_level"],
            difficulty=data["difficulty"]
        )
        for cid in concept_ids:
            self._add_edge(cid, q_id, "has_question")

        # 4. Method nodes
        for m in data.get("methods", []):
            if not m.strip():
                continue
            mid = self._add_node(_normalize(m), "method", label=m)
            self._add_edge(q_id, mid, "requires_method")

        # 5. Prerequisite edges
        # A prerequisite concept must already exist in the learner's knowledge
        # before the current concept can be meaningfully understood.
        for pre in data.get("prerequisites", []):
            if not pre.strip():
                continue
            pre_id = self._add_node(_normalize(pre), "concept", label=pre)
            for cid in concept_ids:
                if pre_id != cid:
                    self._add_edge(pre_id, cid, "prerequisite")

        # 6. Successor edges
        # Successor concepts naturally follow after mastering the current concept.
        for suc in data.get("successors", []):
            if not suc.strip():
                continue
            suc_id = self._add_node(_normalize(suc), "concept", label=suc)
            for cid in concept_ids:
                if suc_id != cid:
                    self._add_edge(cid, suc_id, "successor")

        # 7. Peer edges — bidirectional co-occurrence within the same question
        for i in range(len(concept_ids)):
            for j in range(i + 1, len(concept_ids)):
                self._add_edge(concept_ids[i], concept_ids[j], "peer")
                self._add_edge(concept_ids[j], concept_ids[i], "peer")

    def get_graph(self) -> nx.DiGraph:
        """
        Return the completed knowledge graph.

        Before returning, annotates all concept nodes with their accumulated
        'frequency' attribute — the number of distinct questions that reference
        each concept. This is useful for identifying central or high-coverage
        concepts in downstream analysis.
        """
        for nid, count in self._concept_occurrences.items():
            if nid in self.G.nodes:
                self.G.nodes[nid]["frequency"] = count
        return self.G

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _add_node(self, node_id: str, node_type: str, **attrs) -> str:
        """
        Add a node to the graph or update its attributes if it already exists.
        Returns the node_id for chaining.
        """
        if node_id not in self.G:
            self.G.add_node(node_id, type=node_type, **attrs)
        else:
            # Update attributes without overwriting the node type
            self.G.nodes[node_id].update(attrs)
        return node_id

    def _add_edge(self, src: str, dst: str, relation: str):
        """
        Add a directed edge between two nodes with a relation label.
        If the edge already exists, increment its weight counter.
        Weight reflects the number of extractions that produced the same edge,
        which can be used as a confidence signal in downstream graph reasoning.
        """
        if not self.G.has_edge(src, dst):
            self.G.add_edge(src, dst, relation=relation, weight=1)
        else:
            self.G[src][dst]["weight"] += 1
