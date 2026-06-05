"""
retrieval/cgir_pipeline.py
==========================
CG-IR pipeline (FAISS version):

  keyword + mastery
       │
       ▼
  GraphNavigator  →  enriched query + candidate question IDs
       │
       ▼
  FAISSIndex      →  semantic search (dalam candidate pool jika ada)
       │
       ▼
  Bloom filter    →  pastikan level kognitif sesuai mastery
       │
       ▼
  top-1 soal

Dua pipeline tersedia:
  CGIRPipeline   : Graph + FAISS  (sistem utama)
  FAISSBaseline  : FAISS only     (baseline untuk evaluasi)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

import networkx as nx

from .faiss_index import FAISSIndex
from .graph_navigator import GraphNavigator, QueryContext, mastery_to_bloom


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Output dari satu retrieval query."""
    question: Dict
    score: float
    rank: int
    target_bloom: int
    retrieved_bloom: int

    query_context: Optional[QueryContext] = None
    all_candidates: List[Dict]            = field(default_factory=list)
    latency_ms: float                     = 0.0
    pipeline: str                         = "cgir"
    graph_filtered: bool                  = False

    @property
    def bloom_match(self) -> bool:
        return self.retrieved_bloom == self.target_bloom

    @property
    def bloom_distance(self) -> int:
        return abs(self.retrieved_bloom - self.target_bloom)


# ── Baseline: FAISS only ──────────────────────────────────────────────────────

class FAISSBaseline:
    """
    Baseline pipeline — hanya FAISS, tanpa graph navigation.
    Digunakan sebagai pembanding untuk mengukur kontribusi graph.
    """

    def __init__(self, model_name: str = None):
        self.faiss = FAISSIndex(model_name=model_name)

    def build(self, documents: List[Dict], **kwargs):
        self.faiss.build(documents, **kwargs)

    def save(self, dir_path: str):
        self.faiss.save(dir_path)

    def load(self, dir_path: str):
        self.faiss.load(dir_path)

    def retrieve(
        self,
        keyword: str,
        mastery: float   = 0.0,
        bloom_window: int = 1,
    ) -> RetrievalResult:
        t0           = time.time()
        target_bloom = mastery_to_bloom(mastery)
        bloom_filter = list(range(
            max(1, target_bloom - bloom_window),
            min(6, target_bloom + bloom_window) + 1,
        ))

        results = self.faiss.search(keyword, top_k=20, bloom_filter=bloom_filter)
        if not results:
            results = self.faiss.search(keyword, top_k=20)

        latency = (time.time() - t0) * 1000

        if not results:
            return RetrievalResult(
                question={}, score=0.0, rank=0,
                target_bloom=target_bloom, retrieved_bloom=0,
                latency_ms=latency, pipeline="faiss_baseline",
            )

        _, best_score, best_doc = results[0]
        return RetrievalResult(
            question=best_doc,
            score=best_score,
            rank=1,
            target_bloom=target_bloom,
            retrieved_bloom=best_doc.get("bloom_level", 0),
            all_candidates=[r[2] for r in results[:10]],
            latency_ms=latency,
            pipeline="faiss_baseline",
        )

    def retrieve_candidates(
        self, keyword: str, mastery: float = 0.0, top_k: int = 10
    ) -> List[RetrievalResult]:
        target_bloom = mastery_to_bloom(mastery)
        bloom_filter = list(range(
            max(1, target_bloom - 1),
            min(6, target_bloom + 1) + 1,
        ))
        t0 = time.time()
        results = self.faiss.search(keyword, top_k=top_k, bloom_filter=bloom_filter)
        if not results:
            results = self.faiss.search(keyword, top_k=top_k)
        latency = (time.time() - t0) * 1000

        return [
            RetrievalResult(
                question=doc, score=score, rank=rank + 1,
                target_bloom=target_bloom,
                retrieved_bloom=doc.get("bloom_level", 0),
                latency_ms=latency,
                pipeline="faiss_baseline",
            )
            for rank, (_, score, doc) in enumerate(results)
        ]


# ── Main: CG-IR pipeline ──────────────────────────────────────────────────────

class CGIRPipeline:
    """
    CG-IR: Knowledge Graph navigation + FAISS dense retrieval.

    Kontribusi graph:
    1. Query enrichment  — label nodes yang di-traverse ditambahkan ke query
                           sebelum di-encode, sehingga embedding-nya lebih
                           representatif secara topik
    2. Candidate pool    — question IDs dari has_question edges dipakai
                           sebagai pre-filter, FAISS search hanya di dalamnya

    Args:
        graph           : NetworkX DiGraph dari data_loader.load_graph()
        doc_to_graph_id : mapping doc_id → graph question node id
                          dari data_loader.link_corpus_to_graph()
        model_name      : sentence-transformers model
        bloom_window    : toleransi Bloom level (target ± window)
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        doc_to_graph_id: Optional[Dict[str, str]] = None,
        model_name: str   = None,
        bloom_window: int = 1,
    ):
        self.graph_nav       = GraphNavigator(graph, bloom_window=bloom_window)
        self.faiss           = FAISSIndex(model_name=model_name)
        self.bloom_window    = bloom_window
        self.doc_to_graph_id = doc_to_graph_id or {}
        self._graph_to_docs: Dict[str, List[int]] = {}

    def build(self, documents: List[Dict], **kwargs):
        """Build FAISS index dan inverted graph ID mapping."""
        self.faiss.build(documents, **kwargs)

        for doc_idx, doc in enumerate(documents):
            graph_id = self.doc_to_graph_id.get(doc["id"])
            if graph_id:
                self._graph_to_docs.setdefault(graph_id, []).append(doc_idx)

        linked = len(self._graph_to_docs)
        print(f"   Graph linking: {linked} question nodes → doc indices "
              f"({'pre-filtering aktif' if linked > 0 else 'hanya query enrichment'})")

    def save(self, dir_path: str):
        self.faiss.save(dir_path)

    def load(self, dir_path: str):
        self.faiss.load(dir_path)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _resolve_candidate_indices(self, ctx: QueryContext) -> Optional[List[int]]:
        """Konversi graph question IDs → doc indices untuk FAISS subset search."""
        if not ctx.candidate_question_ids or not self._graph_to_docs:
            return None
        indices: Set[int] = set()
        for qid in ctx.candidate_question_ids:
            indices.update(self._graph_to_docs.get(qid, []))
        return list(indices) if indices else None

    # ── Retrieve ───────────────────────────────────────────────────────────

    def retrieve(self, keyword: str, mastery: float = 0.0) -> RetrievalResult:
        """
        Full CG-IR retrieval → 1 soal terbaik.

        Args:
            keyword : kata kunci dari user
            mastery : mastery score learner [0, 1]
        """
        t0 = time.time()

        # ── Step 1: Graph navigation ───────────────────────────────────────
        ctx = self.graph_nav.formulate(keyword, mastery)

        # ── Step 2: Resolve candidate pool dari graph ──────────────────────
        candidate_indices = self._resolve_candidate_indices(ctx)
        graph_filtered    = candidate_indices is not None

        # ── Step 3: FAISS search dengan enriched query ─────────────────────
        # Enriched query = keyword + label nodes yang di-traverse
        # Ini yang membedakan CG-IR dari plain FAISS:
        # embedding-nya merepresentasikan semantic context yang lebih luas
        results = self.faiss.search(
            ctx.expanded_query,
            top_k=20,
            bloom_filter=ctx.bloom_filter,
            candidate_indices=candidate_indices,
        )

        # Fallback bertahap jika hasil terlalu sedikit
        if len(results) < 3 and candidate_indices:
            # Coba tanpa graph pre-filter tapi masih pakai enriched query
            results = self.faiss.search(
                ctx.expanded_query,
                top_k=20,
                bloom_filter=ctx.bloom_filter,
            )
            graph_filtered = False

        if len(results) < 3:
            # Last resort: plain keyword, no bloom filter
            results = self.faiss.search(keyword, top_k=20)

        if not results:
            latency = (time.time() - t0) * 1000
            return RetrievalResult(
                question={}, score=0.0, rank=0,
                target_bloom=ctx.target_bloom, retrieved_bloom=0,
                query_context=ctx, latency_ms=latency,
            )

        _, best_score, best_doc = results[0]
        latency = (time.time() - t0) * 1000

        return RetrievalResult(
            question=best_doc,
            score=best_score,
            rank=1,
            target_bloom=ctx.target_bloom,
            retrieved_bloom=best_doc.get("bloom_level", 0),
            query_context=ctx,
            all_candidates=[r[2] for r in results[:10]],
            latency_ms=latency,
            pipeline="cgir",
            graph_filtered=graph_filtered,
        )

    def retrieve_candidates(
        self, keyword: str, mastery: float = 0.0, top_k: int = 10
    ) -> List[RetrievalResult]:
        """Top-k results — untuk evaluasi NDCG, MAP, dll."""
        t0  = time.time()
        ctx = self.graph_nav.formulate(keyword, mastery)

        candidate_indices = self._resolve_candidate_indices(ctx)
        results = self.faiss.search(
            ctx.expanded_query,
            top_k=top_k,
            bloom_filter=ctx.bloom_filter,
            candidate_indices=candidate_indices,
        )
        if len(results) < 3:
            results = self.faiss.search(ctx.expanded_query, top_k=top_k, bloom_filter=ctx.bloom_filter)
        if not results:
            results = self.faiss.search(keyword, top_k=top_k)

        latency = (time.time() - t0) * 1000
        return [
            RetrievalResult(
                question=doc, score=score, rank=rank + 1,
                target_bloom=ctx.target_bloom,
                retrieved_bloom=doc.get("bloom_level", 0),
                query_context=ctx,
                latency_ms=latency,
                pipeline="cgir",
                graph_filtered=candidate_indices is not None,
            )
            for rank, (_, score, doc) in enumerate(results)
        ]
