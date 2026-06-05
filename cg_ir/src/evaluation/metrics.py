"""
evaluation/metrics.py
=====================
Implementasi metrik evaluasi IR standar:
  - Precision@k, Recall@k, F1@k
  - NDCG@k (graded relevance)
  - MRR (Mean Reciprocal Rank)
  - MAP (Mean Average Precision)
  - Bloom accuracy & MAE
  - Latency percentiles

Semua fungsi menerima input sederhana (list of IDs / grades)
agar mudah dipakai dari notebook.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ── Per-query metric helpers ───────────────────────────────────────────────────

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """P@k = |ret_k ∩ rel| / k"""
    if k == 0:
        return 0.0
    rel_set = set(relevant)
    hits = sum(1 for d in retrieved[:k] if d in rel_set)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """R@k = |ret_k ∩ rel| / |rel|"""
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    hits = sum(1 for d in retrieved[:k] if d in rel_set)
    return hits / len(relevant)


def f1_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def ndcg_at_k(retrieved: List[str], grades: Dict[str, int], k: int) -> float:
    """
    NDCG@k = DCG@k / IDCG@k

    Args:
        grades : doc_id → relevance grade (int, higher = more relevant)
    """
    def dcg(ranking: List[str], g: Dict[str, int], k: int) -> float:
        return sum(
            (2 ** g.get(doc, 0) - 1) / np.log2(i + 2)
            for i, doc in enumerate(ranking[:k])
        )

    actual_dcg = dcg(retrieved, grades, k)
    ideal_ranking = sorted(grades.keys(), key=lambda d: grades[d], reverse=True)
    ideal_dcg = dcg(ideal_ranking, grades, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """AP = Σ_k P@k · rel(k) / |rel|"""
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    hits, ap = 0, 0.0
    for k, doc in enumerate(retrieved, start=1):
        if doc in rel_set:
            hits += 1
            ap += hits / k
    return ap / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """1 / rank_of_first_relevant (0 if none found)"""
    rel_set = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in rel_set:
            return 1.0 / rank
    return 0.0


# ── Aggregate metrics ──────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    """Container untuk semua metrik, siap print/export."""
    system_name: str
    n_queries: int = 0

    # Standard IR
    mrr: float = 0.0
    map_score: float = 0.0

    precision: Dict[int, float] = field(default_factory=dict)  # P@{1,5,10}
    recall:    Dict[int, float] = field(default_factory=dict)  # R@{1,5,10}
    f1:        Dict[int, float] = field(default_factory=dict)  # F1@{1,5,10}
    ndcg:      Dict[int, float] = field(default_factory=dict)  # NDCG@{1,5,10}

    # Bloom-level accuracy
    bloom_accuracy: float = 0.0    # exact match
    bloom_mae: float = 0.0         # mean |target - retrieved|

    # Graph coverage
    graph_coverage: float = 0.0    # fraction of queries that used graph

    # Latency
    latency_mean: float = 0.0
    latency_p50:  float = 0.0
    latency_p95:  float = 0.0

    def to_dict(self) -> Dict:
        d = {
            "system": self.system_name,
            "n_queries": self.n_queries,
            "MRR": round(self.mrr, 4),
            "MAP": round(self.map_score, 4),
            "Bloom_Acc": round(self.bloom_accuracy, 4),
            "Bloom_MAE": round(self.bloom_mae, 4),
            "Graph_Cov": round(self.graph_coverage, 4),
            "Latency_mean_ms": round(self.latency_mean, 2),
            "Latency_p95_ms":  round(self.latency_p95, 2),
        }
        for k in sorted(self.precision):
            d[f"P@{k}"]    = round(self.precision[k], 4)
            d[f"R@{k}"]    = round(self.recall[k], 4)
            d[f"F1@{k}"]   = round(self.f1[k], 4)
            d[f"NDCG@{k}"] = round(self.ndcg[k], 4)
        return d

    def __str__(self) -> str:
        d = self.to_dict()
        lines = [f"\n{'─'*50}", f"  {d['system']}  (n={d['n_queries']})", f"{'─'*50}"]
        for key, val in d.items():
            if key not in ("system", "n_queries"):
                lines.append(f"  {key:<22} {val}")
        return "\n".join(lines)


class IREvaluator:
    """
    Evaluator untuk CG-IR dan baseline BM25Pipeline.

    Usage:
        evaluator = IREvaluator(k_values=[1, 5, 10])
        report = evaluator.evaluate(system_name, results, ground_truth)
        print(report)
        df = evaluator.compare([report_bm25, report_cgir])
    """

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 5, 10]

    def evaluate(
        self,
        system_name: str,
        results: List[Dict],
        ground_truth: List[Dict],
    ) -> EvalReport:
        """
        Evaluate one system against ground truth.

        Args:
            results: List of dicts per query:
                {
                  "query_id"     : str,
                  "retrieved_ids": List[str],   # ranked
                  "target_bloom" : int,
                  "retrieved_bloom": int,
                  "graph_used"   : bool,
                  "latency_ms"   : float,
                }
            ground_truth: List of dicts per query:
                {
                  "query_id"    : str,
                  "relevant_ids": List[str],    # binary relevant
                  "grades"      : Dict[str, int] (optional, for NDCG)
                }
        """
        gt = {g["query_id"]: g for g in ground_truth}

        ap_scores, rr_scores = [], []
        p_scores = {k: [] for k in self.k_values}
        r_scores = {k: [] for k in self.k_values}
        f_scores = {k: [] for k in self.k_values}
        n_scores = {k: [] for k in self.k_values}

        bloom_targets, bloom_retrieved = [], []
        graph_flags, latencies = [], []

        for res in results:
            qid = res["query_id"]
            if qid not in gt:
                continue

            g      = gt[qid]
            ret    = res.get("retrieved_ids", [])
            rel    = g.get("relevant_ids", [])
            grades = g.get("grades", {d: 1 for d in rel})

            ap_scores.append(average_precision(ret, rel))
            rr_scores.append(reciprocal_rank(ret, rel))

            for k in self.k_values:
                p_scores[k].append(precision_at_k(ret, rel, k))
                r_scores[k].append(recall_at_k(ret, rel, k))
                f_scores[k].append(f1_at_k(ret, rel, k))
                n_scores[k].append(ndcg_at_k(ret, grades, k))

            bloom_targets.append(res.get("target_bloom", 0))
            bloom_retrieved.append(res.get("retrieved_bloom", 0))
            graph_flags.append(int(res.get("graph_used", False)))
            latencies.append(res.get("latency_ms", 0.0))

        n = len(ap_scores)
        bt = np.array(bloom_targets)
        br = np.array(bloom_retrieved)

        report = EvalReport(
            system_name=system_name,
            n_queries=n,
            mrr=float(np.mean(rr_scores)) if rr_scores else 0.0,
            map_score=float(np.mean(ap_scores)) if ap_scores else 0.0,
            bloom_accuracy=float(np.mean(bt == br)) if len(bt) > 0 else 0.0,
            bloom_mae=float(np.mean(np.abs(bt - br))) if len(bt) > 0 else 0.0,
            graph_coverage=float(np.mean(graph_flags)) if graph_flags else 0.0,
            latency_mean=float(np.mean(latencies)) if latencies else 0.0,
            latency_p50=float(np.percentile(latencies, 50)) if latencies else 0.0,
            latency_p95=float(np.percentile(latencies, 95)) if latencies else 0.0,
        )

        for k in self.k_values:
            report.precision[k] = float(np.mean(p_scores[k]))
            report.recall[k]    = float(np.mean(r_scores[k]))
            report.f1[k]        = float(np.mean(f_scores[k]))
            report.ndcg[k]      = float(np.mean(n_scores[k]))

        return report

    def compare(self, reports: List[EvalReport]) -> "pd.DataFrame":
        """Return a pandas DataFrame comparing multiple EvalReports."""
        import pandas as pd
        rows = [r.to_dict() for r in reports]
        df = pd.DataFrame(rows).set_index("system")
        return df
