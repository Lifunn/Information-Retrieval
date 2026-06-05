"""
evaluation/benchmark.py
=======================
Benchmark runner — membandingkan FAISSBaseline vs CGIRPipeline.

Evaluasi Bloom accuracy tidak butuh relevance judgments.
Evaluasi P@k, NDCG@k, MRR, MAP butuh synthetic ground truth
dari LLM judge (generate sekali, simpan, reuse).
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from ..retrieval.cgir_pipeline import CGIRPipeline, FAISSBaseline
from .metrics import IREvaluator, EvalReport


# ── Format results for evaluator ──────────────────────────────────────────────

def _format_results(pipeline, test_queries: List[Dict], top_k: int) -> List[Dict]:
    """Run pipeline.retrieve_candidates() dan format hasilnya."""
    results = []
    for q in test_queries:
        candidates = pipeline.retrieve_candidates(
            q["keyword"],
            mastery=q.get("mastery", 0.0),
            top_k=top_k,
        )
        results.append({
            "query_id":        q["query_id"],
            "retrieved_ids":   [c.question.get("id", "") for c in candidates if c.question],
            "target_bloom":    candidates[0].target_bloom if candidates else 1,
            "retrieved_bloom": candidates[0].retrieved_bloom if candidates else 0,
            "graph_used":      (
                candidates[0].query_context.graph_used
                if candidates and candidates[0].query_context else False
            ),
            "graph_filtered":  candidates[0].graph_filtered if candidates else False,
            "latency_ms":      sum(c.latency_ms for c in candidates) if candidates else 0.0,
            "top1_question":   candidates[0].question if candidates else {},
        })
    return results


# ── Benchmark runner ──────────────────────────────────────────────────────────

class Benchmark:
    """
    Evaluasi offline: FAISSBaseline vs CGIRPipeline.

    Usage:
        bench  = Benchmark(cgir, faiss_baseline, evaluator)
        output = bench.run(test_queries, ground_truth)
        Benchmark.print_report(output)
        Benchmark.export(output, "output/eval")
    """

    def __init__(
        self,
        cgir_pipeline:  CGIRPipeline,
        faiss_pipeline: FAISSBaseline,
        evaluator:      Optional[IREvaluator] = None,
        top_k: int = 10,
    ):
        self.cgir      = cgir_pipeline
        self.faiss_bl  = faiss_pipeline
        self.evaluator = evaluator or IREvaluator(k_values=[1, 5, 10])
        self.top_k     = top_k

    def run(self, test_queries: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Run full benchmark.

        Args:
            test_queries: [{"query_id", "keyword", "mastery"}]
            ground_truth: [{"query_id", "relevant_ids", "grades"}]

        Returns dict dengan EvalReport per sistem + comparison DataFrame.
        """
        print("⏱️  Running FAISS baseline...")
        faiss_results = _format_results(self.faiss_bl, test_queries, self.top_k)
        faiss_report  = self.evaluator.evaluate("FAISS (baseline)", faiss_results, ground_truth)

        print("⏱️  Running CG-IR (Graph + FAISS)...")
        cgir_results  = _format_results(self.cgir, test_queries, self.top_k)
        cgir_report   = self.evaluator.evaluate("CG-IR (Graph + FAISS)", cgir_results, ground_truth)

        return {
            "faiss_report":  faiss_report,
            "cgir_report":   cgir_report,
            "faiss_results": faiss_results,
            "cgir_results":  cgir_results,
            "comparison_df": self.evaluator.compare([faiss_report, cgir_report]),
        }

    @staticmethod
    def print_report(output: Dict):
        print(output["faiss_report"])
        print(output["cgir_report"])
        print("\n" + "=" * 55)
        print("  COMPARISON: FAISS Baseline vs CG-IR (Graph + FAISS)")
        print("=" * 55)
        try:
            from tabulate import tabulate
            print(tabulate(
                output["comparison_df"].T,
                headers="keys",
                tablefmt="rounded_outline",
                floatfmt=".4f",
            ))
        except ImportError:
            print(output["comparison_df"].T.to_string())

    @staticmethod
    def export(output: Dict, output_dir: str = "output/eval"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output["comparison_df"].to_csv(f"{output_dir}/comparison.csv")

        for name, results in [("faiss_baseline", output["faiss_results"]),
                               ("cgir",           output["cgir_results"])]:
            rows = [{
                "query_id":        r["query_id"],
                "retrieved_bloom": r["retrieved_bloom"],
                "target_bloom":    r["target_bloom"],
                "graph_used":      r.get("graph_used", False),
                "graph_filtered":  r.get("graph_filtered", False),
                "latency_ms":      round(r["latency_ms"], 2),
                "top1_question":   r.get("top1_question", {}).get("question", "")[:80],
            } for r in results]
            pd.DataFrame(rows).to_csv(f"{output_dir}/{name}_per_query.csv", index=False)

        print(f"✅ Exported → {output_dir}/")


# ── Bloom-only eval (no relevance judgments needed) ───────────────────────────

def evaluate_bloom_accuracy(pipeline, test_queries: List[Dict], verbose: bool = True) -> Dict:
    """
    Quick Bloom-level accuracy check — tidak butuh ground truth.

    Args:
        test_queries: [{"query_id", "keyword", "mastery", "expected_bloom"}]
    """
    correct, errors = 0, []
    for q in test_queries:
        res = pipeline.retrieve(q["keyword"], mastery=q["mastery"])
        expected = q.get("expected_bloom", res.target_bloom)
        got = res.retrieved_bloom
        if got == expected:
            correct += 1
        else:
            errors.append({
                "query":    q["keyword"],
                "mastery":  q["mastery"],
                "expected": expected,
                "got":      got,
                "question": res.question.get("question", "")[:60],
            })

    accuracy = correct / len(test_queries) if test_queries else 0.0
    if verbose:
        print(f"Bloom Accuracy: {correct}/{len(test_queries)} = {accuracy:.1%}")
        for e in errors[:5]:
            print(f"  ✗ [{e['query']} | mastery={e['mastery']:.2f}] "
                  f"expected C{e['expected']}, got C{e['got']}: {e['question']}")
    return {"accuracy": accuracy, "errors": errors}
