"""
evaluate_graph.py — Evaluasi kualitas Knowledge Graph hasil Auto-HKG.

Menghasilkan laporan evaluasi lengkap yang bisa dibandingkan antar model
(E2B vs 4B, dsb). Output disimpan ke evaluation_report.json dan evaluation_report.txt.

Cara pakai:
    python evaluate_graph.py --graph output/graph/knowledge_graph.json --model gemma-4-E2B
    python evaluate_graph.py --graph output/graph/knowledge_graph.json --model gemma-4-4B
"""

import json
import argparse
import networkx as nx
from collections import Counter
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Load graph
# ---------------------------------------------------------------------------

def load_graph(path: str) -> nx.DiGraph:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data)


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------

def evaluate(G: nx.DiGraph, model_name: str) -> dict:
    results = {
        "model"     : model_name,
        "timestamp" : datetime.now().isoformat(),
    }

    # ── 1. Basic stats ───────────────────────────────────────────────────────
    results["basic"] = {
        "total_nodes" : G.number_of_nodes(),
        "total_edges" : G.number_of_edges(),
        "density"     : round(nx.density(G), 6),
    }

    # ── 2. Node type distribution ────────────────────────────────────────────
    type_counts = Counter(d.get("type", "unknown") for _, d in G.nodes(data=True))
    results["node_types"] = dict(type_counts)

    # ── 3. Edge relation distribution ────────────────────────────────────────
    rel_counts = Counter(d.get("relation", "unknown") for _, _, d in G.edges(data=True))
    results["edge_relations"] = dict(rel_counts)

    # ── 4. Coverage ──────────────────────────────────────────────────────────
    n_questions = type_counts.get("question", 0)
    n_concepts  = type_counts.get("concept", 0)
    n_methods   = type_counts.get("method", 0)
    results["coverage"] = {
        "questions_processed"   : n_questions,
        "coverage_pct"          : round(n_questions / 1200 * 100, 1),
        "concepts_per_question" : round(n_concepts / n_questions, 2) if n_questions else 0,
        "topic_fine_per_coarse" : round(
            type_counts.get("topic_fine", 0) / max(type_counts.get("topic_coarse", 1), 1), 2
        ),
        "method_fill_rate_pct"  : round(n_methods / max(n_questions, 1) * 100, 1),
    }

    # ── 5. Connectivity ──────────────────────────────────────────────────────
    G_und    = G.to_undirected()
    comps    = list(nx.connected_components(G_und))
    largest  = len(max(comps, key=len))
    results["connectivity"] = {
        "connected_components"      : len(comps),
        "largest_component_nodes"   : largest,
        "largest_component_pct"     : round(largest / G.number_of_nodes() * 100, 1),
        "is_fully_connected"        : len(comps) == 1,
    }

    # ── 6. Degree stats ──────────────────────────────────────────────────────
    degrees = [d for _, d in G.degree()]
    results["degree_stats"] = {
        "avg_degree" : round(sum(degrees) / len(degrees), 2),
        "max_degree" : max(degrees),
        "min_degree" : min(degrees),
    }

    # ── 7. Top 10 hub nodes ──────────────────────────────────────────────────
    top_nodes = sorted(G.degree(), key=lambda x: -x[1])[:10]
    results["top_hub_nodes"] = [
        {
            "node"   : node,
            "type"   : G.nodes[node].get("type", "?"),
            "degree" : deg,
        }
        for node, deg in top_nodes
    ]

    # ── 8. Bloom level distribution ──────────────────────────────────────────
    bloom_counts = Counter(
        d.get("bloom_level", "?")
        for _, d in G.nodes(data=True)
        if d.get("type") == "question"
    )
    results["bloom_distribution"] = dict(sorted(bloom_counts.items()))

    # ── 9. Difficulty distribution ───────────────────────────────────────────
    diff_counts = Counter(
        str(d.get("difficulty", "?"))
        for _, d in G.nodes(data=True)
        if d.get("type") == "question"
    )
    results["difficulty_distribution"] = dict(sorted(diff_counts.items()))

    # ── 10. Concept uniqueness ───────────────────────────────────────────────
    concept_ids = [n for n, d in G.nodes(data=True) if d.get("type") == "concept"]
    results["concept_quality"] = {
        "total_concepts"  : len(concept_ids),
        "unique_concepts" : len(set(concept_ids)),
        "duplicate_count" : len(concept_ids) - len(set(concept_ids)),
    }

    # ── 11. Relational richness ──────────────────────────────────────────────
    n_prerequisite = rel_counts.get("prerequisite", 0)
    n_successor    = rel_counts.get("successor", 0)
    n_peer         = rel_counts.get("peer", 0)
    results["relational_richness"] = {
        "prerequisite_edges"      : n_prerequisite,
        "successor_edges"         : n_successor,
        "peer_edges"              : n_peer,
        "avg_prerequisites_per_q" : round(n_prerequisite / max(n_questions, 1), 2),
        "avg_successors_per_q"    : round(n_successor / max(n_questions, 1), 2),
    }

    # ── 12. Skor ringkasan (0-100) ───────────────────────────────────────────
    score = 0
    score += min(results["coverage"]["coverage_pct"], 100) * 0.30           # 30% coverage
    score += (1 if results["connectivity"]["is_fully_connected"] else 0) * 20  # 20% connectivity
    score += min(results["coverage"]["concepts_per_question"] / 4 * 100, 100) * 0.20  # 20% concept richness
    score += min(results["relational_richness"]["avg_prerequisites_per_q"] / 2 * 100, 100) * 0.15  # 15% prerequisites
    score += min(results["relational_richness"]["avg_successors_per_q"] / 2 * 100, 100) * 0.15    # 15% successors
    results["summary_score"] = round(score, 1)

    return results


# ---------------------------------------------------------------------------
# Format laporan teks
# ---------------------------------------------------------------------------

def format_report(r: dict) -> str:
    sep = "=" * 60
    lines = [
        sep,
        f"  AUTO-HKG EVALUATION REPORT",
        f"  Model     : {r['model']}",
        f"  Timestamp : {r['timestamp']}",
        sep,
        "",
        "[ 1. Basic Statistics ]",
        f"  Nodes   : {r['basic']['total_nodes']:,}",
        f"  Edges   : {r['basic']['total_edges']:,}",
        f"  Density : {r['basic']['density']}",
        "",
        "[ 2. Node Types ]",
        *[f"  {k:20s}: {v:,}" for k, v in r["node_types"].items()],
        "",
        "[ 3. Edge Relations ]",
        *[f"  {k:20s}: {v:,}" for k, v in r["edge_relations"].items()],
        "",
        "[ 4. Coverage ]",
        f"  Questions processed   : {r['coverage']['questions_processed']} / 1200 ({r['coverage']['coverage_pct']}%)",
        f"  Concepts per question : {r['coverage']['concepts_per_question']}",
        f"  Topic fine per coarse : {r['coverage']['topic_fine_per_coarse']}",
        f"  Method fill rate      : {r['coverage']['method_fill_rate_pct']}%",
        "",
        "[ 5. Connectivity ]",
        f"  Connected components  : {r['connectivity']['connected_components']}",
        f"  Largest component     : {r['connectivity']['largest_component_nodes']:,} nodes ({r['connectivity']['largest_component_pct']}%)",
        f"  Fully connected       : {'✅ Yes' if r['connectivity']['is_fully_connected'] else '❌ No'}",
        "",
        "[ 6. Degree Statistics ]",
        f"  Avg degree : {r['degree_stats']['avg_degree']}",
        f"  Max degree : {r['degree_stats']['max_degree']}",
        f"  Min degree : {r['degree_stats']['min_degree']}",
        "",
        "[ 7. Top 10 Hub Nodes ]",
        *[f"  [{n['type']:12s}] {n['node'][:40]:40s} degree={n['degree']}" for n in r["top_hub_nodes"]],
        "",
        "[ 8. Bloom Level Distribution ]",
        *[f"  {k}: {v}" for k, v in r["bloom_distribution"].items()],
        "",
        "[ 9. Difficulty Distribution ]",
        *[f"  difficulty {k}: {v}" for k, v in r["difficulty_distribution"].items()],
        "",
        "[ 10. Concept Quality ]",
        f"  Total concepts   : {r['concept_quality']['total_concepts']:,}",
        f"  Unique concepts  : {r['concept_quality']['unique_concepts']:,}",
        f"  Duplicates       : {r['concept_quality']['duplicate_count']}",
        "",
        "[ 11. Relational Richness ]",
        f"  Prerequisite edges        : {r['relational_richness']['prerequisite_edges']:,}",
        f"  Successor edges           : {r['relational_richness']['successor_edges']:,}",
        f"  Peer edges                : {r['relational_richness']['peer_edges']:,}",
        f"  Avg prerequisites / soal  : {r['relational_richness']['avg_prerequisites_per_q']}",
        f"  Avg successors / soal     : {r['relational_richness']['avg_successors_per_q']}",
        "",
        sep,
        f"  SUMMARY SCORE : {r['summary_score']} / 100",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compare dua model
# ---------------------------------------------------------------------------

def compare(r1: dict, r2: dict) -> str:
    sep = "=" * 60
    def diff(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = b - a
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            return f"{arrow} {abs(delta):.2f}"
        return "—"

    lines = [
        sep,
        f"  COMPARISON: {r1['model']}  vs  {r2['model']}",
        sep,
        f"  {'Metric':35s} {'Model 1':>12s} {'Model 2':>12s} {'Diff':>10s}",
        "-" * 75,
        f"  {'Summary Score':35s} {r1['summary_score']:>12} {r2['summary_score']:>12} {diff(r1['summary_score'], r2['summary_score']):>10}",
        f"  {'Total Nodes':35s} {r1['basic']['total_nodes']:>12,} {r2['basic']['total_nodes']:>12,} {diff(r1['basic']['total_nodes'], r2['basic']['total_nodes']):>10}",
        f"  {'Total Edges':35s} {r1['basic']['total_edges']:>12,} {r2['basic']['total_edges']:>12,} {diff(r1['basic']['total_edges'], r2['basic']['total_edges']):>10}",
        f"  {'Coverage (%)':35s} {r1['coverage']['coverage_pct']:>12} {r2['coverage']['coverage_pct']:>12} {diff(r1['coverage']['coverage_pct'], r2['coverage']['coverage_pct']):>10}",
        f"  {'Concepts per Question':35s} {r1['coverage']['concepts_per_question']:>12} {r2['coverage']['concepts_per_question']:>12} {diff(r1['coverage']['concepts_per_question'], r2['coverage']['concepts_per_question']):>10}",
        f"  {'Method Fill Rate (%)':35s} {r1['coverage']['method_fill_rate_pct']:>12} {r2['coverage']['method_fill_rate_pct']:>12} {diff(r1['coverage']['method_fill_rate_pct'], r2['coverage']['method_fill_rate_pct']):>10}",
        f"  {'Fully Connected':35s} {'Yes' if r1['connectivity']['is_fully_connected'] else 'No':>12} {'Yes' if r2['connectivity']['is_fully_connected'] else 'No':>12} {'':>10}",
        f"  {'Avg Degree':35s} {r1['degree_stats']['avg_degree']:>12} {r2['degree_stats']['avg_degree']:>12} {diff(r1['degree_stats']['avg_degree'], r2['degree_stats']['avg_degree']):>10}",
        f"  {'Prerequisite Edges':35s} {r1['relational_richness']['prerequisite_edges']:>12,} {r2['relational_richness']['prerequisite_edges']:>12,} {diff(r1['relational_richness']['prerequisite_edges'], r2['relational_richness']['prerequisite_edges']):>10}",
        f"  {'Successor Edges':35s} {r1['relational_richness']['successor_edges']:>12,} {r2['relational_richness']['successor_edges']:>12,} {diff(r1['relational_richness']['successor_edges'], r2['relational_richness']['successor_edges']):>10}",
        f"  {'Concept Duplicates':35s} {r1['concept_quality']['duplicate_count']:>12,} {r2['concept_quality']['duplicate_count']:>12,} {diff(r1['concept_quality']['duplicate_count'], r2['concept_quality']['duplicate_count']):>10}",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Auto-HKG output graph.")
    parser.add_argument("--graph",   required=True, help="Path to knowledge_graph.json")
    parser.add_argument("--model",   required=True, help="Model name label (e.g. gemma-4-E2B)")
    parser.add_argument("--compare", default=None,  help="Path to second report JSON for comparison")
    parser.add_argument("--out",     default=".",   help="Output directory for report files")
    args = parser.parse_args()

    print(f"Loading graph from: {args.graph}")
    G = load_graph(args.graph)

    print("Evaluating...")
    report = evaluate(G, args.model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simpan JSON
    json_path = out_dir / f"evaluation_{args.model.replace('/', '_')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON report saved: {json_path}")

    # Simpan TXT
    txt_path = out_dir / f"evaluation_{args.model.replace('/', '_')}.txt"
    report_text = format_report(report)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✅ TXT report saved: {txt_path}")

    # Print ke layar
    print("\n" + report_text)

    # Compare jika ada
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            other_report = json.load(f)
        compare_text = compare(other_report, report)
        cmp_path = out_dir / f"comparison_{other_report['model'].replace('/', '_')}_vs_{args.model.replace('/', '_')}.txt"
        with open(cmp_path, "w", encoding="utf-8") as f:
            f.write(compare_text)
        print("\n" + compare_text)
        print(f"✅ Comparison saved: {cmp_path}")
