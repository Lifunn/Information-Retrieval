"""
manual_evaluation.py — Evaluasi akurasi manual Auto-HKG (50 soal sample).

Mengikuti metodologi paper Wang et al. (2026):
- Akurasi coarse: apakah topic_coarse relevan dengan soal?
- Akurasi fine: apakah topic_fine relevan dengan soal?
- Akurasi concept: apakah concepts yang diekstrak relevan?

Output: manual_evaluation_template.csv (template untuk dinilai manual)
        manual_evaluation_result.csv   (setelah diisi, jalankan hitung_akurasi())

Cara pakai:
    1. Jalankan generate_template() → buka CSV di Excel/Sheets → isi kolom penilaian
    2. Simpan CSV → jalankan hitung_akurasi() → lihat hasil akurasi
"""

import json
import random
import pandas as pd
import networkx as nx
from pathlib import Path


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data(graph_path: str, dataset_path: str):
    with open(graph_path, encoding="utf-8") as f:
        G_data = json.load(f)
    G = nx.node_link_graph(G_data)

    df = pd.read_csv(dataset_path, sep=";")
    df.columns = [c.strip() for c in df.columns]

    return G, df


# ---------------------------------------------------------------------------
# Ekstrak info dari graph untuk satu question node
# ---------------------------------------------------------------------------

def get_question_info(G: nx.DiGraph, q_node_id: str) -> dict:
    """Ambil topic_coarse, topic_fine, concepts, methods dari graph untuk satu soal."""
    info = {
        "topic_coarse" : [],
        "topic_fine"   : [],
        "concepts"     : [],
        "methods"      : [],
        "bloom_level"  : G.nodes[q_node_id].get("bloom_level", ""),
        "difficulty"   : G.nodes[q_node_id].get("difficulty", ""),
    }

    # Cari node yang punya relasi has_question ke soal ini
    for pred in G.predecessors(q_node_id):
        pred_data = G.nodes[pred]
        pred_type = pred_data.get("type", "")
        rel       = G.edges[pred, q_node_id].get("relation", "")

        if rel == "has_question":
            if pred_type == "topic_coarse":
                info["topic_coarse"].append(pred_data.get("label", pred))
            elif pred_type == "topic_fine":
                info["topic_fine"].append(pred_data.get("label", pred))
            elif pred_type == "concept":
                info["concepts"].append(pred_data.get("label", pred))

    # Cari method dari soal ini
    for succ in G.successors(q_node_id):
        succ_data = G.nodes[succ]
        rel       = G.edges[q_node_id, succ].get("relation", "")
        if rel == "requires_method":
            info["methods"].append(succ_data.get("label", succ))

    return info


# ---------------------------------------------------------------------------
# Generate template evaluasi
# ---------------------------------------------------------------------------

def generate_template(
    graph_path: str,
    dataset_path: str,
    output_path: str = "manual_evaluation_template.csv",
    n_sample: int = 50,
    random_seed: int = 42,
):
    print(f"Loading graph dari: {graph_path}")
    G, df = load_data(graph_path, dataset_path)

    # Ambil semua question nodes
    q_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "question"]
    print(f"Total question nodes: {len(q_nodes)}")

    # Sample 50 soal
    random.seed(random_seed)
    sampled = random.sample(q_nodes, min(n_sample, len(q_nodes)))

    rows = []
    for q_id, q_data in sampled:
        info = get_question_info(G, q_id)

        rows.append({
            # Info soal
            "no"              : len(rows) + 1,
            "question_id"     : q_id,
            "pertanyaan"      : q_data.get("label", ""),
            "bloom_level"     : q_data.get("bloom_level", ""),
            "difficulty"      : q_data.get("difficulty", ""),

            # Hasil ekstraksi graph
            "topic_coarse"    : " | ".join(info["topic_coarse"]) if info["topic_coarse"] else "-",
            "topic_fine"      : " | ".join(info["topic_fine"])   if info["topic_fine"]   else "-",
            "concepts"        : " | ".join(info["concepts"])     if info["concepts"]     else "-",
            "methods"         : " | ".join(info["methods"])      if info["methods"]      else "-",

            # Kolom penilaian manual — ISI DENGAN: 1 (benar) atau 0 (salah)
            "coarse_correct"  : "",   # 1 = topic_coarse relevan, 0 = tidak relevan
            "fine_correct"    : "",   # 1 = topic_fine relevan, 0 = tidak relevan
            "concept_correct" : "",   # 1 = minimal 1 concept relevan, 0 = tidak ada yang relevan
            "catatan"         : "",   # catatan opsional
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Template tersimpan di: {output_path}")
    print(f"   Total soal: {len(rows)}")
    print(f"\nPetunjuk pengisian:")
    print("  - Buka file CSV di Excel atau Google Sheets")
    print("  - Isi kolom 'coarse_correct', 'fine_correct', 'concept_correct'")
    print("  - Isi dengan: 1 (relevan/benar) atau 0 (tidak relevan/salah)")
    print("  - Simpan file lalu jalankan hitung_akurasi()")

    return result_df


# ---------------------------------------------------------------------------
# Hitung akurasi dari template yang sudah diisi
# ---------------------------------------------------------------------------

def hitung_akurasi(
    template_path: str = "manual_evaluation_template.csv",
    model_name: str = "gemma-4-E2B",
    output_path: str = "manual_evaluation_result.json",
):
    df = pd.read_csv(template_path, encoding="utf-8-sig")

    # Filter baris yang sudah diisi
    filled = df[df["coarse_correct"].notna() & (df["coarse_correct"] != "")]
    n = len(filled)

    if n == 0:
        print("❌ Belum ada penilaian yang diisi.")
        return

    coarse_acc  = filled["coarse_correct"].astype(int).sum() / n * 100
    fine_acc    = filled["fine_correct"].astype(int).sum()   / n * 100
    concept_acc = filled["concept_correct"].astype(int).sum()/ n * 100
    avg_acc     = (coarse_acc + fine_acc + concept_acc) / 3

    result = {
        "model"               : model_name,
        "n_evaluated"         : n,
        "coarse_accuracy_pct" : round(coarse_acc, 1),
        "fine_accuracy_pct"   : round(fine_acc, 1),
        "concept_accuracy_pct": round(concept_acc, 1),
        "average_accuracy_pct": round(avg_acc, 1),
    }

    # Bandingkan dengan paper
    sep = "=" * 55
    print(sep)
    print(f"  HASIL EVALUASI MANUAL — {model_name}")
    print(sep)
    print(f"  Soal dievaluasi     : {n}")
    print(f"  Akurasi coarse      : {coarse_acc:.1f}%  (paper: 75%)")
    print(f"  Akurasi fine        : {fine_acc:.1f}%  (paper: 90%)")
    print(f"  Akurasi concept     : {concept_acc:.1f}%")
    print(f"  Rata-rata akurasi   : {avg_acc:.1f}%")
    print(sep)

    # Interpretasi vs paper
    print("\n  Perbandingan dengan paper (Wang et al., 2026):")
    if coarse_acc >= 75:
        print(f"  ✅ Coarse accuracy {coarse_acc:.1f}% ≥ paper 75%")
    else:
        print(f"  ⚠️  Coarse accuracy {coarse_acc:.1f}% < paper 75%")

    if fine_acc >= 90:
        print(f"  ✅ Fine accuracy {fine_acc:.1f}% ≥ paper 90%")
    else:
        print(f"  ⚠️  Fine accuracy {fine_acc:.1f}% < paper 90%")

    # Simpan result
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Result tersimpan di: {output_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual evaluation for Auto-HKG")
    parser.add_argument("--mode",    choices=["generate", "evaluate"], required=True)
    parser.add_argument("--graph",   default="output/graph/knowledge_graph.json")
    parser.add_argument("--dataset", default="data/Knowledge_Base_Sample.csv")
    parser.add_argument("--model",   default="gemma-4-E2B")
    parser.add_argument("--template",default="manual_evaluation_template.csv")
    parser.add_argument("--out",     default=".")
    args = parser.parse_args()

    if args.mode == "generate":
        generate_template(
            graph_path   = args.graph,
            dataset_path = args.dataset,
            output_path  = f"{args.out}/manual_evaluation_template.csv",
        )
    elif args.mode == "evaluate":
        hitung_akurasi(
            template_path = args.template,
            model_name    = args.model,
            output_path   = f"{args.out}/manual_evaluation_result_{args.model}.json",
        )
