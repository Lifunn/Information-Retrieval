# CG-IR Adaptive Quiz System

**Context-Guided Information Retrieval: Knowledge Graph + BM25 + RM3**

Final Project — Mata Kuliah Information Retrieval

---

## Overview

Sistem adaptive quiz yang menggunakan **CG-IR** (Context-Guided IR) untuk memilih soal secara dinamis berdasarkan mastery learner. Pipeline terdiri dari:

```
keyword + mastery
     │
     ▼
[GraphNavigator]   KG traversal → enriched query terms + Bloom target
     │
     ▼
[BM25 Initial]     top-K candidates dengan Bloom filter
     │
     ▼
[RM3 Expansion]    pseudo-relevance feedback → expanded query
     │
     ▼
[BM25 Re-rank]     final ranked list
     │
     ▼
 best question     → dikirim ke learner
```

Setelah learner menjawab, **LLM Judge** menilai jawaban dan **BKT** meng-update mastery score untuk menentukan soal berikutnya.

---

## Project Structure

```
cgir-adaptive-quiz/
├── src/
│   ├── data_loader.py           # Load CSV + graph JSON
│   ├── retrieval/
│   │   ├── bm25_index.py        # BM25Okapi dengan preprocessing BI
│   │   ├── rm3_expander.py      # RM3 query expansion
│   │   ├── graph_navigator.py   # KG traversal + query formulation
│   │   └── cgir_pipeline.py     # Pipeline utama + BM25 baseline
│   ├── tracing/
│   │   └── bkt.py               # Bayesian Knowledge Tracing
│   ├── judge/
│   │   └── llm_judge.py         # LLM answer judge + relevance judge
│   └── evaluation/
│       ├── metrics.py           # P@k, R@k, NDCG@k, MRR, MAP, Bloom acc
│       └── benchmark.py         # CG-IR vs BM25 comparison runner
├── notebooks/
│   └── CG_IR_Pipeline.py        # Notebook utama (salin ke Colab)
├── data/
│   └── Knowledge_Base_Update.csv
├── output/
│   ├── graph/                   # Output auto-HKG
│   └── eval/                    # Hasil evaluasi
└── requirements.txt
```

---


## Evaluation Metrics

| Metrik | Keterangan |
|--------|-----------|
| P@k | Precision at k |
| R@k | Recall at k |
| NDCG@k | Normalized Discounted Cumulative Gain |
| MRR | Mean Reciprocal Rank |
| MAP | Mean Average Precision |
| Bloom Acc | % soal yang Bloom level-nya tepat |
| Bloom MAE | Mean Absolute Error Bloom level |
| Latency | Rata-rata + P95 latency retrieval |

Karena dataset tidak memiliki kunci jawaban, relevance judgments dibuat menggunakan **LLM judge** (synthetic ground truth). Bloom-level accuracy dievaluasi secara langsung tanpa ground truth tambahan.

---

## Hyperparameters

| Parameter | Default | Keterangan |
|-----------|---------|-----------|
| BM25 k1 | 1.5 | Term frequency saturation |
| BM25 b | 0.75 | Document length normalization |
| RM3 fb_docs | 10 | Jumlah pseudo-relevant docs |
| RM3 fb_terms | 15 | Jumlah expansion terms |
| RM3 alpha | 0.5 | Interpolation weight (query vs RM1) |
| Bloom window | 1 | Toleransi filter (target ± window) |

---

## Dependencies

- `rank-bm25` — BM25 implementation
- `networkx` — Knowledge graph traversal
- `Sastrawi` — Indonesian text stemming
- `anthropic` — LLM judge & relevance judgment generation
- `scikit-learn`, `numpy`, `pandas` — Utilities
