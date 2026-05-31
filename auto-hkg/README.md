# Auto-HKG 🧠
**Automated Hierarchical Knowledge Graph Constructor**

Implementasi modul Auto-HKG dari paper:
> *"Beyond Static Question Banks: Dynamic Knowledge Expansion via LLM-Automated Graph Construction and Adaptive Generation"* (Wang et al., 2026)

Diadaptasi untuk dataset Geografi/IPS SMA dengan pipeline **CG-IR** (pengganti RAG) untuk final project mata kuliah **Information Retrieval**.

---

## Struktur Proyek

```
auto-hkg/
├── src/
│   ├── auto_hkg.py          # pipeline utama
│   ├── llm_client.py        # abstraksi provider LLM
│   ├── schema_validator.py  # validasi output JSON
│   ├── graph_builder.py     # perakitan graph (NetworkX)
│   └── visualize_graph.py   # visualisasi PNG & HTML
├── data/
│   └── Knowledge_Base_Update.csv
├── output/
│   ├── graph/               # hasil: .json, .csv, .png, .html
│   └── logs/                # checkpoint & log
├── notebooks/
│   └── Auto_HKG_Colab.ipynb
├── requirements.txt
└── README.md
```

---

## Output

Setelah pipeline selesai, `output/graph/` berisi:

```
knowledge_graph.json   # full graph (node-link format, untuk CG-IR)
nodes.csv              # semua node beserta atributnya
edges.csv              # semua edge + relasi
stats.json             # ringkasan statistik
graph_static.png       # visualisasi PNG (untuk laporan)
graph_interactive.html # visualisasi interaktif (buka di browser)
```

### Struktur Graph

```
topic_coarse  ──has_subtopic──>  topic_fine
topic_fine    ──has_concept──>   concept
concept       ──has_question──>  question
question      ──requires_method──> method
concept       ──prerequisite──>  concept
concept       ──successor──>     concept
concept       ──peer──>          concept
```

---

## Checkpoint & Resume

Pipeline otomatis menyimpan progress ke `output/logs/checkpoint.json`.
Jika Colab disconnect di tengah jalan, cukup jalankan ulang — proses akan dilanjutkan dari baris terakhir yang berhasil.

---

## Referensi

Wang, Y., Wei, T., Li, Q., & Zeng, L. (2026). *Beyond Static Question Banks: Dynamic Knowledge Expansion via LLM-Automated Graph Construction and Adaptive Generation.* PVLDB.
