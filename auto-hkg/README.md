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

## Cara Pakai (Google Colab)

### 1 — Buka notebook
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/auto-hkg/blob/main/notebooks/Auto_HKG_Colab.ipynb)

### 2 — Siapkan API key (pilih salah satu)

| Provider | Daftar | Harga |
|---|---|---|
| **Groq** *(recommended)* | [console.groq.com](https://console.groq.com) | ✅ Gratis |
| Google Gemini | [aistudio.google.com](https://aistudio.google.com) | ✅ Gratis (tier) |
| OpenAI | [platform.openai.com](https://platform.openai.com) | 💰 ~$0.0002/1K token |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | 💰 Murah |
| Ollama | [ollama.com](https://ollama.com) | ✅ Lokal, gratis |

### 3 — Jalankan pipeline

```bash
# Lokal
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
python src/auto_hkg.py --provider groq --batch-size 10

# Dengan provider lain
python src/auto_hkg.py --provider openai --model gpt-4o-mini
python src/auto_hkg.py --provider gemini --model gemini-2.0-flash
```

---

## Model LLM yang Didukung

### 🆓 Gratis

| Provider | Model | Kecepatan | Kualitas JSON |
|---|---|---|---|
| Groq | `llama-3.3-70b-versatile` | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| Groq | `mixtral-8x7b-32768` | ⚡⚡⚡ | ⭐⭐⭐ |
| Gemini | `gemini-2.0-flash` | ⚡⚡ | ⭐⭐⭐⭐ |
| Ollama | `llama3.2` (lokal) | ⚡ | ⭐⭐⭐ |

### 💰 Berbayar (murah)

| Provider | Model | Biaya per ~11K baris |
|---|---|---|
| OpenAI | `gpt-4o-mini` | ~$0.50 |
| Anthropic | `claude-haiku-4-5-20251001` | ~$0.40 |
| OpenAI | `gpt-4o` | ~$5.00 |

> **Rekomendasi untuk Colab:** Groq `llama-3.3-70b-versatile` — gratis, cepat, output JSON-nya konsisten.

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
