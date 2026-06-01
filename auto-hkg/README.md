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

# Auto-HKG — Automated Hierarchical Knowledge Graph Constructor

Implementasi Auto-HKG dari paper Wang et al. (2026) "Beyond Static Question Banks: Dynamic Knowledge Expansion via LLM-Automated Graph Construction and Adaptive Generation".

Pipeline ini secara otomatis membangun hierarchical knowledge graph dari kumpulan soal ujian menggunakan LLM lokal (Gemma 4 via Unsloth), tanpa butuh anotasi manual dari ahli.

---

## Apa yang Dilakukan Pipeline Ini

Setiap soal diproses oleh LLM untuk mengekstrak:
- **topic_coarse** — kategori topik luas (setara bab)
- **topic_fine** — sub-topik spesifik (setara sub-bab)
- **concepts** — konsep pengetahuan yang diuji
- **methods** — skill kognitif yang dibutuhkan untuk menjawab
- **prerequisites** — konsep yang harus dipahami sebelumnya
- **successors** — konsep lanjutan setelah ini
- **difficulty** — tingkat kesulitan berdasarkan level Bloom

Hasil ekstraksi dirangkai menjadi directed graph (NetworkX DiGraph) yang merepresentasikan hierarki dan ketergantungan antar konsep.

---

## Struktur Graph

Graph terdiri dari 5 jenis node dan 7 jenis relasi.

### Node

| Node | Deskripsi | Contoh |
|---|---|---|
| topic_coarse | Kategori topik luas, setara bab | Letak Geografis Indonesia |
| topic_fine | Sub-topik spesifik, setara sub-bab | Lokasi Absolut dan Relatif |
| concept | Unit pengetahuan terkecil yang diuji soal | letak astronomis |
| question | Soal ujian dari dataset | Sebutkan letak astronomis Indonesia |
| method | Skill kognitif untuk menjawab soal | membaca peta |

### Relasi

| Relasi | Dari | Ke | Artinya |
|---|---|---|---|
| has_subtopic | topic_coarse | topic_fine | Bab ini punya sub-bab ini |
| has_concept | topic_fine | concept | Sub-bab ini mengandung konsep ini |
| has_question | concept | question | Konsep ini diuji oleh soal ini |
| prerequisite | concept | concept | Harus paham A dulu sebelum belajar B |
| successor | concept | concept | Setelah paham A, lanjut ke B |
| peer | concept | concept | A dan B adalah konsep yang setara |
| requires_method | question | method | Soal ini butuh skill kognitif ini |

## Referensi

Wang, Y., Wei, T., Li, Q., & Zeng, L. (2026). *Beyond Static Question Banks: Dynamic Knowledge Expansion via LLM-Automated Graph Construction and Adaptive Generation.* PVLDB.
