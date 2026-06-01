## ⚙️ System Pipeline Architecture (CG-IR & Hybrid LLM Judge)

Karena dataset bank soal tidak memiliki kunci jawaban, sistem menggunakan pendekatan **Hybrid Context-Retrieval LLM Judge**. Sistem akan mencari materi teori yang relevan secara real-time untuk dijadikan "buku panduan" bagi LLM saat mengoreksi jawaban siswa.

```mermaid
flowchart TD
    %% ── Node styles ──────────────────────────────────────────────
    classDef offline  fill:#DBEAFE,stroke:#1D4ED8,stroke-width:1.5px,color:#1e3a5f,rx:8
    classDef cgir     fill:#DCFCE7,stroke:#15803D,stroke-width:1.5px,color:#14532d,rx:8
    classDef judge    fill:#FEF9C3,stroke:#CA8A04,stroke-width:1.5px,color:#713f12,rx:8
    classDef state    fill:#F3E8FF,stroke:#7E22CE,stroke-width:1.5px,color:#3b0764,rx:8
    classDef io       fill:#F1F5F9,stroke:#64748B,stroke-width:2px,color:#0f172a,rx:8
    classDef terminal fill:#E2E8F0,stroke:#475569,stroke-width:2px,color:#0f172a

    %% ── PHASE 1 — Offline Indexing ───────────────────────────────
    subgraph P1["Phase 1 — Offline Processing & Indexing"]
        direction TB
        DS[("Dataset\nKonteks & Pertanyaan")]:::io
        HKG["Auto-HKG Construction\nEkstrak node & edge KG"]:::offline
        KG{{"Knowledge Graph\nNode: Topik › Konsep › Soal\nEdge: prerequisite · peer · successor"}}:::offline
        IDX["BM25 Search Index\nInverted index seluruh corpus"]:::offline

        DS --> HKG
        HKG --> KG
        DS --> IDX
    end

    %% ── PHASE 2-3 — CG-IR Retrieval ─────────────────────────────
    subgraph P23["Phase 2 & 3 — CG-IR Question Retrieval"]
        direction TB
        LS(["Learner State\nMastery vector  s_c ∈ [0, 1]"]):::cgir
        QF["Graph-guided Query Formulation\nPilih node KG + target level Bloom C1–C6"]:::cgir
        BM["BM25 Retrieval\n+ Hard filter level Bloom"]:::cgir
        PRF["Pseudo-Relevance Feedback\nQuery expansion · re-ranking"]:::cgir
        REC(["1 Soal Terbaik\nSesuai mastery & level kognitif"]):::cgir

        LS --> QF
        KG -. "navigasi graph\n(prerequisite / peer / successor)" .-> QF
        QF --> BM
        IDX -. "kandidat soal" .-> BM
        BM --> PRF
        PRF --> REC
    end

    %% ── User interaction ─────────────────────────────────────────
    ANS[/"Siswa Menjawab Soal\nteks esai atau jawaban singkat"/]:::io

    %% ── PHASE 4 — Hybrid LLM Judge ───────────────────────────────
    subgraph P4["Phase 4 — Hybrid LLM Judge"]
        direction TB
        CTX["Retrieve Context\nAmbil paragraf teori dari baris soal yang sama"]:::judge
        LLM["LLM Judge Prompting\nInput: soal + jawaban siswa + teori konteks"]:::judge
        OUT{{"JSON Output\nscore: 0 atau 1\nfeedback: kalimat singkat"}}:::judge

        CTX --> LLM
        LLM --> OUT
    end

    DS -. "context retrieval" .-> CTX

    %% ── PHASE 5 — BKT & Loop ─────────────────────────────────────
    subgraph P5["Phase 5 — Dynamic Knowledge Tracing & Exit"]
        direction TB
        CHK{"Kondisi berhenti?\nscore > 0.9\natau jumlah soal > batas"}:::state
        BKT["BKT Update\nHitung ulang mastery s_c\npakai model Bayesian"]:::state
        CHK -->|"Belum tercapai"| BKT
    end

    %% ── Entry & exit ─────────────────────────────────────────────
    START(["▶  Mulai\nSiswa pilih topik"]):::terminal
    DONE(["⏹  Selesai\nSesi berakhir / lulus"]):::terminal

    %% ── Main flow ────────────────────────────────────────────────
    START --> LS
    REC   --> ANS
    ANS   --> CTX
    OUT   --> CHK
    BKT   -->|"Update state s_c"| LS
    CHK   -->|"Tercapai"| DONE
```

# Auto-HKG 
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


