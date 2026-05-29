## ⚙️ System Pipeline Architecture (CG-IR & Hybrid LLM Judge)

Karena dataset bank soal tidak memiliki kunci jawaban, sistem menggunakan pendekatan **Hybrid Context-Retrieval LLM Judge**. Sistem akan mencari materi teori yang relevan secara real-time untuk dijadikan "buku panduan" bagi LLM saat mengoreksi jawaban siswa.

```mermaid
flowchart TD
    %% Styling Kotak & Warna Teks Hitam Pekat
    classDef offline fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:black;
    classDef cgir fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:black;
    classDef judge fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:black;
    classDef state fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:black;
    classDef startend fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:black;

    %% Memaksa Judul Phase (Subgraph) berwarna Hitam
    style Phase1 color:black;
    style Phase2_3 color:black;
    style Phase4 color:black;
    style Phase5 color:black;

    %% Entry Point
    START(("▶️ MULAI:\nSiswa Input\nKeyword Topik")):::startend

    %% Phase 1: Offline
    subgraph Phase1 ["Phase 1: Offline Processing & Indexing"]
        A[("Dataset Kaggle:\nKonteks & Pertanyaan")] ==> B["Auto-HKG Construction"]
        B ==> C{"Knowledge Graph \n Nodes & Edges"}
        A ==> D["BM25 Search Index"]
    end
    class Phase1 offline

    %% Phase 2 & 3: CG-IR
    subgraph Phase2_3 ["Phase 2 & 3: CG-IR Question Recommendation"]
        START ==> G
        G(("Learner State \n Mastery s_c")) ==> H{"Query Formulation \n & Bloom C1-C6 Targeting"}
        C -.->|Prerequisite / Peer / Successor| H
        
        H ==>|Graph Query + Target Bloom| I["BM25 Retrieval + \n Hard Filter Bloom"]
        D -.->|Search Candidates| I
        
        I ==> J["Pseudo-Relevance Feedback \n PRF Re-ranking"]
        J ==> K(["Rekomendasi 1 Soal Terbaik"])
    end
    class Phase2_3 cgir

    %% User Interaction
    K ==> L[/"Siswa Menjawab Soal \n Teks Esai / Singkat"/]

    %% Phase 4: Hybrid LLM Judge
    subgraph Phase4 ["Phase 4: Hybrid LLM Judge - Evaluation"]
        L ==> M["Ambil 'Konteks' Teori \ndari Baris Soal yang Sama"]
        A -.->|Retrieve Context| M
        
        M ==>|Soal + Jawaban + Teori Konteks| N["LLM Judge Prompting"]
        N ==> O{"JSON Output: \n Score 0/1 & Feedback"}
    end
    class Phase4 judge

    %% Phase 5: BKT & Stop Condition
    subgraph Phase5 ["Phase 5: Dynamic Knowledge Tracing & Exit"]
        O ==> P{"Cek Batas?\nSkor > 0.9 ATAU\nSoal > Max"}
        P ==>|Belum, Lanjut| Q["Bayesian Knowledge Tracing \n BKT Update"]
        Q ==>|Update State s_c| G
    end
    class Phase5 state

    %% Exit Point
    P ==>|Ya, Selesai| END(("🛑 SELESAI:\nSesi Berakhir / Lulus")):::startend

    %% Menebalkan semua garis dan memberinya warna hitam (menggunakan kata 'black' agar tidak error di GitHub)
    linkStyle default stroke-width:4px,stroke:black;
