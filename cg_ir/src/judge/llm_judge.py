"""
judge/llm_judge.py
==================
LLM judge menggunakan Groq API.
- judge_answer      : nilai jawaban siswa (skor kontinu 0.0–1.0)
- judge_relevance   : nilai relevansi dokumen terhadap query (untuk evaluasi IR)
"""

import os
import json
import re
from typing import Dict
from groq import Groq

_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
MODEL   = "llama-3.3-70b-versatile"


# ── Answer Judge ──────────────────────────────────────────────────────────────

ANSWER_JUDGE_PROMPT = """\
Kamu adalah penilai jawaban siswa untuk pelajaran IPS/Geografi SMA.
Gunakan konteks berikut sebagai referensi utama:
<konteks>
{context}
</konteks>
Soal:
{question}
Jawaban siswa:
{answer}
Nilai jawaban dalam skala 0.0 hingga 1.0:
  1.00 = benar sempurna dan lengkap
  0.75 = sebagian besar benar, ada yang kurang
  0.50 = setengah benar, inti ada tapi banyak yang hilang
  0.25 = sedikit benar, kebanyakan salah
  0.00 = salah total atau tidak relevan
Berikan penilaian dalam format JSON berikut (tanpa teks lain):
{{"score": 0.0 hingga 1.0, "feedback": "kalimat singkat 1-2 kalimat menjelaskan kenapa benar/salah"}}
"""


def judge_answer(
    question: str,
    answer: str,
    context: str = "",
    subject: str = "IPS/Geografi SMA",
) -> Dict:
    """
    Nilai jawaban siswa menggunakan Groq LLM.

    Returns:
        {"score": float 0.0–1.0, "feedback": str}
    """
    prompt = ANSWER_JUDGE_PROMPT.format(
        context=context[:2000] if context else "Tidak ada konteks tersedia.",
        question=question,
        answer=answer,
    )
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            max_tokens=256,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        raw    = resp.choices[0].message.content.strip()
        raw    = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        score  = float(result.get("score", 0.0))
        return {
            "score":    min(max(score, 0.0), 1.0),
            "feedback": str(result.get("feedback", "")),
        }
    except Exception as e:
        return {"score": 0.0, "feedback": f"[Error: {e}]"}


# ── Relevance Judge (untuk evaluasi IR offline) ───────────────────────────────

RELEVANCE_JUDGE_PROMPT = """\
Kamu adalah expert evaluasi information retrieval untuk soal IPS/Geografi SMA.
Nilai RELEVANSI antara query dan soal berikut.

Query: "{query}"
Bloom level yang diinginkan: C{target_bloom}

Soal: {question}
Topik soal: {doc_topic}
Konsep soal: {doc_concept}
Bloom level soal: C{doc_bloom}

Skala:
  0 = tidak relevan sama sekali
  1 = sedikit relevan (topik berkaitan jauh)
  2 = cukup relevan (topik sesuai tapi bloom level jauh)
  3 = sangat relevan (topik sesuai DAN bloom level sesuai)

Jawab hanya dengan angka 0, 1, 2, atau 3. Tidak ada teks lain.
"""

BLOOM_DESCRIPTIONS = {
    1: "Mengingat", 2: "Memahami", 3: "Mengaplikasikan",
    4: "Menganalisis", 5: "Mengevaluasi", 6: "Mencipta",
}


def judge_relevance(
    query: str,
    document: Dict,
    target_bloom: int,
    topic: str = "",
) -> int:
    """
    Nilai relevansi dokumen terhadap query (skala 0–3).
    Digunakan untuk membuat synthetic relevance judgments.
    """
    prompt = RELEVANCE_JUDGE_PROMPT.format(
        query=query,
        target_bloom=target_bloom,
        question=document.get("question", "")[:400],
        doc_topic=document.get("topic", ""),
        doc_concept=document.get("concept", ""),
        doc_bloom=document.get("bloom_level", "?"),
    )
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            max_tokens=8,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw   = resp.choices[0].message.content.strip()
        grade = int(raw[0])
        return min(max(grade, 0), 3)
    except Exception:
        return 0


def generate_relevance_judgments(
    test_queries: list,
    candidates: list,
    output_path: str = None,
    verbose: bool = True,
) -> list:
    """
    Generate synthetic relevance judgments menggunakan LLM judge.
    Simpan ke JSON untuk dipakai berulang kali.
    """
    judgments = []
    total     = len(test_queries) * len(candidates)

    for q_idx, query in enumerate(test_queries):
        for d_idx, doc in enumerate(candidates):
            if verbose:
                print(f"\r[{q_idx * len(candidates) + d_idx + 1}/{total}] Judging...", end="")
            grade = judge_relevance(
                query=query["keyword"],
                document=doc,
                target_bloom=query.get("target_bloom", 1),
            )
            judgments.append({
                "query_id": query["query_id"],
                "doc_id":   doc["id"],
                "grade":    grade,
            })

    if verbose:
        print(f"\nGenerated {len(judgments)} judgments")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(judgments, f, indent=2, ensure_ascii=False)
        print(f"Saved → {output_path}")

    return judgments
