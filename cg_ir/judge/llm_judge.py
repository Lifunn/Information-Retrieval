"""
judge/llm_judge.py
==================
LLM-based answer judge menggunakan Anthropic API.

Input : soal + jawaban siswa + konteks teori (dari dataset)
Output: {"score": 0|1, "feedback": "..."}

Juga menyediakan LLMRelevanceJudge — digunakan untuk menghasilkan
synthetic relevance judgments untuk evaluasi IR offline.
"""

import json
import re
from typing import Dict, Optional, Tuple
import anthropic

_client = anthropic.Anthropic()
MODEL   = "claude-sonnet-4-20250514"


# ── Answer Judge ──────────────────────────────────────────────────────────────

ANSWER_JUDGE_PROMPT = """\
Kamu adalah asisten evaluasi soal untuk pelajaran {subject}.
Tugasmu: nilai apakah jawaban siswa BENAR (score: 1) atau SALAH (score: 0).

Gunakan konteks teori berikut sebagai referensi utama:
<konteks>
{context}
</konteks>

Soal:
{question}

Jawaban siswa:
{answer}

Berikan penilaian dalam format JSON berikut (tanpa teks lain):
{{"score": 0 atau 1, "feedback": "kalimat singkat 1-2 kalimat menjelaskan kenapa benar/salah"}}
"""


def judge_answer(
    question: str,
    answer: str,
    context: str = "",
    subject: str = "IPS/Geografi SMA",
) -> Dict:
    """
    Nilai jawaban siswa menggunakan LLM.

    Returns:
        {"score": 0|1, "feedback": str}
    """
    prompt = ANSWER_JUDGE_PROMPT.format(
        subject=subject,
        context=context[:1500] if context else "Tidak ada konteks tersedia.",
        question=question,
        answer=answer,
    )

    try:
        response = _client.messages.create(
            model=MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)

        return {
            "score":    int(result.get("score", 0)),
            "feedback": str(result.get("feedback", "")),
        }

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {"score": 0, "feedback": f"[Parse error: {e}] Raw: {raw[:100]}"}
    except anthropic.APIError as e:
        return {"score": 0, "feedback": f"[API error: {e}]"}


# ── Relevance Judge (untuk evaluasi IR) ───────────────────────────────────────

RELEVANCE_JUDGE_PROMPT = """\
Kamu adalah expert evaluasi information retrieval untuk soal pelajaran IPS/Geografi SMA.
Tugasmu: nilai RELEVANSI antara satu query dan satu soal.

Query: "{query}"
Topic target: "{topic}"
Bloom level yang diinginkan: C{target_bloom} ({bloom_desc})

Soal yang dievaluasi:
{question}

Topik soal: {doc_topic}
Konsep soal: {doc_concept}
Bloom level soal: C{doc_bloom}

Skala relevansi:
  0 = tidak relevan sama sekali
  1 = sedikit relevan (topik berkaitan jauh)
  2 = cukup relevan (topik sesuai tapi bloom level jauh)
  3 = sangat relevan (topik sesuai DAN bloom level sesuai)

Jawab hanya dengan angka 0, 1, 2, atau 3. Tidak ada teks lain.
"""

BLOOM_DESCRIPTIONS = {
    1: "Mengingat (recall facts)",
    2: "Memahami (explain concepts)",
    3: "Mengaplikasikan (apply to new situation)",
    4: "Menganalisis (analyze relationships)",
    5: "Mengevaluasi (make judgments)",
    6: "Mencipta (create new ideas)",
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

    Returns:
        Relevance grade 0–3
    """
    prompt = RELEVANCE_JUDGE_PROMPT.format(
        query=query,
        topic=topic or query,
        target_bloom=target_bloom,
        bloom_desc=BLOOM_DESCRIPTIONS.get(target_bloom, ""),
        question=document.get("question", "")[:400],
        doc_topic=document.get("topic", ""),
        doc_concept=document.get("concept", ""),
        doc_bloom=document.get("bloom_level", "?"),
    )

    try:
        response = _client.messages.create(
            model=MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        grade = int(raw[0])
        return min(max(grade, 0), 3)
    except Exception:
        return 0


def generate_relevance_judgments(
    test_queries: list,       # List of {"query_id": str, "keyword": str, "target_bloom": int}
    candidates: list,         # List of document dicts (from retrieve_candidates)
    output_path: str = None,
    verbose: bool = True,
) -> list:
    """
    Generate synthetic relevance judgments menggunakan LLM judge.
    Simpan ke JSON untuk dipakai berulang kali (mahal di token!).

    Returns:
        List of {"query_id", "doc_id", "grade"}
    """
    judgments = []
    total = len(test_queries) * len(candidates)

    for q_idx, query in enumerate(test_queries):
        for d_idx, doc in enumerate(candidates):
            if verbose:
                print(f"\r[{q_idx * len(candidates) + d_idx + 1}/{total}] Judging...", end="")

            grade = judge_relevance(
                query=query["keyword"],
                document=doc,
                target_bloom=query.get("target_bloom", 1),
                topic=query.get("keyword", ""),
            )
            judgments.append({
                "query_id": query["query_id"],
                "doc_id":   doc["id"],
                "grade":    grade,
            })

    if verbose:
        print(f"\n✅ Generated {len(judgments)} relevance judgments")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(judgments, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved → {output_path}")

    return judgments
