"""
judge/llm_judge.py
==================
LLM-based judge menggunakan Groq API.

Model default: llama-3.3-70b-versatile
  - Free tier tersedia di console.groq.com
  - Sangat cepat (Groq LPU inference)
  - Cukup capable untuk menilai jawaban essay pendek

Dua fungsi utama:
  judge_answer()           - nilai jawaban siswa (score 0/1 + feedback)
  generate_relevance_judgments() - buat synthetic ground truth untuk evaluasi IR
"""

import os
import json
import re
from typing import Dict, List
from groq import Groq

# ── Client & model config ──────────────────────────────────────────────────────
# Lazy init — client dibuat saat pertama kali dipakai,
# sehingga import tidak gagal kalau GROQ_API_KEY belum di-set
_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY belum di-set.\n"
                "Set dulu: os.environ['GROQ_API_KEY'] = 'gsk_...'\n"
                "Daftar gratis di https://console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client

# Pilihan model Groq (semua gratis di free tier):
#   llama-3.3-70b-versatile  → kualitas terbaik, default
#   llama-3.1-8b-instant     → paling cepat, untuk relevance judging massal
#   mixtral-8x7b-32768       → context window besar (32k)
JUDGE_MODEL     = "llama-3.3-70b-versatile"
RELEVANCE_MODEL = "llama-3.1-8b-instant"   # lebih cepat untuk judging massal


# ── Answer Judge ──────────────────────────────────────────────────────────────

ANSWER_JUDGE_PROMPT = """\
Kamu adalah asisten evaluasi soal pelajaran {subject}.
Nilai apakah jawaban siswa BENAR atau SALAH berdasarkan konteks teori berikut.

<konteks>
{context}
</konteks>

Soal:
{question}

Jawaban siswa:
{answer}

Balas HANYA dengan JSON berikut, tanpa teks lain:
{{"score": 0 atau 1, "feedback": "1-2 kalimat penjelasan singkat"}}"""


def judge_answer(
    question: str,
    answer: str,
    context: str = "",
    subject: str = "IPS/Geografi SMA",
) -> Dict:
    """
    Nilai jawaban siswa.

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
        response = _get_client().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,   # rendah supaya konsisten
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        return {
            "score":    int(result.get("score", 0)),
            "feedback": str(result.get("feedback", "")),
        }

    except json.JSONDecodeError:
        # Fallback: coba parse manual jika JSON malformed
        score = 1 if re.search(r"\bbenar\b|\bcorrect\b|score.*1", raw.lower()) else 0
        return {"score": score, "feedback": raw[:200]}
    except Exception as e:
        return {"score": 0, "feedback": f"[Error: {e}]"}


# ── Relevance Judge (untuk evaluasi IR) ───────────────────────────────────────

RELEVANCE_JUDGE_PROMPT = """\
Kamu adalah evaluator information retrieval untuk soal IPS/Geografi SMA.
Nilai relevansi soal berikut terhadap query pengguna.

Query: "{query}"
Bloom level yang diinginkan: C{target_bloom}

Soal:
{question}
Topik soal: {topic} | Bloom soal: C{doc_bloom}

Skala:
0 = tidak relevan
1 = sedikit relevan (topik berkaitan jauh)
2 = cukup relevan (topik sesuai, bloom level jauh)
3 = sangat relevan (topik sesuai DAN bloom level sesuai)

Balas HANYA dengan angka 0, 1, 2, atau 3."""


def judge_relevance(
    query: str,
    document: Dict,
    target_bloom: int,
) -> int:
    """
    Nilai relevansi dokumen terhadap query. Skala 0–3.
    Dipakai untuk membuat synthetic ground truth evaluasi IR.
    """
    prompt = RELEVANCE_JUDGE_PROMPT.format(
        query=query,
        target_bloom=target_bloom,
        question=document.get("question", "")[:300],
        topic=document.get("topic", ""),
        doc_bloom=document.get("bloom_level", "?"),
    )

    try:
        response = _get_client().chat.completions.create(
            model=RELEVANCE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        grade = int(re.search(r"[0-3]", raw).group())
        return min(max(grade, 0), 3)
    except Exception:
        return 0


def generate_relevance_judgments(
    test_queries: List[Dict],
    candidates: List[Dict],
    output_path: str = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Generate synthetic relevance judgments untuk semua pasangan (query, dokumen).
    Simpan ke JSON — jalankan sekali, reuse untuk semua evaluasi.

    Args:
        test_queries : [{"query_id", "keyword", "target_bloom"}]
        candidates   : list dokumen yang akan di-judge
        output_path  : path untuk simpan hasil (opsional)

    Returns:
        [{"query_id", "doc_id", "grade"}]
    """
    judgments = []
    total = len(test_queries) * len(candidates)

    for q_idx, query in enumerate(test_queries):
        for d_idx, doc in enumerate(candidates):
            n = q_idx * len(candidates) + d_idx + 1
            if verbose:
                print(f"\r  Judging {n}/{total}...", end="", flush=True)

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
        print(f"\n✅ Generated {len(judgments)} judgments")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(judgments, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved → {output_path}")

    return judgments
