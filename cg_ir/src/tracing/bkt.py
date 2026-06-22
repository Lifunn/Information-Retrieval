"""
tracing/bkt.py
==============
Bayesian Knowledge Tracing (BKT) — Corbett & Anderson (1994).

Parameter:
  P(L0) : prior probability menguasai konsep
  P(T)  : probabilitas transisi tidak tahu → tahu
  P(S)  : probabilitas slip (tahu tapi salah)
  P(G)  : probabilitas guess (tidak tahu tapi benar)

Dua mode update tersedia:
  update()            : binary (correct=True/False) — sesuai paper
  update_continuous() : weighted (score=0.0–1.0) — ekstensi kontinu
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BKTParams:
    p_init:    float = 0.10
    p_transit: float = 0.08
    p_slip:    float = 0.15
    p_guess:   float = 0.45


@dataclass
class ObservationRecord:
    concept:        str
    correct:        bool
    mastery_before: float
    mastery_after:  float
    bloom_level:    int = 1
    question_id:    str = ""


class BayesianKnowledgeTracing:
    """
    Per-concept BKT tracker.

    Usage:
        bkt = BayesianKnowledgeTracing()

        # Binary update (sesuai paper)
        new_mastery = bkt.update("pancasila", correct=True)

        # Continuous update (ekstensi)
        new_mastery = bkt.update_continuous("pancasila", score=0.75)
    """

    def __init__(self, default_params: Optional[BKTParams] = None):
        self.default_params  = default_params or BKTParams()
        self.concept_params: Dict[str, BKTParams] = {}
        self._mastery: Dict[str, float]            = {}
        self.history: List[ObservationRecord]      = []

    # ── Mastery access ─────────────────────────────────────────────────────────

    def get_mastery(self, concept: str) -> float:
        return self._mastery.get(
            concept,
            self.concept_params.get(concept, self.default_params).p_init
        )

    def set_params(self, concept: str, params: BKTParams):
        self.concept_params[concept] = params

    # ── Binary update (sesuai paper Corbett & Anderson 1994) ──────────────────

    def update(
        self,
        concept: str,
        correct: bool,
        bloom_level: int = 1,
        question_id: str = "",
    ) -> float:
        """
        Update mastery P(Ln) setelah satu observasi biner.

        Returns:
            Updated mastery score [0, 1]
        """
        params = self.concept_params.get(concept, self.default_params)
        p_l    = self.get_mastery(concept)

        # Step 1: Bayesian posterior
        if correct:
            p_obs_l1 = 1.0 - params.p_slip
            p_obs_l0 = params.p_guess
        else:
            p_obs_l1 = params.p_slip
            p_obs_l0 = 1.0 - params.p_guess

        p_obs = p_obs_l1 * p_l + p_obs_l0 * (1.0 - p_l)
        if p_obs < 1e-12:
            p_obs = 1e-12

        p_l_given_obs = (p_obs_l1 * p_l) / p_obs

        # Step 2: Learning transition
        p_l_next = p_l_given_obs + (1.0 - p_l_given_obs) * params.p_transit
        p_l_next = min(max(p_l_next, 0.0), 1.0)

        self._mastery[concept] = p_l_next
        self.history.append(ObservationRecord(
            concept=concept,
            correct=correct,
            mastery_before=p_l,
            mastery_after=p_l_next,
            bloom_level=bloom_level,
            question_id=question_id,
        ))
        return p_l_next

    # ── Continuous update (ekstensi weighted BKT) ─────────────────────────────

    def update_continuous(
        self,
        concept: str,
        score: float,
        bloom_level: int = 1,
        question_id: str = "",
    ) -> float:
        """
        Update mastery dengan skor kontinu 0.0–1.0 dari LLM judge.

        Menghitung p_l_next untuk kasus benar dan salah secara terpisah,
        lalu menginterpolasi berdasarkan score:
            p_l_next = score × p_next_correct + (1-score) × p_next_wrong

        Returns:
            Updated mastery score [0, 1]
        """
        params = self.concept_params.get(concept, self.default_params)
        p_l    = self.get_mastery(concept)

        # p_l_next jika benar sempurna
        p_obs_c        = (1 - params.p_slip) * p_l + params.p_guess * (1 - p_l)
        p_l_if_correct = ((1 - params.p_slip) * p_l) / max(p_obs_c, 1e-12)
        p_next_correct = p_l_if_correct + (1 - p_l_if_correct) * params.p_transit

        # p_l_next jika salah total
        p_obs_w       = params.p_slip * p_l + (1 - params.p_guess) * (1 - p_l)
        p_l_if_wrong  = (params.p_slip * p_l) / max(p_obs_w, 1e-12)
        p_next_wrong  = p_l_if_wrong + (1 - p_l_if_wrong) * params.p_transit

        # Interpolasi dengan score LLM sebagai bobot
        p_l_next = score * p_next_correct + (1 - score) * p_next_wrong
        p_l_next = min(max(p_l_next, 0.0), 1.0)

        self._mastery[concept] = p_l_next
        self.history.append(ObservationRecord(
            concept=concept,
            correct=score >= 0.5,
            mastery_before=p_l,
            mastery_after=p_l_next,
            bloom_level=bloom_level,
            question_id=question_id,
        ))
        return p_l_next

    # ── Helpers ────────────────────────────────────────────────────────────────

    def get_topic_mastery(self, concepts: List[str]) -> float:
        if not concepts:
            return 0.0
        return sum(self.get_mastery(c) for c in concepts) / len(concepts)

    def all_masteries(self) -> Dict[str, float]:
        return dict(self._mastery)

    def concept_history(self, concept: str) -> List[ObservationRecord]:
        return [r for r in self.history if r.concept == concept]

    def summary(self) -> str:
        lines = ["BKT Mastery Summary:"]
        for concept, mastery in sorted(self._mastery.items(), key=lambda x: -x[1]):
            hist     = self.concept_history(concept)
            attempts = len(hist)
            correct  = sum(1 for r in hist if r.correct)
            lines.append(
                f"  {concept:<30} {mastery:.3f}  ({correct}/{attempts} correct)"
            )
        return "\n".join(lines)
