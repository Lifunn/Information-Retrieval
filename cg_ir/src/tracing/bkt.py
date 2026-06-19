"""
tracing/bkt.py
==============
Bayesian Knowledge Tracing (BKT) untuk melacak mastery learner per konsep.

Model BKT klasik (Corbett & Anderson, 1994):
  - P(L₀) : prior probability menguasai konsep
  - P(T)  : probability transisi dari tidak tahu → tahu setelah satu kesempatan
  - P(S)  : probability "slip" (tahu tapi salah)
  - P(G)  : probability "guess" (tidak tahu tapi benar)

Update setelah observasi:
  1. P(Lₙ | obs) ∝ P(obs | Lₙ) · P(Lₙ₋₁)
  2. P(Lₙ₊₁) = P(Lₙ | obs) + (1 − P(Lₙ | obs)) · P(T)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BKTParams:
    """BKT parameter set untuk satu concept/skill."""
    p_init:    float = 0.10   # P(L₀): prior mastery
    p_transit: float = 0.05   # P(T): learning rate
    p_slip:    float = 0.20   # P(S): slip probability
    p_guess:   float = 0.10   # P(G): guess probability


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
        new_mastery = bkt.update("pancasila", correct=True)
        print(bkt.get_mastery("pancasila"))
    """

    def __init__(self, default_params: Optional[BKTParams] = None):
        self.default_params = default_params or BKTParams()
        self.concept_params: Dict[str, BKTParams] = {}
        self._mastery: Dict[str, float]             = {}
        self.history: List[ObservationRecord]        = []

    # ── Mastery access ─────────────────────────────────────────────────────

    def get_mastery(self, concept: str) -> float:
        """Return P(Lₙ) untuk concept (prior P(L₀) jika belum ada record)."""
        return self._mastery.get(
            concept,
            self.concept_params.get(concept, self.default_params).p_init
        )

    def set_params(self, concept: str, params: BKTParams):
        """Set custom BKT params untuk konsep tertentu."""
        self.concept_params[concept] = params

    # ── Update ─────────────────────────────────────────────────────────────

    def update(
        self,
        concept: str,
        correct: bool,
        bloom_level: int = 1,
        question_id: str = "",
    ) -> float:
        """
        Update mastery P(Lₙ) setelah satu observasi jawaban.

        Returns:
            Updated mastery score [0, 1]
        """
        params = self.concept_params.get(concept, self.default_params)
        p_l = self.get_mastery(concept)

        # ── Step 1: Bayesian update P(L | obs) ────────────────────────────
        if correct:
            p_obs_l1 = 1.0 - params.p_slip   # P(correct | know)
            p_obs_l0 = params.p_guess         # P(correct | not know)
        else:
            p_obs_l1 = params.p_slip          # P(wrong | know)
            p_obs_l0 = 1.0 - params.p_guess   # P(wrong | not know)

        p_obs = p_obs_l1 * p_l + p_obs_l0 * (1.0 - p_l)
        if p_obs < 1e-12:
            p_obs = 1e-12  # avoid division by zero

        p_l_given_obs = (p_obs_l1 * p_l) / p_obs

        # ── Step 2: Learning transition ────────────────────────────────────
        p_l_next = p_l_given_obs + (1.0 - p_l_given_obs) * params.p_transit
        p_l_next = min(max(p_l_next, 0.0), 1.0)   # clamp to [0,1]

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

    # ── Aggregate helpers ──────────────────────────────────────────────────

    def get_topic_mastery(self, concepts: List[str]) -> float:
        """Average mastery across a list of concepts (for a topic)."""
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
            attempts = len(self.concept_history(concept))
            correct  = sum(1 for r in self.concept_history(concept) if r.correct)
            lines.append(f"  {concept:<30} {mastery:.3f}  ({correct}/{attempts} correct)")
        return "\n".join(lines)
