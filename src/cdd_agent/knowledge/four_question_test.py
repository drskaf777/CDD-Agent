"""Bain's four-question test - the hard constraint in the ToT Critic.

Design specification s I: this reframes the exercise from "is this a good company"
to "will the plan in the model come true". Checkpoint 4.1 s 2.3 makes it a rule-based
pass/fail check rather than a scored criterion, which is why it lives here as data the
Critic can evaluate deterministically instead of prose in a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    # Markers used by the deterministic half of the Critic's check. Deliberately broad:
    # this screens for whether a framing *addresses* the question, not whether it
    # answers it well - the latter is the scored `four_question_alignment` criterion.
    markers: tuple[str, ...]


FOUR_QUESTIONS: tuple[Question, ...] = (
    Question(
        key="market_growing",
        text="Is the market genuinely growing?",
        markers=(
            "market", "tam", "sam", "demand", "growth rate", "cagr", "segment",
            "industry", "adoption", "penetration", "catchment", "volume",
        ),
    ),
    Question(
        key="target_keeps_winning",
        text="Can the target keep winning share within it?",
        markers=(
            "share", "win", "competitive", "competitor", "differentiat", "moat",
            "positioning", "churn", "retention", "nrr", "grr", "switching",
            "displacement", "referral", "cross-sell", "upsell", "pipeline",
        ),
    ),
    Question(
        key="unit_economics_hold",
        text="Do the underlying unit economics hold up?",
        markers=(
            "unit economic", "margin", "cac", "ltv", "payback", "gross margin",
            "pricing", "cost", "ebitda", "contribution", "rule of 40", "utilization",
            "revenue per", "reimbursement rate",
        ),
    ),
    Question(
        key="what_breaks_the_deal",
        text="What specific findings would break the deal?",
        markers=(
            "risk", "concentration", "break", "downside", "dependency", "key-person",
            "key person", "regulat", "compliance", "litigation", "covenant",
            "attrition", "step-down", "cliff", "expiry", "renegotiat",
        ),
    ),
)

QUESTION_KEYS: tuple[str, ...] = tuple(q.key for q in FOUR_QUESTIONS)


def classify(text: str) -> set[str]:
    """Return which of the four questions a hypothesis statement speaks to.

    A hypothesis may map to more than one - that is expected and not penalised. What
    is penalised (with a hard prune) is a *framing* that leaves a question unmapped
    by every one of its hypotheses.
    """
    lowered = text.lower()
    return {q.key for q in FOUR_QUESTIONS if any(m in lowered for m in q.markers)}


def question(key: str) -> Question:
    return next(q for q in FOUR_QUESTIONS if q.key == key)
