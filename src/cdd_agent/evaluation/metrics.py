"""Detective layer: the six evaluation metrics from Checkpoint 6.1.

Groundedness, calibration, escalation/fallback rate, numeric correctness, latency
(particularly across the Analyst-Risk Auditor loop), and Risk Register coverage.

Two of these cannot be computed from a run alone and are honest about it:

* **Calibration** asks whether sampled "Confirmed" tags hold up under human review.
  Only a human can answer that, so this module produces the sample and stores the
  verdicts - it never scores itself.
* **Numeric correctness** spot-checks computation output against source documents; the
  check is generated here, the comparison is recorded from review.

Metrics are meant to feed back into tightening guardrails, not just get reported, so
each result names the guardrail it bears on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

from cdd_agent.knowledge.risk_taxonomy import applicable_categories
from cdd_agent.schemas.common import ConfidenceTag
from cdd_agent.schemas.deck import Deck
from cdd_agent.schemas.evidence import EvidenceMatrix
from cdd_agent.schemas.hypothesis import HypothesisTree
from cdd_agent.schemas.risk import RiskRegister
from cdd_agent.state.store import Collection, StateStore


@dataclass
class Metric:
    name: str
    value: Optional[float]
    detail: str = ""
    guardrail: str = ""
    needs_human: bool = False

    def render(self) -> str:
        if self.value is None:
            shown = "pending human review" if self.needs_human else "n/a"
        elif 0.0 <= self.value <= 1.0 and self.name != "latency_seconds":
            shown = f"{self.value:.1%}"
        else:
            shown = f"{self.value:,.2f}"
        return f"{self.name:<28} {shown:>22}  {self.detail}"


@dataclass
class CalibrationSample:
    """A Confirmed claim pulled for human review."""

    hypothesis_id: str
    claim: str
    citations: list[str]
    independently_triangulated: bool
    human_verdict: Optional[str] = None  # "holds" | "does not hold"


@dataclass
class MetricSet:
    metrics: list[Metric] = field(default_factory=list)
    calibration_sample: list[CalibrationSample] = field(default_factory=list)
    numeric_checks: list[dict[str, Any]] = field(default_factory=list)

    def get(self, name: str) -> Optional[Metric]:
        return next((m for m in self.metrics if m.name == name), None)

    def render(self) -> str:
        return "\n".join(m.render() for m in self.metrics)


def groundedness(deck: Deck) -> Metric:
    """Share of output claims carrying a valid citation. Target: near 100%.

    Claims tagged No Data are excluded from the denominator - an explicit gap is a
    correct output, not an ungrounded one, and counting it as a miss would create
    pressure to assert rather than to log.
    """
    claims = [c for c in deck.all_claims() if c.tag is not ConfidenceTag.NO_DATA]
    if not claims:
        return Metric(
            "groundedness", None, "no assertive claims in the deck",
            guardrail="output constraints",
        )
    cited = sum(1 for c in claims if c.citations)
    gaps = len(deck.all_claims()) - len(claims)
    return Metric(
        "groundedness",
        cited / len(claims),
        f"{cited}/{len(claims)} assertive claims cited ({gaps} logged as No Data)",
        guardrail="output constraints",
    )


def current_version_citations(deck: Deck, matrix: EvidenceMatrix) -> Metric:
    """Share of citations that point at a chunk still present in the evidence matrix.

    The stricter half of groundedness: Checkpoint 6.1 asks for a *current-version*
    citation, which is what the grounded-but-wrong failure mode defeats.
    """
    live = {c.chunk_id for i in matrix.items for c in i.citations if c.chunk_id}
    cites = [
        citation
        for claim in deck.all_claims()
        for citation in claim.citations
        if citation.chunk_id
    ]
    if not cites:
        return Metric("current_version_citations", None, "no chunk-level citations",
                      guardrail="source verification")
    ok = sum(1 for c in cites if c.chunk_id in live)
    return Metric(
        "current_version_citations",
        ok / len(cites),
        f"{ok}/{len(cites)} citations trace to a live matrix chunk",
        guardrail="source verification",
    )


def risk_register_coverage(register: RiskRegister, strategic_buyer: bool) -> Metric:
    """Share of the standing taxonomy actually evaluated - not just what surfaced."""
    applicable = applicable_categories(strategic_buyer)
    evaluated = register.categories_evaluated()
    covered = [c for c in applicable if c in evaluated]
    missing = [c.value for c in applicable if c not in evaluated]
    return Metric(
        "risk_register_coverage",
        len(covered) / len(applicable) if applicable else None,
        f"uncovered: {', '.join(missing) if missing else 'none'}",
        guardrail="risk taxonomy screening",
    )


def escalation_rate(store: StateStore, engagement_id: str) -> Metric:
    """How often the system correctly declined to auto-resolve.

    Reported as a count, not a ratio against a target: over-triggering is as much a
    failure as under-triggering, so this is a number for a human to read in context.
    """
    escalations = store.list(engagement_id, Collection.ESCALATION)
    by_trigger: dict[str, int] = {}
    for _, doc in escalations:
        by_trigger[doc.get("trigger", "?")] = by_trigger.get(doc.get("trigger", "?"), 0) + 1
    detail = ", ".join(f"{k}={v}" for k, v in sorted(by_trigger.items())) or "none raised"
    return Metric(
        "escalations_raised", float(len(escalations)), detail,
        guardrail="escalation rules",
    )


def latency(timings: list[tuple[str, float]]) -> list[Metric]:
    """Wall-clock per phase, with the Analyst-Auditor loop called out."""
    out = [
        Metric("latency_seconds", round(sum(t for _, t in timings), 2),
               "total pipeline", guardrail="runtime monitoring")
    ]
    for phase, seconds in timings:
        if "evidence" in phase.lower() or "audit" in phase.lower():
            out.append(
                Metric(
                    "latency_analyst_auditor",
                    round(seconds, 2),
                    "deliberate latency-for-reliability trade (Checkpoint 5.1)",
                    guardrail="runtime monitoring",
                )
            )
    return out


def build_calibration_sample(
    deck: Deck, matrix: EvidenceMatrix, *, n: int = 5, seed: int = 0
) -> list[CalibrationSample]:
    """Draw Confirmed claims for human review. The system does not grade itself."""
    confirmed = [c for c in deck.all_claims() if c.tag is ConfidenceTag.CONFIRMED]
    rng = random.Random(seed)
    picked = confirmed if len(confirmed) <= n else rng.sample(confirmed, n)
    return [
        CalibrationSample(
            hypothesis_id=c.hypothesis_id or "-",
            claim=c.text,
            citations=[cit.short() for cit in c.citations],
            independently_triangulated=bool(
                c.hypothesis_id and matrix.triangulated(c.hypothesis_id)
            ),
        )
        for c in picked
    ]


def build_numeric_checks(matrix: EvidenceMatrix, *, n: int = 5) -> list[dict[str, Any]]:
    """Spot-checks of computation-tool output against the source document."""
    from cdd_agent.schemas.common import SourceKind

    computed = [
        i for i in matrix.items
        if any(c.source_kind is SourceKind.COMPUTATION for c in i.citations)
    ][:n]
    return [
        {
            "evidence_id": i.id,
            "hypothesis_id": i.hypothesis_id,
            "claim": i.claim,
            "computed_from": [c.short() for c in i.citations],
            "human_verified": None,
        }
        for i in computed
    ]


def evaluate(
    *,
    deck: Optional[Deck],
    matrix: EvidenceMatrix,
    register: RiskRegister,
    tree: Optional[HypothesisTree],
    store: StateStore,
    engagement_id: str,
    strategic_buyer: bool,
    timings: Optional[list[tuple[str, float]]] = None,
) -> MetricSet:
    """Compute the full metric set for one run and persist it."""
    result = MetricSet()

    if deck is not None:
        result.metrics.append(groundedness(deck))
        result.metrics.append(current_version_citations(deck, matrix))
        result.calibration_sample = build_calibration_sample(deck, matrix)
        result.metrics.append(
            Metric(
                "calibration",
                None,
                f"{len(result.calibration_sample)} Confirmed claim(s) sampled for review",
                guardrail="output constraints",
                needs_human=True,
            )
        )
    result.metrics.append(risk_register_coverage(register, strategic_buyer))
    result.metrics.append(escalation_rate(store, engagement_id))
    result.numeric_checks = build_numeric_checks(matrix)
    result.metrics.append(
        Metric(
            "numeric_correctness",
            None,
            f"{len(result.numeric_checks)} computed figure(s) queued for spot-check",
            guardrail="structured computation",
            needs_human=True,
        )
    )
    if tree is not None:
        tier1 = tree.tier_1()
        resolved = sum(
            1 for h in tier1 if matrix.rating(h.id) is not ConfidenceTag.NO_DATA
        )
        result.metrics.append(
            Metric(
                "tier1_resolution",
                resolved / len(tier1) if tier1 else None,
                f"{resolved}/{len(tier1)} Tier-1 hypotheses carry evidence",
                guardrail="synthesis gate",
            )
        )
    if timings:
        result.metrics.extend(latency(timings))

    store.put(
        engagement_id,
        Collection.METRICS,
        "evaluation",
        {
            "metrics": [
                {"name": m.name, "value": m.value, "detail": m.detail,
                 "guardrail": m.guardrail, "needs_human": m.needs_human}
                for m in result.metrics
            ],
            "calibration_sample": [s.__dict__ for s in result.calibration_sample],
            "numeric_checks": result.numeric_checks,
        },
        agent="Evaluation",
    )
    return result
