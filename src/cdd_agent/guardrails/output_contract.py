"""Preventive guardrail: output constraints (Checkpoint 6.1).

"Every claim in the Synthesizer's output must carry a citation and a
Confirmed/Partial/Contradicted/No Data tag; unqualified assertions are a schema
violation, not a style choice."

The Pydantic models already refuse to construct an ungrounded claim. This module is
the second half: a whole-deck check that also catches the failures a per-claim
validator cannot see - a superseded citation, a Confirmed tag resting only on
management-supplied data, a Tier-1 hypothesis rendered without either a rating or a
dated gap.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional

from cdd_agent.schemas.common import ConfidenceTag, SourceKind
from cdd_agent.schemas.deck import Claim, Deck
from cdd_agent.schemas.evidence import EvidenceMatrix
from cdd_agent.schemas.hypothesis import HypothesisTree
from cdd_agent.schemas.risk import RiskRegister


class SchemaViolation(ValueError):
    """An output that breaks the citation/confidence contract."""


@dataclass
class ContractReport:
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    claims_checked: int = 0
    claims_cited: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def groundedness(self) -> float:
        return self.claims_cited / self.claims_checked if self.claims_checked else 1.0

    def raise_if_violated(self) -> None:
        if self.violations:
            raise SchemaViolation(
                "output contract violated:\n- " + "\n- ".join(self.violations)
            )


def check_claim(claim: Claim, *, current_sources: Optional[set[str]] = None) -> list[str]:
    """Per-claim checks. Returns violation strings (empty when clean)."""
    problems: list[str] = []
    text = claim.text[:70]

    if claim.tag is not ConfidenceTag.NO_DATA and not claim.citations:
        problems.append(f"uncited claim: {text!r}")

    if claim.tag is ConfidenceTag.NO_DATA and claim.citations:
        problems.append(
            f"claim tagged No Data but carries citations - re-tag or drop: {text!r}"
        )

    if claim.tag is ConfidenceTag.CONFIRMED:
        kinds = {c.source_kind for c in claim.citations}
        if kinds and kinds.issubset(
            {SourceKind.DATA_ROOM, SourceKind.STRUCTURED_DATA}
        ) and not claim.management_data_only:
            problems.append(
                f"claim tagged Confirmed on management-supplied data alone without the "
                f"bias flag set: {text!r}"
            )

    if current_sources is not None:
        for c in claim.citations:
            if c.chunk_id and c.chunk_id not in current_sources:
                problems.append(
                    f"citation {c.short()} is not the current version of that source: {text!r}"
                )
    return problems


def check_deck(
    deck: Deck,
    *,
    tree: Optional[HypothesisTree] = None,
    matrix: Optional[EvidenceMatrix] = None,
    register: Optional[RiskRegister] = None,
    current_sources: Optional[set[str]] = None,
) -> ContractReport:
    """Full pre-delivery check. The Synthesizer runs this before the deck is saved."""
    report = ContractReport()

    for slide in deck.slides:
        for claim in slide.claims:
            report.claims_checked += 1
            if claim.citations:
                report.claims_cited += 1
            for problem in check_claim(claim, current_sources=current_sources):
                report.violations.append(f"s{slide.section_number}: {problem}")
        if not slide.so_what_headline.strip():
            report.violations.append(
                f"s{slide.section_number}: missing so-what headline (a topic label is "
                "not a headline)"
            )

    if not deck.draft_notice.strip():
        report.violations.append(
            "draft-status notice removed - the deck must not present as a final "
            "IC recommendation (design spec s VIII)"
        )

    # The synthesis gate, restated at the output boundary as a defence in depth.
    if tree and matrix:
        dated_gaps = _dated_gap_hypotheses(register)
        for h in tree.tier_1():
            rating = matrix.rating(h.id)
            if rating is ConfidenceTag.NO_DATA and h.id not in dated_gaps:
                report.violations.append(
                    f"Tier-1 hypothesis {h.id} is below Partially Confirmed and has no "
                    f"dated information gap: {h.statement[:70]!r}"
                )
            elif rating is not ConfidenceTag.NO_DATA and not matrix.triangulated(h.id):
                report.warnings.append(
                    f"Tier-1 hypothesis {h.id} rests only on management-supplied data - "
                    "flag in the Risk Register per design spec s VIII (bias disclosure)"
                )
    return report


def _dated_gap_hypotheses(register: Optional[RiskRegister]) -> set[str]:
    if not register:
        return set()
    return {
        g.hypothesis_id
        for g in register.gaps
        if g.hypothesis_id and g.target_close_date is not None
    }


def format_report(report: ContractReport) -> str:
    lines = [
        f"claims checked: {report.claims_checked}",
        f"groundedness:   {report.groundedness:.1%}",
    ]
    if report.violations:
        lines.append("violations:")
        lines += [f"  - {v}" for v in report.violations]
    if report.warnings:
        lines.append("warnings:")
        lines += [f"  - {w}" for w in report.warnings]
    if report.ok and not report.warnings:
        lines.append("output contract: clean")
    return "\n".join(lines)


def today() -> _dt.date:
    return _dt.date.today()
