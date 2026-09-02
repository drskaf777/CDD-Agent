"""Risk Auditor - a separate persona from the Analyst.

Checkpoint 5.1 is explicit about why this is its own agent: the role gathering evidence
and the role auditing it for staleness or gaps need the same separation the Critic has
from the Generator, "otherwise the audit is worthless". So this agent has no outreach
tool and does not generate new evidence - it reads what the Analyst produced, screens it
against the standing taxonomy, and pushes back.

The failure mode it exists to catch is the "grounded-but-wrong" case from Checkpoint
3.1: a real, correctly cited, but superseded figure. That is more dangerous than a
hallucination because it arrives with a citation and therefore looks legitimate. The
Analyst-Auditor round trip is a deliberate latency-for-reliability trade.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional

from cdd_agent.agents.base import Agent, AgentContext
from cdd_agent.guardrails.authorization import AgentRole
from cdd_agent.knowledge.risk_taxonomy import TAXONOMY, applicable_categories, matches
from cdd_agent.schemas.common import ConfidenceTag
from cdd_agent.schemas.evidence import EvidenceMatrix
from cdd_agent.schemas.hypothesis import HypothesisTree
from cdd_agent.schemas.risk import (
    GapOwner,
    InformationGap,
    RiskCategory,
    RiskItem,
    RiskRegister,
)

# How stale a citation may be before it is questioned. Deliberately generous: the point
# is to flag "this is a year-old number presented as current", not to churn on freshness.
_STALENESS_DAYS = 400


@dataclass
class AuditReport:
    """What the Auditor found, and what it wants the Analyst to do about it."""

    conflicts: list[str] = field(default_factory=list)
    stale_citations: list[str] = field(default_factory=list)
    untriangulated: list[str] = field(default_factory=list)
    uncovered_categories: list[RiskCategory] = field(default_factory=list)
    risks_raised: list[RiskItem] = field(default_factory=list)
    gaps_raised: list[InformationGap] = field(default_factory=list)
    routed_back: list[str] = field(default_factory=list)

    @property
    def requires_analyst_rework(self) -> bool:
        """Loop 2: a flagged gap routes back to the Analyst before Synthesis starts."""
        return bool(self.routed_back)

    def summary(self) -> str:
        return (
            f"{len(self.risks_raised)} risk(s), {len(self.gaps_raised)} gap(s), "
            f"{len(self.conflicts)} source conflict(s), "
            f"{len(self.routed_back)} hypothesis(es) routed back to the Analyst"
        )


class RiskAuditor(Agent):
    role = AgentRole.RISK_AUDITOR

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._risk_counter = 0
        self._gap_counter = 1000  # keep ids distinct from the Analyst's

    def audit(
        self,
        tree: HypothesisTree,
        matrix: EvidenceMatrix,
        *,
        register: Optional[RiskRegister] = None,
        save: bool = True,
    ) -> tuple[RiskRegister, AuditReport]:
        register = register or self.context.memory.risk_register()
        report = AuditReport()
        strategic = self.context.deal_shape

        self._check_source_conflicts(matrix, register, report)
        self._check_staleness(matrix, report)
        self._screen_taxonomy(tree, matrix, register, report, strategic)
        self._check_triangulation(tree, matrix, register, report)
        self._check_coverage(register, report, strategic)

        if save:
            self.context.memory.save_risk_register(register, agent=self.name)
        return register, report

    # ------------------------------------------------------------ individual screens
    def _check_source_conflicts(
        self, matrix: EvidenceMatrix, register: RiskRegister, report: AuditReport
    ) -> None:
        """Multiple live versions of one source - escalation trigger 3.

        Retrieval already filters superseded versions by date. This catches what that
        filter cannot: two *undated* documents in the same version group, where nothing
        establishes which one is current.
        """
        groups: dict[str, set[str]] = {}
        undated: dict[str, set[str]] = {}
        for item in matrix.items:
            for c in item.citations:
                group = c.chunk_id.split("::")[0] if c.chunk_id else c.source_file
                groups.setdefault(group, set()).add(c.source_file)
                if c.document_date is None:
                    undated.setdefault(group, set()).add(c.source_file)

        for group, files in groups.items():
            if len(files) > 1 and len(undated.get(group, set())) > 1:
                conflict = (
                    f"{group}: {len(files)} versions cited with no date to order them "
                    f"({', '.join(sorted(files))})"
                )
                report.conflicts.append(conflict)
                if conflict not in register.source_conflicts:
                    register.source_conflicts.append(conflict)
                self._raise_risk(
                    register,
                    report,
                    RiskCategory.DATA_ROOM_INTEGRITY,
                    f"Conflicting document versions cited without dates: {group}",
                    severity=4,
                    likelihood=3,
                )

    def _check_staleness(self, matrix: EvidenceMatrix, report: AuditReport) -> None:
        cutoff = _dt.date.today() - _dt.timedelta(days=_STALENESS_DAYS)
        for item in matrix.items:
            for c in item.citations:
                if c.document_date and c.document_date < cutoff:
                    report.stale_citations.append(
                        f"{item.hypothesis_id}: {c.short()} predates the staleness window"
                    )

    def _screen_taxonomy(
        self,
        tree: HypothesisTree,
        matrix: EvidenceMatrix,
        register: RiskRegister,
        report: AuditReport,
        strategic: bool,
    ) -> None:
        """Scan the Evidence Matrix against the standing taxonomy."""
        for category in applicable_categories(strategic):
            for item in matrix.items:
                text = f"{item.claim} {' '.join(c.quoted_text for c in item.citations)}"
                hits = matches(text, category, strategic)
                if not hits:
                    continue
                severity = 4 if item.tag is ConfidenceTag.CONTRADICTED else 3
                self._raise_risk(
                    register,
                    report,
                    category,
                    hits[0].description,
                    severity=severity,
                    likelihood=3 if item.tag is ConfidenceTag.CONTRADICTED else 2,
                    hypothesis_ids=[item.hypothesis_id],
                    evidence_ids=[item.id],
                    management_data_only=item.source_kind.is_management_supplied,
                )

        # A contradicted Tier-1 hypothesis is a red flag in its own right.
        for h in tree.tier_1():
            if matrix.rating(h.id) is ConfidenceTag.CONTRADICTED:
                self._raise_risk(
                    register,
                    report,
                    _category_for(h.statement),
                    f"Tier-1 hypothesis contradicted by evidence: {h.statement}",
                    severity=5,
                    likelihood=4,
                    hypothesis_ids=[h.id],
                )

    def _check_triangulation(
        self,
        tree: HypothesisTree,
        matrix: EvidenceMatrix,
        register: RiskRegister,
        report: AuditReport,
    ) -> None:
        """Bias disclosure, design spec s VIII.

        A finding resting solely on management-provided data is flagged as such. Where
        the hypothesis is Tier-1 and independent triangulation was available but not
        obtained, it also routes back to the Analyst.
        """
        for h in tree.tier_1():
            items = matrix.for_hypothesis(h.id)
            if not items:
                continue
            if matrix.triangulated(h.id):
                continue
            report.untriangulated.append(h.id)
            self._raise_risk(
                register,
                report,
                _category_for(h.statement),
                f"{h.id} rests solely on management-supplied data with no independent "
                f"triangulation: {h.statement}",
                severity=3,
                likelihood=3,
                hypothesis_ids=[h.id],
                evidence_ids=[i.id for i in items],
                management_data_only=True,
            )
            if matrix.rating(h.id) is ConfidenceTag.CONFIRMED:
                # A Confirmed rating on management data alone is exactly the
                # grounded-but-wrong shape. Push it back.
                report.routed_back.append(h.id)

        for h in tree.tier_1():
            if matrix.rating(h.id) is ConfidenceTag.NO_DATA:
                dated = any(
                    g.hypothesis_id == h.id and g.target_close_date
                    for g in register.gaps
                )
                if not dated:
                    report.routed_back.append(h.id)
                    self._gap_counter += 1
                    gap = InformationGap(
                        id=f"GAP-{self._gap_counter:03d}",
                        engagement_id=self.context.engagement_id,
                        created_by=self.name,
                        hypothesis_id=h.id,
                        request=(
                            f"No evidence and no dated gap for Tier-1 hypothesis {h.id}: "
                            f"{h.statement}"
                        ),
                        owner=GapOwner.DEAL_TEAM,
                        target_close_date=_dt.date.today() + _dt.timedelta(days=7),
                        blocking=True,
                    )
                    register.gaps.append(gap)
                    report.gaps_raised.append(gap)

    def _check_coverage(
        self, register: RiskRegister, report: AuditReport, strategic: bool
    ) -> None:
        """Risk Register coverage: what share of the taxonomy was actually evaluated."""
        evaluated = register.categories_evaluated()
        report.uncovered_categories = [
            c for c in applicable_categories(strategic) if c not in evaluated
        ]

    # ------------------------------------------------------------------ helpers
    def _raise_risk(
        self,
        register: RiskRegister,
        report: AuditReport,
        category: RiskCategory,
        description: str,
        *,
        severity: int,
        likelihood: int,
        hypothesis_ids: Optional[list[str]] = None,
        evidence_ids: Optional[list[str]] = None,
        management_data_only: bool = False,
    ) -> None:
        # One screen firing on three pieces of evidence is one risk with three
        # supporting items, not three risks. A register that lists the same finding
        # repeatedly is worse than useless at the top of a severity ranking.
        existing = next(
            (r for r in register.risks
             if r.category is category and r.description == description),
            None,
        )
        if existing is not None:
            for hid in hypothesis_ids or []:
                if hid not in existing.hypothesis_ids:
                    existing.hypothesis_ids.append(hid)
            for eid in evidence_ids or []:
                if eid not in existing.evidence_ids:
                    existing.evidence_ids.append(eid)
            existing.severity = max(existing.severity, severity)
            existing.likelihood = max(existing.likelihood, likelihood)
            # The flag only survives if *every* supporting item is management-supplied.
            existing.management_data_only = (
                existing.management_data_only and management_data_only
            )
            return
        self._risk_counter += 1
        risk = RiskItem(
            id=f"RISK-{self._risk_counter:03d}",
            engagement_id=self.context.engagement_id,
            created_by=self.name,
            category=category,
            description=description,
            severity=severity,
            likelihood=likelihood,
            hypothesis_ids=hypothesis_ids or [],
            evidence_ids=evidence_ids or [],
            management_data_only=management_data_only,
        )
        register.risks.append(risk)
        report.risks_raised.append(risk)


def _category_for(statement: str) -> RiskCategory:
    """Route a hypothesis to its most likely taxonomy category."""
    lowered = statement.lower()
    for category, screens in TAXONOMY.items():
        if any(m in lowered for s in screens for m in s.markers):
            return category
    return RiskCategory.GROWTH_SUSTAINABILITY
