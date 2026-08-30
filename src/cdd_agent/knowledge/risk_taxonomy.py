"""The standing risk taxonomy - design specification s VII.A.

Each category carries the screens the Risk Auditor runs against the Evidence Matrix.
Coverage of this taxonomy per deal is itself an evaluation metric (Checkpoint 6.1),
which is why the screens are enumerated rather than left to whatever the model happens
to surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from cdd_agent.schemas.risk import RiskCategory


@dataclass(frozen=True)
class Screen:
    description: str
    # Terms that make a piece of evidence relevant to this screen. Used to decide
    # whether a category was actually *evaluated*, separately from whether a risk fired.
    markers: tuple[str, ...]
    # True where the screen applies only to strategic buyers.
    strategic_only: bool = False


TAXONOMY: dict[RiskCategory, tuple[Screen, ...]] = {
    RiskCategory.REVENUE_QUALITY: (
        Screen("Top-5/10/20 customer dependency",
               ("concentration", "top-5", "top 5", "top-10", "top 10", "top-20",
                "top 20", "largest customer")),
        Screen("One-time or non-recurring items inflating run-rate revenue",
               ("one-time", "non-recurring", "run-rate", "run rate", "perpetual "
                "license", "professional services revenue")),
        Screen("Channel-stuffing or booking-vs-billing gaps",
               ("bookings", "billings", "channel", "deferred revenue", "rev rec",
                "revenue recognition")),
    ),
    RiskCategory.GROWTH_SUSTAINABILITY: (
        Screen("TAM/SAM that does not reconcile to bottom-up customer-acquisition math",
               ("tam", "sam", "bottom-up", "addressable", "penetration")),
        Screen("CAGR assumptions exceeding independently sourced market-growth data",
               ("cagr", "growth rate", "market growth", "forecast", "projection")),
    ),
    RiskCategory.COMPETITIVE_DISRUPTION: (
        Screen("New entrants, substitutes, or business-model shifts absent from the base case",
               ("entrant", "substitute", "disruption", "displacement", "open-source",
                "open source", "ai ", "commoditiz", "low-cost")),
    ),
    RiskCategory.UNIT_ECONOMICS: (
        Screen("CAC payback lengthening",
               ("cac", "payback", "acquisition cost", "magic number")),
        Screen("LTV assumptions unsupported by cohort data",
               ("ltv", "lifetime value", "cohort")),
        Screen("Gross-margin trends diverging from segment mix",
               ("gross margin", "margin", "mix", "contribution")),
    ),
    RiskCategory.KEY_PERSON: (
        Screen("Revenue or relationships concentrated with individuals lacking retention "
               "agreements",
               ("key-person", "key person", "founder", "retention agreement",
                "non-compete", "rainmaker")),
        Screen("Thin bench beneath the CEO/founder",
               ("succession", "bench", "leadership depth", "second line")),
    ),
    RiskCategory.RETENTION: (
        Screen("Logo churn trending against reported NRR",
               ("churn", "logo", "nrr", "grr", "retention", "renewal")),
        Screen("Attrition spikes in revenue-generating roles",
               ("attrition", "turnover", "quota", "sales headcount", "clinician")),
    ),
    RiskCategory.REGULATORY: (
        Screen("Pending litigation, license renewals, or sector rule changes not priced in",
               ("litigation", "license", "permit", "accreditation", "regulat",
                "reimbursement", "privacy", "gdpr", "hipaa", "compliance")),
    ),
    RiskCategory.DATA_ROOM_INTEGRITY: (
        Screen("Inconsistent figures across documents",
               ("inconsisten", "conflict", "discrepan", "does not reconcile",
                "differs from")),
        Screen("Unaudited numbers presented as final",
               ("unaudited", "draft", "management estimate", "preliminary")),
        Screen("Cohorts or comparison periods selected to flatter the trend",
               ("cherry", "selected period", "restated", "rebased", "excludes")),
    ),
    RiskCategory.INTEGRATION_SYNERGY: (
        Screen("Synergy assumptions not independently validated",
               ("synergy", "integration", "cost take-out", "cross-sell synergy"),
               strategic_only=True),
        Screen("Cultural or systems-integration risk absent from the plan",
               ("culture", "systems integration", "erp", "migration plan"),
               strategic_only=True),
    ),
}


def screens_for(category: RiskCategory, strategic_buyer: bool) -> tuple[Screen, ...]:
    return tuple(
        s for s in TAXONOMY[category] if strategic_buyer or not s.strategic_only
    )


def applicable_categories(strategic_buyer: bool) -> list[RiskCategory]:
    """Integration/synergy is scoped to strategic buyers, so sponsor deals are not
    penalised on coverage for a category that does not apply to them."""
    return [
        c for c in RiskCategory
        if screens_for(c, strategic_buyer)
    ]


def matches(text: str, category: RiskCategory, strategic_buyer: bool) -> list[Screen]:
    lowered = text.lower()
    return [
        s for s in screens_for(category, strategic_buyer)
        if any(m in lowered for m in s.markers)
    ]
