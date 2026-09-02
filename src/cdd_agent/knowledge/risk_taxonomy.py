"""The standing risk taxonomy - design specification s VII.A.

Each category carries the screens the Risk Auditor runs against the Evidence Matrix.
Coverage of this taxonomy per deal is itself an evaluation metric (Checkpoint 6.1),
which is why the screens are enumerated rather than left to whatever the model happens
to surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from cdd_agent.schemas.deal_profile import DealShape
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
    # --- Listed targets ------------------------------------------------------
    # A public company is already being valued continuously by people who read the
    # same filings. The commercial question is therefore not "is this a good
    # business" but "what do we believe that the price does not" - so the screens
    # test the differential, not the fundamentals a second time.
    RiskCategory.MARKET_EXPECTATIONS: (
        Screen("Base case does not clear the unaffected price plus the premium the "
               "board could recommend",
               ("premium", "unaffected", "share price", "offer price", "52-week",
                "market capitalisation", "market capitalization")),
        Screen("Thesis restates published consensus, so the buyer is paying for "
               "growth the market has already priced",
               ("consensus", "analyst", "sell-side", "sell side", "street",
                "guidance", "estimate")),
        Screen("Management guidance to the market diverges from the plan shown in "
               "the data room",
               ("guidance", "investor day", "earnings call", "outlook",
                "management plan", "budget")),
    ),
    RiskCategory.GOVERNANCE_CONTROL: (
        Screen("Value-creation plan requires decisions a minority holder cannot compel",
               ("minority", "board seat", "governance", "control", "veto",
                "standstill", "shareholder agreement", "consent")),
        Screen("Continuing minority shareholders constrain related-party actions and "
               "capital allocation",
               ("related party", "minority shareholder", "free float", "squeeze-out",
                "squeeze out", "tag-along", "drag-along", "fiduciary")),
        Screen("No ongoing information rights after close, so the plan cannot be "
               "monitored",
               ("information rights", "reporting", "disclosure", "observer",
                "quarterly reporting")),
        Screen("Founder, dual-class, or insider holdings decide the outcome regardless "
               "of the stake acquired",
               ("dual-class", "dual class", "founder", "insider", "voting rights",
                "super-voting")),
    ),
    RiskCategory.DEAL_COMPLETION: (
        Screen("Shareholder vote or acceptance threshold not secured",
               ("shareholder approval", "shareholder vote", "acceptance",
                "irrevocable", "proxy", "scheme of arrangement")),
        Screen("Regulatory or foreign-investment clearance conditions the timetable",
               ("antitrust", "hsr", "cfius", "merger control", "clearance",
                "regulatory approval", "national security")),
        Screen("Interloper, activist, or arbitrage pressure on price and timetable",
               ("activist", "competing bid", "interloper", "go-shop", "arbitrage",
                "fiduciary out", "topping bid")),
        Screen("Defences that make the approach unactionable without board support",
               ("poison pill", "rights plan", "staggered board", "classified board",
                "supermajority")),
    ),
}


def category_applies(category: RiskCategory, shape: DealShape) -> bool:
    """Whether this category is in scope for this deal at all.

    Coverage is a headline metric, so an out-of-scope category must be excluded
    rather than left permanently uncovered - otherwise every private deal reports a
    hole it could never fill, and the metric stops meaning anything.
    """
    if category is RiskCategory.INTEGRATION_SYNERGY:
        return shape.strategic_buyer
    if category is RiskCategory.MARKET_EXPECTATIONS:
        # A traded price to argue with is the whole point of the category.
        return shape.public_target
    if category is RiskCategory.GOVERNANCE_CONTROL:
        # Minority holders on one side of the table or the other: either the buyer
        # is one, or it must live with the ones who remain.
        return shape.public_target and (
            shape.retains_listing or not shape.confers_control
        )
    if category is RiskCategory.DEAL_COMPLETION:
        # Only a control transaction has a completion condition to fail.
        return shape.public_target and shape.confers_control
    return True


def screens_for(category: RiskCategory, deal: "DealShape | bool") -> tuple[Screen, ...]:
    strategic = DealShape.coerce(deal).strategic_buyer
    return tuple(
        s for s in TAXONOMY[category] if strategic or not s.strategic_only
    )


def applicable_categories(deal: "DealShape | bool") -> list[RiskCategory]:
    """The taxonomy actually in scope for this deal.

    Accepts the older `strategic_buyer` boolean so call sites without a profile in
    hand keep working; they simply see a private financial-sponsor deal.
    """
    shape = DealShape.coerce(deal)
    return [c for c in RiskCategory if category_applies(c, shape)]


def matches(text: str, category: RiskCategory,
            deal: "DealShape | bool") -> list[Screen]:
    lowered = text.lower()
    return [
        s for s in screens_for(category, deal)
        if any(m in lowered for m in s.markers)
    ]
