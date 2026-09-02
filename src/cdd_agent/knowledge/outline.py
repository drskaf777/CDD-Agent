"""The enhanced master outline - design specification s IV.

Section 0 (Deal Thesis & Hypothesis Tree), Section 5 (Management, Organization &
Execution Capability) and Section 8 (Risk Register & Outstanding Information Gaps)
are the three additions; Sections 1 and 4 are the two re-framings. Those flags are
carried in the data because the Synthesizer treats the new sections differently:
Section 0 renders the evidence-status dashboard rather than prose, and Section 8 is
generated from the Risk Register rather than written.
"""

from __future__ import annotations

from cdd_agent.schemas.common import OutlineSection
from cdd_agent.schemas.deal_profile import DealShape, TransactionStructure

UNIVERSAL_OUTLINE: tuple[OutlineSection, ...] = (
    OutlineSection(
        number=0,
        title="Deal Thesis & Hypothesis Tree",
        key_elements=(
            "Lead hypothesis and supporting assumptions",
            "Evidence-status dashboard (Confirmed / Partially Confirmed / "
            "Contradicted / No Data)",
            "Four-question screening summary",
        ),
        is_new=True,
    ),
    OutlineSection(
        number=1,
        title="Executive Summary & Investment Thesis",
        key_elements=(
            "So-what headline",
            "Core thesis pillars",
            "Four-question verdict",
            "Key red flags",
            "Value-creation upside",
        ),
        is_enhanced=True,
    ),
    OutlineSection(
        number=2,
        title="Market Dynamics & Attractiveness",
        key_elements=(
            "TAM/SAM breakdown",
            "Historical and projected CAGR",
            "Macro driver heatmap",
            "Cyclicality assessment",
        ),
    ),
    OutlineSection(
        number=3,
        title="Competitive Landscape & Target Positioning",
        key_elements=(
            "Market-share matrix",
            "Feature/capability scorecard",
            "Competitor moat evaluation",
            "Win/loss analysis",
        ),
    ),
    OutlineSection(
        number=4,
        title="Customer & Voice-of-Market Analysis",
        key_elements=(
            "NPS/CSAT by cohort",
            "Customer concentration risk",
            "Churn and retention cohorts",
            "Purchasing-criteria rankings",
            "Primary-research methodology disclosure (sample size, expert/customer mix, "
            "independence from management-supplied lists)",
        ),
        is_enhanced=True,
    ),
    OutlineSection(
        number=5,
        title="Management, Organization & Execution Capability",
        key_elements=(
            "Leadership-team track record and depth",
            "Key-person / succession risk",
            "Incentive and retention structure",
            "Organizational readiness for the value-creation plan",
        ),
        is_new=True,
    ),
    OutlineSection(
        number=6,
        title="Financial & Operational Assessment",
        key_elements=(
            "Unit economics",
            "Pricing power",
            "Sales-pipeline health",
            "Historical bridge analysis",
            "Margin/cost-structure levers, read jointly with the commercial thesis",
        ),
    ),
    OutlineSection(
        number=7,
        title="Valuation, Sensitivities & Growth Levers",
        key_elements=(
            "Base/upside/downside scenarios",
            "Sensitivity matrix",
            "100-day value-creation playbook",
            "Precedent-transaction and sponsor-to-sponsor multiple benchmarking",
            "Exit hypotheses",
        ),
    ),
    OutlineSection(
        number=8,
        title="Risk Register & Outstanding Information Gaps",
        key_elements=(
            "Consolidated red-flag list ranked by severity x likelihood",
            "Open information gaps with owner and target-close date",
            "Items requiring confirmatory (post-LOI) diligence",
        ),
        is_new=True,
    ),
)


# --- Tailored modules: plug into the universal spine at Sections 2-7 ---------

SAAS_MODULE: dict[int, tuple[str, ...]] = {
    1: (
        "Product-market fit, platform scalability, technology modernness",
        "Rule of 40 and LTV/CAC at a glance",
    ),
    2: (
        "Segmented TAM by company size",
        "Generative-AI / open-source displacement risk",
        "Remaining on-premise-to-cloud migration pool",
    ),
    3: (
        "Capability matrix vs. legacy vendors and point solutions",
        "Switching-cost / data lock-in index",
        "R&D efficiency benchmarking",
    ),
    4: (
        "NRR and GRR by cohort",
        "Magic number and CAC payback",
    ),
    6: (
        "Professional-services revenue drag",
        "Seat- vs. consumption-based pricing risk",
        "White-space cross-sell/upsell quantification",
        # The spec's SaaS enhancement: the single most-scrutinized exhibit in SaaS CDD.
        "Contract-level ARR waterfall reconciling gross new, expansion, contraction "
        "and churn",
    ),
}

HEALTHCARE_MODULE: dict[int, tuple[str, ...]] = {
    1: (
        "Clinical reputation, provider capacity, regulatory/compliance safety",
        "Geographic-moat overview",
    ),
    2: (
        "Catchment-area demographics",
        "Inpatient-to-outpatient shift",
        "Wait-time and capacity bottlenecks",
    ),
    3: (
        "Physician referral concentration and leakage",
        "Medicare/Medicaid/commercial payer split",
        "Reimbursement-rate trend risk",
    ),
    5: (
        "Physician/nurse turnover and wage inflation",
        "Revenue-per-FTE productivity",
        "Clinical non-compete risk",
    ),
    6: (
        "Regulatory-funding shock sensitivity",
        "De Novo unit economics",
        "Billing/coding compliance audit",
        # The spec's healthcare enhancement.
        "Payer-contract renegotiation calendar and rate-lock expiry schedule",
    ),
}

# --- Listed targets: merged on top of the sub-sector module --------------------
# A public company has already been analysed by everyone who reads its filings and
# is repriced daily on the result. Restating that analysis is not diligence, so the
# public module points every section at the differential: what the buyer believes
# that the price does not already reflect.

PUBLIC_COMPANY_MODULE: dict[int, tuple[str, ...]] = {
    1: (
        "The differentiated view: what this work concludes that published consensus "
        "does not, stated explicitly",
    ),
    2: (
        "Market sizing reconciled to the reported segment disclosure, not built beside it",
    ),
    4: (
        "Customer and channel contact constrained pre-announcement: what could be "
        "asked, and what was deferred to confirmatory diligence",
    ),
    6: (
        "Reported vs. adjusted earnings bridge, with each adjustment named and sourced "
        "to the filing",
        "Guidance history against delivery: the plan's credibility measured on the "
        "company's own public record",
    ),
    7: (
        "Unaffected share price, reference date, and the 52-week range",
        "Consensus estimates vs. management plan vs. this base case, on one axis",
    ),
    8: (
        "MNPI and wall-crossing status; what the restriction prevents until announcement",
    ),
}

MINORITY_STAKE_MODULE: dict[int, tuple[str, ...]] = {
    5: (
        "Incumbent management is the execution vehicle: the plan being underwritten is "
        "theirs to run, and cannot be replaced",
    ),
    7: (
        "Value if the plan is never adopted: what the stake is worth on the "
        "status-quo trajectory",
        "Exit path for a block this size measured against average daily volume",
    ),
    8: (
        "Influence rights actually secured: board seat, observer, consent rights, or none",
        "Ongoing information rights after close - without them the plan cannot be "
        "monitored, and any that are granted re-restrict trading",
        "Disclosure threshold and standstill: the stake at which the holding becomes "
        "public and the build-up must stop",
    ),
}

CONTROL_STAKE_MODULE: dict[int, tuple[str, ...]] = {
    5: (
        "Board composition after close and the controlled-company governance regime",
    ),
    6: (
        "Cost of remaining listed, retained in the base case rather than assumed away",
    ),
    7: (
        "Value capture available to a controlling holder without prejudicing minority "
        "shareholders: related-party limits on the synergy case",
    ),
    8: (
        "Free-float and index-eligibility consequences of the stake acquired",
        "Continuing minority shareholders as a constraint on capital allocation",
    ),
}

TAKE_PRIVATE_MODULE: dict[int, tuple[str, ...]] = {
    6: (
        "Public-company cost base removed on delisting, sized and sourced rather than "
        "asserted as a round number",
    ),
    7: (
        "Premium to the unaffected price that the base case must clear, and the "
        "premium a board could recommend",
        "Financing package and the leverage the cash flows actually support",
    ),
    8: (
        "Completion conditions: shareholder vote, regulatory and foreign-investment "
        "clearance, financing certainty",
        "Interloper and activist risk: go-shop, fiduciary out, and topping-bid exposure",
    ),
}

STRUCTURE_MODULES: dict[str, dict[int, tuple[str, ...]]] = {
    TransactionStructure.PUBLIC_MINORITY_STAKE.value: MINORITY_STAKE_MODULE,
    TransactionStructure.PUBLIC_CONTROL_STAKE.value: CONTROL_STAKE_MODULE,
    TransactionStructure.TAKE_PRIVATE.value: TAKE_PRIVATE_MODULE,
}


PREBUILT_MODULES: dict[str, dict[int, tuple[str, ...]]] = {
    "saas": SAAS_MODULE,
    "healthcare": HEALTHCARE_MODULE,
}


def module_for_sub_sector(sub_sector: str, business_model: str = "") -> str | None:
    """Route a free-text sub-sector to a pre-built module, or None.

    Returning None is a real answer, not a failure: design spec s IV.D says an
    unmatched sub-sector gets the universal spine plus 3-5 diagnostic metrics
    generated the same way the pre-built modules were - see `sub_sector.py`.
    """
    text = f"{sub_sector} {business_model}".lower()
    saas_markers = ("saas", "software", "subscription", "platform", "cloud")
    health_markers = ("health", "clinic", "dental", "provider", "medical", "physician",
                      "hospital", "care")
    if any(m in text for m in health_markers):
        return "healthcare"
    if any(m in text for m in saas_markers):
        return "saas"
    return None


def tailored_outline(sub_sector: str, business_model: str = "",
                     shape: "DealShape | None" = None) -> list[OutlineSection]:
    """The universal spine, with the sub-sector and deal-structure modules merged in.

    Order matters: sub-sector first, then the public-company module, then the module
    for the specific structure. A take-private and a minority stake in the same
    company ask different questions of it, and the last module in is the one closest
    to the decision being taken.
    """
    module_name = module_for_sub_sector(sub_sector, business_model)
    layers: list[dict[int, tuple[str, ...]]] = [
        PREBUILT_MODULES.get(module_name or "", {})
    ]
    if shape is not None and shape.public_target:
        layers.append(PUBLIC_COMPANY_MODULE)
        layers.append(STRUCTURE_MODULES.get(shape.structure.value, {}))

    out: list[OutlineSection] = []
    for section in UNIVERSAL_OUTLINE:
        extra: tuple[str, ...] = ()
        for layer in layers:
            extra += layer.get(section.number, ())
        out.append(
            section.model_copy(update={"key_elements": section.key_elements + extra})
            if extra
            else section
        )
    return out


def section(number: int) -> OutlineSection:
    return next(s for s in UNIVERSAL_OUTLINE if s.number == number)
