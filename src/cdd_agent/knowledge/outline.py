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


def tailored_outline(sub_sector: str, business_model: str = "") -> list[OutlineSection]:
    """The universal spine with the matching module's elements merged in."""
    module_name = module_for_sub_sector(sub_sector, business_model)
    module = PREBUILT_MODULES.get(module_name or "", {})
    out: list[OutlineSection] = []
    for section in UNIVERSAL_OUTLINE:
        extra = module.get(section.number, ())
        out.append(
            section.model_copy(update={"key_elements": section.key_elements + extra})
            if extra
            else section
        )
    return out


def section(number: int) -> OutlineSection:
    return next(s for s in UNIVERSAL_OUTLINE if s.number == number)
