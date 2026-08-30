"""Universal data-request catalogue and sub-sector add-ons - design spec s V.

Tiering rule, restated from s V so it is applied consistently rather than re-decided
per deal: Tier 1 is what must be received before any hypothesis can be rated Confirmed
or Contradicted; Tier 2 builds presentation-ready depth; Tier 3 enriches an exhibit
without ever blocking.
"""

from __future__ import annotations

from dataclasses import dataclass

from cdd_agent.schemas.common import Tier


@dataclass(frozen=True)
class CatalogItem:
    category: str
    item: str
    tier: Tier
    sub_sector_specific: bool = False
    rationale: str = ""


UNIVERSAL_CATALOG: tuple[CatalogItem, ...] = (
    # --- Corporate & Legal ---
    CatalogItem("Corporate & Legal", "Cap table and ownership structure", Tier.DEPTH_BUILDING),
    CatalogItem("Corporate & Legal", "Material contracts", Tier.DEAL_CRITICAL),
    CatalogItem("Corporate & Legal", "Litigation and regulatory-inquiry log",
                Tier.DEPTH_BUILDING),
    CatalogItem("Corporate & Legal", "IP/patent register", Tier.ENRICHMENT),
    CatalogItem("Corporate & Legal", "Organizational chart with legal-entity map",
                Tier.DEPTH_BUILDING),
    # --- Financial Records ---
    CatalogItem("Financial Records", "Audited/reviewed financials (3-5 years)",
                Tier.DEAL_CRITICAL,
                rationale="Tier-1 by definition: no rating is Confirmed without it."),
    CatalogItem("Financial Records", "Monthly management accounts", Tier.DEPTH_BUILDING),
    CatalogItem("Financial Records", "Current-year budget vs. actual", Tier.DEPTH_BUILDING),
    CatalogItem("Financial Records", "Working-capital detail", Tier.DEPTH_BUILDING),
    CatalogItem("Financial Records", "Debt schedule and covenant headroom",
                Tier.DEPTH_BUILDING),
    CatalogItem("Financial Records", "Current business plan / operating model",
                Tier.DEAL_CRITICAL,
                rationale="The plan being tested. Phase 3 cannot start without it."),
    # --- Commercial / Sales ---
    CatalogItem("Commercial / Sales", "Revenue by product, geography, and customer",
                Tier.DEAL_CRITICAL),
    CatalogItem("Commercial / Sales", "CRM pipeline export", Tier.DEPTH_BUILDING),
    CatalogItem("Commercial / Sales", "Win/loss log", Tier.DEPTH_BUILDING),
    CatalogItem("Commercial / Sales", "Pricing schedules and discount history",
                Tier.DEPTH_BUILDING),
    CatalogItem("Commercial / Sales", "Sales-team headcount and quota attainment",
                Tier.DEPTH_BUILDING),
    # --- Customer Data ---
    CatalogItem("Customer Data", "Customer-level billing/contract detail", Tier.DEAL_CRITICAL),
    CatalogItem("Customer Data", "Cohort retention data", Tier.DEPTH_BUILDING),
    CatalogItem("Customer Data", "NPS/CSAT survey history", Tier.DEPTH_BUILDING),
    CatalogItem("Customer Data", "Top-20 customer contracts and renewal dates",
                Tier.DEPTH_BUILDING),
    CatalogItem("Customer Data", "Customer concentration schedule", Tier.DEAL_CRITICAL),
    # --- Competitive & Market Intelligence (internally held) ---
    CatalogItem("Competitive & Market Intelligence",
                "Board decks referencing competitive positioning", Tier.ENRICHMENT),
    CatalogItem("Competitive & Market Intelligence",
                "Prior market studies or sell-side vendor due diligence", Tier.ENRICHMENT),
    CatalogItem("Competitive & Market Intelligence", "Win/loss commentary", Tier.ENRICHMENT),
    CatalogItem("Competitive & Market Intelligence",
                "Analyst or industry-report subscriptions", Tier.ENRICHMENT),
    # --- Operations ---
    CatalogItem("Operations", "Cost structure by function", Tier.DEPTH_BUILDING),
    CatalogItem("Operations", "Capacity/utilization data", Tier.DEPTH_BUILDING),
    CatalogItem("Operations", "Supplier and vendor concentration", Tier.DEPTH_BUILDING),
    CatalogItem("Operations", "Key operational KPIs and trend history", Tier.DEPTH_BUILDING),
    # --- Management & HR ---
    CatalogItem("Management & HR", "Organization chart with tenure", Tier.DEPTH_BUILDING),
    CatalogItem("Management & HR", "Compensation and incentive plan summary",
                Tier.DEPTH_BUILDING),
    CatalogItem("Management & HR", "Employee turnover by function", Tier.DEPTH_BUILDING),
    CatalogItem("Management & HR", "Key-person / non-compete agreements", Tier.DEAL_CRITICAL),
    CatalogItem("Management & HR", "Succession plans", Tier.ENRICHMENT),
    # --- IT & Data Infrastructure ---
    CatalogItem("IT & Data Infrastructure", "Systems architecture overview", Tier.ENRICHMENT),
    CatalogItem("IT & Data Infrastructure", "Data-security and compliance certifications",
                Tier.DEPTH_BUILDING),
    CatalogItem("IT & Data Infrastructure",
                "Technical-debt or platform-modernization roadmap", Tier.DEPTH_BUILDING),
    # --- ESG & Regulatory ---
    CatalogItem("ESG & Regulatory", "Licenses, permits, and accreditation status",
                Tier.DEPTH_BUILDING),
    CatalogItem("ESG & Regulatory", "Environmental/health-and-safety incident log",
                Tier.ENRICHMENT),
    CatalogItem("ESG & Regulatory", "Sector-specific regulatory filings", Tier.DEPTH_BUILDING),
)


# Design spec s V.B - the illustrative B2B SaaS tailored output.
SAAS_ADDONS: tuple[CatalogItem, ...] = (
    CatalogItem("Customer Data",
                "Contract-level ARR/MRR waterfall (new, expansion, contraction, churn) "
                "by month, reconciled to recognized revenue",
                Tier.DEAL_CRITICAL, True,
                "Fastest way to catch bookings-vs-billings or rev-rec issues before "
                "confirmatory diligence."),
    CatalogItem("Customer Data",
                "Cohort-level NRR and GRR, minimum trailing 12 quarters, segmented by "
                "customer size band",
                Tier.DEAL_CRITICAL, True),
    CatalogItem("Commercial / Sales",
                "Product usage / telemetry data by account",
                Tier.DEPTH_BUILDING, True,
                "Tests stickiness independent of contractual lock-in."),
    CatalogItem("Commercial / Sales",
                "Full CRM pipeline export with stage-by-stage conversion history and "
                "average sales-cycle length",
                Tier.DEPTH_BUILDING, True),
    CatalogItem("Operations",
                "Customer support ticket volume and resolution-time trends",
                Tier.DEPTH_BUILDING, True,
                "Proxy for product-quality risk."),
    CatalogItem("IT & Data Infrastructure",
                "Engineering roadmap, release velocity, and technical-debt register",
                Tier.DEPTH_BUILDING, True,
                "Material where the thesis depends on platform extensibility."),
    CatalogItem("IT & Data Infrastructure",
                "Security and compliance certifications (SOC 2, ISO 27001) and any "
                "breach/incident history",
                Tier.DEAL_CRITICAL, True,
                "Elevated priority in the cybersecurity sub-sector."),
)

HEALTHCARE_ADDONS: tuple[CatalogItem, ...] = (
    CatalogItem("Customer Data",
                "Payer-contract renegotiation calendar and rate-lock expiry schedule",
                Tier.DEAL_CRITICAL, True,
                "Reimbursement step-downs are the most common source of post-close "
                "EBITDA misses in provider roll-ups and are rarely visible in headline "
                "financials."),
    CatalogItem("Customer Data", "Payer mix by volume and by revenue (Medicare / "
                "Medicaid / commercial), trailing 12 quarters",
                Tier.DEAL_CRITICAL, True),
    CatalogItem("Commercial / Sales",
                "Physician referral source concentration and leakage analysis",
                Tier.DEAL_CRITICAL, True),
    CatalogItem("Management & HR",
                "Clinician turnover, wage inflation, and non-compete coverage by site",
                Tier.DEPTH_BUILDING, True),
    CatalogItem("Operations", "Revenue-per-FTE and capacity utilization by site",
                Tier.DEPTH_BUILDING, True),
    CatalogItem("ESG & Regulatory", "Billing/coding compliance audit results",
                Tier.DEPTH_BUILDING, True),
    CatalogItem("Operations", "De Novo unit economics and ramp curves for recent openings",
                Tier.DEPTH_BUILDING, True),
)

ADDONS_BY_MODULE: dict[str, tuple[CatalogItem, ...]] = {
    "saas": SAAS_ADDONS,
    "healthcare": HEALTHCARE_ADDONS,
}

CATEGORIES: tuple[str, ...] = (
    "Corporate & Legal",
    "Financial Records",
    "Commercial / Sales",
    "Customer Data",
    "Competitive & Market Intelligence",
    "Operations",
    "Management & HR",
    "IT & Data Infrastructure",
    "ESG & Regulatory",
)
