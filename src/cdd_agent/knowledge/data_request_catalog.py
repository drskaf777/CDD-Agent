"""Universal data-request catalogue and sub-sector add-ons - design spec s V.

Tiering rule, restated from s V so it is applied consistently rather than re-decided
per deal: Tier 1 is what must be received before any hypothesis can be rated Confirmed
or Contradicted; Tier 2 builds presentation-ready depth; Tier 3 enriches an exhibit
without ever blocking.
"""

from __future__ import annotations

from dataclasses import dataclass

from cdd_agent.schemas.common import Tier
from cdd_agent.schemas.deal_profile import DealShape, TransactionStructure


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
    "Public Record",
)


# --- Listed targets -----------------------------------------------------------
# Two things change when the target is public. First, much of the standard request
# is already answered in the public record, and asking management for it wastes the
# one scarce resource in a live process - their attention - while signalling that
# nobody read the filings. Second, a set of questions opens up that exists only
# because there is a share price and there are other shareholders.

# Catalogue items the public record normally answers. Matched on a distinctive
# fragment of the item text, so rewording an item does not silently re-request it.
PUBLICLY_ANSWERABLE: dict[str, str] = {
    "Audited/reviewed financials": "Annual report / 10-K, audited and filed",
    "Revenue by product, geography": "Segment and geographic disclosure in the "
                                     "annual report - note the reported segments are "
                                     "usually coarser than the commercial question",
    "Litigation and regulatory-inquiry log": "Legal proceedings and risk factors in "
                                             "the annual report",
    "Cap table and ownership structure": "Proxy statement and institutional holdings "
                                         "filings",
    "Organizational chart with legal-entity map": "Subsidiary list filed as an "
                                                  "exhibit to the annual report",
    "IP/patent register": "Public patent register and the IP discussion in the "
                          "annual report",
    "Debt schedule and covenant headroom": "Debt footnote and covenant discussion in "
                                           "the filings",
}

PUBLIC_TARGET_ADDONS: tuple[CatalogItem, ...] = (
    CatalogItem("Public Record", "Last three annual reports and every interim report "
                "since", Tier.DEAL_CRITICAL,
                rationale="The baseline the market already prices. Retrieved, not "
                          "requested."),
    CatalogItem("Public Record", "Earnings-call transcripts and investor-day "
                "materials for the last eight quarters", Tier.DEPTH_BUILDING,
                rationale="Guidance against delivery is the cheapest available test "
                          "of management credibility."),
    CatalogItem("Public Record", "Proxy statement: board composition, executive "
                "incentives, and ownership", Tier.DEPTH_BUILDING),
    CatalogItem("Public Record", "Published consensus estimates and the dispersion "
                "around them", Tier.DEPTH_BUILDING,
                rationale="Sets the bar the thesis must beat. Not corroboration - "
                          "analysts are guided by the company."),
    CatalogItem("Commercial / Sales", "Reconciliation of reported segments to the "
                "internal commercial view (product, motion, segment)",
                Tier.DEAL_CRITICAL,
                rationale="Reported segments are built for disclosure, not for "
                          "diligence - the commercial question sits below them."),
    CatalogItem("Financial Records", "Bridge from reported to the adjusted figures "
                "used in the equity story, each adjustment named", Tier.DEAL_CRITICAL,
                rationale="The gap between reported and adjusted is where a listed "
                          "SaaS story is most often made."),
    CatalogItem("Financial Records", "Cost of being a listed company, itemised",
                Tier.DEPTH_BUILDING,
                rationale="A real saving on a take-private, and a cost that stays on "
                          "any structure keeping the listing."),
)

# Structure-specific. Requested only for the structure that needs them.
MINORITY_STAKE_REQUESTS: tuple[CatalogItem, ...] = (
    CatalogItem("Corporate & Legal", "Governance terms on offer: board seats, "
                "consent rights, standstill", Tier.DEAL_CRITICAL,
                rationale="Decides whether the plan being underwritten can be "
                          "influenced at all, or only hoped for."),
    CatalogItem("Corporate & Legal", "Information rights offered post-close, and "
                "whether accepting them restricts trading", Tier.DEAL_CRITICAL,
                rationale="Monitoring rights and tradability pull against each "
                          "other - taking the first forfeits the second."),
    CatalogItem("Commercial / Sales", "Average daily traded volume and free-float "
                "history", Tier.DEPTH_BUILDING,
                rationale="The exit constraint on a block of this size."),
)

CONTROL_STAKE_REQUESTS: tuple[CatalogItem, ...] = (
    CatalogItem("Corporate & Legal", "Related-party transaction policy and the "
                "minority protections that constrain value capture",
                Tier.DEAL_CRITICAL),
    CatalogItem("Corporate & Legal", "Continued-listing and free-float requirements "
                "of the exchange", Tier.DEPTH_BUILDING),
    CatalogItem("Management & HR", "Board composition and independence requirements "
                "under the controlled-company regime", Tier.DEPTH_BUILDING),
)

TAKE_PRIVATE_REQUESTS: tuple[CatalogItem, ...] = (
    CatalogItem("Corporate & Legal", "Constitutional documents: approval thresholds, "
                "defences, and change-of-control provisions", Tier.DEAL_CRITICAL),
    CatalogItem("Corporate & Legal", "Change-of-control and consent provisions in the "
                "top customer and partner contracts", Tier.DEAL_CRITICAL,
                rationale="A commercial question, not only a legal one: consents "
                          "that can be withheld are revenue at risk on close."),
    CatalogItem("Management & HR", "Management rollover intentions and retention "
                "terms post-delisting", Tier.DEAL_CRITICAL),
    CatalogItem("Financial Records", "Financing package terms and the leverage the "
                "plan actually supports", Tier.DEPTH_BUILDING),
)

REQUESTS_BY_STRUCTURE: dict[str, tuple[CatalogItem, ...]] = {
    TransactionStructure.PUBLIC_MINORITY_STAKE.value: MINORITY_STAKE_REQUESTS,
    TransactionStructure.PUBLIC_CONTROL_STAKE.value: CONTROL_STAKE_REQUESTS,
    TransactionStructure.TAKE_PRIVATE.value: TAKE_PRIVATE_REQUESTS,
}


def public_record_note(item: str) -> str:
    """Where the public record already answers a catalogue item, if it does."""
    for fragment, where in PUBLICLY_ANSWERABLE.items():
        if fragment.lower() in item.lower():
            return where
    return ""


def catalog_for(shape: DealShape) -> tuple[CatalogItem, ...]:
    """The listed-target additions in scope for this structure."""
    if not shape.public_target:
        return ()
    return PUBLIC_TARGET_ADDONS + REQUESTS_BY_STRUCTURE.get(shape.structure.value, ())
