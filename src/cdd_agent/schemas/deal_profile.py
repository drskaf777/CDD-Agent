"""Deal Profile Brief - the Intake Agent's output artifact.

Mirrors the diagnostic intake protocol, design specification s III, categories A-G.
Category F (Data Access and Constraints) is load-bearing beyond documentation: it is
read directly by the authorization guardrail to decide which tools may run at all.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from cdd_agent.schemas.common import Stamped


class TransactionStructure(str, Enum):
    MAJORITY_BUYOUT = "majority buyout"
    MINORITY_GROWTH = "minority growth investment"
    CARVE_OUT = "carve-out"
    ASSET_PURCHASE = "asset purchase"
    # Public-market structures. Kept distinct from their private analogues because
    # what changes is not the size of the stake but what the buyer can compel: a
    # minority holder cannot execute the value-creation plan, and a company that
    # stays listed keeps minority shareholders whose interests constrain it.
    PUBLIC_MINORITY_STAKE = "significant minority stake (listed)"
    PUBLIC_CONTROL_STAKE = "controlling stake, listing retained"
    TAKE_PRIVATE = "take-private"
    UNKNOWN = "unknown"

    @property
    def is_public_market(self) -> bool:
        return self in (
            TransactionStructure.PUBLIC_MINORITY_STAKE,
            TransactionStructure.PUBLIC_CONTROL_STAKE,
            TransactionStructure.TAKE_PRIVATE,
        )

    @property
    def retains_listing(self) -> bool:
        """The target is still a listed company the day after close."""
        return self in (
            TransactionStructure.PUBLIC_MINORITY_STAKE,
            TransactionStructure.PUBLIC_CONTROL_STAKE,
        )

    @property
    def confers_control(self) -> bool:
        """Whether the buyer can actually direct the plan it is underwriting."""
        return self in (
            TransactionStructure.MAJORITY_BUYOUT,
            TransactionStructure.CARVE_OUT,
            TransactionStructure.ASSET_PURCHASE,
            TransactionStructure.PUBLIC_CONTROL_STAKE,
            TransactionStructure.TAKE_PRIVATE,
        )

    @property
    def requires_shareholder_approval(self) -> bool:
        """A vote the buyer does not control is a live execution risk."""
        return self is TransactionStructure.TAKE_PRIVATE


class DealStage(str, Enum):
    EARLY_SCREENING = "early screening"
    SIGNED_LOI = "signed LOI"
    EXCLUSIVITY = "exclusivity / confirmatory diligence"
    POST_CLOSE = "post-close review"


class BusinessModel(str, Enum):
    SAAS = "subscription/SaaS"
    PROFESSIONAL_SERVICES = "professional services"
    TECH_ENABLED_SERVICES = "tech-enabled services"
    PRODUCT_MANUFACTURING = "product/manufacturing"
    MARKETPLACE = "marketplace"
    FRANCHISE = "franchise"
    CLINICAL_PROVIDER = "clinical/provider"
    OTHER = "other"


class BuyerType(str, Enum):
    FINANCIAL_SPONSOR = "financial sponsor"
    STRATEGIC = "corporate strategic acquirer"


class ProcessType(str, Enum):
    AUCTION = "competitive auction"
    BILATERAL = "bilateral negotiation"
    PROPRIETARY = "proprietary sourced"


class VDRAccess(str, Enum):
    VDR_LINK = "VDR link"
    FILE_UPLOAD = "direct file upload"
    INTERVIEW_NOTES_ONLY = "management-interview notes only"
    NONE = "none"


# --- Category A ---
class TargetIdentification(BaseModel):
    legal_name: str
    trading_names: list[str] = Field(default_factory=list)
    website: Optional[str] = None
    headquarters: Optional[str] = None
    revenue_geographies: list[str] = Field(default_factory=list)
    transaction_structure: TransactionStructure = TransactionStructure.UNKNOWN
    publicly_traded: Optional[bool] = None
    deal_stage: DealStage = DealStage.EARLY_SCREENING
    ic_date: Optional[_dt.date] = Field(
        default=None, description="Drives target-resolution dates on every logged gap."
    )


# --- Category A (public targets): the market context the deal is priced against ---
class PublicMarketContext(BaseModel):
    """What is knowable about a listed target before the data room opens.

    Every field is optional and defaults to unknown. That is deliberate: this block
    feeds exhibits, and a free-float or premium figure the intake never supplied must
    read as absent, not as zero. Nothing here is inferred from the ticker.
    """

    ticker: str = ""
    exchange: str = ""
    unaffected_price_date: Optional[_dt.date] = Field(
        default=None,
        description="Last date the market was uninformed of the approach. Every "
        "premium calculation is measured from this date, so a wrong one silently "
        "misstates the premium.",
    )
    unaffected_share_price: Optional[float] = None
    currency: str = ""
    shares_outstanding_m: Optional[float] = None
    free_float_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    insider_or_founder_stake_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    dual_class_shares: Optional[bool] = None
    analyst_coverage_count: Optional[int] = None
    consensus_available: Optional[bool] = None
    activist_holder_present: Optional[bool] = None
    index_memberships: list[str] = Field(default_factory=list)
    filings_available: list[str] = Field(
        default_factory=list,
        description="Named filings the engagement actually holds, e.g. "
        "'10-K FY2025'. Drives the public-record side of the data request: the "
        "agent should not ask management for what it can already read.",
    )
    shareholder_approval_threshold_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Vote needed to carry a take-private under the governing law.",
    )
    disclosure_threshold_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Stake at which the holding becomes publicly reportable "
        "(13D/13G in the US, TR-1 in the UK). Crossing it ends a discreet build-up.",
    )

    @property
    def is_populated(self) -> bool:
        return bool(self.ticker or self.exchange or self.filings_available)


# --- Category B ---
class SectorDefinition(BaseModel):
    sub_sector: str = Field(description="Specific, not the umbrella sector.")
    business_model: BusinessModel = BusinessModel.OTHER
    revenue_is_recurring: Optional[bool] = None
    customer_type: Optional[str] = Field(default=None, description="B2B, B2C, or B2B2C")
    company_stage: Optional[str] = None


# --- Category C ---
class InvestmentThesis(BaseModel):
    one_sentence_thesis: str
    critical_model_assumptions: list[str] = Field(
        default_factory=list,
        description="The 2-3 assumptions that, if wrong, change the recommendation.",
    )
    base_case_organic_growth: Optional[str] = None
    base_case_margin_expansion: Optional[str] = None
    hold_period_years: Optional[float] = None
    exit_route: Optional[str] = None


# --- Category D ---
class BuyerProfile(BaseModel):
    buyer_type: BuyerType = BuyerType.FINANCIAL_SPONSOR
    platform_or_bolt_on: Optional[str] = None
    strategic_core_business: Optional[str] = None
    strategic_relationship: Optional[str] = None
    decision_criteria: list[str] = Field(
        default_factory=list,
        description="What drives go/no-go: cash-flow stability, growth optionality, "
        "proprietary technology, customer-base access, talent. Scored by the ToT Critic.",
    )


# --- Category E ---
class ProcessContext(BaseModel):
    process_type: ProcessType = ProcessType.BILATERAL
    known_bidders: list[str] = Field(default_factory=list)
    prior_diligence_report_exists: bool = False
    prior_report_notes: Optional[str] = None


# --- Category F: read by guardrails/authorization.py ---
class AccessConstraints(BaseModel):
    """Constraints that gate tool authorization before any run.

    A False here is a hard block, not a warning (Checkpoint 6.1, human-intervention
    triggers: "any action that would exceed intake's NDA/access constraints").
    """

    vdr_access: VDRAccess = VDRAccess.NONE
    above_the_line: bool = Field(
        default=True, description="False = blind/discreet process; restricts outreach."
    )
    customer_contact_permitted: bool = True
    competitor_contact_permitted: bool = False
    top5_customer_contact_permitted_pre_signing: bool = False
    expert_calls_permitted: bool = True
    external_web_research_permitted: bool = True
    # --- Public targets: MNPI and selective-disclosure control ---
    # Diligence on a listed company hands the buyer material non-public information.
    # From that moment the buyer and everyone it briefs are restricted from trading
    # the security, and the issuer is exposed under Reg FD if the agent talks to
    # insiders. These are not documentation fields: authorization.py reads them.
    mnpi_expected: bool = Field(
        default=False,
        description="True once a data room on a listed target is opened. Anything "
        "beyond the public record on a public company should be assumed material "
        "until counsel says otherwise.",
    )
    trading_restriction_acknowledged: bool = Field(
        default=False,
        description="Compliance has recorded the restriction and wall-crossed the "
        "team. Until this is True, work that creates MNPI is blocked outright.",
    )
    wall_crossed_parties: list[str] = Field(default_factory=list)
    issuer_contact_permitted: bool = Field(
        default=False,
        description="Contact with company insiders or IR. Default False: an "
        "unscripted call with an officer of a listed company is how selective "
        "disclosure happens, and the exposure is the issuer's, not the buyer's.",
    )
    nda_constraints: list[str] = Field(default_factory=list)
    data_retention_policy: str = Field(
        default="purge engagement index at close",
        description="Carried through Phases 3-5; drives index teardown.",
    )


# --- Category G ---
class DeliverableParameters(BaseModel):
    primary_audience: str = "investment committee"
    output_format: str = "slide presentation"
    house_style_notes: Optional[str] = None
    length_limit_pages: Optional[int] = None


class DealProfile(Stamped):
    """The complete Phase-0 artifact. Gates advance to Phase 1."""

    engagement_id: str
    target: TargetIdentification
    public_market: PublicMarketContext = Field(default_factory=PublicMarketContext)
    sector: SectorDefinition
    thesis: InvestmentThesis
    buyer: BuyerProfile
    process: ProcessContext = Field(default_factory=ProcessContext)
    access: AccessConstraints = Field(default_factory=AccessConstraints)
    deliverable: DeliverableParameters = Field(default_factory=DeliverableParameters)
    open_intake_questions: list[str] = Field(
        default_factory=list,
        description="Unanswered scoping questions. A non-empty Category C entry here "
        "blocks Phase 1, since the thesis is what gets decomposed.",
    )

    # ------------------------------------------------------------------ shape
    @property
    def is_public_target(self) -> bool:
        """Listed either by the intake flag or by the structure being contemplated."""
        return bool(
            self.target.publicly_traded
            or self.target.transaction_structure.is_public_market
        )

    def is_ready_for_phase_1(self) -> tuple[bool, list[str]]:
        """Phase 1 needs a thesis, a sub-sector, and stated buyer decision criteria."""
        missing: list[str] = []
        if not self.thesis.one_sentence_thesis.strip():
            missing.append("Category C: one-sentence investment thesis")
        if not self.sector.sub_sector.strip():
            missing.append("Category B: specific sub-sector")
        if not self.buyer.decision_criteria:
            missing.append("Category D: buyer decision criteria")
        if self.is_public_target:
            # A listed target with no structure named is not a scoping detail. The
            # three public structures ask different questions of the same company -
            # a minority holder cannot execute the plan it is underwriting - so a
            # decomposition written before the structure is known tests the wrong
            # thing.
            if not self.target.transaction_structure.is_public_market:
                missing.append(
                    "Category A: which public structure - significant minority "
                    "stake, controlling stake with the listing retained, or "
                    "take-private"
                )
            if not self.public_market.ticker.strip():
                missing.append("Category A: ticker and exchange for the listed target")
            if self.access.mnpi_expected and not self.access.trading_restriction_acknowledged:
                missing.append(
                    "Category F: compliance has not acknowledged the trading "
                    "restriction that diligence on a listed target creates"
                )
        return (not missing, missing)


@dataclass(frozen=True)
class DealShape:
    """The few facts about a deal that change what the system does, in one object.

    Sub-sector tailoring already flows through the outline; this carries the other
    axis - who the buyer is, whether the target is listed, and how much of it the
    buyer will actually control. Passed instead of a widening list of booleans so
    that adding a structure does not mean editing eight signatures.
    """

    strategic_buyer: bool = False
    public_target: bool = False
    structure: TransactionStructure = TransactionStructure.UNKNOWN

    @classmethod
    def from_profile(cls, profile: Optional["DealProfile"]) -> "DealShape":
        if profile is None:
            return cls()
        return cls(
            strategic_buyer=profile.buyer.buyer_type is BuyerType.STRATEGIC,
            public_target=profile.is_public_target,
            structure=profile.target.transaction_structure,
        )

    @classmethod
    def coerce(cls, value: "DealShape | bool | None") -> "DealShape":
        """Accept the older `strategic_buyer` boolean at call sites not yet updated."""
        if isinstance(value, DealShape):
            return value
        return cls(strategic_buyer=bool(value))

    @property
    def retains_listing(self) -> bool:
        return self.public_target and self.structure.retains_listing

    @property
    def confers_control(self) -> bool:
        return self.structure.confers_control

    @property
    def is_public_minority(self) -> bool:
        """The buyer underwrites a plan it cannot execute - the defining constraint."""
        return self.structure is TransactionStructure.PUBLIC_MINORITY_STAKE

    @property
    def is_take_private(self) -> bool:
        return self.structure is TransactionStructure.TAKE_PRIVATE

    def label(self) -> str:
        # The public structures already name themselves as listed.
        if self.public_target and not self.structure.is_public_market:
            return f"{self.structure.value} (listed target)"
        return self.structure.value
