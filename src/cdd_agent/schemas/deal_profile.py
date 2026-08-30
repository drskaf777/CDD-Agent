"""Deal Profile Brief - the Intake Agent's output artifact.

Mirrors the diagnostic intake protocol, design specification s III, categories A-G.
Category F (Data Access and Constraints) is load-bearing beyond documentation: it is
read directly by the authorization guardrail to decide which tools may run at all.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from cdd_agent.schemas.common import Stamped


class TransactionStructure(str, Enum):
    MAJORITY_BUYOUT = "majority buyout"
    MINORITY_GROWTH = "minority growth investment"
    CARVE_OUT = "carve-out"
    ASSET_PURCHASE = "asset purchase"
    UNKNOWN = "unknown"


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

    def is_ready_for_phase_1(self) -> tuple[bool, list[str]]:
        """Phase 1 needs a thesis, a sub-sector, and stated buyer decision criteria."""
        missing: list[str] = []
        if not self.thesis.one_sentence_thesis.strip():
            missing.append("Category C: one-sentence investment thesis")
        if not self.sector.sub_sector.strip():
            missing.append("Category B: specific sub-sector")
        if not self.buyer.decision_criteria:
            missing.append("Category D: buyer decision criteria")
        return (not missing, missing)
