"""The diagnostic intake protocol - design specification s III, categories A-G.

Held as data rather than baked into a prompt so the Intake Agent can report exactly
which questions remain unanswered, and so Category F can be read by the authorization
guardrail without re-parsing conversation text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntakeCategory:
    key: str
    title: str
    questions: tuple[str, ...]
    required_for_phase_1: bool = False


INTAKE_PROTOCOL: tuple[IntakeCategory, ...] = (
    IntakeCategory(
        key="A",
        title="Target Identification",
        questions=(
            "What is the target's legal name, trading name(s), and website / domain?",
            "Where is the company headquartered, and in which geographies does it "
            "generate meaningful revenue?",
            "What is the transaction structure - majority buyout, minority growth "
            "investment, carve-out, or asset purchase?",
            "Is the target a privately held company (or asset) or is it publicly traded?",
            "What stage is the deal at - early screening, signed LOI, exclusivity / "
            "confirmatory diligence, or post-close review?",
            "Is there an IC date or reporting deadline the output needs to meet?",
        ),
    ),
    IntakeCategory(
        key="B",
        title="Sector and Sub-Sector Definition",
        required_for_phase_1=True,
        questions=(
            "What is the specific sub-sector (e.g. B2B cybersecurity SaaS vs. horizontal "
            "IT-security software; specialized dental clinics vs. multi-specialty dental "
            "service organizations)?",
            "What is the core business model - subscription/SaaS, professional services, "
            "tech-enabled services, product/manufacturing, marketplace, franchise, or "
            "clinical/provider?",
            "Is revenue predominantly recurring or transactional, and is the customer "
            "base B2B, B2C, or B2B2C?",
            "What stage is the company at - early growth, scaled/mature, or "
            "turnaround/underperforming?",
        ),
    ),
    IntakeCategory(
        key="C",
        title="Investment Thesis and Value-Creation Logic",
        required_for_phase_1=True,
        questions=(
            "What is the investment thesis in one sentence (e.g. regional roll-up, "
            "cross-sell of a new product line, geographic expansion, buy-and-build "
            "platform, turnaround, technology-disruption play)?",
            "Which two or three assumptions does the financial model most depend on - "
            "the ones that, if wrong, change the recommendation?",
            "What organic growth rate and margin expansion are embedded in the base "
            "case, and over what hold period?",
            "What is the anticipated exit route (strategic sale, sponsor-to-sponsor, "
            "IPO), if known?",
        ),
    ),
    IntakeCategory(
        key="D",
        title="Buyer Profile and Decision Criteria",
        required_for_phase_1=True,
        questions=(
            "Who is the buyer - a financial sponsor or a corporate strategic acquirer?",
            "If a sponsor: is this a new platform investment or a bolt-on/add-on to an "
            "existing portfolio company?",
            "If a strategic: what is the acquirer's core business, and how does the "
            "target relate to it - adjacency, vertical integration, or capability "
            "acquisition?",
            "What matters most to this buyer's go/no-go decision - cash-flow stability, "
            "growth optionality, proprietary technology, customer-base access, or talent?",
        ),
    ),
    IntakeCategory(
        key="E",
        title="Competitive Field and Process Context",
        questions=(
            "Is this a competitive auction, a bilateral negotiation, or a proprietary "
            "sourced deal?",
            "Are other bidders known, and how does that affect the depth and speed "
            "required?",
            "Does a prior diligence report exist (e.g. a sell-side vendor due diligence "
            "report) that the agent should reconcile against rather than duplicate?",
        ),
    ),
    IntakeCategory(
        key="F",
        title="Data Access and Constraints",
        questions=(
            "What data-room access will be provided - VDR link, direct file upload, or "
            "management-interview notes only?",
            "Is management aware diligence is underway (above the line) or is this a "
            "blind / discreet process (below the line)?",
            "Are customer and industry-expert reference calls permitted, and are there "
            "restrictions (e.g. no direct competitor contact, no top-5-customer contact "
            "pre-signing)?",
            "Are there NDA or confidentiality constraints the agent must observe in how "
            "it stores, cites, or externally researches information?",
        ),
    ),
    IntakeCategory(
        key="G",
        title="Deliverable Parameters",
        questions=(
            "Who is the primary audience - investment committee, board, or internal "
            "working team?",
            "Is the required output a slide presentation, a written memo, or both?",
            "Beyond the master structure already provided, are there house style, "
            "branding, or length requirements?",
        ),
    ),
)


def category(key: str) -> IntakeCategory:
    return next(c for c in INTAKE_PROTOCOL if c.key == key.upper())


def all_questions() -> list[tuple[str, str]]:
    return [(c.key, q) for c in INTAKE_PROTOCOL for q in c.questions]
