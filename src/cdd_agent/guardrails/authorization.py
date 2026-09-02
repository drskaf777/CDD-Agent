"""Preventive guardrail: input checks and tool access limits (Checkpoint 6.1).

Two rules, both enforced here rather than in prompts:

1. **Input checks.** Intake's NDA/access constraints (Category F) gate tool
   authorization *before any run*. A "no customer contact pre-signing" flag disables
   the interview tool outright - the tool is not offered to the model at all, so there
   is nothing for it to be talked into.
2. **Tool access limits.** Each agent's tool scope is narrow by role. The Intake Agent
   has no data-room access; the Analyst can query indexes but cannot initiate outreach
   beyond what intake authorized.

An attempted violation raises rather than warns. Checkpoint 6.1 lists "any action that
would exceed intake's NDA/access constraints" as a hard block, not a warning, and the
distinction only means anything if the code refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from cdd_agent.schemas.deal_profile import AccessConstraints, DealProfile, DealStage


class AgentRole(str, Enum):
    INTAKE = "Intake Agent"
    THESIS_ARCHITECT = "Thesis Architect"
    ANALYST = "Analyst"
    RISK_AUDITOR = "Risk Auditor"
    SYNTHESIZER = "Synthesizer"
    CONTROLLER = "Controller"


class ToolName(str, Enum):
    DOCUMENT_RETRIEVAL = "document_retrieval"
    STRUCTURED_COMPUTATION = "structured_computation"
    MARKET_SEARCH = "market_search"
    PRIMARY_RESEARCH = "primary_research"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"


class AuthorizationError(PermissionError):
    """Raised when an action would exceed intake's NDA/access constraints."""


# Role -> tool scope. Deliberately narrow; widening a row is a design decision.
ROLE_TOOLS: dict[AgentRole, frozenset[ToolName]] = {
    # Intake runs a conversation. It has no data-room access at all.
    AgentRole.INTAKE: frozenset({ToolName.STATE_READ, ToolName.STATE_WRITE}),
    # The Thesis Architect checks sub-sector fit against the Knowledge-Base Index.
    # It has no Data-Room access: Phase 1 runs before any data request goes out.
    AgentRole.THESIS_ARCHITECT: frozenset(
        {ToolName.MARKET_SEARCH, ToolName.STATE_READ, ToolName.STATE_WRITE}
    ),
    AgentRole.ANALYST: frozenset(
        {
            ToolName.DOCUMENT_RETRIEVAL,
            ToolName.STRUCTURED_COMPUTATION,
            ToolName.MARKET_SEARCH,
            ToolName.PRIMARY_RESEARCH,
            ToolName.STATE_READ,
            ToolName.STATE_WRITE,
        }
    ),
    # The Auditor reads and re-queries to verify, but never commissions new outreach:
    # an auditor that can generate its own evidence is auditing its own work.
    AgentRole.RISK_AUDITOR: frozenset(
        {
            ToolName.DOCUMENT_RETRIEVAL,
            ToolName.STRUCTURED_COMPUTATION,
            ToolName.STATE_READ,
            ToolName.STATE_WRITE,
        }
    ),
    # The Synthesizer writes from artifacts already in the store. It cannot go and
    # find new evidence mid-deck, which is how uncited assertions get in.
    AgentRole.SYNTHESIZER: frozenset({ToolName.STATE_READ, ToolName.STATE_WRITE}),
    AgentRole.CONTROLLER: frozenset({ToolName.STATE_READ, ToolName.STATE_WRITE}),
}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""

    def raise_if_denied(self) -> None:
        if not self.allowed:
            raise AuthorizationError(self.reason)


class ToolAuthorization:
    """Resolves role scope and intake Category F into an allow/deny per tool call."""

    def __init__(self, profile: Optional[DealProfile]) -> None:
        self.profile = profile
        self.access: AccessConstraints = (
            profile.access if profile else AccessConstraints()
        )
        self.pre_signing = bool(
            profile
            and profile.target.deal_stage
            in (DealStage.EARLY_SCREENING, DealStage.SIGNED_LOI)
        )
        self.public_target = bool(profile and profile.is_public_target)

    # ------------------------------------------------------------- MNPI gate
    @property
    def mnpi_block(self) -> Optional[str]:
        """Why work that touches non-public material is barred, if it is.

        Diligence on a listed company hands the buyer material non-public
        information, and from that moment everyone who has seen it is restricted
        from trading the security. The restriction is not the agent's to accept on
        the firm's behalf, so until compliance has recorded it the tools that would
        create the exposure do not run. This is the same class of rule as the
        NDA/access constraints - a hard block, not a warning.
        """
        if not self.public_target or not self.access.mnpi_expected:
            return None
        if self.access.trading_restriction_acknowledged:
            return None
        return (
            "This is a listed target and intake Category F expects the data room to "
            "carry material non-public information, but compliance has not recorded "
            "the trading restriction. Reading it would put the firm in possession of "
            "MNPI without a wall-crossing on file. Obtain the acknowledgement, then "
            "re-run."
        )

    # ------------------------------------------------------------------ checks
    def check(
        self,
        role: AgentRole,
        tool: ToolName,
        *,
        contact_type: str | None = None,
        is_top5_customer: bool = False,
    ) -> Decision:
        scope = ROLE_TOOLS.get(role, frozenset())
        if tool not in scope:
            return Decision(
                False,
                f"{role.value} is not authorized to call {tool.value}. "
                f"Role scope: {sorted(t.value for t in scope)}.",
            )

        if tool in (ToolName.DOCUMENT_RETRIEVAL, ToolName.STRUCTURED_COMPUTATION):
            blocked = self.mnpi_block
            if blocked:
                return Decision(False, blocked)

        if tool is ToolName.DOCUMENT_RETRIEVAL:
            if self.access.vdr_access.value == "none":
                return Decision(
                    False,
                    "Intake Category F records no data-room access, so document "
                    "retrieval cannot run. Resolve access before Phase 3.",
                )

        if tool is ToolName.MARKET_SEARCH and not self.access.external_web_research_permitted:
            return Decision(
                False,
                "Intake Category F prohibits external research on this engagement.",
            )

        if tool is ToolName.PRIMARY_RESEARCH:
            return self._check_outreach(contact_type, is_top5_customer)

        return Decision(True)

    def _check_outreach(self, contact_type: str | None, is_top5_customer: bool) -> Decision:
        kind = (contact_type or "customer").lower()

        if not self.access.above_the_line:
            return Decision(
                False,
                "This is a below-the-line (discreet) process. Outreach that would "
                "reveal the transaction is a hard block; carry the question forward "
                "as a confirmatory-diligence item instead.",
            )
        if kind.startswith(("issuer", "insider", "management", "investor relations",
                            "ir", "officer", "director")):
            # Reg FD: an unscripted call with an officer of a listed company is how
            # selective disclosure happens, and the exposure lands on the issuer,
            # which is also the asset being bought. Default deny, and note that the
            # public record is the sanctioned channel.
            if not self.public_target:
                return Decision(True)
            if not self.access.issuer_contact_permitted:
                return Decision(
                    False,
                    "Contact with insiders of a listed issuer is not authorized. "
                    "Selective disclosure would expose the issuer under Reg FD and "
                    "taint the process. Use the public record - filings, earnings "
                    "calls, investor days - or route the question through the "
                    "company's advisers as a data request.",
                )
            return Decision(True)
        if kind.startswith("customer"):
            if not self.access.customer_contact_permitted:
                return Decision(False, "Intake Category F prohibits customer contact.")
            if (
                is_top5_customer
                and self.pre_signing
                and not self.access.top5_customer_contact_permitted_pre_signing
            ):
                return Decision(
                    False,
                    "Top-5 customer contact is restricted pre-signing under intake "
                    "Category F. Carry forward as a confirmatory-diligence item.",
                )
            return Decision(True)
        if kind.startswith("competitor"):
            if not self.access.competitor_contact_permitted:
                return Decision(
                    False, "Intake Category F prohibits direct competitor contact."
                )
            return Decision(True)
        if kind.startswith("expert"):
            if not self.access.expert_calls_permitted:
                return Decision(False, "Intake Category F prohibits expert calls.")
            return Decision(True)
        return Decision(False, f"Unrecognized contact type {contact_type!r}; denied by default.")

    def authorize(self, role: AgentRole, tool: ToolName, **kwargs: object) -> None:
        """Raise unless the call is permitted."""
        self.check(role, tool, **kwargs).raise_if_denied()  # type: ignore[arg-type]

    def available_tools(self, role: AgentRole) -> list[ToolName]:
        """Tools this role may actually use on this engagement.

        Used to build the model's tool list, so a forbidden tool is never presented.
        """
        return [t for t in ROLE_TOOLS.get(role, frozenset()) if self.check(role, t).allowed]
