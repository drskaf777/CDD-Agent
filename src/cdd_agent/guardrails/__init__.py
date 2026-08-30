"""Three layers around the pipeline (Checkpoint 6.1, Architecture v6.7 slide 4).

* Preventive - `authorization` (input checks, tool access limits) and
  `output_contract` (citation + confidence tag on every claim).
* Detective - `cdd_agent.evaluation.metrics`.
* Corrective - `escalation` (the five human-intervention triggers).

The runtime-monitoring guardrail lives in `cdd_agent.state.store`, since attribution
is only meaningful if it is impossible to write without it.
"""

from cdd_agent.guardrails.authorization import (
    AgentRole,
    AuthorizationError,
    Decision,
    ToolAuthorization,
    ToolName,
)
from cdd_agent.guardrails.escalation import (
    Escalation,
    Trigger,
    access_boundary,
    check_phase1,
    check_source_conflicts,
    check_tier1_evidence,
    final_recommendation_review,
)
from cdd_agent.guardrails.output_contract import (
    ContractReport,
    SchemaViolation,
    check_claim,
    check_deck,
    format_report,
)

__all__ = [
    "AgentRole", "AuthorizationError", "ContractReport", "Decision", "Escalation",
    "SchemaViolation", "ToolAuthorization", "ToolName", "Trigger", "access_boundary",
    "check_claim", "check_deck", "check_phase1", "check_source_conflicts",
    "check_tier1_evidence", "final_recommendation_review", "format_report",
]
