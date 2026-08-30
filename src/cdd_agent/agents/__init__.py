"""The five role-bearing agents (Checkpoint 5.1).

    Intake -> Thesis Architect -> Analyst <-> Risk Auditor -> Synthesizer

Five, not more and not fewer, because the count maps competencies and bias-guards
rather than phases: Phase-2 data-request generation folds into the Analyst, and a
standalone computation agent was rejected as coordination cost with no distinct
bias-guard behind it.

The Controller is not here - it is routing logic, not a persona. See
`cdd_agent.orchestration.controller`.
"""

from cdd_agent.agents.analyst import Analyst, LoopReport
from cdd_agent.agents.base import Agent, AgentContext
from cdd_agent.agents.intake import IntakeAgent
from cdd_agent.agents.risk_auditor import AuditReport, RiskAuditor
from cdd_agent.agents.synthesizer import Synthesizer
from cdd_agent.agents.thesis_architect import ThesisArchitect

__all__ = [
    "Agent", "AgentContext", "Analyst", "AuditReport", "IntakeAgent", "LoopReport",
    "RiskAuditor", "Synthesizer", "ThesisArchitect",
]
