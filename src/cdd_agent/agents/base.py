"""Shared agent scaffolding.

Every agent writes its output artifact to the shared state store under its own role
name. That is what makes the hand-off auditable (Checkpoint 5.1) and what satisfies the
runtime-monitoring guardrail (Checkpoint 6.1) - not a convention agents are asked to
follow, but the only write path available to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cdd_agent.config import Settings, get_settings
from cdd_agent.guardrails.authorization import AgentRole, ToolAuthorization
from cdd_agent.schemas.deal_profile import DealProfile, DealShape
from cdd_agent.state.memory import LongTermMemory
from cdd_agent.state.store import Collection, StateStore
from cdd_agent.tools.registry import ToolBundle, ToolRegistry


def save_structured_tables(store: "StateStore", engagement_id: str, tables, *,
                           agent: str = "Controller") -> int:
    """Persist parsed tabular files so later phases can still compute from them."""
    payload = [
        {"source_file": t.source_file, "name": t.name, "columns": list(t.columns),
         "rows": t.rows,
         "document_date": t.document_date.isoformat() if t.document_date else None}
        for t in tables
    ]
    store.put(engagement_id, Collection.STRUCTURED, "tables", {"tables": payload},
              agent=agent)
    return len(payload)


def load_structured_tables(store: "StateStore", engagement_id: str) -> list:
    """Rebuild the parsed tables saved at ingestion."""
    import datetime as _dt

    from cdd_agent.retrieval.ingestion import StructuredTable

    stored = store.get(engagement_id, Collection.STRUCTURED, "tables")
    if not stored:
        return []
    out = []
    for raw in stored.get("tables", []):
        date = raw.get("document_date")
        out.append(StructuredTable(
            source_file=raw["source_file"], name=raw["name"],
            columns=list(raw.get("columns", [])), rows=raw.get("rows", []),
            document_date=_dt.date.fromisoformat(date) if date else None,
        ))
    return out


@dataclass
class AgentContext:
    """Everything an agent is handed. Nothing is reached for globally."""

    engagement_id: str
    store: StateStore
    memory: LongTermMemory
    settings: Settings
    profile: Optional[DealProfile] = None
    registry: Optional[ToolRegistry] = None

    @classmethod
    def create(
        cls,
        engagement_id: str,
        *,
        store: Optional[StateStore] = None,
        profile: Optional[DealProfile] = None,
        tables: tuple = (),
    ) -> "AgentContext":
        store = store or StateStore()
        memory = LongTermMemory(store, engagement_id)
        if not tables:
            # Parsed tables used to live only in the ingesting process, so a restart
            # silently stripped the computation tool of its inputs and every
            # quantitative exhibit degraded to a gap. They are engagement data, so
            # they belong in the store like every other artifact.
            tables = tuple(load_structured_tables(store, engagement_id))
        profile = profile if profile is not None else memory.deal_profile()
        authorization = ToolAuthorization(profile)
        return cls(
            engagement_id=engagement_id,
            store=store,
            memory=memory,
            settings=get_settings(),
            profile=profile,
            registry=ToolRegistry(engagement_id, authorization, tables=tables),
        )

    @property
    def authorization(self) -> ToolAuthorization:
        assert self.registry is not None
        return self.registry.authorization

    def tools_for(self, role: AgentRole) -> ToolBundle:
        assert self.registry is not None, "no tool registry on this context"
        return self.registry.bundle_for(role)

    @property
    def sub_sector(self) -> str:
        return self.profile.sector.sub_sector if self.profile else ""

    @property
    def is_strategic_buyer(self) -> bool:
        return bool(
            self.profile and self.profile.buyer.buyer_type.value.startswith("corporate")
        )

    @property
    def deal_shape(self) -> DealShape:
        """Buyer type, listing status and structure - the axes that change behaviour.

        Passed wherever `is_strategic_buyer` used to go, so a listed target reaches
        the risk taxonomy, the outline and the exhibits without a second parameter
        being threaded through every signature.
        """
        return DealShape.from_profile(self.profile)


class Agent:
    """Base class. `role` doubles as the audit-log attribution string."""

    role: AgentRole

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    @property
    def name(self) -> str:
        return self.role.value

    @property
    def offline(self) -> bool:
        return self.context.settings.offline

    def tools(self) -> ToolBundle:
        return self.context.tools_for(self.role)
