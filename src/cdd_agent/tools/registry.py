"""Role-scoped tool registry.

The tool-access-limits guardrail is enforced at construction, not at call time: a role
is handed only the tools it may use on this engagement, so a forbidden tool is never
in the model's tool list to begin with. Call-time authorization stays as a second
check for code paths that reach a tool directly.

Also exposes the tools as LangChain `StructuredTool`s, which is what lets the same
objects be bound to an LCEL chain and to a CrewAI agent without two implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from cdd_agent.guardrails.authorization import (
    AgentRole,
    AuthorizationError,
    ToolAuthorization,
    ToolName,
)
from cdd_agent.retrieval.ingestion import StructuredTable
from cdd_agent.tools.primary_research import PrimaryResearchTool
from cdd_agent.tools.retrieval_tools import DocumentRetrievalTool, MarketSearchTool
from cdd_agent.tools.structured_computation import StructuredComputationTool


@dataclass
class ToolBundle:
    """The tools one role may actually use on one engagement."""

    role: AgentRole
    document_retrieval: Optional[DocumentRetrievalTool] = None
    market_search: Optional[MarketSearchTool] = None
    computation: Optional[StructuredComputationTool] = None
    primary_research: Optional[PrimaryResearchTool] = None
    denied: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.denied is None:
            self.denied = {}

    def names(self) -> list[str]:
        return [
            name
            for name, tool in (
                ("document_retrieval", self.document_retrieval),
                ("market_search", self.market_search),
                ("structured_computation", self.computation),
                ("primary_research", self.primary_research),
            )
            if tool is not None
        ]

    def require(self, name: str) -> Any:
        tool = {
            "document_retrieval": self.document_retrieval,
            "market_search": self.market_search,
            "structured_computation": self.computation,
            "primary_research": self.primary_research,
        }.get(name)
        if tool is None:
            reason = self.denied.get(name, f"{self.role.value} has no {name} in scope")
            raise AuthorizationError(reason)
        return tool


class ToolRegistry:
    """Builds role-scoped bundles from one authorization decision set."""

    def __init__(
        self,
        engagement_id: str,
        authorization: ToolAuthorization,
        *,
        tables: Sequence[StructuredTable] = (),
    ) -> None:
        self.engagement_id = engagement_id
        self.authorization = authorization
        self.tables = list(tables)

    def bundle_for(self, role: AgentRole) -> ToolBundle:
        bundle = ToolBundle(role=role)
        for tool_name in (
            ToolName.DOCUMENT_RETRIEVAL,
            ToolName.MARKET_SEARCH,
            ToolName.STRUCTURED_COMPUTATION,
            ToolName.PRIMARY_RESEARCH,
        ):
            decision = self.authorization.check(role, tool_name)
            if not decision.allowed:
                bundle.denied[tool_name.value] = decision.reason
                continue
            if tool_name is ToolName.DOCUMENT_RETRIEVAL:
                bundle.document_retrieval = DocumentRetrievalTool(self.engagement_id)
            elif tool_name is ToolName.MARKET_SEARCH:
                bundle.market_search = MarketSearchTool()
            elif tool_name is ToolName.STRUCTURED_COMPUTATION:
                bundle.computation = StructuredComputationTool(self.tables)
            elif tool_name is ToolName.PRIMARY_RESEARCH:
                bundle.primary_research = PrimaryResearchTool(self.authorization, role)
        return bundle

    # ------------------------------------------------------- framework adapters
    def langchain_tools(self, role: AgentRole) -> list[Any]:
        """Expose the bundle as LangChain StructuredTools.

        Imported lazily so the core package stays importable (and testable) without
        LangChain installed.
        """
        from langchain_core.tools import StructuredTool

        bundle = self.bundle_for(role)
        tools: list[Any] = []

        if bundle.document_retrieval is not None:
            tools.append(
                StructuredTool.from_function(
                    func=_wrap_retrieval(bundle.document_retrieval),
                    name="document_retrieval",
                    description=DocumentRetrievalTool.description,
                )
            )
        if bundle.market_search is not None:
            tools.append(
                StructuredTool.from_function(
                    func=_wrap_market(bundle.market_search),
                    name="market_search",
                    description=MarketSearchTool.description,
                )
            )
        if bundle.computation is not None:
            tools.append(
                StructuredTool.from_function(
                    func=_wrap_computation(bundle.computation),
                    name="structured_computation",
                    description=StructuredComputationTool.description
                    + f" Available tables: {bundle.computation.available()}.",
                )
            )
        if bundle.primary_research is not None:
            tools.append(
                StructuredTool.from_function(
                    func=_wrap_research(bundle.primary_research),
                    name="primary_research",
                    description=PrimaryResearchTool.description,
                )
            )
        return tools


def _wrap_retrieval(tool: DocumentRetrievalTool) -> Callable[..., str]:
    def document_retrieval(query: str, doc_type: str = "") -> str:
        """Semantic search over the engagement's data room. Returns cited passages."""
        return tool(query, doc_type=doc_type or None).render()

    return document_retrieval


def _wrap_market(tool: MarketSearchTool) -> Callable[..., str]:
    def market_search(query: str, sub_sector: str = "") -> str:
        """Search the cross-engagement knowledge base. Returns cited passages."""
        return tool(query, sub_sector=sub_sector or None).render()

    return market_search


def _wrap_computation(tool: StructuredComputationTool) -> Callable[..., str]:
    def structured_computation(table: str, column: str, how: str = "sum") -> str:
        """Aggregate a numeric column of a parsed data-room table."""
        return tool.aggregate(table, column, how).render()

    return structured_computation


def _wrap_research(tool: PrimaryResearchTool) -> Callable[..., str]:
    def primary_research(
        hypothesis_id: str, contact_type: str, objective: str, sample_size: int = 8
    ) -> str:
        """Commission independently-sourced interviews, subject to Category F limits."""
        req = tool.commission(
            hypothesis_id=hypothesis_id,
            contact_type=contact_type,
            objective=objective,
            target_sample_size=sample_size,
        )
        return f"{req.id} commissioned. {req.methodology_disclosure()}"

    return primary_research
