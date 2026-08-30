"""The four tool categories (Checkpoint 2.1, extended by Checkpoint 3.1).

Document Extraction & Retrieval and Market & Competitive Search are the two RAG tools,
querying the Data-Room and Knowledge-Base indexes respectively. Structured Computation
operates on parsed, schema'd data. Independent Primary Research commissions original
interviews rather than retrieving existing ones.
"""

from cdd_agent.tools.primary_research import (
    InterviewNote,
    PrimaryResearchTool,
    ResearchRequest,
)
from cdd_agent.tools.registry import ToolBundle, ToolRegistry
from cdd_agent.tools.retrieval_tools import (
    DocumentRetrievalTool,
    MarketSearchTool,
    RetrievalObservation,
)
from cdd_agent.tools.structured_computation import (
    ComputationError,
    ComputationResult,
    StructuredComputationTool,
)

__all__ = [
    "ComputationError", "ComputationResult", "DocumentRetrievalTool", "InterviewNote",
    "MarketSearchTool", "PrimaryResearchTool", "ResearchRequest", "RetrievalObservation",
    "StructuredComputationTool", "ToolBundle", "ToolRegistry",
]
