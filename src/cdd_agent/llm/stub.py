"""Offline mode.

`CDD_OFFLINE=1` runs the full pipeline with no API calls, so the orchestration,
guardrails, retrieval plumbing, and artifact contracts can be exercised end to end in
CI or on a machine with no credentials.

Offline behaviour is *not* a fake language model. Each agent has a deterministic,
rule-based path (`_offline_*` methods) built from the knowledge modules - the catalogue,
the taxonomy, the four-question markers. Those paths produce structurally valid
artifacts with real citations from the real indexes; what they do not produce is
judgment. Output from an offline run is labelled as such and must never be presented
as diligence.

These stub classes exist only to fail loudly if some code path reaches for a model in
offline mode, rather than silently producing a plausible-looking answer.
"""

from __future__ import annotations

from typing import Any


class OfflineModeError(RuntimeError):
    """A model call was attempted while CDD_OFFLINE=1."""


_MESSAGE = (
    "Model call attempted in offline mode. Offline runs use each agent's deterministic "
    "path; if you reached this, that path is missing for the step you just ran. Unset "
    "CDD_OFFLINE (and set ANTHROPIC_API_KEY or run `ant auth login`) to use the model."
)


class StubChatModel:
    """Stands in for the LangChain chat model in offline mode."""

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        raise OfflineModeError(_MESSAGE)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "StubChatModel":
        return self

    def bind_tools(self, *args: Any, **kwargs: Any) -> "StubChatModel":
        return self

    def __or__(self, other: Any) -> "StubChatModel":
        return self


class StubCrewLLM(StubChatModel):
    """Stands in for the CrewAI LLM in offline mode."""


def is_stub(model: Any) -> bool:
    return isinstance(model, StubChatModel)
