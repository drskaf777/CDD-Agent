"""Model construction for the two frameworks in play.

Checkpoint 4.1 s 2.4 assigns frameworks by role, and this module keeps that split
honest: the Generator and Controller get a LangChain chat model, the Critic gets a
CrewAI LLM. They are separate objects even when they name the same model, because the
whole point of the separation is that the Critic must not see the Generator's context.

Model default is `claude-opus-5`, configurable via CDD_MODEL.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from cdd_agent.config import get_settings


def get_chat_model(
    *, model: Optional[str] = None, temperature: Optional[float] = None
) -> Any:
    """LangChain chat model for the Generator, Controller, Analyst, and Synthesizer."""
    settings = get_settings()
    if settings.offline:
        from cdd_agent.llm.stub import StubChatModel

        return StubChatModel()

    from langchain_anthropic import ChatAnthropic

    kwargs: dict[str, Any] = {
        "model": model or settings.model,
        "max_tokens": settings.max_tokens,
    }
    # Sampling parameters are rejected on Opus 5 / Sonnet 5 and the 4.7+ family;
    # only pass one when a caller explicitly asks and the model still accepts it.
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatAnthropic(**kwargs)


def get_crew_llm(*, model: Optional[str] = None) -> Any:
    """CrewAI LLM for the Critic and the Risk Auditor personas.

    CrewAI routes through LiteLLM, so the model id takes the `anthropic/` prefix.
    """
    settings = get_settings()
    if settings.offline:
        from cdd_agent.llm.stub import StubCrewLLM

        return StubCrewLLM()

    try:
        from crewai import LLM
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "The Critic runs on CrewAI, which is an optional extra because it does "
            "not support Python 3.14. Install it with `pip install -e \".[critic]\"` "
            "on a 3.11-3.13 interpreter. It is not substituted with anything else: a "
            "Critic that is quietly not the Critic defeats the separation it exists "
            "for (Checkpoint 4.1 s 2.4)."
        ) from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        # CrewAI's native Anthropic provider reads the environment variable directly
        # and does not consult an `ant auth login` profile. Left alone it raises
        # ImportError("Error importing native provider: ANTHROPIC_API_KEY is required"),
        # which sends you looking for a broken install rather than a missing key.
        raise RuntimeError(
            "The Critic needs ANTHROPIC_API_KEY set in the environment. CrewAI's "
            "Anthropic provider reads that variable directly - unlike the rest of the "
            "pipeline, it will not pick up an `ant auth login` profile."
        )

    return LLM(model=f"anthropic/{model or settings.critic_model}")


def structured(model: Any, schema: type) -> Any:
    """Bind a Pydantic schema to a chat model, falling back for the offline stub."""
    if hasattr(model, "with_structured_output"):
        return model.with_structured_output(schema)
    raise TypeError(f"{type(model).__name__} does not support structured output")
