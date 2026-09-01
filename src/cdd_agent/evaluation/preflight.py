"""Preflight checks for a live, model-backed run.

Everything that can silently ruin a live demo, checked in the order it would bite:
credentials, both model paths, the vector index, and the seeded knowledge base.

The point is to fail here rather than three phases into a recording. Each check
reports what to do about it, not just that it failed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cdd_agent.config import get_settings


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    fatal: bool = True

    @property
    def mark(self) -> str:
        return "OK  " if self.ok else ("FAIL" if self.fatal else "WARN")


@dataclass
class Preflight:
    checks: list[Check] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(c.ok for c in self.checks if c.fatal)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check


def _check_mode(pf: Preflight) -> bool:
    settings = get_settings()
    pf.add(Check(
        "run mode",
        not settings.offline,
        "offline" if settings.offline else f"live, model {settings.model}",
        fix="set CDD_OFFLINE=0 (PowerShell: $env:CDD_OFFLINE=\"0\")",
    ))
    return not settings.offline


def _check_credentials(pf: Preflight) -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    present = bool(key.strip())
    # Never print the key. Its length and prefix are enough to spot a truncated paste.
    shape = f"set, {len(key)} chars, starts {key[:7]}…" if present else "not set"
    pf.add(Check(
        "ANTHROPIC_API_KEY",
        present,
        shape,
        fix=(
            'PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-…"  (this session only), or '
            "add it to .env. CrewAI's provider reads this variable directly and will "
            "not use an `ant auth login` profile, so the variable is required even if "
            "the CLI is already authenticated."
        ),
    ))
    return present


def _check_langchain(pf: Preflight, live_call: bool) -> None:
    try:
        from cdd_agent.llm.models import get_chat_model

        model = get_chat_model()
    except Exception as exc:
        pf.add(Check("Generator / Analyst model", False, f"{type(exc).__name__}: {exc}",
                     fix="check langchain-anthropic is installed"))
        return
    pf.add(Check("Generator / Analyst model", True,
                 f"{type(model).__name__}(model={getattr(model, 'model', '?')})"))

    if not live_call:
        pf.add(Check("live model call", True, "skipped (--no-call)", fatal=False))
        return
    try:
        # One tiny call. This is the only check that proves the credential actually
        # works rather than merely being present.
        reply = model.invoke("Reply with the single word: ready")
        text = getattr(reply, "content", "")
        text = text if isinstance(text, str) else str(text)
        pf.add(Check("live model call", True, f"answered {text.strip()[:40]!r}"))
    except Exception as exc:
        pf.add(Check("live model call", False, f"{type(exc).__name__}: {str(exc)[:200]}",
                     fix="check the key is valid and has credit"))


def _check_critic(pf: Preflight) -> None:
    try:
        import crewai  # noqa: F401
    except ImportError:
        pf.add(Check(
            "Critic (CrewAI)", False, "crewai not installed on this interpreter",
            fix='use a Python 3.11-3.13 environment and `pip install -e ".[dev,critic]"` '
                "- CrewAI publishes no wheels for 3.14",
        ))
        return
    try:
        from cdd_agent.llm.models import get_crew_llm

        llm = get_crew_llm()
    except Exception as exc:
        pf.add(Check("Critic (CrewAI)", False, f"{type(exc).__name__}: {exc}"))
        return
    pf.add(Check("Critic (CrewAI)", True,
                 f"crewai {crewai.__version__}, {type(llm).__name__}"
                 f"(model={getattr(llm, 'model', '?')})"))


def _check_index(pf: Preflight) -> None:
    settings = get_settings()
    try:
        import chromadb

        from cdd_agent.retrieval.indexes import KnowledgeBaseIndex

        kb = KnowledgeBaseIndex()
        count = kb.count()
    except Exception as exc:
        pf.add(Check(
            "vector index", False, f"{type(exc).__name__}: {str(exc)[:200]}",
            fix=f"delete {settings.chroma_dir} and re-run `cdd seed-kb`",
        ))
        return
    pf.add(Check("vector index", True,
                 f"chromadb {chromadb.__version__}, dir {settings.chroma_dir}"))
    pf.add(Check(
        "knowledge base seeded", count > 0, f"{count} reference chunk(s)",
        fix="run `cdd seed-kb`",
    ))


def _check_data_room(pf: Preflight, data_room: Optional[Path]) -> None:
    if data_room is None:
        return
    exists = data_room.is_dir()
    files = len([p for p in data_room.rglob("*") if p.is_file()]) if exists else 0
    pf.add(Check(
        "data room", exists and files > 0,
        f"{files} file(s) at {data_room}" if exists else f"not a directory: {data_room}",
        fix="point --data-room at the folder holding the deal's documents",
    ))


def run_preflight(
    *, live_call: bool = True, data_room: Optional[Path] = None
) -> Preflight:
    """Run every check. Returns the results rather than raising."""
    pf = Preflight()
    live = _check_mode(pf)
    has_key = _check_credentials(pf)
    if live and has_key:
        _check_langchain(pf, live_call)
        _check_critic(pf)
    else:
        pf.add(Check("model paths", False, "skipped - fix the mode and key first"))
    _check_index(pf)
    _check_data_room(pf, data_room)
    return pf
