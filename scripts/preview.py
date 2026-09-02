"""Launch the web app for the preview pane.

`.claude/launch.json` lives at the workspace root, one level above this project, so a
server started from it would inherit that directory as its working directory. That
matters: `CDD_DATA_DIR`, `CDD_CHROMA_DIR` and `.env` are all resolved relative to the
working directory, so the app would look for engagements in the wrong place and quietly
create an empty store beside them.

This pins the working directory to the project root first, then fills in the defaults
that a 3.13 interpreter needs, then starts the server. Precedence is preserved: a value
already exported in the shell wins over `.env`, which wins over the defaults here.
"""

from __future__ import annotations

import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(PROJECT_ROOT)

    # Load .env before applying defaults, so anything the operator put there wins.
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass

    # CrewAI pins chromadb ~=1.1.0, so a 3.13 environment cannot read an index written
    # by the 3.14 one. Give it its own directories rather than letting it fail on a
    # version mismatch the first time it opens the store.
    if _running_under_critic_env():
        os.environ.setdefault("CDD_DATA_DIR", "./data313")
        os.environ.setdefault("CDD_CHROMA_DIR", "./data/chroma-313")

    # State the tracing preference so CrewAI never blocks on its interactive prompt.
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "true")

    import uvicorn

    host = os.environ.get("CDD_HOST", "127.0.0.1")
    port = int(os.environ.get("CDD_PORT", "8000"))
    print(f"CDD Agent on http://{host}:{port}  (cwd {PROJECT_ROOT})")
    print(f"  data dir   : {os.environ.get('CDD_DATA_DIR', './data')}")
    print(f"  offline    : {os.environ.get('CDD_OFFLINE', '0 (live)')}")
    print(f"  key present: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
    uvicorn.run("cdd_agent.web.api:app", host=host, port=port)


def _running_under_critic_env() -> bool:
    """True when this interpreter has CrewAI, and therefore the older chromadb."""
    import importlib.util

    return importlib.util.find_spec("crewai") is not None


if __name__ == "__main__":
    sys.exit(main())
