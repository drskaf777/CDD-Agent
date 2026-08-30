"""State access protocol - the seam MCP plugs into.

Checkpoint 4.1 s 2.4 is precise about the split, and this module preserves it: the
*store* holds the branches and scores (it is the same Long-Term Memory persistence
layer, not new infrastructure); *MCP* is layered on top solely to expose that store's
read/write operations as one protocol both the LangChain generator/controller and the
CrewAI critic can call, since those roles otherwise sit in separate frameworks with no
shared interface.

Two implementations satisfy the same protocol:

* ``LocalStateAccess`` - in-process, used when everything runs in one interpreter.
* ``MCPStateAccess`` - talks to ``cdd_agent.state.mcp_server`` over stdio, used when
  the Critic runs out-of-process (the case the MCP layer actually exists for).

Because both satisfy ``StateAccess``, the Thesis Architect does not know or care which
is in play, which is the point of putting a protocol here rather than a client.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol, runtime_checkable

from cdd_agent.state.store import Collection, StateStore


@runtime_checkable
class StateAccess(Protocol):
    """The four operations the ToT search and the agent hand-offs need."""

    def read(self, engagement_id: str, collection: str, key: str) -> Optional[dict[str, Any]]:
        ...

    def write(
        self, engagement_id: str, collection: str, key: str, document: Any, *, agent: str
    ) -> int:
        ...

    def list_keys(self, engagement_id: str, collection: str) -> list[str]:
        ...

    def list_documents(
        self, engagement_id: str, collection: str
    ) -> list[tuple[str, dict[str, Any]]]:
        ...


class LocalStateAccess:
    """Direct, in-process access to the state store."""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def read(self, engagement_id: str, collection: str, key: str) -> Optional[dict[str, Any]]:
        return self.store.get(engagement_id, collection, key)

    def write(
        self, engagement_id: str, collection: str, key: str, document: Any, *, agent: str
    ) -> int:
        return self.store.put(engagement_id, collection, key, document, agent=agent)

    def list_keys(self, engagement_id: str, collection: str) -> list[str]:
        return [k for k, _ in self.store.list(engagement_id, collection)]

    def list_documents(
        self, engagement_id: str, collection: str
    ) -> list[tuple[str, dict[str, Any]]]:
        return self.store.list(engagement_id, collection)


class MCPStateAccess:
    """Access the same store through the MCP protocol.

    Used when the CrewAI Critic runs in a separate process from the LangChain
    Generator/Controller. Requires an already-connected ``mcp.ClientSession``; see
    ``cdd_agent.state.mcp_server`` for the server side and ``connect_stdio`` below
    for the usual way to obtain one.
    """

    def __init__(self, session: Any) -> None:
        # Typed as Any to keep `mcp` an optional import at module load time.
        self._session = session

    def _call(self, tool: str, **arguments: Any) -> Any:
        import anyio

        async def _run() -> Any:
            result = await self._session.call_tool(tool, arguments)
            if not result.content:
                return None
            text = getattr(result.content[0], "text", None)
            return json.loads(text) if text else None

        return anyio.from_thread.run(_run) if _in_worker_thread() else anyio.run(_run)

    def read(self, engagement_id: str, collection: str, key: str) -> Optional[dict[str, Any]]:
        return self._call(
            "state_read", engagement_id=engagement_id, collection=collection, key=key
        )

    def write(
        self, engagement_id: str, collection: str, key: str, document: Any, *, agent: str
    ) -> int:
        body = document.model_dump(mode="json") if hasattr(document, "model_dump") else document
        return int(
            self._call(
                "state_write",
                engagement_id=engagement_id,
                collection=collection,
                key=key,
                document=body,
                agent=agent,
            )
            or 0
        )

    def list_keys(self, engagement_id: str, collection: str) -> list[str]:
        return list(
            self._call("state_list_keys", engagement_id=engagement_id, collection=collection)
            or []
        )

    def list_documents(
        self, engagement_id: str, collection: str
    ) -> list[tuple[str, dict[str, Any]]]:
        docs = self._call(
            "state_list_documents", engagement_id=engagement_id, collection=collection
        )
        return [(d["key"], d["document"]) for d in (docs or [])]


def _in_worker_thread() -> bool:
    try:
        import anyio.from_thread  # noqa: F401
        import sniffio

        sniffio.current_async_library()
        return True
    except Exception:
        return False


def default_access(store: StateStore | None = None) -> StateAccess:
    """The in-process default. Swap for MCPStateAccess to cross a process boundary."""
    return LocalStateAccess(store or StateStore())


__all__ = [
    "Collection",
    "LocalStateAccess",
    "MCPStateAccess",
    "StateAccess",
    "default_access",
]
