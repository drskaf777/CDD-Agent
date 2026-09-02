"""Runtime configuration.

Every tunable named in the checkpoints is a setting here rather than a literal
buried in code, so the design parameters (beam width, prune threshold, tie band,
chunk size, top-k, similarity floor) are auditable in one place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="CDD_", extra="ignore", case_sensitive=False
    )

    # --- Models (Checkpoint 4.1: Generator and Critic are separate personas) ---
    model: str = "claude-opus-5"
    critic_model: str = "claude-opus-5"
    effort: str = "high"
    max_tokens: int = 16_000
    # Identity-linked API keys must name the workspace they act in; the API rejects
    # the request otherwise. Console -> Settings -> Workspaces, id looks like wrkspc_...
    workspace_id: str = ""
    # CrewAI execution traces for the Critic. On by default - they are the Critic's
    # own view of its reasoning, which is worth having next to ours.
    crewai_tracing: bool = True

    # --- Storage ---
    data_dir: Path = Path("./data")
    chroma_dir: Path = Path("./data/chroma")
    embeddings: str = "default"  # "default" (Chroma ONNX) | "hash" (deterministic)

    # --- Tree of Thought, Checkpoint 4.1 § 2.2-2.3 ---
    beam_width: int = 3          # 3 candidate framings at depth 1
    # The tree *artifact* reaches depth 2 (root -> Tier-1 hypotheses -> assumptions),
    # but the *search* expands one level only and is not iteratively deepened
    # (Checkpoint 4.1 s 2.2 vs s 2.3; Architecture v6.7 slide 2, "depth capped at 1").
    tree_max_depth: int = 2
    search_expansion_levels: int = 1
    tier1_min: int = 3           # a framing must yield >= 3 Tier-1 hypotheses
    tier1_max: int = 5           # ...and <= 5
    prune_threshold: float = 3.0  # average < 3/5 on soft criteria -> pruned
    tie_band: float = 0.5        # framings within 0.5 pts go to the user, not a reranker

    # --- Retrieval, Checkpoint 3.1 § 4 ---
    chunk_target_tokens: int = 650   # ~500-800 band
    chunk_min_tokens: int = 500
    chunk_max_tokens: int = 800
    chunk_overlap_ratio: float = 0.15
    top_k: int = 6                   # 5-8 band
    similarity_floor: float = 0.35   # below this -> "No Data", never a low-confidence match

    # --- ReAct evidence loop, Checkpoint 2.1 ---
    max_react_steps: int = 40
    max_auditor_rounds: int = 3      # Analyst <-> Risk Auditor feedback loop bound

    # --- Execution ---
    offline: bool = Field(default=False, description="Use scripted stub LLM responses.")

    @property
    def engagements_dir(self) -> Path:
        return self.data_dir / "engagements"

    @property
    def state_db(self) -> Path:
        return self.data_dir / "state.sqlite3"

    @property
    def knowledge_base_dir(self) -> Path:
        return self.data_dir / "knowledge_base"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.chroma_dir, self.engagements_dir, self.knowledge_base_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Test hook: re-read the environment."""
    get_settings.cache_clear()
