"""Test fixtures.

Every test runs against a temp data directory with the deterministic hashing embedder
and offline mode on, so the suite needs no credentials and makes no network calls.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from cdd_agent.config import get_settings, reset_settings_cache


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CDD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CDD_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CDD_EMBEDDINGS", "hash")
    monkeypatch.setenv("CDD_OFFLINE", "1")
    reset_settings_cache()
    settings = get_settings()
    settings.ensure_dirs()
    yield settings
    reset_settings_cache()


@pytest.fixture
def store():
    from cdd_agent.state.store import StateStore

    s = StateStore()
    yield s
    s.close()


@pytest.fixture
def profile():
    from cdd_agent.schemas.deal_profile import (
        AccessConstraints,
        BuyerProfile,
        DealProfile,
        DealStage,
        InvestmentThesis,
        SectorDefinition,
        TargetIdentification,
        VDRAccess,
    )

    return DealProfile(
        engagement_id="test-eng",
        created_by="test",
        target=TargetIdentification(
            legal_name="Sentinel Secure Ltd",
            deal_stage=DealStage.SIGNED_LOI,
            ic_date=_dt.date.today() + _dt.timedelta(days=45),
        ),
        sector=SectorDefinition(
            sub_sector="B2B cybersecurity SaaS", revenue_is_recurring=True
        ),
        thesis=InvestmentThesis(
            one_sentence_thesis="Cross-sell an identity product into the installed base",
            critical_model_assumptions=["NRR stays above 115%", "30% attach in 24 months"],
        ),
        buyer=BuyerProfile(
            decision_criteria=[
                "cash-flow stability",
                "growth optionality",
                "proprietary technology",
            ]
        ),
        access=AccessConstraints(
            vdr_access=VDRAccess.FILE_UPLOAD,
            customer_contact_permitted=True,
            top5_customer_contact_permitted_pre_signing=False,
            competitor_contact_permitted=False,
        ),
    )


@pytest.fixture
def tree(profile):
    """A four-question-complete tree, used wherever the search itself is not under test."""
    from cdd_agent.schemas.common import Tier
    from cdd_agent.schemas.hypothesis import Hypothesis, HypothesisTree

    return HypothesisTree(
        engagement_id=profile.engagement_id,
        created_by="test",
        branch_id="growth",
        framing_label="growth-led",
        root_thesis=profile.thesis.one_sentence_thesis,
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="The addressable market is growing at the rate the base case assumes",
                tier=Tier.DEAL_CRITICAL,
                depth=1,
                required_evidence=["Third-party market sizing with stated methodology"],
            ),
            Hypothesis(
                id="H2",
                statement="The target keeps winning share against competitive substitutes",
                tier=Tier.DEAL_CRITICAL,
                depth=1,
                required_evidence=["Win/loss log with reasons and churn by cohort"],
            ),
            Hypothesis(
                id="H3",
                statement="Unit economics hold as mix shifts: CAC payback and gross margin",
                tier=Tier.DEAL_CRITICAL,
                depth=1,
                required_evidence=["CAC payback by cohort and gross margin bridge"],
            ),
            Hypothesis(
                id="H4",
                statement="No customer concentration or contract step-down would break the plan",
                tier=Tier.DEAL_CRITICAL,
                depth=1,
                required_evidence=["Top-20 contracts with renewal dates and step-downs"],
            ),
        ],
    )


@pytest.fixture
def context(store, profile):
    """An engagement with a saved profile, for tests that exercise a whole agent."""
    from cdd_agent.agents.base import AgentContext

    ctx = AgentContext.create(profile.engagement_id, store=store, profile=profile)
    ctx.memory.save_deal_profile(profile, agent="test")
    return ctx
