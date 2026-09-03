"""Static domain knowledge lifted from the design specification.

Everything here is non-client-confidential and therefore also the seed corpus for the
cross-engagement Knowledge-Base Index (Checkpoint 3.1 s 2): sub-sector diagnostic
frameworks, the standing risk taxonomy, and the enhanced master outline.
"""

from cdd_agent.knowledge.data_request_catalog import (
    ADDONS_BY_MODULE,
    CATEGORIES,
    UNIVERSAL_CATALOG,
    CatalogItem,
)
from cdd_agent.knowledge.four_question_test import (
    FOUR_QUESTIONS,
    QUESTION_KEYS,
    classify,
)
from cdd_agent.knowledge.intake_questions import INTAKE_PROTOCOL, IntakeCategory
from cdd_agent.knowledge.outline import (
    PREBUILT_MODULES,
    UNIVERSAL_OUTLINE,
    module_for_sub_sector,
    tailored_outline,
)
from cdd_agent.knowledge.risk_taxonomy import TAXONOMY, Screen, applicable_categories

__all__ = [
    "ADDONS_BY_MODULE",
    "CATEGORIES",
    "FOUR_QUESTIONS",
    "INTAKE_PROTOCOL",
    "PREBUILT_MODULES",
    "QUESTION_KEYS",
    "TAXONOMY",
    "UNIVERSAL_CATALOG",
    "UNIVERSAL_OUTLINE",
    "CatalogItem",
    "IntakeCategory",
    "Screen",
    "applicable_categories",
    "classify",
    "module_for_sub_sector",
    "tailored_outline",
]
