"""
Pytest configuration for CatBoost + DAGMA Version 2 tests.
Ensures project roots are properly injected into sys.path.
"""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
CATBOOST_DAGMA_DIR = TESTS_DIR.parent
PROJECT_ROOT = CATBOOST_DAGMA_DIR.parent

for p in [str(PROJECT_ROOT), str(CATBOOST_DAGMA_DIR), str(PROJECT_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)
