"""Shared fixtures: a small generated database, and a scripted fake LLM.

Nothing here needs an API key or network access.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import db  # noqa: E402


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_data", ROOT / "data" / "generate_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def netops_db(tmp_path_factory):
    """Generate the synthetic dataset once, into a temp file, and point db at it."""
    path = tmp_path_factory.mktemp("data") / "netops.db"
    gen = _load_generator()
    gen.DB_PATH = str(path)
    gen.generate(days=180, seed=7)
    original = db.DB_PATH
    db.DB_PATH = str(path)
    yield path
    db.DB_PATH = original


class FakeLLM:
    """Returns scripted replies in order and records every prompt it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, max_tokens=1024):
        self.calls.append({"system": system, "user": user})
        return self.replies.pop(0)


@pytest.fixture
def fake_llm():
    return FakeLLM
