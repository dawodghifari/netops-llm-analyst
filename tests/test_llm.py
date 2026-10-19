"""Provider selection in the vendor-agnostic client. No API calls are made."""

import pytest

from src import llm


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)


def test_no_key_is_a_clear_error():
    with pytest.raises(RuntimeError, match="No LLM API key"):
        llm.LLMClient()


def test_picks_anthropic_from_its_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    client = llm.LLMClient()
    assert client.provider == "anthropic"
    assert client.model == llm.DEFAULT_MODELS["anthropic"]


def test_picks_openai_from_its_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert llm.LLMClient().provider == "openai"


def test_provider_and_model_can_be_forced(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    client = llm.LLMClient()
    assert (client.provider, client.model) == ("openai", "some-model")
