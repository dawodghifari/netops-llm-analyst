"""
Provider-agnostic LLM client.

Supports Anthropic (Claude) and OpenAI (GPT). The provider is chosen automatically
from whichever API key is present, or forced with LLM_PROVIDER=anthropic|openai.

Env vars:
    ANTHROPIC_API_KEY   -> uses Anthropic
    OPENAI_API_KEY      -> uses OpenAI
    LLM_PROVIDER        -> optional override ("anthropic" or "openai")
    LLM_MODEL           -> optional model override

Keeping this behind one tiny interface means the rest of the app never cares which
vendor is in use — a clean abstraction that's easy to talk about in an interview.
"""

import os

try:  # load .env if python-dotenv is installed (optional dependency at runtime)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

DEFAULT_MODELS = {
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o-mini",
}


def _resolve_provider() -> str:
    forced = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if forced in DEFAULT_MODELS:
        return forced
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
        "(copy .env.example to .env and fill it in)."
    )


class LLMClient:
    """One method, `complete(system, user)`, returning the model's text."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or _resolve_provider()
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS[self.provider]

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if self.provider == "anthropic":
            return self._anthropic(system, user, max_tokens)
        return self._openai(system, user, max_tokens)

    def _anthropic(self, system: str, user: str, max_tokens: int) -> str:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()

    def _openai(self, system: str, user: str, max_tokens: int) -> str:
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
