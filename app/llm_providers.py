"""LLM abstraction. Gemini is the single supported provider.

The manager keeps chain.py behind a stable generate/stream interface and
provides the MockProvider path for CI/offline work — which is refused in
production (see `_init_providers`) so a missing key fails loud instead of
silently serving mock text behind a green /health.
"""

import logging
from dataclasses import dataclass

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Normalised response, so vendor payload shapes never leak inward."""

    text: str
    provider: str
    model: str


# ─── Gemini Provider ──────────────────────────────────────────
class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None = None):
        # Imported lazily so the SDK is only loaded when actually used.
        from google import genai

        settings = get_settings()
        # BYOK: api_key (when given) is scoped to this one throwaway client —
        # never written back to settings or any shared state.
        self.client = genai.Client(api_key=api_key or settings.google_api_key)
        self.model = settings.gemini_model

    def _config(self, system: str, max_tokens: int):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

    def generate(self, system: str, user_message: str, max_tokens: int = 1024) -> LLMResponse:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=self._config(system, max_tokens),
        )
        return LLMResponse(
            text=(resp.text or "").strip(),
            provider=self.name,
            model=self.model,
        )

    async def stream(self, system: str, user_message: str, max_tokens: int = 1024):
        # google-genai's sync stream is a blocking generator; the async client
        # yields without occupying the event loop between chunks.
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=user_message,
            config=self._config(system, max_tokens),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text


# ─── OpenRouter Provider ──────────────────────────────────────
class OpenRouterProvider:
    name = "openrouter"

    def __init__(self, api_key: str | None = None):
        import openai
        settings = get_settings()
        self.client = openai.OpenAI(
            api_key=api_key or settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key or settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = settings.openrouter_model

    def generate(self, system: str, user_message: str, max_tokens: int = 1024) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            provider=self.name,
            model=self.model,
        )

    async def stream(self, system: str, user_message: str, max_tokens: int = 1024):
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ─── Mock Provider (no API key — CI / offline only) ───────────
class MockProvider:
    name = "mock"

    def __init__(self):
        self.model = "mock-model"

    def generate(self, system: str, user_message: str, max_tokens: int = 1024) -> LLMResponse:
        return LLMResponse(
            text="Mock response (testing mode)",
            provider=self.name,
            model=self.model,
        )

    async def stream(self, system: str, user_message: str, max_tokens: int = 1024):
        yield "Mock response (testing mode)"


# ─── Manager ─────────────────────────────────────────────────
class LLMManager:
    """Holds the provider chain and applies fallback semantics."""

    def __init__(self):
        self.providers = []
        self._init_providers()
        log.info("LLM providers initialised: %s", " → ".join(p.name for p in self.providers))

    def _init_providers(self):
        settings = get_settings()
        if settings.google_api_key:
            try:
                self.providers.append(GeminiProvider())
                log.info("Gemini provider ready (%s)", settings.gemini_model)
            except Exception as e:
                log.warning("Gemini provider failed to init: %s", e)

        if settings.openrouter_api_key:
            try:
                self.providers.append(OpenRouterProvider())
                log.info("OpenRouter provider ready (%s)", settings.openrouter_model)
            except Exception as e:
                log.warning("OpenRouter provider failed to init: %s", e)

        if self.providers:
            return

        # Fail closed: a typo'd GOOGLE_API_KEY in production must not degrade
        # into serving mock text with a 200 and a green /health.
        if settings.is_production:
            raise RuntimeError(
                "No LLM provider available: GOOGLE_API_KEY is unset or the Gemini "
                "client failed to initialise, and APP_ENV=production forbids the "
                "mock provider. Refusing to start."
            )
        log.warning("No GOOGLE_API_KEY set — using MockProvider for offline/CI testing.")
        self.providers.append(MockProvider())

    @property
    def active_provider(self) -> str:
        return self.providers[0].name if self.providers else "none"

    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    def _byok_provider(self, api_key: str) -> GeminiProvider:
        """One-off Gemini provider bound to the caller's own key. A separate
        method so tests can monkeypatch just this construction step."""
        return GeminiProvider(api_key=api_key)

    def generate(
        self, system: str, user_message: str, max_tokens: int = 1024, api_key: str | None = None
    ) -> LLMResponse:
        """`api_key` (BYOK): if given, use only a per-request Gemini client
        built from that key — no fallback chain."""
        if api_key:
            provider = self._byok_provider(api_key)
            result = provider.generate(system, user_message, max_tokens)
            log.info("LLM response from %s (BYOK)", provider.name)
            return result

        errors = []
        for provider in self.providers:
            try:
                result = provider.generate(system, user_message, max_tokens)
                log.info("LLM response from %s (%s)", provider.name, result.model)
                return result
            except Exception as e:
                log.warning("Provider %s failed: %s — trying next", provider.name, e)
                errors.append((provider.name, str(e)))

        raise RuntimeError(
            "All LLM providers failed. Errors: "
            + "; ".join(f"{name}: {err}" for name, err in errors)
        )

    async def stream(
        self, system: str, user_message: str, max_tokens: int = 1024, api_key: str | None = None
    ):
        """Async streaming with fallback — but only *before* the first token.
        The `yielded` guard stops a mid-stream failure from restarting on
        another provider and duplicating already-sent output.

        `api_key` (BYOK): if given, stream only from a per-request Gemini
        client built from that key — no fallback chain.
        """
        if api_key:
            provider = self._byok_provider(api_key)
            async for token in provider.stream(system, user_message, max_tokens):
                yield token
            return

        errors = []
        yielded = False
        for provider in self.providers:
            try:
                async for token in provider.stream(system, user_message, max_tokens):
                    yielded = True
                    yield token
                return  # success
            except Exception as e:
                log.warning("Provider %s stream failed: %s", provider.name, e)
                errors.append((provider.name, str(e)))
                if yielded:
                    raise RuntimeError(
                        f"Stream interrupted after partial output ({provider.name}: {e})"
                    ) from e

        raise RuntimeError(
            "All LLM providers failed. Errors: "
            + "; ".join(f"{name}: {err}" for name, err in errors)
        )


# ── Singleton ─────────────────────────────────────────────────
llm_manager = LLMManager()
