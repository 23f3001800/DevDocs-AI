"""
Multi-provider LLM abstraction with automatic fallback.

Fallback order: Anthropic → Gemini → Grok (xAI)
Each provider only initialises if its API key is set.
If the primary provider fails (rate limit, credit exhaustion, outage),
the next available provider is tried automatically.

Usage:
    from app.llm_providers import llm_manager
    response = llm_manager.generate(system, user_message)
    async for token in llm_manager.stream(system, user_message):
        ...
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider."""

    text: str
    provider: str
    model: str


# ─── Anthropic Provider ───────────────────────────────────────
class AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        from anthropic import Anthropic, AsyncAnthropic

        self.client = Anthropic()
        self.async_client = AsyncAnthropic()
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    def generate(self, system: str, user_message: str, max_tokens: int = 1024) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return LLMResponse(
            text=resp.content[0].text.strip(),
            provider=self.name,
            model=self.model,
        )

    async def stream(self, system: str, user_message: str, max_tokens: int = 1024):
        async with self.async_client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


# ─── Gemini Provider ──────────────────────────────────────────
class GeminiProvider:
    name = "gemini"

    def __init__(self):
        from google import genai

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, system: str, user_message: str, max_tokens: int = 1024) -> LLMResponse:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        return LLMResponse(
            text=resp.text.strip(),
            provider=self.name,
            model=self.model,
        )

    async def stream(self, system: str, user_message: str, max_tokens: int = 1024):
        from google.genai import types

        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text


# ─── Grok Provider (xAI — OpenAI-compatible API) ─────────────
class GrokProvider:
    name = "grok"

    def __init__(self):
        from openai import AsyncOpenAI, OpenAI

        self.client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        self.model = os.getenv("GROK_MODEL", "grok-3-mini-fast")

    def generate(self, system: str, user_message: str, max_tokens: int = 1024) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        return LLMResponse(
            text=resp.choices[0].message.content.strip(),
            provider=self.name,
            model=self.model,
        )

    async def stream(self, system: str, user_message: str, max_tokens: int = 1024):
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# ─── Fallback Manager ────────────────────────────────────────
class LLMManager:
    """Manages multiple LLM providers with automatic fallback.

    Tries each provider in order. If one fails (rate limit, credit
    exhaustion, network error), transparently falls back to the next.
    """

    def __init__(self):
        self.providers = []
        self._init_providers()
        if not self.providers:
            # No real providers found – fall back to a simple mock for testing/CI
            log.warning("No LLM API keys set. Using MockProvider for offline testing.")
            self.providers.append(MockProvider())
        log.info(
            "LLM providers initialised: %s",
            " → ".join(p.name for p in self.providers),
        )

    def _init_providers(self):
        """Initialise only providers whose API keys are set."""
        provider_configs = [
            ("ANTHROPIC_API_KEY", AnthropicProvider),
            ("GOOGLE_API_KEY", GeminiProvider),
            ("XAI_API_KEY", GrokProvider),
        ]
        for env_var, cls in provider_configs:
            if os.getenv(env_var):
                try:
                    self.providers.append(cls())
                    log.info("✅ %s provider ready", cls.name)
                except Exception as e:
                    log.warning("⚠️  %s provider failed to init: %s", cls.name, e)

    @property
    def active_provider(self) -> str:
        """Name of the primary (first) provider."""
        return self.providers[0].name if self.providers else "none"

    def provider_names(self) -> list[str]:
        """List of all available provider names."""
        return [p.name for p in self.providers]

    def generate(self, system: str, user_message: str, max_tokens: int = 1024) -> LLMResponse:
        """Synchronous generation with automatic fallback."""
        errors = []
        for provider in self.providers:
            try:
                result = provider.generate(system, user_message, max_tokens)
                log.info("LLM response from %s (%s)", provider.name, result.model)
                return result
            except Exception as e:
                log.warning("Provider %s failed: %s — trying next", provider.name, e)
                errors.append((provider.name, str(e)))

        # All providers failed
        error_summary = "; ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(f"All LLM providers failed. Errors: {error_summary}")

    async def stream(self, system: str, user_message: str, max_tokens: int = 1024):
        """Async streaming with automatic fallback."""
        errors = []
        for provider in self.providers:
            try:
                async for token in provider.stream(system, user_message, max_tokens):
                    yield token
                return  # success — stop trying other providers
            except Exception as e:
                log.warning("Provider %s stream failed: %s — trying next", provider.name, e)
                errors.append((provider.name, str(e)))

        # All providers failed
        error_summary = "; ".join(f"{name}: {err}" for name, err in errors)
        yield f"\n\n⚠️ All LLM providers failed: {error_summary}"


# ── Singleton ─────────────────────────────────────────────────
llm_manager = LLMManager()

# Mock provider used when no real API keys are set (useful for CI/tests)
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

