"""Pure-SDK LLM factory — no LangChain dependency.

Provider is selected by settings.LLM_PROVIDER.
Tier ("low"/"medium"/"high") resolves to the correct model for the active provider.

Usage:
    llm = get_llm("low")
    resp = llm.invoke("What is 2+2?")        # resp.content → "4"
    resp = await llm.ainvoke("What is 2+2?")  # async version
    obj  = llm.with_structured_output(MyModel).invoke("...")  # returns Pydantic model
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List, Type

from pydantic import BaseModel

from config.settings import settings
from src.utils.cost_tracker import cost_tracker


# ---------------------------------------------------------------------------
# Lightweight response wrapper (replaces LangChain AIMessage)
# ---------------------------------------------------------------------------

class LLMResponse:
    """Uniform response object — agents access .content like before."""
    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content

    def __str__(self) -> str:
        return self.content


# ---------------------------------------------------------------------------
# Provider-agnostic LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper that delegates to native SDKs per provider."""

    def __init__(self, provider: str, model: str, temperature: float = 1.0, **kwargs: Any):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._kwargs = kwargs
        self._structured_model: Type[BaseModel] | None = None

    def with_structured_output(self, model_class: Type[BaseModel], **_: Any) -> LLMClient:
        """Return a copy that parses output into a Pydantic model."""
        clone = LLMClient(self.provider, self.model, self.temperature, **self._kwargs)
        clone._structured_model = model_class
        return clone

    def invoke(self, messages: Any) -> Any:
        """Synchronous LLM call."""
        msgs = _normalize_messages(messages)
        dispatch = {
            "openai": self._call_openai,
            "anthropic": self._call_anthropic,
            "sarvam": self._call_sarvam,
            "gemini": self._call_gemini,
        }
        fn = dispatch.get(self.provider)
        if not fn:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        return fn(msgs)

    async def ainvoke(self, messages: Any) -> Any:
        """Async wrapper — runs sync call in thread pool."""
        return await asyncio.to_thread(self.invoke, messages)

    # ── OpenAI ────────────────────────────────────────────────────────────
    def _call_openai(self, messages: List[dict]) -> Any:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        if self._structured_model:
            resp = client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format=self._structured_model,
            )
            _track(self.model, resp.usage)
            return resp.choices[0].message.parsed

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        _track(self.model, resp.usage)
        return LLMResponse(resp.choices[0].message.content)

    # ── Anthropic ─────────────────────────────────────────────────────────
    def _call_anthropic(self, messages: List[dict]) -> Any:
        import os
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        system_text, user_msgs = _split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": user_msgs,
            "temperature": self.temperature,
            "max_tokens": self._kwargs.get("max_tokens", 4096),
        }
        if system_text:
            kwargs["system"] = system_text

        if self._structured_model:
            schema = json.dumps(self._structured_model.model_json_schema(), indent=2)
            _append_to_last(user_msgs, f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema}")

        resp = client.messages.create(**kwargs)
        _track(self.model, resp.usage)
        raw = resp.content[0].text

        if self._structured_model:
            return self._structured_model.model_validate_json(_strip_json_fences(raw))
        return LLMResponse(raw)

    # ── Sarvam (native SDK) ───────────────────────────────────────────────
    def _call_sarvam(self, messages: List[dict]) -> Any:
        import os
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

        if self._structured_model:
            schema = json.dumps(self._structured_model.model_json_schema(), indent=2)
            _append_to_last(messages, f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema}")

        resp = client.chat.completions(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self._kwargs.get("max_tokens", 4096),
        )
        _track(self.model, getattr(resp, "usage", None))
        raw = resp.choices[0].message.content
        finish_reason = getattr(resp.choices[0], "finish_reason", "unknown")

        if raw is None:
            raise RuntimeError(
                f"Sarvam returned empty content (finish_reason={finish_reason!r}). "
                "This may indicate the prompt was filtered, too long, or the API hit a limit."
            )

        if finish_reason == "length":
            print(f"   ⚠️  Sarvam hit max_tokens limit ({self._kwargs.get('max_tokens', 4096)}). Response may be truncated.")

        if self._structured_model:
            return self._structured_model.model_validate_json(_strip_json_fences(raw))
        return LLMResponse(raw)

    # ── Gemini (Vertex AI) ────────────────────────────────────────────────
    def _call_gemini(self, messages: List[dict]) -> Any:
        from google import genai
        from google.genai import types

        client = genai.Client(
            vertexai=True,
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )

        system_text, user_msgs = _split_system(messages)
        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part.from_text(m["content"])],
            )
            for m in user_msgs
        ]

        config_kwargs: dict[str, Any] = {"temperature": self.temperature}
        if system_text:
            config_kwargs["system_instruction"] = system_text
        if self._structured_model:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = self._structured_model

        resp = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        usage = resp.usage_metadata
        if usage:
            cost_tracker.track(
                self.model,
                getattr(usage, "prompt_token_count", 0),
                getattr(usage, "candidates_token_count", 0),
            )

        raw = resp.text
        if self._structured_model:
            return self._structured_model.model_validate_json(_strip_json_fences(raw))
        return LLMResponse(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_messages(messages: Any) -> list[dict]:
    """Accept str, list[dict], or LangChain-style message objects."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    result = []
    for m in messages:
        if isinstance(m, dict):
            result.append(m)
        elif hasattr(m, "content") and hasattr(m, "type"):
            role_map = {"human": "user", "system": "system", "ai": "assistant"}
            result.append({"role": role_map.get(m.type, m.type), "content": m.content})
        else:
            result.append({"role": "user", "content": str(m)})
    return result


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Separate system message (Anthropic / Gemini need it outside messages)."""
    system = None
    rest = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            rest.append(m)
    return system, rest


def _append_to_last(messages: list[dict], text: str) -> None:
    """Append text to the last user message (for JSON schema instruction)."""
    for m in reversed(messages):
        if m["role"] == "user":
            m["content"] += text
            return
    messages.append({"role": "user", "content": text})


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```) around JSON."""
    import re
    text = text.strip()
    # Match ```json\n...\n``` or ```\n...\n```
    match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```\s*$', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _track(model: str, usage: Any) -> None:
    """Extract token counts from any SDK response and record."""
    if usage is None:
        return
    inp = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
    cost_tracker.track(model, inp, out)


def _resolve_model(provider: str, tier: str) -> str:
    """Map tier name → actual model name for the given provider."""
    attr = f"{provider.upper()}_MODEL_{tier.upper()}"
    model = getattr(settings, attr, None)
    if not model:
        raise ValueError(f"No model configured for {provider}/{tier} (settings.{attr})")
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_llm(tier: str = "low", temperature: float = 1.0, **kwargs: Any) -> LLMClient:
    """Return a configured LLM client for the active provider.

    Args:
        tier: "low", "medium", or "high"
        temperature: LLM temperature
    """
    provider = settings.LLM_PROVIDER
    model = _resolve_model(provider, tier)
    return LLMClient(provider=provider, model=model, temperature=temperature, **kwargs)

