"""Strict, provider-neutral experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any

from .errors import ConfigurationError


TRANSPORTS = {"openai_chat", "openai_responses"}
CURRENCIES = {"CNY", "USD"}


def _number(table: dict[str, Any], key: str, *, minimum: float = 0.0) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{key} must be numeric")
    result = float(value)
    if result < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}")
    return result


def _positive_int(table: dict[str, Any], key: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{key} must be a positive integer")
    return value


@dataclass(frozen=True)
class Pricing:
    currency: str
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float
    cache_creation_per_million: float = 0.0

    def cost(
        self,
        *,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
    ) -> float:
        values = (input_tokens, cached_input_tokens, output_tokens, cache_creation_tokens)
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise ConfigurationError("token counts must be non-negative integers")
        return (
            input_tokens * self.input_per_million
            + cached_input_tokens * self.cached_input_per_million
            + output_tokens * self.output_per_million
            + cache_creation_tokens * self.cache_creation_per_million
        ) / 1_000_000.0


@dataclass(frozen=True)
class SafeRelease:
    enabled: bool
    cost_threshold: float | None
    max_turns: int
    fallback_turn: int
    allow_tools: bool


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    model_id: str
    transport: str
    base_url_env: str
    api_key_env: str
    context_window: int
    reasoning_effort: str | None
    pricing: Pricing
    safe_release: SafeRelease

    @property
    def base_url(self) -> str | None:
        return os.getenv(self.base_url_env)

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


def load_model_profile(path: str | Path) -> ModelProfile:
    source = Path(path)
    with source.open("rb") as stream:
        payload = tomllib.load(stream)
    profile = payload.get("profile")
    prices = payload.get("pricing")
    release = payload.get("safe_release")
    if not all(isinstance(value, dict) for value in (profile, prices, release)):
        raise ConfigurationError(f"profile, pricing and safe_release tables are required: {source}")
    transport = str(profile.get("transport", ""))
    if transport not in TRANSPORTS:
        raise ConfigurationError(f"unsupported transport: {transport}")
    currency = str(prices.get("currency", ""))
    if currency not in CURRENCIES:
        raise ConfigurationError(f"unsupported currency: {currency}")
    enabled = release.get("enabled")
    if not isinstance(enabled, bool):
        raise ConfigurationError("safe_release.enabled must be boolean")
    threshold = release.get("cost_threshold")
    if enabled:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ConfigurationError("enabled safe release requires a positive cost_threshold")
        threshold = float(threshold)
    elif threshold is not None:
        raise ConfigurationError("disabled safe release must use cost_threshold = null")
    max_turns = _positive_int(release, "max_turns")
    fallback_turn = _positive_int(release, "fallback_turn")
    if fallback_turn > max_turns:
        raise ConfigurationError("fallback_turn cannot exceed max_turns")
    allow_tools = release.get("allow_tools")
    if not isinstance(allow_tools, bool):
        raise ConfigurationError("safe_release.allow_tools must be boolean")
    reasoning = profile.get("reasoning_effort")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ConfigurationError("reasoning_effort must be a string or null")
    return ModelProfile(
        profile_id=str(profile["id"]),
        model_id=str(profile["model_id"]),
        transport=transport,
        base_url_env=str(profile["base_url_env"]),
        api_key_env=str(profile["api_key_env"]),
        context_window=_positive_int(profile, "context_window"),
        reasoning_effort=reasoning,
        pricing=Pricing(
            currency=currency,
            input_per_million=_number(prices, "input_per_million"),
            cached_input_per_million=_number(prices, "cached_input_per_million"),
            output_per_million=_number(prices, "output_per_million"),
            cache_creation_per_million=_number(prices, "cache_creation_per_million"),
        ),
        safe_release=SafeRelease(
            enabled=enabled,
            cost_threshold=threshold,
            max_turns=max_turns,
            fallback_turn=fallback_turn,
            allow_tools=allow_tools,
        ),
    )
