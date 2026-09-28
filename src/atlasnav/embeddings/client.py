"""Provider-neutral OpenAI-compatible embedding client with bounded retries."""

from __future__ import annotations

import asyncio
import base64
import random
from dataclasses import dataclass
from typing import Any, Sequence

import httpx
import numpy as np


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[np.ndarray]
    input_tokens: int
    model: str
    attempts: int


def decode_vector(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        raw = base64.b64decode(value, validate=True)
        if len(raw) % 4:
            raise ValueError("base64 vector is not float32 aligned")
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    raise TypeError(f"unsupported embedding representation: {type(value).__name__}")


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("embedding has zero or non-finite norm")
    return vector / norm


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 180.0,
        maximum_retries: int = 12,
    ) -> None:
        if not base_url or not api_key or not model:
            raise ValueError("embedding base URL, API key, and model are required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.maximum_retries = maximum_retries
        self.client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self.client.aclose()

    async def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts:
            raise ValueError("embedding batch is empty")
        for attempt in range(1, self.maximum_retries + 1):
            try:
                response = await self.client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": list(texts), "encoding_format": "float"},
                )
                if response.status_code in RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError("retryable status", request=response.request, response=response)
                response.raise_for_status()
                payload = response.json()
                rows = sorted(payload.get("data") or [], key=lambda row: int(row.get("index", 0)))
                if len(rows) != len(texts):
                    raise RuntimeError("embedding response cardinality mismatch")
                vectors = [normalize(decode_vector(row["embedding"])) for row in rows]
                tokens = int((payload.get("usage") or {}).get("prompt_tokens") or (payload.get("usage") or {}).get("total_tokens") or 0)
                return EmbeddingBatch(vectors, tokens, str(payload.get("model") or self.model), attempt)
            except (httpx.HTTPError, RuntimeError, ValueError):
                if attempt >= self.maximum_retries:
                    raise
                delay = min(60.0, 0.8 * (2 ** min(attempt - 1, 6)))
                await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        raise AssertionError("unreachable")

