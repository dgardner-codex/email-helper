"""OpenAI embeddings client for Phase 2A."""

import json
import os
import random
import socket
import time
from typing import Any
from urllib import error, request

from Constants import EMBED_MODEL_NAME, MAX_EMBED_BATCH, REQUEST_TIMEOUT_SECS


class EmbeddingError(Exception):
    """Raised when embeddings are unavailable or persistently failing."""


def _batched(items: list[str], size: int) -> list[list[str]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _is_transient_http(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _is_transient_exception(exc: Exception) -> bool:
    if isinstance(exc, error.HTTPError):
        return _is_transient_http(exc.code)
    if isinstance(exc, error.URLError):
        return isinstance(exc.reason, TimeoutError | socket.timeout)
    return isinstance(exc, TimeoutError | socket.timeout)


def _request_embeddings(batch: list[str], api_key: str) -> list[list[float]]:
    payload = json.dumps({"model": EMBED_MODEL_NAME, "input": batch}).encode("utf-8")
    req = request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECS) as response:
                body = response.read().decode("utf-8")
            parsed: dict[str, Any] = json.loads(body)
            data = parsed.get("data")
            if not isinstance(data, list):
                raise EmbeddingError("Embeddings API response missing data")

            vectors: list[list[float]] = []
            for item in data:
                if not isinstance(item, dict):
                    raise EmbeddingError("Embeddings API response item malformed")
                embedding = item.get("embedding")
                if not isinstance(embedding, list):
                    raise EmbeddingError("Embeddings API response missing embedding")
                vectors.append([float(value) for value in embedding])

            if len(vectors) != len(batch):
                raise EmbeddingError("Embeddings API returned mismatched vector count")
            return vectors
        except Exception as exc:
            if not _is_transient_exception(exc) or attempt == max_attempts:
                if isinstance(exc, EmbeddingError):
                    raise
                raise EmbeddingError(f"Embeddings request failed: {exc}") from exc

            sleep_for = (2 ** (attempt - 1)) + random.uniform(0.0, 0.25)
            time.sleep(sleep_for)

    raise EmbeddingError("Embeddings request exhausted retries")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed input texts using OpenAI text-embedding-3-small."""
    if not texts:
        return []

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EmbeddingError("OPENAI_API_KEY is missing")

    all_vectors: list[list[float]] = []
    for batch in _batched(texts, MAX_EMBED_BATCH):
        vectors = _request_embeddings(batch, api_key)
        all_vectors.extend(vectors)
    return all_vectors
