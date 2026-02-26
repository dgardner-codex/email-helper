"""Samples embeddings index and cache management for Phase 2A."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from typing import Any

from Constants import (
    EMBED_BODY_SNIPPET_CHARS,
    EMBED_MODEL_NAME,
    SAMPLE_EMBED_CACHE_PATH,
    SAMPLES_PATH,
)
from openai_embeddings import embed_texts
from trace import _trace

_WARNED_PRIORITY_NORMALIZATION = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _samples_path() -> Path:
    return (_repo_root() / SAMPLES_PATH).resolve()


def _cache_path() -> Path:
    return (_repo_root() / SAMPLE_EMBED_CACHE_PATH).resolve()


def _canonical_text(record: dict[str, Any]) -> str:
    from_field = str(record.get("from", ""))
    subject = str(record.get("subject", ""))
    body = str(record.get("body", ""))[:EMBED_BODY_SNIPPET_CHARS]
    return f"FROM: {from_field}\nSUBJECT: {subject}\nBODY: {body}"


def _normalize(vector: list[float]) -> list[float] | None:
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return None
    return [value / norm for value in vector]


def _normalize_priority(value: Any) -> str:
    global _WARNED_PRIORITY_NORMALIZATION
    normalized = str(value).strip().lower()
    if normalized in {"high", "normal"}:
        return normalized
    if not _WARNED_PRIORITY_NORMALIZATION:
        _WARNED_PRIORITY_NORMALIZATION = True
        _trace("samples priority normalization: non-standard value mapped to normal")
    return "normal"


def _parse_cache(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    samples = payload.get("samples")
    if not isinstance(samples, list):
        return []

    parsed: list[dict[str, Any]] = []
    for item in samples:
        if not isinstance(item, dict):
            continue
        embedding = item.get("embedding")
        category = str(item.get("category", ""))
        priority = _normalize_priority(item.get("priority", "normal"))
        if not isinstance(embedding, list) or not category:
            continue
        try:
            normalized = [float(value) for value in embedding]
        except (TypeError, ValueError):
            continue
        parsed.append(
            {
                "embedding": normalized,
                "category": category,
                "priority": priority,
                "from": str(item.get("from", "")),
                "subject": str(item.get("subject", "")),
            }
        )
    return parsed


def _build_index(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    texts = [_canonical_text(record) for record in records]
    vectors = embed_texts(texts)

    built: list[dict[str, Any]] = []
    for record, vector in zip(records, vectors):
        normalized = _normalize(vector)
        if normalized is None:
            _trace("samples index warning: skipped sample with zero-norm embedding")
            continue
        built.append(
            {
                "embedding": normalized,
                "category": str(record.get("category", "")).strip(),
                "priority": _normalize_priority(record.get("priority", "normal")),
                "from": str(record.get("from", "")),
                "subject": str(record.get("subject", "")),
            }
        )
    return built


def build_or_load_samples_index(categories: list[str]) -> list[dict[str, Any]]:
    samples_path = _samples_path()
    raw = samples_path.read_text(encoding="utf-8")
    sample_hash = sha256(raw.encode("utf-8")).hexdigest()

    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        _trace("samples index cache miss: samples.json not a list")
        return []

    allowed = set(categories)
    records = [
        record
        for record in parsed
        if isinstance(record, dict) and str(record.get("category", "")).strip() in allowed
    ]

    cache_path = _cache_path()
    cache_payload: dict[str, Any] | None = None
    if cache_path.exists():
        try:
            cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            cache_payload = None

    rebuild_reason = ""
    if cache_payload is None:
        rebuild_reason = "cache_missing_or_invalid"
    elif str(cache_payload.get("sample_hash", "")) != sample_hash:
        rebuild_reason = "sample_hash_changed"
    elif str(cache_payload.get("model_name", "")) != EMBED_MODEL_NAME:
        rebuild_reason = "model_changed"

    if rebuild_reason == "":
        _trace("samples index cache hit")
        return _parse_cache(cache_payload)

    _trace(f"samples index cache miss ({rebuild_reason})")
    index_samples = _build_index(records)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_hash": sample_hash,
        "model_name": EMBED_MODEL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": index_samples,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _trace(f"samples index rebuilt: {len(index_samples)} samples")
    return index_samples
