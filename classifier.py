"""Classifier placeholder for Phase 1 scaffold."""

from __future__ import annotations

from typing import Any

from Constants import PLACEHOLDER_CATEGORY, PLACEHOLDER_PRIORITY


def label_email(
    email: dict[str, str],
    categories: list[str],
    samples: Any = None,
) -> tuple[str, str, dict[str, str]]:
    del email, samples

    if PLACEHOLDER_CATEGORY not in categories:
        raise ValueError(
            f"Placeholder classifier requires '{PLACEHOLDER_CATEGORY}' in categories"
        )

    return (
        PLACEHOLDER_CATEGORY,
        PLACEHOLDER_PRIORITY,
        {
            "method": "placeholder",
            "confidence": "low",
        },
    )
