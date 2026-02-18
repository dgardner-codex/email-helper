"""Deterministic heuristic classifier for Phase 1B."""

from typing import Any
import re

from Constants import (
    ALLOWED_PRIORITIES,
    BODY_SNIPPET_CHARS,
    LINK_DENSITY_THRESHOLD,
    MIN_CATEGORY_MARGIN,
    MIN_CATEGORY_SCORE,
    OPERATIONAL_CATEGORIES_TO_SKIP,
    SPECIAL_CATEGORY_ARCHIVE,
    SPECIAL_CATEGORY_JUNK,
    W_BODY,
    W_DOMAIN,
    W_FROM,
    W_SUBJECT,
)
from trace import _trace

PROMOTIONAL_WORDS = (
    "deal",
    "limited time",
    "offer",
    "sale",
    "discount",
    "coupon",
    "promo",
    "save now",
    "shop now",
)

SCAM_PHRASES = (
    "urgent action required",
    "verify your account",
    "suspended account",
    "wire transfer",
    "claim your prize",
    "you have won",
    "bitcoin payment",
    "gift card",
)

HIGH_PRIORITY_PHRASES = (
    "please respond",
    "asap",
    "urgent",
    "deadline",
    "confirm",
    "action required",
    "need your",
)


def _token_pattern(text: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(text)}(?![a-z0-9])")


def _has_boundary_match(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    pattern = _token_pattern(needle)
    return pattern.search(haystack) is not None


def _extract_sender_parts(from_field: str) -> tuple[str, str, str]:
    value = from_field.strip()
    if "<" in value and ">" in value:
        left, _, right = value.partition("<")
        email_part = right.split(">", 1)[0].strip()
        display = left.strip().strip('"')
    else:
        display = value
        email_part = value

    if "@" in email_part:
        domain = email_part.split("@", 1)[1]
    else:
        domain = ""

    return display.lower(), email_part.lower(), domain.lower()


def _operational_candidates(categories: list[str]) -> list[str]:
    return [
        category
        for category in categories
        if category not in OPERATIONAL_CATEGORIES_TO_SKIP
        and category not in {SPECIAL_CATEGORY_JUNK, SPECIAL_CATEGORY_ARCHIVE}
    ]


def _score_candidate(
    category: str,
    sender_display: str,
    sender_email: str,
    sender_domain: str,
    subject: str,
    body_snippet: str,
) -> int:
    term = category.lower()
    score = 0

    if _has_boundary_match(sender_display, term) or _has_boundary_match(sender_email, term):
        score += W_FROM
    if _has_boundary_match(sender_domain, term):
        score += W_DOMAIN
    if _has_boundary_match(subject, term):
        score += W_SUBJECT
    if _has_boundary_match(body_snippet, term):
        score += W_BODY

    return score


def _top_candidates_summary(scored: list[tuple[str, int]], limit: int = 3) -> str:
    top = scored[:limit]
    return ", ".join(f"{name}:{score}" for name, score in top) if top else ""


def _is_junk(subject: str, body_snippet: str) -> tuple[bool, str]:
    combined = f"{subject} {body_snippet}".lower()
    has_unsubscribe = "unsubscribe" in combined
    has_promo = any(word in combined for word in PROMOTIONAL_WORDS)
    if has_unsubscribe and has_promo:
        return True, "unsubscribe + promotional language"

    if any(phrase in combined for phrase in SCAM_PHRASES):
        return True, "obvious scam phrase detected"

    link_count = combined.count("http") + combined.count("www")
    if link_count >= LINK_DENSITY_THRESHOLD:
        return True, f"high link density ({link_count})"

    return False, "no junk indicators"


def _priority_for_email(subject: str, body_snippet: str) -> tuple[str, str]:
    combined = f"{subject} {body_snippet}".lower()
    has_question = "?" in subject or "?" in body_snippet
    has_action_language = any(phrase in combined for phrase in HIGH_PRIORITY_PHRASES)

    if has_question or has_action_language:
        return "high", "question/action language detected"
    return "normal", "no urgency indicators"


def _validate_required_categories(categories: list[str]) -> None:
    if SPECIAL_CATEGORY_JUNK not in categories:
        raise ValueError(f"Required category missing from categories.json: {SPECIAL_CATEGORY_JUNK}")
    if SPECIAL_CATEGORY_ARCHIVE not in categories:
        raise ValueError(f"Required category missing from categories.json: {SPECIAL_CATEGORY_ARCHIVE}")


def label_email(
    email: dict[str, str],
    categories: list[str],
    samples: Any = None,
) -> tuple[str, str, dict[str, str]]:
    _ = samples
    _validate_required_categories(categories)

    sender_display, sender_email, sender_domain = _extract_sender_parts(email["from"])
    subject = email["subject"].lower()
    body_snippet = email["body"][:BODY_SNIPPET_CHARS].lower()

    is_junk, junk_reason = _is_junk(subject, body_snippet)
    _trace(f"junk decision: {is_junk} ({junk_reason})")

    if is_junk:
        _trace("top candidates: Junk override")
        _trace("final category: Junk (high-confidence override)")
        _trace("priority decision: normal (junk override)")
        return (
            SPECIAL_CATEGORY_JUNK,
            "normal",
            {
                "method": "heuristic",
                "confidence": "high",
                "reason": junk_reason,
                "top_candidates": f"{SPECIAL_CATEGORY_JUNK}:override",
            },
        )

    scored = [
        (
            category,
            _score_candidate(
                category,
                sender_display,
                sender_email,
                sender_domain,
                subject,
                body_snippet,
            ),
        )
        for category in _operational_candidates(categories)
    ]

    scored.sort(key=lambda item: item[1], reverse=True)
    top_summary = _top_candidates_summary(scored)

    best_category = SPECIAL_CATEGORY_ARCHIVE
    best_score = 0
    second_best_score = 0

    if scored:
        best_category, best_score = scored[0]
        second_best_score = scored[1][1] if len(scored) > 1 else 0

    low_confidence = (
        best_score < MIN_CATEGORY_SCORE
        or (best_score - second_best_score) < MIN_CATEGORY_MARGIN
    )

    if low_confidence:
        selected_category = SPECIAL_CATEGORY_ARCHIVE
        confidence = "low"
        reason = (
            f"low-confidence match (best={best_score}, margin={best_score - second_best_score}); archived"
        )
    else:
        selected_category = best_category
        confidence = "high"
        reason = f"strongest category score: {best_category} ({best_score})"

    if selected_category not in categories:
        raise ValueError(f"Classifier selected unknown category: {selected_category}")

    priority, priority_reason = _priority_for_email(subject, body_snippet)
    if priority not in ALLOWED_PRIORITIES:
        raise ValueError(f"Classifier selected invalid priority: {priority}")

    _trace(f"top candidates: {top_summary}")
    _trace(f"final category: {selected_category} ({reason})")
    _trace(f"priority decision: {priority} ({priority_reason})")

    return (
        selected_category,
        priority,
        {
            "method": "heuristic",
            "confidence": confidence,
            "reason": reason,
            "top_candidates": top_summary,
        },
    )
