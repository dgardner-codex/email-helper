"""Deterministic heuristic classifier for Phase 1B."""

from typing import Any
from pathlib import Path
import json
import re

from Constants import (
    ALLOWED_PRIORITIES,
    BODY_SNIPPET_CHARS,
    LINK_DENSITY_THRESHOLD,
    MIN_CATEGORY_MARGIN,
    MIN_CATEGORY_SCORE,
    MIN_DOMAIN_HITS,
    MIN_DOMAIN_RATIO,
    MIN_FROM_HITS,
    OPERATIONAL_CATEGORIES_TO_SKIP,
    SAMPLES_PATH,
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

_DOMAIN_TO_CATEGORY_CACHE: dict[str, tuple[str, float, int]] | None = None
_FROM_TO_CATEGORY_CACHE: dict[str, tuple[str, int]] | None = None
_CACHE_KEY: tuple[str, ...] | None = None


CAMEL_CASE_BOUNDARY = re.compile(r"([a-z])([A-Z])")
ALNUM_CHUNKS = re.compile(r"[a-z0-9]+")


def _token_pattern(text: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(text)}(?![a-z0-9])")


def _has_boundary_match(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    pattern = _token_pattern(needle)
    return pattern.search(haystack) is not None


def extract_sender_email(from_field: str) -> str:
    value = from_field
    if "<" in value and ">" in value:
        _, _, right = value.partition("<")
        value = right.split(">", 1)[0]
    return value.strip().lower()


def extract_domain(sender_email: str) -> str:
    if "@" not in sender_email:
        return ""
    return sender_email.split("@", 1)[1].strip().lower()


def _split_sender_tokens(text: str) -> list[str]:
    normalized = CAMEL_CASE_BOUNDARY.sub(r"\1 \2", text)
    chunks = ALNUM_CHUNKS.findall(normalized.lower())
    tokens: list[str] = []
    for chunk in chunks:
        tokens.extend(re.findall(r"[a-z]+|\d+", chunk))
    return tokens


def _extract_sender_parts(from_field: str) -> tuple[str, str, str, str]:
    value = from_field.strip()
    if "<" in value and ">" in value:
        left, _, _ = value.partition("<")
        display = left.strip().strip('"')
    else:
        display = value

    sender_email = extract_sender_email(from_field)
    domain = extract_domain(sender_email)
    local_part = sender_email.split("@", 1)[0] if "@" in sender_email else sender_email

    token_parts = _split_sender_tokens(display)
    token_parts.extend(_split_sender_tokens(local_part))
    sender_token_joined = " ".join(token_parts)

    return display.lower(), sender_email, domain, sender_token_joined


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
    sender_token_joined: str,
    subject: str,
    body_snippet: str,
) -> int:
    term = category.lower()
    score = 0

    if (
        _has_boundary_match(sender_display, term)
        or _has_boundary_match(sender_email, term)
        or _has_boundary_match(sender_token_joined, term)
    ):
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


def _is_learnable_category(category: str, categories: list[str]) -> bool:
    if category not in categories:
        return False
    if category in OPERATIONAL_CATEGORIES_TO_SKIP:
        return False
    return True


def load_samples_map(
    samples_path: Path | str,
    categories: list[str],
) -> tuple[dict[str, tuple[str, float, int]], dict[str, tuple[str, int]]]:
    path = Path(samples_path)
    try:
        raw = path.read_text(encoding="utf-8")
        records = json.loads(raw)
    except FileNotFoundError:
        _trace(f"samples warning: missing {path}; fallback to heuristics-only")
        return {}, {}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _trace(f"samples warning: invalid {path} ({exc}); fallback to heuristics-only")
        return {}, {}

    if not isinstance(records, list):
        _trace(f"samples warning: expected list in {path}; fallback to heuristics-only")
        return {}, {}

    category_counts_by_domain: dict[str, dict[str, int]] = {}
    category_counts_by_from: dict[str, dict[str, int]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        category = str(record.get("category", "")).strip()
        if not _is_learnable_category(category, categories):
            continue

        from_field = str(record.get("from", ""))
        sender_email = extract_sender_email(from_field)
        sender_domain = extract_domain(sender_email)

        if sender_email:
            by_from = category_counts_by_from.setdefault(sender_email, {})
            by_from[category] = by_from.get(category, 0) + 1

        if sender_domain:
            by_domain = category_counts_by_domain.setdefault(sender_domain, {})
            by_domain[category] = by_domain.get(category, 0) + 1

    from_map: dict[str, tuple[str, int]] = {}
    for sender_email, by_category in category_counts_by_from.items():
        top_category, top_count = max(by_category.items(), key=lambda item: (item[1], item[0]))
        from_map[sender_email] = (top_category, top_count)

    domain_map: dict[str, tuple[str, float, int]] = {}
    for sender_domain, by_category in category_counts_by_domain.items():
        total_count = sum(by_category.values())
        top_category, top_count = max(by_category.items(), key=lambda item: (item[1], item[0]))
        ratio = (top_count / total_count) if total_count else 0.0
        domain_map[sender_domain] = (top_category, ratio, total_count)

    return domain_map, from_map


def _load_learned_maps(
    categories: list[str],
    samples: Any,
) -> tuple[dict[str, tuple[str, float, int]], dict[str, tuple[str, int]]]:
    global _CACHE_KEY, _DOMAIN_TO_CATEGORY_CACHE, _FROM_TO_CATEGORY_CACHE

    if samples is not None:
        return _build_maps_from_sample_records(samples, categories)

    cache_key = tuple(categories)
    if _DOMAIN_TO_CATEGORY_CACHE is not None and _FROM_TO_CATEGORY_CACHE is not None and _CACHE_KEY == cache_key:
        return _DOMAIN_TO_CATEGORY_CACHE, _FROM_TO_CATEGORY_CACHE

    domain_map, from_map = load_samples_map(SAMPLES_PATH, categories)
    _DOMAIN_TO_CATEGORY_CACHE = domain_map
    _FROM_TO_CATEGORY_CACHE = from_map
    _CACHE_KEY = cache_key
    return domain_map, from_map


def _build_maps_from_sample_records(
    records: Any,
    categories: list[str],
) -> tuple[dict[str, tuple[str, float, int]], dict[str, tuple[str, int]]]:
    if not isinstance(records, list):
        return {}, {}

    category_counts_by_domain: dict[str, dict[str, int]] = {}
    category_counts_by_from: dict[str, dict[str, int]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        category = str(record.get("category", "")).strip()
        if not _is_learnable_category(category, categories):
            continue

        sender_email = extract_sender_email(str(record.get("from", "")))
        sender_domain = extract_domain(sender_email)

        if sender_email:
            by_from = category_counts_by_from.setdefault(sender_email, {})
            by_from[category] = by_from.get(category, 0) + 1

        if sender_domain:
            by_domain = category_counts_by_domain.setdefault(sender_domain, {})
            by_domain[category] = by_domain.get(category, 0) + 1

    from_map: dict[str, tuple[str, int]] = {}
    for sender_email, by_category in category_counts_by_from.items():
        top_category, top_count = max(by_category.items(), key=lambda item: (item[1], item[0]))
        from_map[sender_email] = (top_category, top_count)

    domain_map: dict[str, tuple[str, float, int]] = {}
    for sender_domain, by_category in category_counts_by_domain.items():
        total_count = sum(by_category.values())
        top_category, top_count = max(by_category.items(), key=lambda item: (item[1], item[0]))
        ratio = (top_count / total_count) if total_count else 0.0
        domain_map[sender_domain] = (top_category, ratio, total_count)

    return domain_map, from_map


def label_email(
    email: dict[str, str],
    categories: list[str],
    samples: Any = None,
) -> tuple[str, str, dict[str, str]]:
    _validate_required_categories(categories)

    sender_display, sender_email, sender_domain, sender_token_joined = _extract_sender_parts(email["from"])
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

    domain_map, from_map = _load_learned_maps(categories, samples)

    from_match = from_map.get(sender_email)
    if from_match is not None:
        learned_category, hit_count = from_match
        if hit_count >= MIN_FROM_HITS:
            priority, priority_reason = _priority_for_email(subject, body_snippet)
            _trace(f"learned from-override: {sender_email} -> {learned_category} (hits={hit_count})")
            _trace(f"final category: {learned_category} (from-override)")
            _trace(f"priority decision: {priority} ({priority_reason})")
            return (
                learned_category,
                priority,
                {
                    "method": "heuristic+learned_from",
                    "confidence": "high",
                    "reason": f"from-override ({sender_email}, hits={hit_count})",
                    "top_candidates": "",
                },
            )

    domain_match = domain_map.get(sender_domain)
    if domain_match is not None:
        learned_category, ratio, total_count = domain_match
        if total_count >= MIN_DOMAIN_HITS and ratio >= MIN_DOMAIN_RATIO:
            priority, priority_reason = _priority_for_email(subject, body_snippet)
            _trace(
                f"learned domain-override: {sender_domain} -> {learned_category} "
                f"(hits={total_count}, ratio={ratio:.2f})"
            )
            _trace(f"final category: {learned_category} (domain-override)")
            _trace(f"priority decision: {priority} ({priority_reason})")
            return (
                learned_category,
                priority,
                {
                    "method": "heuristic+learned_from",
                    "confidence": "high",
                    "reason": f"domain-override ({sender_domain}, hits={total_count}, ratio={ratio:.2f})",
                    "top_candidates": "",
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
                sender_token_joined,
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
