"""Validation functions for scaffold phase."""

from typing import Any

from Constants import ALLOWED_MUTABLE_FIELDS, REQUIRED_EMAIL_FIELDS


def validate_categories(categories: Any) -> list[str]:
    if not isinstance(categories, list) or not categories:
        raise ValueError("categories.json must contain a non-empty JSON list of strings")

    invalid = [item for item in categories if not isinstance(item, str)]
    if invalid:
        raise ValueError("categories.json entries must all be strings")

    return categories


def validate_email_record(email: Any) -> dict[str, str]:
    if not isinstance(email, dict):
        raise ValueError("Each email record must be a JSON object")

    missing = [key for key in REQUIRED_EMAIL_FIELDS if key not in email]
    if missing:
        raise ValueError(f"Email record missing required keys: {missing}")

    for key in REQUIRED_EMAIL_FIELDS:
        value = email[key]
        if not isinstance(value, str):
            raise ValueError(f"Email field '{key}' must be a string")
        if key not in {"category", "priority"} and value == "":
            raise ValueError(f"Email field '{key}' may not be empty")

    return email


def validate_input_emails(emails: Any) -> list[dict[str, str]]:
    if not isinstance(emails, list):
        raise ValueError("Input file must contain a JSON list of email objects")
    return [validate_email_record(email) for email in emails]


def enforce_only_labels_changed(
    original: dict[str, Any],
    labeled: dict[str, Any],
) -> None:
    if set(original.keys()) != set(labeled.keys()):
        raise ValueError("Labeled email must preserve exactly the same keys")

    for key, original_value in original.items():
        labeled_value = labeled[key]
        if key in ALLOWED_MUTABLE_FIELDS:
            continue
        if original_value != labeled_value:
            raise ValueError(
                f"Only category/priority may change; field '{key}' was modified"
            )

    for field in ALLOWED_MUTABLE_FIELDS:
        value = labeled.get(field)
        if not isinstance(value, str) or value == "":
            raise ValueError(f"Labeled field '{field}' must be a non-empty string")
