"""Project constants for Phase 1 scaffold."""

from pathlib import Path

CATEGORIES_FILE = Path("categories.json")
TRACE_FILE = Path("trace.txt")
ALLOWED_MUTABLE_FIELDS = {"category", "priority"}
REQUIRED_EMAIL_FIELDS = ("date", "from", "subject", "priority", "category", "body")
PLACEHOLDER_CATEGORY = "Archive"
PLACEHOLDER_PRIORITY = "normal"
