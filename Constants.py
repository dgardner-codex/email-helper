"""Project constants (Phase 1 scaffold + Phase 1B heuristics)."""

from pathlib import Path

CATEGORIES_FILE = Path("categories.json")
TRACE_FILE = Path("trace.txt")
ALLOWED_MUTABLE_FIELDS = {"category", "priority"}
REQUIRED_EMAIL_FIELDS = ("date", "from", "subject", "priority", "category", "body")
PLACEHOLDER_CATEGORY = "Archive"
PLACEHOLDER_PRIORITY = "normal"

OPERATIONAL_CATEGORIES_TO_SKIP = {"Inbox", "Drafts", "Sent", "Trash"}
SPECIAL_CATEGORY_JUNK = "Junk"
SPECIAL_CATEGORY_ARCHIVE = "Archive"
ALLOWED_PRIORITIES = {"high", "normal"}

MIN_CATEGORY_SCORE = 3
MIN_CATEGORY_MARGIN = 2
BODY_SNIPPET_CHARS = 280

W_FROM = 4
W_DOMAIN = 5
W_SUBJECT = 2
W_BODY = 1

LINK_DENSITY_THRESHOLD = 3
