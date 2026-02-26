"""Project constants (Phase 2A embeddings + kNN)."""

from pathlib import Path

CATEGORIES_FILE = Path("categories.json")
TRACE_FILE = Path("trace.txt")
SAMPLES_PATH = "samples.json"
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

MIN_FROM_HITS = 1
MIN_DOMAIN_HITS = 2
MIN_DOMAIN_RATIO = 0.6

SAMPLE_EMBED_CACHE_PATH = ".cache/samples_embeddings.json"
EMBED_MODEL_NAME = "text-embedding-3-small"
EMBED_BODY_SNIPPET_CHARS = 500
K_NEIGHBORS = 5
MIN_EMBED_SIMILARITY = 0.80
MIN_EMBED_SIM_MARGIN = 0.05
MIN_KNN_TOPCAT_WEIGHT = 0.65
MAX_EMBED_BATCH = 64
REQUEST_TIMEOUT_SECS = 30
