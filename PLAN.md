# Email Labeling Project Plan

## Project goal

Build a Python 3.12 CLI tool that reads unsolved email records, assigns each email a valid `category` and `priority`, and writes a solved output JSON file while preserving all non-label fields.

## Scope and constraints

- Classification must use only categories present in `categories.json`.
- Every input email must be solved.
- If not junk and no category can be confidently chosen, fallback is `Archive`.
- If junk, final category is `Junk` and priority is always `normal`.
- Priority values are restricted to `high` or `normal`.
- Input/output JSON shape must remain unchanged apart from writing `category` and `priority`.

## Inputs and outputs

### Inputs

- `categories.json`: list of valid category names.
- `samples.json`: solved examples for optional guidance.
- CLI input file: unsolved email records with empty `category` and `priority`.

### Outputs

- A newly written JSON file containing solved records.
- `trace.txt` append-only trace log.
- Console progress updates using `print()`.

## Processing pipeline

1. Parse CLI args and load files.
2. Validate data shape and required fields.
3. For each email:
   - Normalize fields (`from`, `subject`, `body`).
   - Run category/junk/priority inference.
   - Apply mandatory overrides and fallbacks.
   - Update only `category` and `priority`.
4. Write solved output file.
5. Emit summary metrics to console and trace.

## Module plan

### `main.py`

- CLI entry point and orchestration.
- File I/O and end-to-end workflow management.

### `constants.py`

- File names, allowed labels, thresholds, prompt templates, and shared constants.

### `io_utils.py`

- JSON read/write helpers.
- Input validation and output naming helpers.

### `classifier.py`

- Deterministic heuristics and fallback logic.
- OpenAI-assisted adjudication hooks.
- Guardrails to enforce valid output labels.

### `openai_client.py`

- OpenAI API wrapper with retries/backoff.
- Structured response parsing for category, junk, and reply-needed decisions.

### `trace.py`

- Append-only tracing helpers for `trace.txt`.
- Timestamped human-readable events.

## OpenAI usage plan

Use OpenAI only at uncertainty points:

- Category disambiguation when deterministic score/confidence is low.
- Junk adjudication when deterministic signals are inconclusive.
- Reply-needed inference when priority cannot be confidently inferred.

Guardrails:

- Validate model-selected category against `categories.json`.
- Reject invalid labels and apply deterministic fallback.
- Log retries/failures and continue with degraded heuristic-only behavior.

## Minimal Initial Heuristic Set (Pre-OpenAI Baseline)

This is the smallest deterministic ruleset needed for useful end-to-end behavior in Phase 2, tailored to a personal mailbox. The key principle is: **never map to invented buckets** (e.g., Finance/HR/etc.). Instead, choose only from the existing folder/category names in `categories.json`.

### Category heuristics (non-junk)

Use a category-inventory-driven match-and-score approach:

- Build a normalized text context from:
  - sender display name and email address (`from`)
  - sender domain
  - `subject`
  - (optionally) the first N characters of `body` (keep N small for speed)
- Consider as candidates all categories from `categories.json` except operational folders that should not be assigned by the classifier (e.g., `Inbox`, `Drafts`, `Sent`, `Trash`). Keep `Junk` and `Archive` reserved for override/fallback logic below.
- Score each candidate category by weighted exact/near-exact matches (case-insensitive):
  - Strong match: category name appears in sender display name or sender domain (highest weight)
  - Medium match: category name appears in subject
  - Weak match: category name appears in body snippet
- Resolution policy:
  - Choose the top-scoring category if the score meets a minimal threshold (to avoid random matches).
  - If no candidate meets threshold, leave category unresolved for fallback (`Archive`) in the baseline.

Notes:
- This is designed to work well for personal folders like `Amazon`, `Apple`, `ADT`, `Netflix`, `Venmo`, `GoDaddy`, `2024 Nantucket Trip`, etc.
- Do not invent or infer generic bucket categories; always select from the provided category list only.

Escalation (future OpenAI phase):
- Preserve a “confidence” signal and the top-N candidates (with scores). If no candidate meets threshold OR the top two candidates are within a small margin, mark the case as low-confidence and defer to the LLM category-disambiguation step in the OpenAI phase. In the pre-OpenAI baseline only, low-confidence falls back to `Archive`.

### Junk heuristics

Flag as junk if one or more high-confidence spam signals are present:

- Common promotional/scam phrases (e.g., “winner”, “claim prize”, “urgent account”, “limited offer”, etc.).
- Presence of “unsubscribe” combined with promotional language (“deal”, “offer”, “sale”, “promo”) or unusually high link density.
- Suspicious sender patterns (spoof-like domains, random alphanumeric local parts).

High-confidence non-junk shortcuts (personal-mailbox oriented):

- Existing thread/reply markers such as `Re:` / `Fwd:` plus quoted text markers (suggests a real conversation).
- Clear, specific relationship/context signals in the body (e.g., personalized requests rather than broadcast marketing).

Escalation (future OpenAI phase):
- If junk signals are inconclusive (neither confidently junk nor confidently non-junk), preserve a “junk=uncertain” signal and defer to the LLM junk-adjudication step in the OpenAI phase. In the pre-OpenAI baseline only, treat uncertain as non-junk.

### Priority heuristics

Set `high` when clear response/action is required:

- Direct question(s) to recipient (e.g., subject ends with “?” or body contains direct questions).
- Explicit request language: “please respond”, “can you…”, “confirm”, “approve”, “need your input”, “action required”.
- Time-bound language: “by tomorrow”, “due today”, “deadline”, “ASAP”, “expires today”.

Otherwise default to `normal`. Junk emails always remain `normal`.

Escalation (future OpenAI phase):
- If reply-needed is unclear, preserve a “priority=uncertain” signal and defer to the LLM reply-needed inference step in the OpenAI phase. In the pre-OpenAI baseline only, default to `normal`.

### Mandatory overrides

- If junk=true → force `category = "Junk"` and `priority = "normal"`.
- If category unresolved and junk=false → force `category = "Archive"`.

## Tracing and observability

- Trace all major milestones to console with `print()`.
- Append detailed diagnostics to `trace.txt`.
- Include per-email decision path and fallback markers.
- Keep trace resilient so trace-write errors do not halt processing.

## Validation and testing plan

- Run baseline command-line validation on representative sample input.
- Verify all outputs use allowed labels only.
- Verify unresolved non-junk categories map to `Archive`.
- Verify junk override forces `Junk` + `normal`.
- Confirm only `category` and `priority` fields are changed.

## Delivery phases

1. **Phase 1**: deterministic-only baseline (no API dependency).
2. **Phase 2**: add OpenAI escalation for low-confidence paths.
3. **Phase 3**: refine scoring, prompts, and trace quality using feedback.
