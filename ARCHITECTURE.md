# Email Labeling Project Architecture

## 1) System Overview

This project is a command-line Python 3.12 application that classifies inbound emails into one of the pre-defined categories from `categories.json`, assigns a priority (`high` or `normal`), and writes the solved dataset as a new JSON file.

At a high level, the system is a deterministic + sample-learned + embeddings-assisted classifier, with LLM usage reserved for later low-confidence escalation:

1. Load static reference data (categories and optional solved samples).
2. Load an input JSON file of unsolved emails (`category == ""`, `priority == ""`).
3. For each email, run a classification pipeline:
   - Apply deterministic heuristics and learned sender/domain signals derived from `samples.json` (Phase 1C).
   - Use embeddings + kNN similarity classification against solved samples when needed (Phase 2A).
   - Escalate only low-confidence residuals to LLM adjudication (Phase 2B).
   - Apply guardrails/fallbacks so every email is solved.
4. Persist output JSON with only `category` and `priority` modified.
5. Emit human-readable trace logs (`trace.txt`) and console progress prints.

Design goals:

- **Accuracy with explainability**: heuristic, learned, embedding-neighbor, and (future) LLM decisions are traceable in human-readable logs.
- **Deterministic boundaries**: output values restricted to approved categories and priorities.
- **Completion guarantee**: every input email receives a final category and priority.
- **Simplicity**: functional design, no unnecessary abstraction.

---

## 2) Modules and Responsibilities

The implementation should remain functional (not class-heavy), with small modules that each own a single concern.

### `main.py` (entry and orchestration)

Responsibilities:

- Parse CLI arguments.
- Validate startup prerequisites (files exist, env var exists).
- Call pipeline functions in sequence.
- Handle top-level errors with clear user-facing messages.
- Print progress markers to console.

### `constants.py` (configuration and global constants)

Responsibilities:

- Centralize filenames (default categories/samples paths).
- Define allowed priorities (`{"high", "normal"}`).
- Define canonical special categories used in logic (`"Junk"`, `"Archive"`).
- Store OpenAI model selection constants.
- Store prompt templates and thresholds (if any score thresholds are used).

### `io_layer.py` (JSON and filesystem IO)

Responsibilities:

- Read/parse JSON files (`categories.json`, sample file, input file).
- Validate that structures are list-based and minimally schema-compliant.
- Write solved output JSON file.
- Build output filename strategy (for example: `<input_basename>.labeled.json`).

### `validation.py` (input and output contract checks)

Responsibilities:

- Validate input email records contain expected fields.
- Ensure only `category` and `priority` are changed.
- Enforce output category is in approved category list.
- Enforce output priority in allowed priorities.
- Provide fallback normalization for malformed/empty fields.

### `classifier.py` (core labeling flow)

Responsibilities:

- Prepare normalized email text signals (`from`, `subject`, `body`).
- Run category decision path.
- Run junk decision path.
- Run reply/priority decision path.
- Merge decisions into a final deterministic label set.
- Apply fallback to `Archive` for non-junk uncategorized mail.

Classifier logic includes phased signal layers: Phase 1C sample-derived lookup maps (for sender/from/domain-to-category priors), then Phase 2A embeddings similarity scoring that yields `(category, confidence, neighbors)` for downstream guardrail checks.

### `embedding utilities / index` (conceptual module; Phase 2A)

Responsibilities:

- Generate embeddings for solved samples and inbound emails.
- Maintain a cached/persisted embedding index to avoid full recomputation.
- Perform kNN similarity retrieval and return top-k neighbors with confidence-oriented metadata.

### `openai_client.py` (LLM interaction wrapper; Phase 2B)

Responsibilities:

- Manage OpenAI API calls with a stable interface for the rest of the app once Phase 2B escalation is enabled.
- Build request payloads using prompt templates/constants.
- Parse model responses into structured decisions.
- Handle retry/backoff for transient API failures.
- Return machine-usable decision objects and optional rationale text for tracing.

### `trace.py` (tracing and operator observability)

Responsibilities:

- Append human-readable events to `trace.txt`.
- Provide timestamped trace helper functions.
- Keep trace writing resilient (best effort; should not crash primary flow).

---

## 3) Data Flow (Input → Classification → Output)

### Step A: Input Acquisition

- CLI receives an input filename.
- System reads:
  - category inventory (`categories.json`)
  - optional solved examples (`samples.json`) for contextual guidance
  - target input file to solve
- Validation confirms parsable JSON and required fields.

### Step B: Per-email Classification Pipeline

For each email record:

1. **Normalize content**
   - Coerce missing fields to safe empty strings.
   - Build a compact classification context from `from`, `subject`, and `body`.

2. **Junk assessment**
   - Run deterministic spam/promotional/scam heuristics first.
   - Use LLM junk adjudication only for uncertain cases (Phase 2B).

3. **Category decision**
   - Apply learned sender/from/domain lookup signals derived from solved samples (Phase 1C).
   - Apply heuristic match-and-score logic (Phase 1B baseline).
   - Apply embeddings kNN similarity vs. solved samples and read `(category, confidence, neighbors)` (Phase 2A).
   - Escalate to LLM category disambiguation only if still low-confidence (Phase 2B).

4. **Priority assessment**
   - Determine whether the email likely requires a response using heuristics first.
   - Escalate to LLM only when heuristics are uncertain (Phase 2B).
   - Map response-needed to priority (`high` for needs response, else `normal`).

5. **Resolution guardrails**
   - If junk=true, force category=`Junk` and priority=`normal`.
   - If category unresolved and junk=false, force category=`Archive`.
   - Guarantee final values are from allowed sets.

6. **Record update**
   - Update only `category` and `priority` fields.
   - Preserve all other original email fields unchanged.

### Step C: Output Generation

- Emit solved list to a new JSON output file.
- Write summary traces (counts by category, count of `high`, count of junk).
- Print completion status and output filename to console.

---

## 4) Decision Points for OpenAI API Usage

OpenAI usage is a Phase 2B capability only, triggered after deterministic, sample-learned, and embeddings-based paths are exhausted or uncertain.

### Phase 2A embeddings path (non-LLM)

- Generate and persist embeddings for solved samples, and generate embeddings for each inbound email.
- Run kNN similarity search over the sample embedding index, then threshold confidence to produce candidate category decisions.
- Cache/persist embeddings and index artifacts to avoid repeated recomputation across runs.

### Phase 2B primary API decision points

1. **Category disambiguation**
   - Trigger when deterministic rule matching yields low confidence or conflicts.
   - Prompt includes allowed category list and email context.
   - Model must return one category from the provided list.

2. **Junk adjudication**
   - Trigger when spam heuristics are inconclusive.
   - Model returns a strict boolean-like signal (junk / not junk) with short reason.

3. **Reply-needed inference**
   - Trigger when it is unclear whether user action is required.
   - Model returns yes/no for response-needed.

### API guardrails

- Never trust free-form category strings; validate against approved categories.
- Validate final categories strictly against `categories.json` for both embeddings-derived and LLM-derived decisions.
- Apply deterministic fallback when model output is invalid/empty.
- Prefer lower temperature for consistency.
- Include concise, schema-oriented prompts to reduce parsing ambiguity.
- Record request intent and parsed result in trace logs (without leaking secrets).
- When embeddings are used, trace top-k nearest-neighbor rationale alongside the chosen category/confidence.

### Failure handling at API boundaries

- On timeout/transient errors: retry with bounded attempts.
- On persistent API failure:
  - Use heuristic-only fallback classification.
  - If unresolved and not junk, assign `Archive`.
- Do not leave any email unsolved due to API failure.

---

## 5) Junk Detection and Priority Logic

### Junk Detection Strategy

Junk detection combines deterministic signals and optional LLM adjudication.

Deterministic junk indicators (examples):

- Obvious scam language in subject/body.
- High promotional density with no user relationship signal.
- Sender/domain patterns historically associated with junk.

Decision policy:

- If deterministic confidence is high for junk → set junk immediately.
- If deterministic confidence is high for non-junk → skip junk API call.
- Otherwise, use OpenAI to break tie.

Final junk override rule:

- `junk == true` always forces:
  - `category = "Junk"`
  - `priority = "normal"`

### Priority Logic

Priority is based on whether a reply/action is likely required.

- If response needed → `priority = "high"`
- If no response needed → `priority = "normal"`
- Junk emails always remain `normal` even if text appears urgent.

Signals for response-needed (examples):

- Direct questions addressed to the recipient.
- Requests for confirmation, approval, payment, or scheduling.
- Time-sensitive deadlines requiring user action.

Safety rules:

- Priority must be one of `high` or `normal` only.
- If uncertain after all checks, default to `normal` unless a clear action request exists.

---

## 6) Tracing and Logging Strategy

The project needs operational transparency for learning/debugging, so tracing is always on.

### Trace destinations

- **Console (`print`)**: progress milestones and high-level status.
- **`trace.txt` append-only**: detailed, human-readable step-by-step records.

### What to trace

At minimum:

- Startup context (input filename, category count, email count).
- Per-email processing start/end (index, sender/subject summary).
- Decision path used (heuristic-only vs OpenAI-assisted).
- Intermediate outcomes (junk decision, category candidate, priority decision).
- Fallbacks applied (e.g., forced `Archive`, forced `Junk`).
- API failures/retries and final degraded-mode behavior.
- End-of-run summary metrics.

### Trace hygiene

- Include timestamps for each trace entry.
- Avoid logging secrets (API key, auth headers).
- Keep content concise but diagnosis-friendly.
- Never let trace write errors stop classification flow.

---

## 7) CLI Invocation and Error Handling

### CLI Contract

Expected invocation:

```bash
python main.py <input_json_file>
```

Behavior:

- Reads required reference files (categories and optional samples) from known paths.
- Produces a solved JSON output file with same record structure.
- Prints output path and summary counts.

### Error Handling Principles

Error handling should be baseline but explicit and user-friendly.

1. **Argument errors**
   - Missing filename argument.
   - Too many/invalid arguments.
   - Action: print usage and exit non-zero.

2. **File errors**
   - Input file not found or unreadable.
   - Reference file not found.
   - Action: print clear filename-specific error and exit non-zero.

3. **JSON/schema errors**
   - Invalid JSON syntax.
   - Unexpected top-level shape.
   - Missing required email fields.
   - Action: print descriptive parsing/validation error and exit non-zero.

4. **Environment/config errors**
   - `OPENAI_API_KEY` missing when API path is needed.
   - Action: print remediation guidance and exit non-zero (or run heuristic-only mode if explicitly supported).

5. **Runtime/API errors**
   - Timeout, rate limit, transient service issues.
   - Action: retry bounded times, trace failure, then continue with fallback logic.

6. **Output write errors**
   - Destination not writable.
   - Action: print error and exit non-zero.

### Completion Guarantee

The pipeline must still satisfy the core requirement that every email in the input is solved:

- Never emit empty `category` or `priority`.
- If category cannot be confidently determined and email is not junk, assign `Archive`.
- Keep all non-label fields untouched.

---

## 8) Non-goals for Current Phase

To keep implementation focused, the following are out of scope for now:

- Direct integration with any email client APIs.
- Continuous learning or automated retraining loops.
- Complex persistent stores or external databases.
- Production-grade observability stack beyond `print` + `trace.txt`.

This architecture is intentionally pragmatic: simple enough for iterative learning, while still robust enough to produce consistent labels with transparent decisioning.
