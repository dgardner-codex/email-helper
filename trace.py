"""Best-effort append-only tracing utilities."""

from datetime import datetime, timezone

from Constants import TRACE_FILE


def _trace(msg: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {msg}\n"
    try:
        with TRACE_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        # Tracing should never impact program success.
        return
