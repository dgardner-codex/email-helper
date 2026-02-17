"""JSON IO helpers."""

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: {file_path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read JSON file {file_path}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc


def write_json(path: str | Path, obj: Any) -> None:
    file_path = Path(path)
    payload = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    try:
        file_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Unable to write JSON file {file_path}: {exc}") from exc


def make_output_path(input_path: str | Path) -> Path:
    input_file = Path(input_path)
    return input_file.with_name(f"{input_file.stem}.labeled.json")
