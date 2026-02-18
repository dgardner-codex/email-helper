"""CLI entrypoint for deterministic scaffold."""

import sys
import argparse
from pathlib import Path

from Constants import CATEGORIES_FILE
from classifier import label_email
from io_utils import make_output_path, read_json, write_json
from trace import _trace
from validation import (
    enforce_only_labels_changed,
    validate_categories,
    validate_input_emails,
)


def run(input_json_file: str) -> Path:
    _trace("-------------------------") # run separator
    _trace("startup")
    print("Starting labeling run...")

    categories = validate_categories(read_json(CATEGORIES_FILE))
    _trace(f"loaded categories from {CATEGORIES_FILE}")

    input_path = Path(input_json_file)
    emails = validate_input_emails(read_json(input_path))
    _trace(f"loaded input emails from {input_path}")

    print(f"Loaded {len(categories)} categories")
    print(f"Loaded {len(emails)} emails")

    labeled_emails: list[dict[str, str]] = []
    for index, email in enumerate(emails, start=1):
        _trace(f"email {index} start")
        category, priority, _meta = label_email(email, categories)

        updated_email = dict(email)
        updated_email["category"] = category
        updated_email["priority"] = priority

        enforce_only_labels_changed(email, updated_email)
        labeled_emails.append(updated_email)
        _trace(f"email {index} end")

    output_path = make_output_path(input_path)
    write_json(output_path, labeled_emails)
    _trace(f"wrote output to {output_path}")

    print(f"Wrote labeled output to {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Email labeling CLI application."
    )

    parser.add_argument(
        "input_file",
        help="Path to input JSON file containing unsolved emails.",
    )

    args = parser.parse_args()

    try:
        run(args.input_file)
    except Exception as exc:
        _trace("Error:\n" + traceback.format_exc())
        print(f"Error: {exc}\nSee trace.txt for details.")
        return 1
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
