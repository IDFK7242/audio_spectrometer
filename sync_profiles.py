#!/usr/bin/env python3
"""
Syncs profiles.json into index.html so the Render static site and Python receiver
always share the exact same acoustic protocol definitions.

Usage:
    python3 sync_profiles.py          # Update index.html from profiles.json
    python3 sync_profiles.py --check  # Verify parity (exits 0 if in sync, 1 otherwise)
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES_PATH = ROOT / "profiles.json"
HTML_PATH = ROOT / "index.html"

BEGIN_MARKER = "// === BEGIN_PROFILES_JSON ==="
END_MARKER = "// === END_PROFILES_JSON ==="

BLOCK_RE = re.compile(
    rf"([ \t]*{re.escape(BEGIN_MARKER)}\n)(.*?)(\n[ \t]*{re.escape(END_MARKER)})",
    re.DOTALL,
)


def build_js_block(data: dict, indent: str = "        ") -> str:
    dumped = json.dumps(data, indent=2, ensure_ascii=False)
    lines = dumped.splitlines()
    indented = "\n".join(
        (f"{indent}{line}" if idx > 0 else line) for idx, line in enumerate(lines)
    )
    return f"{indent}const SHARED_CONFIG = {indented};"


def main() -> int:
    check_only = "--check" in sys.argv

    if not PROFILES_PATH.exists():
        print(f"ERROR: {PROFILES_PATH} not found.", file=sys.stderr)
        return 1
    if not HTML_PATH.exists():
        print(f"ERROR: {HTML_PATH} not found.", file=sys.stderr)
        return 1

    profiles_data = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    html_text = HTML_PATH.read_text(encoding="utf-8")

    match = BLOCK_RE.search(html_text)
    if not match:
        print(
            f"ERROR: Could not find sync markers ({BEGIN_MARKER} / {END_MARKER}) in {HTML_PATH.name}.",
            file=sys.stderr,
        )
        return 1

    # Preserve leading indentation of the BEGIN_MARKER line
    marker_line = match.group(1)
    indent = marker_line[: len(marker_line) - len(marker_line.lstrip(" \t"))]

    expected_body = build_js_block(profiles_data, indent=indent)
    current_body = match.group(2)

    if current_body == expected_body:
        print("OK: index.html is in sync with profiles.json.")
        return 0

    if check_only:
        print(
            "OUT OF SYNC: index.html does not match profiles.json.\n"
            "Run `python3 sync_profiles.py` to synchronize them.",
            file=sys.stderr,
        )
        return 1

    updated_html = (
        html_text[: match.start(2)] + expected_body + html_text[match.end(2) :]
    )
    HTML_PATH.write_text(updated_html, encoding="utf-8")
    print(f"SYNCED: Updated {HTML_PATH.name} from {PROFILES_PATH.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
