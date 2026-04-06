#!/usr/bin/env python3
"""
Create a publish-safe artifact bundle from Helios-generated outputs.

This script copies selected generated artifacts into a public bundle while:
- redacting hostnames
- rewriting absolute repo-local paths to $REPO_ROOT-relative placeholders
- rewriting temp/home paths to placeholders

The intent is to preserve proof artifacts without leaking the local machine name,
username, or absolute filesystem layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import socket
from pathlib import Path
from typing import Any


REDACTED_KEYS = {
    "host_hostname",
    "host_nodename",
}

PATH_LIKE_TOKENS = (
    "path",
    "command_line",
)


def normalize_repo_root(repo_root: Path) -> str:
    return str(repo_root.resolve())


def sanitize_string(value: str, repo_root: str, home_dir: str, host_name: str) -> str:
    sanitized = value

    if repo_root and repo_root in sanitized:
        sanitized = sanitized.replace(repo_root, "$REPO_ROOT")

    if home_dir and home_dir in sanitized:
        sanitized = sanitized.replace(home_dir, "$HOME")

    if host_name and host_name in sanitized:
        sanitized = sanitized.replace(host_name, "[redacted-host]")

    temp_prefixes = [
        "/tmp/",
        "/private/tmp/",
        "/var/folders/",
    ]
    for prefix in temp_prefixes:
        if prefix in sanitized:
            start = sanitized.find(prefix)
            tail = sanitized[start:]
            end = tail.find(" ")
            if end == -1:
                sanitized = sanitized.replace(tail, "[temp-path]")
            else:
                sanitized = sanitized.replace(tail[:end], "[temp-path]")

    return sanitized


def sanitize_value(key: str, value: Any, repo_root: str, home_dir: str, host_name: str) -> Any:
    if isinstance(value, dict):
        return {
            child_key: sanitize_value(child_key, child_value, repo_root, home_dir, host_name)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [sanitize_value(key, item, repo_root, home_dir, host_name) for item in value]
    if not isinstance(value, str):
        return value

    if key in REDACTED_KEYS:
        return "[redacted]"

    if any(token in key for token in PATH_LIKE_TOKENS):
        return sanitize_string(value, repo_root, home_dir, host_name)

    return sanitize_string(value, repo_root, home_dir, host_name)


def sanitize_json_file(src: Path, dst: Path, repo_root: str, home_dir: str, host_name: str) -> None:
    with src.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sanitized = sanitize_value("", payload, repo_root, home_dir, host_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as handle:
        json.dump(sanitized, handle, indent=2, sort_keys=False)
        handle.write("\n")


def sanitize_jsonl_file(src: Path, dst: Path, repo_root: str, home_dir: str, host_name: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as input_handle, dst.open("w", encoding="utf-8") as output_handle:
        for line in input_handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sanitized = sanitize_value("", payload, repo_root, home_dir, host_name)
            output_handle.write(json.dumps(sanitized))
            output_handle.write("\n")


def sanitize_csv_file(src: Path, dst: Path, repo_root: str, home_dir: str, host_name: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", newline="") as input_handle, dst.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.reader(input_handle)
        writer = csv.writer(output_handle)
        for row in reader:
            if len(row) >= 2:
                key = row[0]
                row[1] = str(sanitize_value(key, row[1], repo_root, home_dir, host_name))
            writer.writerow(row)


def sanitize_text_file(src: Path, dst: Path, repo_root: str, home_dir: str, host_name: str) -> None:
    text = src.read_text(encoding="utf-8")
    sanitized = sanitize_string(text, repo_root, home_dir, host_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(sanitized, encoding="utf-8")


def copy_sanitized_tree(src_root: Path, dst_root: Path, repo_root: str, home_dir: str, host_name: str) -> None:
    if not src_root.exists():
        return

    for src in sorted(path for path in src_root.rglob("*") if path.is_file()):
        relative = src.relative_to(src_root)
        dst = dst_root / relative
        if src.suffix == ".json":
            sanitize_json_file(src, dst, repo_root, home_dir, host_name)
        elif src.suffix == ".jsonl":
            sanitize_jsonl_file(src, dst, repo_root, home_dir, host_name)
        elif src.suffix == ".csv":
            sanitize_csv_file(src, dst, repo_root, home_dir, host_name)
        elif src.suffix in {".md", ".txt"}:
            sanitize_text_file(src, dst, repo_root, home_dir, host_name)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a sanitized public artifact bundle.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the Helios repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="public_artifacts",
        help="Directory where the sanitized bundle will be written.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    home_dir = str(Path.home())
    host_name = socket.gethostname()
    repo_root_str = normalize_repo_root(repo_root)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("results", repo_root / "results"),
        ("manifests", repo_root / "data" / "processed" / "manifests"),
        ("docs", repo_root / "docs"),
    ]

    for label, source_dir in sources:
        copy_sanitized_tree(source_dir, output_dir / label, repo_root_str, home_dir, host_name)

    summary = {
        "bundle_kind": "helios_public_artifacts_v1",
        "source_repo_root": "$REPO_ROOT",
        "sanitized_sources": [label for label, _ in sources if _.exists()],
        "notes": [
            "host_hostname and host_nodename were redacted",
            "absolute repo-local paths were rewritten to $REPO_ROOT",
            "home-directory paths were rewritten to $HOME",
            "temp paths were rewritten to [temp-path]",
        ],
    }
    with (output_dir / "SANITIZE_SUMMARY.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Sanitized public bundle written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
