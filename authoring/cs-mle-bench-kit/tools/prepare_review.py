#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
ATTEMPTS = ROOT / "attempts"
REVIEWS = ROOT / "reviews"
REVIEW_MODE = ROOT / "modes" / "REVIEW_MODE.md"


def resolve_slug(raw: str | None) -> str:
    if raw:
        return raw
    candidates = sorted(
        [p for p in ATTEMPTS.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if ATTEMPTS.exists() else []
    if not candidates:
        raise SystemExit("No attempt workspaces found. Pass a slug explicitly.")
    return candidates[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a post-assessment review workspace with private grader material.")
    parser.add_argument("slug", nargs="?", help="Assessment slug. Defaults to most recent attempt.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing review workspace.")
    args = parser.parse_args()

    slug = resolve_slug(args.slug)
    submission = ATTEMPTS / slug
    private = OUT / slug / "private"
    manifest = OUT / slug / "manifest.json"

    if not submission.is_dir():
        raise SystemExit(f"Attempt workspace not found: {submission}")
    if not private.is_dir():
        raise SystemExit(f"Private grader bundle not found: {private}")

    dest = REVIEWS / slug
    if dest.exists():
        if not args.force:
            raise SystemExit(f"Review workspace already exists: {dest}\nUse --force to replace it.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    shutil.copytree(submission, dest / "submission")
    shutil.copytree(private, dest / "private")
    if manifest.exists():
        shutil.copy2(manifest, dest / "manifest.json")

    (dest / "AGENTS.md").write_text(REVIEW_MODE.read_text(encoding="utf-8"), encoding="utf-8")
    (dest / ".bench_review.json").write_text(
        json.dumps({"slug": slug, "submission": "submission", "private": "private"}, indent=2),
        encoding="utf-8",
    )

    print(f"Prepared review workspace: {dest}")
    print("Next:")
    print(f"  cd {dest}")
    print("  codex")
    print('Then say: "Review my attempt. Read AGENTS.md and run the full grader."')


if __name__ == "__main__":
    main()
