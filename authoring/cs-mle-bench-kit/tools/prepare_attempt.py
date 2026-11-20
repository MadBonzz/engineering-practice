#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
ATTEMPTS = ROOT / "attempts"
ATTEMPT_MODE = ROOT / "modes" / "ATTEMPT_MODE.md"


def resolve_slug(raw: str | None) -> str:
    if raw:
        return raw
    candidates = sorted(
        [p for p in OUT.iterdir() if p.is_dir() and (p / "student").is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if OUT.exists() else []
    if not candidates:
        raise SystemExit("No generated assessments found in out/. Pass a slug explicitly.")
    return candidates[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean student-only assessment workspace.")
    parser.add_argument("slug", nargs="?", help="Assessment slug. Defaults to most recently generated assessment.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing attempt workspace.")
    args = parser.parse_args()

    slug = resolve_slug(args.slug)
    src = OUT / slug / "student"
    if not src.is_dir():
        raise SystemExit(f"Student bundle not found: {src}")

    dest = ATTEMPTS / slug
    if dest.exists():
        if not args.force:
            raise SystemExit(f"Attempt workspace already exists: {dest}\nUse --force to replace it.")
        shutil.rmtree(dest)

    shutil.copytree(src, dest)

    original_agents = dest / "AGENTS.md"
    if original_agents.exists():
        original_agents.rename(dest / "ASSESSMENT_AGENTS.md")

    mode_text = ATTEMPT_MODE.read_text(encoding="utf-8")
    extra = "\n\n# Student repository instructions\n"
    if (dest / "ASSESSMENT_AGENTS.md").exists():
        extra += "Also read `ASSESSMENT_AGENTS.md` for task-specific public repository instructions.\n"
    (dest / "AGENTS.md").write_text(mode_text + extra, encoding="utf-8")

    meta = {
        "slug": slug,
        "source": str(src),
        "attempt_workspace": str(dest),
        "private_material_copied": False,
    }
    (dest / ".bench_attempt.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Prepared clean attempt workspace: {dest}")
    print("Next:")
    print(f"  cd {dest}")
    print("  codex")
    print('Then say: "I am starting the assessment. Read AGENTS.md and README.md."')


if __name__ == "__main__":
    main()
