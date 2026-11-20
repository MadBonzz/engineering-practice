"""Build runnable systems labs, validating the stubs, examples, public suites and hidden cases."""
from pathlib import Path
import runpy

from build_papers import build
from build_sets import PUBLIC, ROOT, compact

SOURCE = Path(__file__).resolve().parent / "systems"
GROUPS = ["Distributed systems", "Database internals", "Operating systems", "Networking", "Compilers and runtimes"]


def main():
    labs = [lab for path in sorted(SOURCE.glob("*.py")) for lab in runpy.run_path(str(path))["LABS"]]
    assert len({lab["slug"] for lab in labs}) == len(labs)
    # Put the requested entry points first and keep the remaining topics browsable.
    def position(lab):
        title = lab["title"].lower().replace("–", "-")
        for i, term in enumerate(("three-phase", "consistent hash", "redis")):
            if term in title:
                return 0, i, title
        return 1, GROUPS.index(lab["group"]), title

    catalog, cases = [], {}
    for lab in sorted(labs, key=position):
        try:
            entry, scenarios = build(lab, systems=True)
        except Exception as exc:
            raise RuntimeError(lab["slug"]) from exc
        catalog.append(entry)
        cases[entry["slug"]] = scenarios
        print(entry["slug"], entry["parts"], len(scenarios), flush=True)
    (PUBLIC / "systems-catalog.json").write_text(compact(catalog))
    (ROOT / "worker" / "systems-cases.json").write_text(compact(cases))


if __name__ == "__main__":
    main()
