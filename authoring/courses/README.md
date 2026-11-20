# Authoring course tasks

These are original, standard-library Python exercises informed by the linked course
lectures, homework handouts and prediction quizzes. They do not reproduce official
assignments, solutions, datasets or quiz answer keys.

## Source and build boundary

- `foundations.py`, `systems.py`, `language.py`, `experiments.py`: complete reference
  functions, explicit contracts, hand-computed oracle cases, experiment inputs and
  four plausible implementation mutations per task. These files are authoring
  material, not browser assets.
- `common.py`: converts function definitions into three editable modules and a
  supplied request gateway. Earlier parts are imported by later parts.
- `curriculum.py`: course/unit mapping and reuse of existing task IDs.
- `../build_courses.py`: generates student-facing files and validates the actual
  grading workload against references and mutants.
- `validation.json`: reproducible validation evidence; no timing-dependent fields.

Run from the repository root:

```sh
python -B authoring/build_courses.py
python -B -m unittest discover -s tests -p test_course_content.py
node --test tests/course-api.test.mjs
npm run build
```

The first command needs only Python 3.10+. The API tests use Node's built-in
TypeScript stripping (`node:module.stripTypeScriptTypes`), available in Node 22.13+.
The site uses its existing npm lockfile and installation/build scripts.

Generated outputs:

- `public/courses/<slug>/exercise.json` and `starter.zip`: identical student files.
- `public/course-new-catalog.json`: the new projects' editable-file allowlists.
- `public/course-catalog.json`: new and reused tasks, deduplicated by stable slug.
- `public/course-map.json`: course coverage, source links and adaptation limits.
- `worker/course-cases.json`: server-held grading expectations; never put in public.

## Task contract

Every new lab has three cumulative parts across three implementation modules,
provided gateway/workload/CLI files, two public checks per part, two worked examples,
private boundary and integration workloads, progressive hints, prerequisite links,
a runnable `experiment.py`, and a `REFLECTION.md` notebook.

The generator runs the public suite and examples against the reference, verifies
that the untouched starter fails because its implementation is unfinished, checks
all declared oracles, then runs the hidden action bank against four distinct
mutations. It also checks input immutability and runs the shipped experiment driver.
Independent regression tests use finite differences and algebraic invariants for
numerical routines, Unicode round trips, token-count invariance, and distributed
update equivalence. API tests exercise the real grading handler with an in-memory
R2 test double; the browser uses the app's normal R2-backed local development path.

The notebook is editable, autosaved, included in submission snapshots and exports,
and intentionally ignored by numerical grading. Prediction quality and written
scientific reasoning require self-review or a human reviewer.

## Extending a task

Specify valid shapes, reductions, ordering, tie-breaking, state updates, limits and
numerical tolerances in the student contracts before adding private cases. Include
independent expected values where practical. A reference-derived expected result
alone is not evidence that the mathematical implementation is right.

Do not ship reference implementations, mutation definitions, or the private action
bank in starter ZIPs. Existing repository code is public on GitHub; “private” here
means excluded from the served student assets, not confidential from repository
readers. Existing grading remains intended for self-practice.

New `course-` tasks use the existing current-edition `labs/` storage namespace,
while reused tasks retain their original slug and progress key. Avoid changing
those IDs after learners have saved work.
