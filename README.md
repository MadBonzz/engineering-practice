# Engineering Practice Bench

The authoring kit is in authoring/cs-mle-bench-kit. Its generator contract, assessment standards, attempt rules, and review rubric are preserved. The first hosted assessment is in assessments/release-comparison, with the same student/private split. Its manifest sets the editable files.

## Practice
The private site serves a multi-file CodeMirror workspace. A fresh web worker runs the vendored CPython 3.12 (Pyodide 0.27.7) on each test or submission. Public tests use unittest, including the learner's tests. Run a part separately or the complete suite. Export attempt produces an ordinary Python project with the full README and no hidden material.

A random workspace ID links the browser draft to an R2 record. Copy resume link carries that ID in the URL fragment. Server-side conditional writes detect competing drafts. Each submission has its own immutable snapshot and feedback record, available from Review. Access is restricted by the private site; treat a resume link as personal.

## Assessment preparation
Supply a topic and learning references to the original generator. Its output comes in student and private directories. The included assessment is a short calibration based on the prior paired-evaluation discussion. The supplied reference and held-back workload bank stay outside public assets.

For the calibration, run:
    python -B assessments/release-comparison/private/grader/validate.py
The validation checks the reference, untouched starter, independent workload oracle and six realistic mutants. It refreshes the student bundle, starter ZIP and worker-only grading fixtures. The action protocol in web/protocol.py is public. Never place the expected values, private tests or reference solution under public/.

## Execution boundary
Submission freezes the editable files server-side. Workloads are disclosed only after submission; CPython replays them in the browser. The server compares the returned observations to its private oracle, records per-part results and shares representative counterexamples. This is a self-practice grader. Native processes, CUDA and independently trusted execution would require an isolated compute service.

For a qualitative postmortem, export the graded snapshot from Review and bring it to Codex with the kit's review rubric. The site itself provides deterministic feedback.
