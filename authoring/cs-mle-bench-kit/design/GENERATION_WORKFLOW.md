# Generation workflow

Follow this sequence. Do not skip directly to writing starter code.

## Phase 1 — Learn the user's learning boundary

Read the supplied study material.

Privately write `out/<slug>/private/design_notes.md` containing:

- concepts explicitly covered;
- important sub-concepts;
- prerequisites the material assumes;
- implementation consequences;
- realistic failure modes;
- common misconceptions;
- concepts that would be unfair because they were not covered and are not standard prerequisites.

This file is private and must not be shown during the attempt.

## Phase 2 — Generate and select candidate task designs

Read `REFERENCE_ARCHETYPES.md` and `TASK_SELECTION_RUBRIC.md`.

Privately create at least four substantially different task concepts. Score them using the rubric. Reject concepts that are contrived, weakly testable, unfair relative to the learning resource, or poorly calibrated.

Choose the strongest task and record why it won in private design notes.

Then choose the most appropriate assessment-archetype blend.

Default preference order:

1. **Progressive framework task** — existing code + cumulative requirements. Best general-purpose choice.
2. **Framework completion / invariant task** — stronger DSLabs flavor when state, concurrency, or distributed behavior dominates.
3. **Proof-of-work build** — useful when the topic is model/data/audio/retrieval experimentation and objective quality can be measured.
4. **Debugging + extension task** — useful for testing, systems, performance, runtime behavior, or code-reading skills.

You may combine them. Example: a starter inference simulator with progressive parts and a hidden workload benchmark.

Do not copy a company's proprietary or candidate-reported question. Borrow assessment *structure and signal*, not leaked content.

## Phase 3 — Design the chosen scenario and progression

Privately define:

- the system being built;
- why the system needs this component;
- what the student inherits;
- what each part adds;
- which requirements intentionally reward good earlier abstractions;
- what behavior is externally observable;
- what common wrong implementations should fail.

Make sure later parts are cumulative.

## Phase 4 — Build the starter repository

Create `out/<slug>/student/`.

Implement the boring infrastructure yourself. Leave only the reasoning-heavy gaps for the student.

The starter code should be coherent and runnable.

Write a student README that explains WHAT must happen without unnecessarily prescribing HOW.

## Phase 5 — Build public tests

Add part-level public tests and useful fixtures.

Run them on the untouched starter code and confirm meaningful failures.

## Phase 6 — Build the private reference implementation

Copy the student repository into `private/reference_solution/` and implement a clean correct solution.

Do not optimize for cleverness; optimize for correctness and clarity.

## Phase 7 — Build hidden tests and grader

Add deeper property, stateful, integration, adversarial, resource, and/or performance checks as appropriate.

The private grader should be runnable with a single documented command.

## Phase 8 — Mutation validation

Create plausible wrong variants and verify hidden/public tests reject them.

Record results in `private/mutation_report.md`.

## Phase 9 — Create private follow-up questions

Create `private/followup_questions.md` with 3–6 questions an interviewer could ask after the implementation. Focus on tradeoffs, failure modes, complexity, testing strategy, or extensions. These are for review mode only.

Questions should probe understanding without assuming the reference implementation is the only valid design.

## Phase 10 — Calibration pass

Estimate how long a strong target candidate would need.

Check student code volume, amount of required reading, and likely debugging time.

If the task is too large, remove routine work rather than deleting the reasoning-heavy part.

## Phase 11 — Final quality gate

Run every item in `QUALITY_GATE.md`.

Only then hand off the assessment.
