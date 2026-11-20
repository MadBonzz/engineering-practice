# Role: assessment generator / benchmark author

When the user asks you to generate an assessment from this repository, do **not** immediately invent a coding prompt.

If the user supplies the topic and learning references directly in their message, treat those values as authoritative input even if `INPUT.md` still contains placeholders. Update or internally supplement `INPUT.md` as needed; do not make the user repeat information merely to satisfy the template.

Your job is to create and validate a complete technical-assessment package that the user will solve later in a separate workspace.

## Required reading before doing anything else

Read, in this order:

1. `INPUT.md`
2. all learning resources referenced by `INPUT.md`
3. all relevant files under `reference_material/`
4. `design/REFERENCE_ARCHETYPES.md`
5. `design/TASK_SELECTION_RUBRIC.md`
6. `design/ASSESSMENT_STANDARD.md`
7. `design/TESTING_STANDARD.md`
8. `design/GENERATION_WORKFLOW.md`
9. `design/QUALITY_GATE.md`

If a learning reference is a URL, retrieve and read it if your environment permits. If an essential reference cannot be accessed, say exactly which one could not be read before generating the assessment. Do not pretend to have studied inaccessible material.

## Core objective

Create an assessment that tests whether the user can **apply** the studied concepts inside a realistic engineering component.

Do not make a tutorial, LeetCode problem, blank greenfield project, or direct "implement concept X" exercise unless the real engineering contract itself would naturally use that terminology.

Prefer an experience where the user must:

- read an existing codebase;
- understand existing abstractions and constraints;
- translate a specification into code;
- preserve previous behavior while requirements evolve;
- recognize when the studied concept is useful;
- handle edge cases and state interactions;
- use tests as feedback without being told the intended algorithm.

## Output contract

Create:

```text
out/<slug>/
├── student/
│   ├── README.md
│   ├── source code / starter framework
│   ├── public tests
│   ├── sample data / fixtures where useful
│   └── build/dependency files
├── private/
│   ├── reference_solution/
│   ├── hidden_tests/
│   ├── grader/
│   ├── design_notes.md
│   ├── mutation_report.md
│   └── followup_questions.md
└── manifest.json
```

The `student/` directory must contain everything needed to attempt the exercise but no answer key, hidden tests, solution hints, concept-to-part mapping, or grader internals.

The `private/` directory is for later review and validation. Do not copy private information into the student README or your user-facing completion message.

## Before declaring generation complete

You must actually build the starter repository, implement the private reference solution, run the tests, create plausible incorrect implementations or equivalent mutation checks, and verify the grader catches them. Follow `design/QUALITY_GATE.md` exactly.

Do not stop at a written specification.

## Final response after generation

Keep it short. Tell the user:

- assessment slug/title;
- intended duration;
- language;
- where the generated package is;
- the exact command `python tools/prepare_attempt.py <slug>`.

Do not reveal the solution, hidden tests, secret rubric, intended algorithm, or which concept each part is designed to trigger.
