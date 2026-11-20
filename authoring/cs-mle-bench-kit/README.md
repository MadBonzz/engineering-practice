# CS / MLE Engineering Bench Kit

This folder is a reusable harness for generating interview-style implementation assessments from topics you have already studied.

The intended use is deliberately simple:

1. Put any local notes, PDFs, papers, code, slides, or docs in `reference_material/`.
2. Either edit the two required fields in `INPUT.md` **or just tell Codex the topic + references in your opening message**.
3. Start Codex from **this folder**.
4. Tell Codex something as short as: **"Topic: KV caching. References: `reference_material/...`. Generate the assessment; read AGENTS.md and follow the repository workflow."**
5. When generation finishes, run:

   ```bash
   python tools/prepare_attempt.py <assessment-slug>
   ```

6. Start a **new Codex session** inside `attempts/<assessment-slug>/` and solve the task yourself.
7. When finished, from the kit root run:

   ```bash
   python tools/prepare_review.py <assessment-slug>
   ```

8. Start a **new Codex session** inside `reviews/<assessment-slug>/` and say: **"Review my attempt."**

The generation session creates both a student-facing task and a private grader/reference bundle. `prepare_attempt.py` copies only the student-facing material into a clean workspace so the assessment session does not need access to the solution or hidden tests.

## What this kit is trying to reproduce

The target is not LeetCode and not a tutorial. It is the style of engineering assessment where you are placed in a small existing codebase, receive requirements in stages, and must make precise changes without breaking previous behavior.

The design standards intentionally borrow from several public hiring patterns, including:

- Perplexity's practical and hands-on coding format.
- CodeSignal's Industry Coding Assessment and progressive project framework.
- DSLabs-style starter frameworks, staged requirements, and invariant-driven testing.
- Sourcegraph's time-bounded zipped projects and job-relevant engineering exercises.
- HRT's emphasis on implementation, testing, corner cases, systems knowledge, and methodical problem solving.
- Sarvam AI candidate-reported proof-of-work ML build tasks.
- Candidate-reported progressive assessments at Anthropic, Dropbox, Ramp, and similar companies.
- Practical/debugging signals reported in Tower Research and Datadog processes.

See `design/REFERENCE_ARCHETYPES.md` for source links, confidence levels, and exactly what the generator should borrow from each one.

## Minimal INPUT.md

Only these two fields are required:

```md
# Topic
KV caching and prefix caching

# Learning references
- ./reference_material/sglang-radix-attention.pdf
- https://example.com/docs-i-used
```

Everything else in `INPUT.md` is optional and has defaults.

## Repository layout

```text
cs-mle-bench-kit/
├── AGENTS.md                    # Codex entry point for assessment generation
├── INPUT.md                     # You edit this
├── README.md                    # This file
├── reference_material/          # Drop learning materials here
├── design/
│   ├── TASK_SELECTION_RUBRIC.md
│   ├── ASSESSMENT_STANDARD.md
│   ├── TESTING_STANDARD.md
│   ├── REFERENCE_ARCHETYPES.md
│   ├── GENERATION_WORKFLOW.md
│   └── QUALITY_GATE.md
├── modes/
│   ├── ATTEMPT_MODE.md
│   └── REVIEW_MODE.md
├── tools/
│   ├── prepare_attempt.py
│   └── prepare_review.py
├── out/                         # Generator writes student + private bundles here
├── attempts/                    # Clean assessment workspaces
└── reviews/                     # Post-assessment review workspaces
```

## Default assessment shape

Unless the topic strongly suggests another format, the generator should produce a 60–120 minute assessment with:

- one coherent real-world system scenario;
- an existing multi-file starter codebase;
- 2–4 cumulative parts;
- roughly 3–8 meaningful TODO methods across 2–5 files;
- public part-level tests;
- hidden integration/adversarial tests;
- a private reference implementation;
- mutation checks proving that plausible wrong solutions are rejected;
- a final review mode that critiques reasoning and design, not merely the score.

The exact test machinery should match the topic. A distributed-systems task might need fault injection and invariant checks. A parser/router task might need fuzzing and property tests. An ML-serving task might use deterministic simulation, traces, cache state, and workload-based metrics.
