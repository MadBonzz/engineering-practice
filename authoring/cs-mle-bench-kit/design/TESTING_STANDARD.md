# Testing and grading standard

Testing is part of the assessment design, not an afterthought.

## 1. Two test layers

Every generated assessment should have:

### Student-visible tests

These should:

- support each part independently;
- verify the basic contract;
- provide useful failure messages;
- show how the interfaces are used;
- make debugging possible;
- include enough edge cases that the student understands the specification is strict.

They should not enumerate every adversarial case or effectively reveal the reference algorithm.

### Hidden tests

These live only in `private/hidden_tests/` and should probe:

- additional boundaries;
- adversarial combinations;
- cross-part integration;
- stateful sequences;
- likely superficially-correct implementations;
- scale/resource behavior where relevant;
- randomized/property checks where appropriate.

A hidden failure should identify the violated high-level property, not immediately reveal the fix.

## 2. Use the strongest applicable test style

Do not blindly imitate distributed-systems testing for every topic. Select test machinery that matches the property being assessed.

Possible mechanisms:

### Exact unit tests
For deterministic local behavior.

### Property/invariant tests
Examples:

- selected route actually contains the queried address;
- a supposedly canonical prefix has all host bits cleared;
- cache usage never exceeds capacity;
- each logical request executes at most once;
- every shard has exactly one owner;
- committed replicas agree at the same log index;
- historical queries never return state created after the query timestamp.

### Stateful sequence tests
Generate or script operation sequences where the result depends on previous state.

### Differential tests against a small oracle
Useful for parsers, routing, caches, schedulers, indexes, accounting, and state machines.

The oracle may be deliberately slow and simple because it is private test infrastructure.

### Randomized / fuzz tests
Useful for parsing, validation, overlapping ranges, malformed input, state-machine transitions, ordering, and idempotency.

Use deterministic seeds.

### Fault/reordering tests
When the topic warrants it, simulate duplication, retries, loss, reordering, stale state, timeouts, worker failure, etc.

### Resource/longevity tests
When relevant, test bounded state, cleanup, memory/capacity, long-running workloads, or accidental unbounded metadata.

### Performance/complexity tests
Only when performance is part of the documented engineering requirement. Use robust thresholds or operation counts rather than fragile microbenchmarks whenever possible.

## 3. Part-level and final tests

Each part needs a runnable command. Example:

```bash
pytest -q tests/public/part1
pytest -q tests/public/part2
pytest -q tests/public/part3
pytest -q tests/public
```

or the language-appropriate equivalent.

The final public suite must rerun earlier requirements as regression tests.

The hidden grader should also be able to run per-part checks plus a complete integrated grade.

## 4. Fairness rule

Every hidden test must be justified by at least one of:

- explicit README requirement;
- public interface/type contract;
- clearly documented invariant;
- normal language semantics;
- a natural consequence of the supplied assumptions.

Do not grade on an undocumented implementation strategy.

If multiple correct designs satisfy the public contract, hidden tests must accept them.

## 5. Test the tests with mutations

Before handoff, create at least 4 plausible wrong implementations or equivalent mutations. Prefer mistakes such as:

- off-by-one boundary;
- missing canonicalization;
- incorrect tie-breaking;
- first match instead of most specific match;
- stale-state reuse;
- missing cleanup;
- duplicate execution under retry;
- only direct dependency handling, missing transitive effect;
- wrong state at historical timestamp;
- cache eviction that works on examples but violates capacity or recency;
- locally correct output that breaks a global invariant;
- quadratic implementation when the documented scale rules it out.

Run the grader against each mutant and record which tests kill it in `private/mutation_report.md`.

If a clearly wrong mutant passes, strengthen the grader before release.

## 6. Validate the starter state

The untouched starter repository must:

- build/import successfully unless a TODO intentionally prevents a particular test;
- fail meaningful student tests;
- not accidentally satisfy most requirements;
- produce clear failures rather than unrelated setup errors.

## 7. Reference solution requirement

Maintain a private, complete reference implementation. Run:

- all public tests;
- all hidden tests;
- lint/type checks if the project uses them;
- any deterministic benchmark/resource checks.

Everything must pass before handoff.
