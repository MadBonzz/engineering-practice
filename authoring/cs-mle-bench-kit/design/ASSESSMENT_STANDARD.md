# Assessment design standard

## 1. The task is an engineering component, not a concept worksheet

The user has already studied theory. The assessment exists to test transfer from theory to implementation.

A poor task says:

> Implement longest-prefix matching.

A stronger task says:

> Complete the route-selection component. Multiple route entries may cover an address; select the valid route according to the documented specificity and policy rules.

A poor task says:

> Implement a KV cache.

A stronger task says:

> Complete the state-reuse layer for an inference worker. Requests may share already-computed token prefixes, worker memory is bounded, and the scheduler must correctly account for reusable and evictable state.

The engineering scenario should have a reason to exist independently of the concept being practiced.

## 2. Existing code is mandatory by default

The student should not start from a blank file.

Unless there is a strong topic-specific reason otherwise, create a starter codebase with roughly:

- 5–15 source/test/support files;
- 300–1000 lines of already-working code, documentation, fixtures, and tests;
- 3–8 meaningful TODO methods/functions;
- TODOs spread across 2–5 implementation files;
- at least one cross-file interaction that requires understanding the architecture.

Most boilerplate must already work. The student's effort should concentrate on the reasoning-heavy behavior.

Good pre-existing elements include:

- domain/data classes;
- interfaces / protocols / abstract classes;
- configuration objects;
- parsers or serializers;
- simulator/event loop;
- request/response models;
- workload generator;
- test fixtures;
- CLI;
- existing components the student's code must cooperate with.

Do not create complexity merely by adding files. Every provided abstraction should serve the scenario.

## 3. Use cumulative parts

Default: 3 parts, acceptable range 2–4.

Parts must build one coherent system. Later parts should force the student to extend, reuse, or refactor earlier behavior rather than solve unrelated mini-problems.

A useful generic shape is:

- **Part A — Local correctness / representation.** Parsing, validation, state representation, core operation, or local functionality.
- **Part B — System behavior.** Integrate the Part A component into routing, scheduling, storage, execution, retrieval, etc.
- **Part C — Interaction / evolving requirement.** Introduce overlap, history, capacity, failure, stale state, retries, dependency, reconfiguration, or another natural complication.
- **Part D — Extension.** Only if appropriate: performance, backward-compatible feature, historical query, production constraint, or quality tradeoff.

Do not force this exact pattern when another progression is more natural.

The important property is that an early design can make later work easier or harder, just as in progressive engineering assessments.

## 4. Hide the intended mechanism when recognition matters

The README must specify **behavior** precisely but should avoid giving away the internal algorithm or data structure unless the implementation strategy is genuinely part of the contract.

Bad TODO:

```python
# Traverse the prefix trie and return the longest matching prefix.
```

Better TODO:

```python
# Return the most specific applicable route for this address.
```

Bad README:

> Use an LRU cache to evict entries.

Better README:

> When capacity is exceeded, evict reusable state according to the documented recency policy.

The user should sometimes have to identify the concept they studied as the natural solution.

## 5. Be precise about the external contract

Do not manufacture difficulty through ambiguity.

The student-facing README must clearly document:

- scenario and system boundary;
- repository architecture;
- files allowed to change;
- public interfaces/signatures;
- inputs and outputs;
- validation/error behavior;
- state-transition rules;
- ordering/tie-breaking rules when externally visible;
- invariants;
- assumptions;
- resource/performance constraints if tested;
- test commands for each part and for the full public suite.

Hidden tests may explore unstated **examples**, but not unstated **semantics**.

## 6. Make code-reading matter

At least some requirements should require the student to notice existing behavior in the codebase rather than reimplement everything in isolation.

Examples:

- an existing parser already canonicalizes a field;
- a scheduler calls the student's method through an interface;
- a model object encodes a state invariant;
- a simulator controls time, so wall-clock calls would be wrong;
- a provided storage abstraction must be respected;
- existing code assumes stable method signatures.

Do not hide arbitrary traps. The codebase should reward careful engineering reading.

## 7. Calibrate for technical-screen implementation, not homework volume

Default target: 60–120 minutes.

Prefer roughly 50–200 lines of thoughtful student code. Some topics may justify more or less.

Difficulty should come from:

- understanding requirements;
- adapting to existing code;
- state modeling;
- edge cases;
- maintaining invariants;
- evolving requirements;
- choosing appropriate abstractions;
- debugging interactions.

Difficulty should **not** mainly come from:

- environment setup;
- giant amounts of boilerplate;
- obscure APIs;
- puzzle algorithms unrelated to the topic;
- undocumented behavior;
- arbitrary syntax tricks;
- huge data downloads;
- external paid services.

For GPU/distributed topics, use a faithful deterministic simulator or CPU-scale model when that preserves the reasoning being tested.

## 8. MLE-specific assessments should still be engineering assessments

For ML/MLE topics, do not default to "train a model and maximize accuracy" unless that is genuinely the concept being practiced.

Good MLE task shapes include:

- serving/scheduling state machine;
- cache-aware routing;
- batching and admission logic;
- retrieval/index update behavior;
- evaluation harness;
- data validation pipeline;
- model-output post-processing;
- experiment/result parser;
- failure recovery around model workers;
- deterministic simulation of memory/latency tradeoffs;
- a small dataset plus a measurable functional objective when the task is genuinely model-facing.

The problem should expose engineering constraints and testable behavior.
