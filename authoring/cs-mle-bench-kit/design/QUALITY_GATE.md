# Final quality gate

Do not release an assessment until every applicable item below is true.

## Learning alignment

- [ ] I read the user's supplied learning material.
- [ ] Core requirements are derivable from the studied material plus normal prerequisites.
- [ ] I did not add unrelated specialist knowledge just to make the task harder.
- [ ] The assessment requires application, not merely recall.

## Task selection

- [ ] I generated at least four candidate task designs.
- [ ] I scored them using `TASK_SELECTION_RUBRIC.md`.
- [ ] The chosen task was not simply the first idea.
- [ ] The private notes explain why this scenario is the best learning/interview simulation for the topic.

## Scenario

- [ ] The system has a plausible real-world purpose.
- [ ] The studied concept appears naturally inside that purpose.
- [ ] The README does not unnecessarily name the intended algorithm/data structure.
- [ ] This does not feel like a LeetCode problem with a story pasted on top.

## Starter code

- [ ] The student receives meaningful existing code to read.
- [ ] The repository has coherent abstractions and realistic names.
- [ ] The TODOs are concentrated on the reasoning-heavy functionality.
- [ ] The intended solution works within the existing architecture.
- [ ] The task cannot be reduced to one isolated blank function unless that is unusually appropriate.

## Progression

- [ ] Parts are cumulative.
- [ ] Earlier design choices influence the ease of later work.
- [ ] Later parts preserve previous contracts unless a documented contract change is part of the exercise.
- [ ] The student can run tests after each part.

## Specification fairness

- [ ] Inputs/outputs/errors/invariants are precise.
- [ ] Tie-breaking and state-transition semantics are documented when observable.
- [ ] Hidden tests do not rely on secret semantics.
- [ ] Multiple valid implementations are accepted.
- [ ] There are no arbitrary gotchas.

## Testing

- [ ] Public tests cover basic contract and give useful feedback.
- [ ] Hidden tests cover materially deeper cases.
- [ ] Stateful behavior is tested with sequences where relevant.
- [ ] Invariants/properties are tested where stronger than example assertions.
- [ ] Resource/performance tests are included only where justified.
- [ ] At least four plausible mutants were tested.
- [ ] The grader catches the important mutants.
- [ ] The untouched starter fails meaningful tests.
- [ ] The reference solution passes everything.

## Difficulty / realism

- [ ] The estimated duration matches `INPUT.md`.
- [ ] The amount of boilerplate the student must write is reasonable.
- [ ] Difficulty comes from engineering reasoning, not setup pain.
- [ ] A strong candidate may reasonably not finish every optional/final extension.
- [ ] The task rewards methodical reading, testing, and debugging.

## Separation

- [ ] `student/` contains no hidden tests or solution hints.
- [ ] `private/` contains the reference solution, hidden tests, grader, design notes, mutation report, and follow-up questions.
- [ ] The generation completion message reveals no private reasoning or answer.
