# Task selection rubric

Do not commit to the first plausible scenario you think of. Before building the assessment, privately generate **at least 4 candidate task designs** and compare them.

Write the comparison into `out/<slug>/private/design_notes.md`; do not expose it during the attempt.

Score each candidate from 1–5 on the dimensions below. A task does not need the highest raw total if there is a compelling qualitative reason, but explain the selection privately.

## 1. Natural concept relevance
Would a real engineer plausibly need the studied concept in this scenario, or is the concept awkwardly pasted onto a story?

5 = the concept is intrinsic to the component.
1 = the scenario is basically an algorithm exercise in disguise.

## 2. Transfer / recognition value
Does the task require turning theory into behavior and possibly recognizing the right idea rather than merely following an instruction that names the solution?

5 = requirements imply the concept without handing over the mechanism.
1 = README says exactly which algorithm/data structure to implement and how.

## 3. Existing-code leverage
Can the task be embedded naturally in a small starter system where reading interfaces/state/fixtures matters?

5 = several meaningful interactions with existing code.
1 = one isolated blank function is the only sensible form.

## 4. Progressive extensibility
Can the system gain 2–4 coherent requirements such that later parts build on earlier choices?

5 = later requirements naturally pressure abstraction/refactoring while preserving contracts.
1 = parts would be unrelated mini-problems.

## 5. Testability
Can correctness be judged strongly and fairly?

5 = clear oracle/invariants/stateful properties/adversarial cases exist.
1 = quality is mostly subjective or impossible to check locally.

## 6. Reasoning density
How much useful engineering reasoning is required per line of student code?

5 = small amount of code with rich state/edge interactions.
1 = mostly boilerplate or repetitive glue.

## 7. Real interview realism
Would this plausibly fit a practical SWE/MLE technical screen or take-home?

5 = self-contained, job-relevant component with realistic constraints.
1 = semester project, research project, or contrived puzzle.

## 8. Fairness from supplied learning material
Can the user solve it using the supplied learning resource plus normal prerequisites?

5 = tightly aligned.
1 = depends on unrelated specialist knowledge.

## 9. Time calibration
Can a strong target candidate make meaningful progress within the requested time?

5 = core signal fits cleanly; final extension may stretch.
1 = either trivial or impossible within the window.

## 10. Failure-mode richness
Are there multiple plausible superficially-correct implementations that strong hidden tests can distinguish?

5 = many meaningful errors around state, ordering, validation, history, capacity, failure, etc.
1 = essentially one happy-path behavior.

# Selection rule

Reject any candidate scoring below 3 on **Natural concept relevance**, **Testability**, **Fairness**, or **Time calibration**, regardless of total score.

Prefer a task that makes a good implementation feel simple *after* the user has the right mental model, but produces realistic bugs when that model is incomplete.
