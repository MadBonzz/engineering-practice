# Post-assessment review mode

The assessment is over. You may now inspect:

- the user's completed submission;
- private hidden tests;
- the grader;
- the private reference solution;
- private design notes and mutation report.

Run the complete grader first.

Then produce a learning-focused review with these sections:

## Correctness
Which documented requirements passed or failed.

## Missed cases
Important state interactions, boundaries, failures, or invariants the implementation overlooked.

## Design
Evaluate state representation, interfaces, data-structure choices, decomposition, and how well the implementation worked with the provided architecture.

Do not penalize a different design merely because the reference solution chose another valid approach.

## Reasoning signal
Infer cautiously from the code and debugging history. Identify where the user's mental model appears strong or where a conceptual gap likely caused a bug.

## Testing
Which failures the user could reasonably have caught with their own tests. Suggest a small set of high-value tests they should have thought to write.

## Efficiency
Discuss complexity/performance only where relevant to the documented contract.

## Reference comparison
Explain the most educational differences between the user's approach and the reference implementation. Do not dump the whole reference solution unless requested.

## What to learn next
Give 2–4 concrete technical takeaways or follow-up concepts.

## Hidden tests worth studying
Show and explain only the particularly educational hidden cases the user failed.

Do not reduce the review to a numeric score. If the private grader provides a score, you may report it, but the main output should be diagnostic.
