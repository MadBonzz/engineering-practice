# Release comparison

Two models answer the same batch of prompts before a release. Evaluation workers can retry, correct, or withdraw a label. Complete the component that turns this stream into a rollout decision.

## Repository
events.py defines the input contract. ledger.py owns live outcomes. report.py builds a release report. Edit ledger.py and report.py. Add your own checks in tests/test_mine.py. All code uses Python's standard library.

## Part 1 · Track updates
An Event has a prompt ID, model ("baseline" or "candidate"), nonnegative integer revision, and correct (True, False, or None). All inputs are valid. Ingest events in any delivery order. For each prompt and model, the greatest revision received is authoritative. A newer None withdraws the label. Repeated deliveries at the same revision have identical contents.
latest(prompt, model) returns the current boolean or None for an absent or withdrawn label. prompts() includes every observed prompt, even after withdrawal. Calls to latest() and pair() are read-only.

## Part 2 · Match outcomes
pair(prompt) returns (baseline_correct, candidate_correct) if both current labels are available. Return None otherwise. A false outcome is a valid label. Reads may occur between any two updates.

## Part 3 · Decide a release
summarize(ledger, min_pairs) returns a dictionary with paired, wins, losses, ties, and decision. Count each currently comparable prompt once. A win means only the candidate is correct, a loss means only the baseline is correct, and equal labels are a tie. Incomplete prompts contribute nothing.
min_pairs is a positive integer. Return "promote" when paired >= min_pairs and wins > losses. Otherwise return "hold". The report must reflect all updates received before the call.

## Run locally
From the student directory, run python -m unittest discover -s tests. For one part, run python -m unittest discover -s tests -k Part1 (or Part2 or Part3). No installation is required.
