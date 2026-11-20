def summarize(ledger, min_pairs):
    wins = losses = ties = 0
    for prompt in ledger.prompts():
        pair = ledger.pair(prompt)
        if pair is None:
            continue
        baseline, candidate = pair
        if baseline == candidate:
            ties += 1
        elif candidate:
            wins += 1
        else:
            losses += 1
    paired = wins + losses + ties
    decision = "promote" if paired >= min_pairs and wins > losses else "hold"
    return dict(paired=paired, wins=wins, losses=losses, ties=ties, decision=decision)
