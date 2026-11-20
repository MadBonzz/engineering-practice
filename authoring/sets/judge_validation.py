def canonical(order, verdict):
    """order is 'AB' or 'BA'; verdict is 'left', 'right', 'tie', or 'invalid'. Return 'A', 'B', 'tie', or None for invalid. A and B identify models regardless of screen position."""
    if verdict == "invalid":
        return None
    if verdict == "tie":
        return "tie"
    return order[0 if verdict == "left" else 1]


def audit(row):
    """Row fields: ab and ba are two judge verdicts with swapped order, human is canonical A/B/tie. Return [valid_pair, stable, agreed]. All three are booleans. Stable requires two valid, equal canonical votes; agreed also requires a match to the human label. Missing/invalid pairs contribute False to all flags."""
    x = canonical("AB",row.get("ab","invalid"))
    y = canonical("BA",row.get("ba","invalid"))
    valid = x is not None and y is not None
    stable = valid and x == y
    return [valid,stable,stable and x == row["human"]]


def summarize(rows):
    """Return a dict with overall, by_task and by_length. Each summary has total, valid, stable, agreement and flip_rate. Agreement is agreed/stable; flip_rate is (valid-stable)/valid; use null for a zero denominator. For length slices use 'equal', 'A-longer' or 'B-longer' comparing len_a and len_b. Include only observed groups."""
    def stats(items):
        flags = [audit(r) for r in items]
        valid,stable,agreed = [sum(row[i] for row in flags) for i in range(3)]
        return dict(total=len(items),valid=valid,stable=stable,agreement=agreed/stable if stable else None,flip_rate=(valid-stable)/valid if valid else None)
    tasks, lengths = {},{}
    for r in rows:
        tasks.setdefault(r["task"],[]).append(r)
        key = "equal" if r["len_a"]==r["len_b"] else ("A-longer" if r["len_a"]>r["len_b"] else "B-longer")
        lengths.setdefault(key,[]).append(r)
    return {"overall":stats(rows),"by_task":{k:stats(v) for k,v in tasks.items()},"by_length":{k:stats(v) for k,v in lengths.items()}}

# AUTHORING
TITLE = "LLM-judge validation"
GROUP = "Evaluation"
INTRO = "A pairwise judge evaluates the same answers in both display orders. Audit it against held-out human labels and locate order sensitivity across tasks and answer lengths."
PARTS = ["canonical","audit","summarize"]
LABELS = ["Normalize the presentation order","Audit a pair of votes","Report agreement by slice"]
HINTS = ["Answer identity travels with the text.|Tie and invalid have distinct meanings.", "Convert both views to the same identity space.|Validity is separate from stability.", "Write down the denominator for each rate.|A slice can contain no stable votes."]
def cases(n):
    votes = ["left","right","tie","invalid"]
    def row(i):
        return dict(ab=votes[i%4],ba=votes[(i//2+1)%4],human=["A","B","tie"][i%3],task=["code","math"][i%2],len_a=i%4,len_b=i%3)
    return [["BA" if n%2 else "AB",votes[n%4]],[row(n)],[[row(n+i) for i in range(n%8+1)] if n!=8 else []]]
MUTANTS = [('order[0 if','"AB"[0 if'),('return "tie"','return None'),('valid and x == y','valid and x != y'),('agreed/stable if stable','agreed/valid if stable')]

