import random


def delta(rows):
    """Each row is [document_id, score_A, score_B]. Return mean(B-A) over rows or None if empty. Each row is a paired measurement."""
    return sum(b-a for _,a,b in rows)/len(rows) if rows else None


def replicate(rows, units, mode):
    """units is a with-replacement list of row indices (mode='row') or document IDs (mode='document'). In document mode each occurrence contributes all its original rows. Compute the row-weighted paired delta across the selected records, including repeated units. Empty selection returns None."""
    if mode == "row":
        sample = [rows[i] for i in units]
    else:
        sample = [row for doc in units for row in rows if row[0] == doc]
    return delta(sample)


def interval(rows, mode, repeats, seed):
    """Return [observed_delta, lower, upper]. Sample as many units as originally present, with replacement, for each of repeats>=1 replicates. Use one random.Random(seed), randrange(len(units)) for each draw, in order. Row units are indices; document units occur in first-appearance order. Use linear interpolation at (N-1)*p of the sorted replicate deltas for p=.025 and .975. Empty dataset returns three nulls."""
    if not rows:
        return [None,None,None]
    units = list(range(len(rows))) if mode == "row" else list(dict.fromkeys(r[0] for r in rows))
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        sample = [units[rng.randrange(len(units))] for _ in units]
        draws.append(replicate(rows,sample,mode))
    draws.sort()
    def quantile(p):
        index = (len(draws)-1)*p
        low = int(index)
        return draws[low]+(index-low)*(draws[min(low+1,len(draws)-1)]-draws[low])
    return [delta(rows),quantile(.025),quantile(.975)]

# AUTHORING
TITLE = "Paired statistical comparison"
GROUP = "Evaluation"
INTRO = "Two releases answered the same questions. Estimate the improvement and its uncertainty when multiple questions share one source document."
PARTS = ["delta","replicate","interval"]
LABELS = ["Compare paired outcomes","Resample rows or documents","Estimate the uncertainty"]
HINTS = ["Keep the direction of improvement consistent.|Pair each answer with its counterpart.", "A repeated unit contributes more than once.|Documents may have unequal sizes.", "Seed once for the whole experiment.|Compare a one-replicate interval to a multi-replicate interval."]
def cases(n):
    r = [["z",1,0],["a",0,1],["z",n%3,0],["b",0,n%4],["z",1,1]]
    mode = "row" if n%2 else "document"
    return [[r if n%6 else []],[r,([1,1,0] if mode=="row" else ["z","a","a"]) if n!=4 else [],mode],
            [r if n!=4 else [],mode,1 if n==3 else 17+n,n]]
MUTANTS = [("sum(b-a","sum(a-b"),("rows[i] for i in units","rows[i] for i in set(units)"),("for doc in units for row","for doc in set(units) for row"),("quantile(.975)","quantile(.75)")]

