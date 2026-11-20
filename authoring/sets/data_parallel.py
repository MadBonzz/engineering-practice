def prediction(weights, x):
    return sum(w*v for w,v in zip(weights,x))


def local(weights, examples):
    """Compute [sample_count, mean_gradient] for loss 0.5*(w.x-y)^2. Each example is [feature_vector, y]. An empty rank contributes count zero and a zero vector the width of weights."""
    grad = [0.]*len(weights)
    for x,y in examples:
        error = prediction(weights,x)-y
        for j,v in enumerate(x):
            grad[j] += error*v
    return [len(examples),[v/max(1,len(examples)) for v in grad]]


def reduce_ranks(reports):
    """Each rank supplies [local_count, local_mean_gradient]. Return the global per-example mean gradient, preserving vector width. At least one report is supplied; if all counts are zero return zeros."""
    total = sum(n for n,_ in reports)
    return [sum(n*g[j] for n,g in reports)/max(1,total) for j in range(len(reports[0][1]))]


def train(weights, steps, lr):
    """Each step contains one examples list per rank (at least one rank), including possible empty shards. All ranks start a step with identical current weights. Apply one global SGD update. Return a weight snapshot after each step, including empty steps, without editing initial weights."""
    history = []
    for shards in steps:
        grad = reduce_ranks([local(weights,shard) for shard in shards])
        weights = [w-lr*g for w,g in zip(weights,grad)]
        history.append(weights)
    return history

# AUTHORING
TITLE = "Distributed data-parallel SGD"
GROUP = "Training"
INTRO = "A training job splits batches unevenly between workers. Implement rank-local gradients, their reduction, and synchronized updates. The browser simulates ranks; a two-process driver is included for local runs."
PARTS = ["local","reduce_ranks","train"]
LABELS = ["Compute a rank report","Reduce uneven shards","Synchronize training steps"]
HINTS = ["The local loss is averaged over examples.|Keep the count with the gradient.", "Try shards of size one and three.|An empty rank has no statistical weight.", "All ranks read the same weight snapshot.|Apply one update per global step."]
def cases(n):
    w = [.2+n/10,-1]
    a = [[[1,2],1],[[2,-3],n],[[0,1],1]]
    b = [[[3,0],1]]
    return [[w,a if n%4 else []],[[[n%3,[1,2]],[1+n%3,[4,-1]],[0,[0,0]]]],
            [w,[[a,b],[[],b],[[],[]],[b,a]],.02*(1+n%4)]]
MUTANTS = [("prediction(weights,x)-y","y-prediction(weights,x)"),("v/max(1,len(examples))","v"),("sum(n*g[j]","sum(g[j]"),("w-lr*g","w+lr*g")]
EXTRA = {"two_process.py": '''"""Run locally: python two_process.py"""
from concurrent.futures import ProcessPoolExecutor
from solution import local, reduce_ranks

if __name__ == "__main__":
    weights = [0.25, -0.5]
    shards = [[([1, 2], 3)], [([0, 1], 2), ([2, 0], -1)]]
    with ProcessPoolExecutor(2) as pool:
        reports = list(pool.map(local, [weights] * 2, shards))
    actual = reduce_ranks(reports)
    expected = local(weights, shards[0] + shards[1])[1]
    assert all(abs(a-b) < 1e-9 for a,b in zip(actual, expected))
    print("Two-rank reduction agrees with a single full batch.")
'''}

