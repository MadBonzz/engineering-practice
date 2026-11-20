import math


def project(hidden, layer, position):
    """Fixed two-component projections. Position is the absolute token offset."""
    x, y = hidden
    angle = position * 0.37
    c, s = math.cos(angle), math.sin(angle)
    def rotate(a, b):
        return [a*c-b*s, a*s+b*c]
    return rotate(x, y+layer), rotate(y, x-layer), [x+y, y-x]


def attend(query, keys, values):
    scores = [sum(a*b for a, b in zip(query, key)) / math.sqrt(2) for key in keys]
    weights = [math.exp(s-max(scores)) for s in scores]
    return [sum(w*v[d] for w, v in zip(weights, values))/sum(weights) for d in range(2)]


def advance(hidden, layer, position, keys, values):
    """Project this token once with project(), append K and V in place, attend over the extended history, and return the input hidden vector plus the attention output elementwise."""
    q, k, v = project(hidden, layer, position)
    keys.append(k)
    values.append(v)
    delta = attend(q, keys, values)
    return [a+b for a, b in zip(hidden, delta)]


def decode(token, cache):
    """Each layer has a mutable [keys, values] pair. Embed integer t as [sin(t), cos(t)]. Pass through layers in ascending zero-based order, using each layer's existing token count as position. Return logits [x, y, x-y] from the final hidden state. Mutate the supplied cache without recomputing earlier projections."""
    hidden = [math.sin(token), math.cos(token)]
    for layer, (keys, values) in enumerate(cache):
        hidden = advance(hidden, layer, len(keys), keys, values)
    x, y = hidden
    return [x, y, x-y]


def serve(chunks, layers):
    """Start an empty multilayer cache. Consume token chunks consecutively, even across empty chunks. Return one list of per-token logit vectors per chunk. Chunk boundaries must not reset positions. Each new token may call project exactly once per layer."""
    cache = [[[], []] for _ in range(layers)]
    return [[decode(token, cache) for token in chunk] for chunk in chunks]

# AUTHORING
TITLE = "Per-layer KV cache"
GROUP = "Inference"
INTRO = "A decoder handles prompt prefill and incremental generation in separate calls. Complete the cache path for a supplied two-dimensional attention model."
PARTS = ["advance", "decode", "serve"]
LABELS = ["Append one attention state", "Decode across layers", "Continue across chunk boundaries"]
HINTS = ["Which keys may the current query see?|Compare attention before and after inserting the current token.", "Layer number and token position have different meanings.|Inspect the length before appending.", "Can an empty chunk change state?|Trace one prompt split across three calls."]
def cases(n):
    t = n % 5
    return [[[1.0+n/10, -0.6], t%3, t, [[0.2,1.]]*t, [[0.4,0.1]]*t],
            [n+1, [[[[.1,-.2]]*t, [[.6,.3]]*t] for _ in range(1+n%3)]],
            [[[n,2],[],[3+t], [1]*t], 1+n%3]]
MUTANTS = [("keys.append(k)", "keys.append(q)"),("len(keys), keys","0, keys"),("x-y]","x+y]"),("enumerate(cache)","enumerate(reversed(cache))")]

EXTRA = {"compare_prefix.py": '''"""Run locally: python compare_prefix.py"""
from math import cos, sin, isclose
from time import perf_counter
from solution import attend, project, serve


def full_prefix(tokens, layers):
    """Uncached model forward. Rebuild every earlier token at every layer."""
    hidden = [[sin(t), cos(t)] for t in tokens]
    for layer in range(layers):
        triples = [project(row, layer, i) for i, row in enumerate(hidden)]
        keys = [k for _, k, _ in triples]
        values = [v for _, _, v in triples]
        next_hidden = []
        for i, ((query, _, _), row) in enumerate(zip(triples, hidden)):
            context = attend(query, keys[:i+1], values[:i+1])
            next_hidden.append([a+b for a, b in zip(row, context)])
        hidden = next_hidden
    x, y = hidden[-1]
    return [x, y, x-y]


if __name__ == "__main__":
    tokens = [2, 5, 1, 7] * 8
    start = perf_counter()
    baseline = [full_prefix(tokens[:i+1], 2) for i in range(len(tokens))]
    full_time = perf_counter()-start
    start = perf_counter()
    result = serve([tokens[:8], tokens[8:]], 2)
    cache_time = perf_counter()-start
    observed = result[0]+result[1]
    assert len(observed)==len(baseline)
    for i, (a, b) in enumerate(zip(observed, baseline)):
        assert len(a)==len(b) and all(isclose(x, y, abs_tol=1e-8, rel_tol=1e-7) for x, y in zip(a, b)), i
    print(f"{len(tokens)} logits checked; full-prefix {full_time:.4f}s, cached {cache_time:.4f}s")
'''}
