import math


def scores(query, keys):
    """Return scaled dot products q.k/sqrt(query width) in key order. Empty keys return []."""
    return [sum(a*b for a,b in zip(query,k))/math.sqrt(len(query)) for k in keys]


def absorb(state, logits, values):
    """State is [max_logit, scaled_mass, weighted_vector]; empty uses None, 0, zeros. Fold in a block and return a new state. Both mass and vector are expressed relative to the maximum logit seen over all blocks (weights exp(logit-max)). Empty blocks are identity. Never exponentiate an unshifted large logit."""
    old,mass,vec = state
    if not logits:
        return state
    high = max(logits) if old is None else max(old,max(logits))
    factor = 0 if old is None else math.exp(old-high)
    weights = [math.exp(s-high) for s in logits]
    total = [factor*x + sum(w*v[d] for w,v in zip(weights,values)) for d,x in enumerate(vec)]
    return [high, mass*factor+sum(weights), total]


def attention(query, blocks, value_width):
    """Blocks are [keys,values] pairs, possibly empty. Produce the normalized attention output using a streaming pass with working storage proportional to one block plus the output width. Return a zero vector when all blocks are empty."""
    state = [None,0,[0]*value_width]
    for keys,values in blocks:
        state = absorb(state,scores(query,keys),values)
    return [x/state[1] for x in state[2]] if state[1] else state[2]

# AUTHORING
TITLE = "Blockwise attention"
GROUP = "Inference"
INTRO = "An inference worker streams key/value blocks that cannot fit in a single score matrix. Compute one attention output while keeping the softmax stable as new maxima arrive."
PARTS = ["scores","absorb","attention"]
LABELS = ["Score a block","Combine softmax state","Read a stream of blocks"]
HINTS = ["Query width sets the scale.|Value width is independent of key width.", "Both accumulators use the same reference maximum.|Try appending a much larger score.", "Normalize once after consuming all blocks.|An empty block can occur in the middle."]
def cases(n):
    q = [1+n, -2.]
    ks = [[1000.,1.],[-1000.,1.],[1001.,-2.]]
    vs = [[1,3],[-2,5],[2,8]]
    ls = [1000.+n,1000.-n]
    return [[q,ks[:1+n%3]],[([None,0,[0,0]] if n%3==0 else [999,2,[3,6]]),ls if n%4 else [],vs[:2] if n%4 else []],
            [q,[[[],[]],[ks[:1],vs[:1]],[ks[1:],vs[1:]]] if n%5 else [],2]]
MUTANTS = [("/math.sqrt(len(query))","/len(query)"),("mass*factor+sum(weights)","mass+sum(weights)"),("factor*x + sum","x + sum"),("x/state[1]","x")]

