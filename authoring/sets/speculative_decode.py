def draw(probabilities, u):
    """Return the first index whose cumulative probability exceeds u in [0,1). Inputs are normalized; final index is a roundoff fallback."""
    total = 0
    for i,p in enumerate(probabilities):
        total += p
        if u < total:
            return i
    return len(probabilities)-1


def greedy(draft, target):
    """Verify one speculative block. target is the target model's greedy token at each draft position plus one bonus position. Return accepted draft tokens followed by exactly one target token: the first correction on mismatch, otherwise the bonus. Empty draft is valid."""
    result = []
    for i,t in enumerate(draft):
        if t != target[i]:
            return result+[target[i]]
        result.append(t)
    return result+[target[-1]]


def verify(token, p, q, u, residual_u):
    """p is the target distribution, q the draft distribution; q[token]>0. Accept the proposal iff u < min(1,p[token]/q[token]). Return [True,token] on acceptance. On rejection return [False, replacement] drawn with draw() from the normalized positive part of p-q using residual_u."""
    if u < min(1,p[token]/q[token]):
        return [True,token]
    mass = [max(0,x-y) for x,y in zip(p,q)]
    return [False,draw([x/sum(mass) for x in mass],residual_u)]


def sample_block(proposals, targets, drafts, uniforms, bonus_u):
    """Each proposal i has distributions targets[i], drafts[i] and uniforms[i]=[u,residual_u]. Return the emitted tokens up to and including the first rejection, or all accepted proposals plus one draw from targets[-1] using bonus_u. targets contains exactly one extra bonus row."""
    result = []
    for i,t in enumerate(proposals):
        ok, emitted = verify(t, targets[i], drafts[i], *uniforms[i])
        result.append(emitted)
        if not ok:
            return result
    return result+[draw(targets[-1],bonus_u)]

# AUTHORING
TITLE = "Speculative decoding"
GROUP = "Inference"
INTRO = "A small draft model proposes tokens in blocks. Complete the target verifier using supplied next-token distributions and deterministic random draws."
PARTS = ["greedy","verify","sample_block"]
LABELS = ["Verify a greedy block","Correct a sampled proposal","Emit a verified block"]
HINTS = ["The first mismatch ends this block.|Check what happens when every proposal agrees.", "Identify where the target assigns more probability.|The acceptance draw and correction draw are independent.", "Stop reading draft tokens after a rejection.|The bonus uses the extra target row."]
def cases(n):
    p,q = [.1,.55,.35],[.6,.2,.2]
    u = [.05,.2,.8,.99][n%4]
    tokens = [0,1] if n%5 else []
    draft = [1,2,n%4]
    tgt = [1,2,n%4,0] if n%3==0 else [1,n%2,3,0]
    return [[draft,tgt],[0,p,q,u,(n%10)/10],[tokens,[p for _ in tokens]+[[.2,.3,.5]],[q for _ in tokens],[[u,(n%10)/10] for _ in tokens],(n%8)/8]]
MUTANTS = [("result+[target[-1]]","result"),("p[token]/q[token]","q[token]/p[token]"),("max(0,x-y)","max(0,y-x)"),("if not ok:","if ok:")]

