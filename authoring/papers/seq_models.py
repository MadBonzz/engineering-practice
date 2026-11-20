"""Original small sequence-model implementations for the paper bench."""
from textwrap import dedent


def module(text):
    return dedent(text).strip() + "\n"


def part(title, file, funcs, requirement, hint1, hint2):
    return dict(title=title, file=file, funcs=funcs, requirement=requirement, hints=[hint1, hint2])


def check(p, fn, args, title):
    return dict(part=p, fn=fn, args=args, title=title)


def example(i, o, why):
    return dict(input=i, output=o, why=why)


def src(title, url):
    return dict(title=title, url=url)


PAPERS = []
GROUP = "Sequence models and embeddings"

STABLE = module("""
    import math


    def sigmoid(x):
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        e = math.exp(x)
        return e / (1 + e)


    def softplus(x):
        return max(x, 0) + math.log1p(math.exp(-abs(x)))


    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))
""")

# Skip-gram negative sampling.
def word_request(tokens, window, centers, contexts, draws, lr):
    return dict(tokens=tokens, window=window, centers=centers, contexts=contexts, negatives=draws, lr=lr)


word_a = word_request([0, 1], 1, [[1], [0]], [[0], [0]], [[], []], 0.2)
word_b = word_request([1], 2, [[0], [1]], [[3], [4]], [], 0.1)
PAPERS.append(dict(
    slug="paper-word2vec", title="Word2Vec · train skip-gram embeddings", group=GROUP, level="Intermediate", duration="45–65 min",
    mission="A text ingestion worker learns two embedding tables from token IDs. Implement the windowing and optimization code so a client can reproduce exactly which examples moved each word vector.",
    inputs="tokens is a list of vocabulary IDs. window is a nonnegative radius. centers and contexts are separate V × D tables. negatives supplies one list of sampled context IDs per ordered positive pair. lr is the learning rate.",
    outputs="Return the ordered center/context pairs, the pre-update loss for each pair, and the final centers and contexts tables. Each loss is a sum over one positive and its negative samples.",
    constraints="Visit center positions left to right, then neighboring positions left to right. Exclude the center position. Equal IDs at distinct positions are valid. Negative IDs can repeat and can coincide with the positive ID. Accumulate all gradient contributions for one pair against its starting tables, then apply one SGD update. Inputs remain unchanged.",
    scope="Implements one deterministic skip-gram negative-sampling training pass. The caller supplies noise samples. Vocabulary construction, phrase detection and a large-corpus training run are outside the exercise.",
    source=[src("Mikolov et al. · Distributed Representations of Words and Phrases and their Compositionality, §2.2","https://arxiv.org/abs/1310.4546")],
    modules={
        "numerics.py":STABLE,
        "context_windows.py":module('''
            def positive_pairs(tokens, radius):
                """Return [center_id, context_id] pairs ordered by center index then context index within radius."""
                pairs = []
                for index, center in enumerate(tokens):
                    for other in range(max(0, index - radius), min(len(tokens), index + radius + 1)):
                        if other != index:
                            pairs.append([center, tokens[other]])
                return pairs
        '''),
        "negative_sampling.py":module('''
            from numerics import dot, sigmoid, softplus


            def update_pair(centers, contexts, pair, negatives, lr):
                """Return loss and copied updated V × D tables as [loss, new_centers, new_contexts]."""
                left, positive = pair
                vector = centers[left]
                center_gradient = [0.0] * len(vector)
                new_centers = [row[:] for row in centers]
                new_contexts = [row[:] for row in contexts]
                loss = 0.0
                for right, sign in [(positive, 1)] + [(i, -1) for i in negatives]:
                    context = contexts[right]
                    score = dot(vector, context)
                    loss += softplus(-sign * score)
                    derivative = -sign * sigmoid(-sign * score)
                    for d in range(len(vector)):
                        center_gradient[d] += derivative * context[d]
                        new_contexts[right][d] -= lr * derivative * vector[d]
                new_centers[left] = [v-lr*g for v,g in zip(vector, center_gradient)]
                return [loss, new_centers, new_contexts]
        '''),
        "gateway.py":module('''
            from context_windows import positive_pairs
            from negative_sampling import update_pair


            def handle(request):
                pairs = positive_pairs(request["tokens"], request["window"])
                centers, contexts = request["centers"], request["contexts"]
                losses = []
                for pair, negatives in zip(pairs, request["negatives"]):
                    loss, centers, contexts = update_pair(centers, contexts, pair, negatives, request["lr"])
                    losses.append(loss)
                return {"pairs": pairs, "losses": losses, "centers": centers, "contexts": contexts}
        ''')
    },
    parts=[
        part("Construct the ordered training stream","context_windows.py",["positive_pairs"],"Enumerate all positions within the radius, skipping the current position. Return IDs so repeated occurrences contribute repeated examples.","Allow both directions of a pair.","Clip each window to the sentence boundaries."),
        part("Differentiate one sampled objective","negative_sampling.py",["update_pair"],"For each signed target s (+1 positive, −1 negative), accumulate softplus(−s u·v). Its derivative with respect to u·v is −s sigmoid(−s u·v). Apply SGD once to the input row and each touched output row. The supplied gateway feeds each update into the next pair.","Use the stable sigmoid and softplus in numerics.py.","When a sampled ID repeats, read its original row for the gradient and add all updates to its destination row.")],
    examples=[
        example(word_a,dict(pairs=[[0,1],[1,0]],losses=[0.6931471805599453]*2,centers=[[1],[0]],contexts=[[0],[0.1]]),"Each initial dot product is zero and costs log(2). The first pair moves context 1 by 0.2 × 0.5 × 1 = 0.1. The second pair has both a zero center and a zero context."),
        example(word_b,dict(pairs=[],losses=[],centers=[[0],[1]],contexts=[[3],[4]]),"The only token has no neighboring position. Both embedding tables pass through unchanged.")],
    checks=[
        check(1,"context_windows.positive_pairs",[[4,4,9],1],"Repeated words at distinct positions"),
        check(1,"context_windows.positive_pairs",[[2,3,4,5],7],"Window extends beyond both boundaries"),
        check(2,"negative_sampling.update_pair",[[[1,2],[-1,1]],[[2,-1],[1,2]],[0,0],[1,1],.1],"Repeated negative table row"),
        check(2,"negative_sampling.update_pair",[[[10]],[[100]],[0,0],[0],.01],"Large dot products and colliding sampled IDs"),
        check(2,"gateway.handle",[word_request([1,0,1],1,[[1,2],[0,-1]],[[1,1],[2,-1]],[[1],[0,1],[0],[1,1]],.05)],"Sequential updates reuse a center"),
        check(2,"gateway.handle",[word_request([0,1,2],0,[[0],[1],[2]],[[0],[2],[1]],[],.1)],"A zero-radius stream"),
        check(2,"gateway.handle",[word_request([0,0],2,[[2]],[[1]],[[0,0],[0]],0)],"Zero learning rate still reports loss")]
))

# Scalar vanilla RNN with full backpropagation through time.
def rnn_req(x, y, keep, p, h, lr):
    return dict(inputs=x, targets=y, scored=keep, params=p, initial=h, lr=lr)


P0 = {"input":0,"recurrent":.5,"bias":0}
P1 = {"input":1.0,"recurrent":-.75,"bias":.2}
PAPERS.append(dict(
    slug="paper-rnn",title="Vanilla RNN · backpropagate through a delayed label",group=GROUP,level="Intermediate",duration="50–70 min",
    mission="A sensor sequence often receives a label only at its last timestep. Train a tiny recurrent predictor and expose the gradient that flowed all the way back to its initial state.",
    inputs="inputs, targets and scored have the same length T. Each input and target is scalar. scored is a Boolean supervision mask. params has scalar input weight, recurrent weight and bias. initial is h₀ and lr is the learning rate.",
    outputs="Return the pre-update hidden sequence, summed pre-update loss, gradient d_initial, and params after one SGD step. With T=0 all gradients and the loss are zero.",
    constraints="Advance the recurrence at every timestep, including unscored ones. At scored steps add 0.5(h−target)² to the loss. Reuse the same weights at every step and accumulate the gradient of their full sequence loss before updating them.",
    scope="Reproduces the feedback link of an Elman recurrent network with one tanh unit, an identity readout and a full backward pass. This small regression task makes every gradient checkable.",
    source=[src("Elman · Finding Structure in Time","https://doi.org/10.1207/s15516709cog1402_1")],
    modules={
        "recurrent_cell.py":module('''
            from math import tanh


            def rollout(inputs, params, initial):
                """Return T hidden values h[t]=tanh(input_weight*x[t]+recurrent_weight*h_prev+bias)."""
                hidden = initial
                states = []
                for value in inputs:
                    hidden = tanh(params["input"] * value + params["recurrent"] * hidden + params["bias"])
                    states.append(hidden)
                return states
        '''),
        "backward_pass.py":module('''
            def gradients(inputs, targets, scored, states, params, initial):
                """Return [loss, {input,recurrent,bias}, d_initial] using full reverse-time differentiation."""
                loss = 0.0
                grad = {"input": 0.0, "recurrent": 0.0, "bias": 0.0}
                future = 0.0
                for t in reversed(range(len(inputs))):
                    error = states[t] - targets[t] if scored[t] else 0.0
                    loss += 0.5 * error**2
                    local = (error + future) * (1.0 - states[t]**2)
                    grad["input"] += local * inputs[t]
                    previous = states[t-1] if t else initial
                    grad["recurrent"] += local * previous
                    grad["bias"] += local
                    future = local * params["recurrent"]
                return [loss, grad, future]
        '''),
        "gateway.py":module('''
            from recurrent_cell import rollout
            from backward_pass import gradients


            def handle(request):
                x, p, h0 = request["inputs"], request["params"], request["initial"]
                states = rollout(x, p, h0)
                loss, grad, d_initial = gradients(x, request["targets"], request["scored"], states, p, h0)
                updated = {k: v-request["lr"]*grad[k] for k,v in p.items()}
                return {"states": states, "loss": loss, "d_initial": d_initial, "params": updated}
        ''')
    },
    parts=[
        part("Roll out the shared recurrent cell","recurrent_cell.py",["rollout"],"Return h₁ through h_T, applying tanh to each affine update. The caller supplies the state preceding the first input.","Use params['input'] for x and params['recurrent'] for the previous hidden value.","Append the new hidden value after applying tanh."),
        part("Send the loss backward through time","backward_pass.py",["gradients"],"At each reverse step, add the local scored error to the hidden derivative coming from the future. Differentiate tanh using its stored output, accumulate shared parameter derivatives, and return the gradient arriving at h₀.","An unscored step has zero local loss but can receive a derivative from later steps.","For the recurrent weight gradient, multiply the preactivation derivative by the previous hidden state.")],
    examples=[
        example(rnn_req([1,2],[0,1],[False,True],P0,0,.1),dict(states=[0,0],loss=.5,d_initial=-.25,params={"input":.25,"recurrent":.5,"bias":.15}),"Only the last step has a direct error of −1. The first receives −0.5 through the recurrent weight. The shared input gradient is −2 − 0.5 = −2.5, and the gradient at the initial state is −0.25."),
        example(rnn_req([],[],[],P1,.3,.2),dict(states=[],loss=0,d_initial=0,params=P1),"Without observations the backward pass contributes no loss or parameter update.")],
    checks=[
        check(1,"recurrent_cell.rollout",[[.5,0,-1],P1,.4],"State carry under nonzero bias"),
        check(1,"recurrent_cell.rollout",[[-3,2],P0,-.8],"Input-independent recurrence with nonzero initial state"),
        check(2,"backward_pass.gradients",[[1,2],[0,1],[False,True],[0,0],P0,0],"Early unscored step carries error"),
        check(2,"backward_pass.gradients",[[3],[.2],[False],[.1],P1,-.3],"Zero supervision suppresses gradients"),
        check(2,"gateway.handle",[rnn_req([.2,-.3,1],[.1,.5,-.4],[True,False,True],P1,-.3,.04)],"Accumulate direct and delayed losses"),
        check(2,"gateway.handle",[rnn_req([1,0],[.5,-.5],[False,False],P1,.75,.4)],"Unlabelled segment still generates states"),
        check(2,"gateway.handle",[rnn_req([-2],[.7],[True],P1,.3,0)],"Report derivatives in a frozen step")]
))

# Packed gates and state preservation in a recurrent inference worker.
def lstm_req(h, c, recurrent, inputs, active):
    return dict(h0=h, c0=c, recurrent=recurrent, drives=inputs, active=active)


U1 = [[0.0],[0.0],[0.0],[0.0]]
U2 = [[.2,-.1],[0,1],[1,.5],[-.7,0],[.25,.5],[-1,.3],[1,0],[0,-1]]
PAPERS.append(dict(
    slug="paper-lstm",title="LSTM · carry hidden and cell state through padded streams",group=GROUP,level="Intermediate",duration="50–65 min",
    mission="Complete an LSTM inference worker that can pause for padding and later resume from exactly the state it last emitted. A user needs both the visible trace and the internal cell state for a continuation.",
    inputs="h0 and c0 each have H components. recurrent is a shared 4H × H matrix. Each drive is the precomputed input projection plus bias, a vector of length 4H in input, forget, candidate, output order. active is a Boolean per drive.",
    outputs="Return trace with one hidden vector per drive (including held steps), and final hidden h and cell c. Return the incoming state when there are no drives.",
    constraints="For active drives compute pre = drive + recurrent × previous_h, split it into four contiguous blocks of width H, then use sigmoid on i, f, o and tanh on g. Apply c' = f ⊙ c + i ⊙ g and h' = o ⊙ tanh(c'). A false mask holds both states and emits a copy of h. No peepholes or projection layer.",
    scope="Implements a single-layer, forget-gated LSTM forward scan from preprojected inputs. The original LSTM introduced the cell and multiplicative gates. The later forget-gate paper supplies the retention mechanism used here.",
    source=[src("Hochreiter and Schmidhuber · Long Short-Term Memory","https://pubmed.ncbi.nlm.nih.gov/9377276/"),src("Gers, Schmidhuber and Cummins · Learning to Forget","https://pubmed.ncbi.nlm.nih.gov/11032042/")],
    modules={
        "numerics.py":STABLE,
        "packed_gates.py":module('''
            from math import tanh
            from numerics import dot, sigmoid


            def unpack(drive, recurrent, previous_h):
                """Return four length-H vectors [i,f,g,o] after adding recurrent × previous_h to the 4H drive."""
                width = len(previous_h)
                pre = [x + dot(row, previous_h) for x,row in zip(drive, recurrent)]
                blocks = []
                for index, activation in enumerate([sigmoid, sigmoid, tanh, sigmoid]):
                    blocks.append([activation(x) for x in pre[index*width:(index+1)*width]])
                return blocks
        '''),
        "cell_state.py":module('''
            from math import tanh


            def advance(previous_c, gates):
                """Given length-H c and activated [i,f,g,o], return [new_h, new_c] without changing either input."""
                i, f, g, o = gates
                cell = [ret*old + write*val for ret,old,write,val in zip(f,previous_c,i,g)]
                hidden = [gate*tanh(value) for gate,value in zip(o,cell)]
                return [hidden, cell]
        '''),
        "padded_scan.py":module('''
            from packed_gates import unpack
            from cell_state import advance


            def scan(h0, c0, recurrent, drives, active):
                """Run active drives, hold both states on padding, return {trace,h,c} including holds in trace."""
                h, c = h0[:], c0[:]
                trace = []
                for drive, is_active in zip(drives,active):
                    if is_active:
                        h, c = advance(c, unpack(drive,recurrent,h))
                    trace.append(h[:])
                return {"trace": trace, "h": h, "c": c}
        '''),
        "gateway.py":module('''
            from padded_scan import scan


            def handle(request):
                return scan(request["h0"],request["c0"],request["recurrent"],request["drives"],request["active"])
        ''')
    },
    parts=[
        part("Decode a fused gate projection","packed_gates.py",["unpack"],"Add each row's recurrent contribution to its drive. Split contiguous blocks and apply each block's activation.","Each row sees the whole previous hidden vector.","Only the third block uses tanh."),
        part("Update memory and expose a hidden vector","cell_state.py",["advance"],"Update the cell coordinatewise using the write and forget gates. Gate the tanh of the new cell to produce h.","The forget gate weights the previous cell directly.","Return h before c in the pair."),
        part("Resume correctly across padding","padded_scan.py",["scan"],"Run both kernels in sequence on active steps. Holding a padded step must preserve h and c even when drive values are large.","Copy emitted vectors so historical steps stay fixed.","A zero-valued drive can still update a state when the mask is true.")],
    examples=[
        example(lstm_req([0],[1],U1,[[0,0,0,0],[4,4,4,4]],[True,False]),{"trace":[[.23105857863000487],[.23105857863000487]],"h":[.23105857863000487],"c":[.5]},"The active step sets every sigmoid gate to 0.5 and the candidate to zero. The cell decays from 1 to 0.5 and h becomes 0.5 tanh(0.5). Padding repeats that result."),
        example(lstm_req([.2,-.3],[1,-2],U2,[],[]),{"trace":[],"h":[.2,-.3],"c":[1,-2]},"An empty continuation returns both incoming state vectors intact.")],
    checks=[
        check(1,"packed_gates.unpack",[[0]*8,U2,[1,-.5]],"Off-diagonal recurrent weights mix hidden features"),
        check(1,"packed_gates.unpack",[[1000,-1000,1,2],U1,[1]],"Independent stable sigmoid branches"),
        check(2,"cell_state.advance",[[3,-1],[[0,1],[1,0],[.5,-.25],[1,0]]],"Retain, replace, and close the output gate"),
        check(2,"cell_state.advance",[[-.4],[[.1],[.6],[-.75],[.9]]],"Partial forget and write"),
        check(3,"padded_scan.scan",[[1],[2],U1,[[0]*4,[0]*4],[False,True]],"Distinguish a held step from an all-zero drive"),
        check(3,"padded_scan.scan",[[1],[2],U1,[[2]*4],[False]],"All-padding stream"),
        check(3,"gateway.handle",[lstm_req([.4,-.2],[0,.3],U2,[[.2]*8,[-.1]*8],[True,True])],"Carry both states in a two-channel recurrent model"),
        check(3,"gateway.handle",[lstm_req([.9],[.7],[[1],[-2],[3],[-4]],[[.1,.2,.3,.4],[1]*4,[-1]*4],[True,False,True])],"Resume after an internal gap"),
        check(3,"gateway.handle",[lstm_req([0],[0],U1,[[0,0,.5,0]],[True])],"A candidate writes to empty memory")]
))

# Reset-before GRU with a fixed prefix state and chunked transport.
def gru_req(h, recurrent, chunks):
    return dict(initial=h,recurrent=recurrent,chunks=chunks)


GR1 = [[0],[0],[0]]
GR2 = [[.5,-1],[1,-.3],[-.5,.2],[.7,1],[0,2],[-1,.5]]
PAPERS.append(dict(
    slug="paper-gru",title="GRU · encode a chunked phrase with reset-before gating",group=GROUP,level="Intermediate",duration="45–60 min",
    mission="A phrase encoder gets tokens in network-sized chunks. Build the recurrent unit that compresses the phrase into a fixed vector and return a checkpoint after each incoming chunk. Splitting a phrase differently must preserve its final embedding.",
    inputs="initial is a length-H hidden vector. Each token in chunks is a packed input projection of length 3H, ordered reset, retention, candidate. recurrent is a 3H × H matrix stored in the same row-block order. Input-side biases are already included in the token drives.",
    outputs="Return final as the encoding after all tokens and checkpoints as one encoding per chunk. Empty chunks repeat the current encoding.",
    constraints="Use the reset-before formulation of Cho et al. Activate r = sigmoid(xr + Ur h) and z = sigmoid(xz + Uz h). Compute g = tanh(xg + Ug (r ⊙ h)) and update h' = z ⊙ h + (1−z) ⊙ g. Matrix rows are output features. Use the same parameters across chunks and preserve the starting vector.",
    scope="Implements the paper's GRU encoder recurrence and phrase-level state carry. Input projections are supplied. The translation decoder and language-pair training corpus are omitted.",
    source=[src("Cho et al. · Learning Phrase Representations using RNN Encoder–Decoder, equations 5–8","https://arxiv.org/abs/1406.1078")],
    modules={
        "numerics.py":STABLE,
        "gru_gates.py":module('''
            from numerics import dot, sigmoid


            def retention(drive, recurrent, h):
                """Return [reset, retention] vectors of width H from the first 2H drive entries and matrix rows."""
                n = len(h)
                gates = [sigmoid(drive[j] + dot(recurrent[j], h)) for j in range(2*n)]
                return [gates[:n], gates[n:]]
        '''),
        "phrase_encoder.py":module('''
            from math import tanh
            from numerics import dot
            from gru_gates import retention


            def step(drive, recurrent, h):
                """Return a length-H hidden vector with reset applied before the candidate recurrent projection."""
                n = len(h)
                r, z = retention(drive, recurrent, h)
                reset_h = [gate * old for gate,old in zip(r,h)]
                candidate = [tanh(drive[2*n+j] + dot(recurrent[2*n+j], reset_h)) for j in range(n)]
                return [keep*old + (1-keep)*new for keep,old,new in zip(z,h,candidate)]


            def encode(initial, recurrent, chunks):
                """Run every projected token through step and return {final, checkpoints} across chunk boundaries."""
                h = initial[:]
                checkpoints = []
                for chunk in chunks:
                    for drive in chunk:
                        h = step(drive, recurrent, h)
                    checkpoints.append(h[:])
                return {"final": h, "checkpoints": checkpoints}
        '''),
        "gateway.py":module('''
            from phrase_encoder import encode


            def handle(request):
                return encode(request["initial"],request["recurrent"],request["chunks"])
        ''')
    },
    parts=[
        part("Produce reset and retention factors","gru_gates.py",["retention"],"Combine the first two input blocks with their rows of the recurrent matrix and the complete old hidden state. Return activated r then z.","Compute both gates from the same incoming state.","One gate scales the candidate's memory input. The other chooses how much old output to keep."),
        part("Encode the phrase across transport boundaries","phrase_encoder.py",["step","encode"],"Implement the candidate and interpolation from the equations, then carry the result through every chunk. Return a checkpoint even for an empty chunk.","Cross-channel candidate weights make resetting before projection different from resetting afterward.","Reset gates are learned factors at each token. The chunk boundary does not reset the encoder.")],
    examples=[
        example(gru_req([1],GR1,[[[0,0,0]],[],[[0,0,0]]]),{"checkpoints":[[.5],[.5],[.25]],"final":[.25]},"The candidate is zero, and retention is 0.5 for each token. Only actual tokens halve the state."),
        example(gru_req([0],GR1,[[[0,0,.5493061443340549]]]),{"checkpoints":[[.25]],"final":[.25]},"The candidate is tanh(atanh(0.5)) = 0.5, so interpolation halfway from zero gives 0.25.")],
    checks=[
        check(1,"gru_gates.retention",[[0]*6,GR2,[1,-.25]],"Independent gates mix multiple old coordinates"),
        check(1,"gru_gates.retention",[[1000,-1000,1],GR1,[1]],"Saturated reset and retention"),
        check(2,"phrase_encoder.step",[[2,-2,0,0,0,0],GR2,[.5,-1]],"Reset-before with a nondiagonal candidate matrix"),
        check(2,"phrase_encoder.encode",[[.5],GR1,[[],[],[]]],"Consecutive empty chunks preserve memory"),
        check(2,"gateway.handle",[gru_req([.3,-.4],GR2,[[[0]*6],[[1,2,3,4,-1,.7]]])],"Cross-feature hidden-state transport"),
        check(2,"gateway.handle",[gru_req([-.5],[[1],[-2],[3]],[[[2,-1,0],[0,.5,1]],[],[[.1,.5,-1]]])],"Multiple chunks and changing gates"),
        check(2,"gateway.handle",[gru_req([1,-1],GR2,[])],"No incoming phrase")]
))

# Recurrent form of the gated delta rule (value rows, key columns).
def token(q, k, v, a, b):
    return {"q":q, "k":k, "v":v, "alpha":a, "beta":b}


def delta_req(state, events):
    return dict(state=state,events=events)


PAPERS.append(dict(
    slug="paper-gated-deltanet",title="Gated DeltaNet · implement the recurrent token mixer",group=GROUP,level="Advanced",duration="55–75 min",
    mission="A streaming language-model worker runs a token mixer with a fixed-size matrix state. Implement the gated delta recurrence and its readout. The worker must be able to checkpoint the matrix and continue with new projected tokens.",
    inputs="state is a Dv × Dk matrix from the previous chunk. Each event has a query q and unit key k of width Dk, a value v of width Dv, retention alpha and write strength beta. q, k, v and activated gates are supplied by an upstream projection.",
    outputs="Return outputs as one Dv-dimensional vector per event and state as the final Dv × Dk matrix. Reading happens after the current write.",
    constraints="Use the value-by-key orientation of equation 10. First D = alpha S. Then p = D k and S' = D + beta (v−p) kᵀ. Emit S' q. Alpha and beta lie in [0,1]. Use the decayed matrix for the prediction, and allocate a fresh state so a caller can retain checkpoints.",
    scope="Implements the sequential matrix-state mixer of Gated DeltaNet for one head. The exercise starts with projected, normalized keys and precomputed gates. Short convolutions, normalization layers, multihead projections and the parallel training kernel are outside this recurrence.",
    source=[src("Yang, Kautz and Hatamizadeh · Gated Delta Networks, §3.1","https://arxiv.org/html/2412.06464v3")],
    modules={
        "state_read.py":module('''
            def project_state(state, vector):
                """Multiply a Dv × Dk matrix by a Dk vector, returning Dv dot products."""
                return [sum(w*x for w,x in zip(row,vector)) for row in state]
        '''),
        "delta_write.py":module('''
            from state_read import project_state


            def write(state, key, value, alpha, beta):
                """Return alpha*S + beta*(value - (alpha*S)@key) outer key with Dv rows and Dk columns."""
                decayed = [[alpha*x for x in row] for row in state]
                estimate = project_state(decayed, key)
                result = []
                for d, row in enumerate(decayed):
                    correction = beta * (value[d] - estimate[d])
                    result.append([weight + correction*k for weight,k in zip(row,key)])
                return result
        '''),
        "token_mixer.py":module('''
            from delta_write import write
            from state_read import project_state


            def mix(state, events):
                """Advance projected events in order and return {outputs, state} with one post-update read per event."""
                outputs = []
                for event in events:
                    state = write(state,event["k"],event["v"],event["alpha"],event["beta"])
                    outputs.append(project_state(state,event["q"]))
                return {"outputs": outputs, "state": state}
        '''),
        "gateway.py":module('''
            from token_mixer import mix


            def handle(request):
                return mix(request["state"],request["events"])
        ''')
    },
    parts=[
        part("Read the matrix-valued recurrent state","state_read.py",["project_state"],"Return S·x. Rows correspond to value coordinates and columns to key coordinates. Use the same operation for a prediction from k and an output from q.","Try a rectangular state to catch accidental transposition.","Each result coordinate is one full row dot the incoming vector."),
        part("Correct a decayed prediction","delta_write.py",["write"],"Decay the entire state and predict the value aligned with the current key. Form the outer product of the prediction error and the key, scale by beta, and add it to the decayed state.","A write of strength zero can still change state through alpha.","Calculate all prediction-error coordinates before updating the corresponding rows."),
        part("Stream post-write outputs","token_mixer.py",["mix"],"Run write once per token and read the updated state along the supplied query. Keep the final matrix so a later request can continue exactly at the boundary.","The key and query can point in different directions.","If the input chunk is empty, preserve the checkpoint.")],
    examples=[
        example(delta_req([[1,2],[3,4]],[token([0,1],[0,1],[5,-1],.5,.5)]),{"outputs":[[3,.5]],"state":[[.5,3],[1.5,.5]]},"After decay, the second column is [1,2]. Half the error to [5,−1] is [2,−1.5]. Add this correction only to the second column, then read that column."),
        example(delta_req([[4,-2]],[token([1,0],[0,1],[99],.25,0)]),{"outputs":[[1]],"state":[[1,-.5]]},"The gate retains one quarter of the old state. A zero write strength discards the proposed correction, and reading column one returns 1.")],
    checks=[
        check(1,"state_read.project_state",[[[1,0,2],[3,-1,1]],[2,1,-1]],"Rectangular state and signed query"),
        check(1,"state_read.project_state",[[[3],[0],[-1]],[-2]],"More value than key dimensions"),
        check(2,"delta_write.write",[[[2,3]],[.6,.8],[1],.5,.25],"Oblique unit key and partial correction"),
        check(2,"delta_write.write",[[[-2],[3]],[1],[4,-1],0,1],"Erase all old state and install new value"),
        check(3,"token_mixer.mix",[[[0,0]],[token([0,1],[1,0],[3],1,1),token([1,0],[0,1],[7],1,1)]],"Independent write and read directions"),
        check(3,"token_mixer.mix",[[[.5],[1]],[]],"Empty continuation"),
        check(3,"gateway.handle",[delta_req([[1,0],[0,1]],[token([1,0],[1,0],[3,4],1,1),token([0,1],[1,0],[2,3],.75,.25)])],"Overwrite an already modified direction"),
        check(3,"gateway.handle",[delta_req([[1,2,3]],[token([.2,.3,.4],[0,.6,.8],[0],.6,.9)])],"Rectangular oblique update"),
        check(3,"gateway.handle",[delta_req([[1],[2]],[token([-1],[1],[0,0],.1,0),token([1],[1],[2,-2],.5,1)])],"Decay followed by full replacement")]
))

# These additional private probes have hand-derived answers independent of the implementation.
# The extra oracle field is authoring metadata; the callable and arguments follow the usual check schema.
import math as _math

def _probe(paper, number, fn, args, name, expected):
    paper["checks"].append(dict(check(number, fn, args, name), oracle=expected))

ln2, ln3, half_angle = _math.log(2), _math.log(3), _math.atanh(.5)
word, rnn, lstm, gru, delta = PAPERS
_probe(word, 2, "negative_sampling.update_pair", [[[1.0]],[[0.0],[0.0]],[0,1],[0,0],.4],
       "Read each repeated negative from the pre-update table", [3*ln2,[[1.0]],[[-.4],[.2]]])
_probe(word, 2, "negative_sampling.update_pair", [[[0.,0.]],[[1.,-1.],[3.,2.]],[0,1],[0],.2],
       "A zero center moves while its old output-row derivatives stay zero", [2*ln2,[[.2,.3]],[[1.,-1.],[3.,2.]]])

_probe(rnn, 2, "gateway.handle", [rnn_req([1,2,4],[.2,42,1],[True,False,True],P0,0,.2)],
       "Bridge an unlabelled timestep between two different supervised errors",
       {"states":[0,0,0],"loss":.52,"d_initial":-.225,"params":{"input":1.09,"recurrent":.5,"bias":.39}})
_probe(rnn, 2, "backward_pass.gradients", [[1,2],[1,1],[True,True],[0,0],{"input":0,"recurrent":-2,"bias":0},0],
       "Opposite-sign adjoints cancel the shared bias gradient", [1,{"input":-1,"recurrent":0,"bias":0},-2])

_probe(lstm, 1, "packed_gates.unpack", [[ln3,-ln3,-ln3,ln3,half_angle,-half_angle,0,0],[[0,0] for _ in range(8)],[2,3]],
       "Distinguish four contiguous blocks from interleaved per-unit gates", [[.75,.25],[.25,.75],[.5,-.5],[.5,.5]])
one, two = .5*_math.tanh(.5), .5*_math.tanh(.25)
_probe(lstm, 3, "gateway.handle", [lstm_req([.5],[1],U1,[[0]*4,[9]*4,[-9]*4,[0]*4],[True,False,False,True])],
       "Carry the cell across two padded steps before a true zero-input update",
       {"trace":[[one],[one],[one],[two]],"h":[two],"c":[.25]})

ug_swap = [[0,0],[0,0],[0,0],[0,0],[0,1],[1,0]]
_probe(gru, 2, "phrase_encoder.step", [[-ln3,ln3,0,0,-3,-.5],ug_swap,[2,4]],
       "Reset then swap channels cancels the candidate drive exactly", [1,2])
_probe(gru, 2, "gateway.handle", [gru_req([.5],GR1,[[[0,ln3,-half_angle]],[],[[0,-ln3,half_angle]]])],
       "Retention changes from three quarters to one quarter after a transport gap",
       {"final":[.4375],"checkpoints":[[.25],[.25],[.4375]]})

_probe(delta, 2, "delta_write.write", [[[1,2],[2,3]],[1,0],[5,6],.25,.5],
       "Prediction error uses the decayed checkpoint rather than the arriving one",
       [[2.625,.5],[3.25,.75]])
_probe(delta, 3, "gateway.handle", [delta_req([[0,0]],[token([0,1],[1,0],[2],1,1),token([1,0],[.6,.8],[0],1,1)])],
       "An oblique overwrite partially erases a previously written direction",
       {"outputs":[[0],[1.28]],"state":[[1.28,-.96]]})
