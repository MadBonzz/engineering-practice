"""Original executable architecture studies. Reference weights are tiny teaching fixtures."""
import math
from textwrap import dedent

PAPERS = []
GROUP = "Transformers and LLMs"


def part(title, file, funcs, requirement, *hints):
    return dict(title=title,file=file,funcs=funcs.split(),requirement=requirement,hints=list(hints))


def example(request, output, why):
    return dict(input=request, output=output, why=why)


def primary(title,url):
    return dict(title=title,url=url)


def add(slug,title,level,duration,mission,inputs,outputs,constraints,scope,source,modules,parts,examples,unit_checks,requests):
    checks = [dict(part=i,fn=fn,args=args,title=name) for i,fn,args,name in unit_checks]
    checks.extend(dict(part=len(parts),fn="gateway.handle",args=[req],title=name) for name,req in requests)
    PAPERS.append(dict(slug="paper-"+slug,title=title,group=GROUP,level=level,duration=duration,mission=mission,inputs=inputs,outputs=outputs,constraints=constraints,scope=scope,source=source,modules={name:dedent(code).strip()+"\n" for name,code in modules.items()},parts=parts,examples=examples,checks=checks))


ops = '''
import math


def dot(a, b):
    return sum(x*y for x, y in zip(a,b))


def plus(a,b):
    return [x+y for x,y in zip(a,b)]


def softmax(scores):
    if not scores:
        return []
    pivot = max(scores)
    terms = [math.exp(value-pivot) for value in scores]
    total = sum(terms)
    return [value/total for value in terms]


def mix(weights, rows):
    return [sum(w * x for w,x in zip(weights,column)) for column in zip(*rows)]


def layer_norm(row, eps=1e-5):
    mean = sum(row)/len(row)
    var = sum((x-mean)**2 for x in row)/len(row)
    scale = math.sqrt(var + eps)
    return [(x-mean)/scale for x in row] if scale else [0.0]*len(row)


def linear(weights,x):
    return [dot(row,x) for row in weights]
'''

add("transformer","Transformer · translate with encoder–decoder attention","Intermediate","60–90 min",
    "An encoder has read a source sentence. A teacher-forced decoder receives a target prefix. Implement one encoder and decoder layer so that every target position can consult the entire source while seeing only its own target history. Return the decoder’s vocabulary scores for all positions.",
    "source and prefix are lists of two-number vectors, already positioned token embeddings. The source contains at least one row. The target prefix may be empty.",
    "Return memory, the encoded source rows, and logits, one three-score vector per target row. The fixed vocabulary head maps [a,b] to [a,b,0].",
    "Use one head with identity Q, K, V and output projections. Scale scores by sqrt(2). This fixture uses post-residual population layer normalization with epsilon zero and maps constant rows to zeros. The supplied feed-forward branch maps [a,b] to [max(0,a+b)/4, max(0,a-b)/4].",
    "One post-norm encoder and decoder layer with bidirectional encoder attention, causal target attention and cross-attention. The supplied width-two checkpoint accepts positioned embeddings. Tokenization, multiple heads, stacked layers and training are outside this exercise.",
    [primary("Vaswani et al. · Attention Is All You Need","https://arxiv.org/abs/1706.03762")],
    {"tiny_ops.py":ops,"attention.py":'''
from tiny_ops import dot, mix, softmax
import math


def attend(queries, keys, values, causal):
    """Map T width-d queries and S matching keys/values to T context vectors. When causal is true, row i can see indices 0 through i inclusive."""
    result = []
    for index, query in enumerate(queries):
        count = index+1 if causal else len(keys)
        weights = softmax([dot(query,key)/math.sqrt(len(query)) for key in keys[:count]])
        result.append(mix(weights,values[:count]))
    return result
''',"sublayers.py":'''
from tiny_ops import plus, layer_norm


def combine(rows, updates):
    return [layer_norm(plus(row,update),0) for row,update in zip(rows,updates)]


def finish(rows):
    updates = [[max(0,a+b)/4,max(0,a-b)/4] for a,b in rows]
    return combine(rows,updates)
''',"translation.py":'''
from attention import attend
from sublayers import combine, finish


def encode(source):
    """Encode S width-two rows. Apply full self-attention, add-and-normalize, then the supplied feed-forward add-and-normalize."""
    return finish(combine(source,attend(source,source,source,False)))


def decode(prefix, memory):
    """Decode T width-two rows. Apply causal self-attention then cross-attention to S memory rows, each followed by add-and-normalize. Finish with the supplied feed-forward sublayer."""
    states = combine(prefix,attend(prefix,prefix,prefix,True))
    return finish(combine(states,attend(states,memory,memory,False)))
''',"gateway.py":'''
from translation import encode, decode


def handle(request):
    memory = encode(request["source"])
    hidden = decode(request["prefix"],memory)
    return {"memory":memory,"logits":[[a,b,0] for a,b in hidden]}
'''},
    [part("Implement scaled attention","attention.py","attend","Use the permitted key indices before normalizing each score row. Values may have a different width from keys.","Form scores with dot(query,key) / sqrt(key width).","A zero query gives all visible values equal weight."),
     part("Connect the translation layers","translation.py","encode decode","Build both residual streams. Encoder memory remains available to every decoder step. Make logits invariant to changing a later prefix row.","combine performs a residual update and normalizes it.","Cross-attention queries come from the first decoder sublayer. Encoder memory supplies both keys and values.")],
    [example({"source":[[2,0]],"prefix":[[0,0]]},{"memory":[[1,-1]],"logits":[[1,-1,0]]},"With one source token, attention copies it. The encoder’s normalized state is [1,-1]. The zero target acquires that state through cross-attention."),
     example({"source":[[0,2]],"prefix":[]},{"memory":[[-1,1]],"logits":[]},"An empty target still encodes the source, but requests no decoder score.")],
    [(1,"attention.attend",[[[0,0],[0,0]],[[1,0],[0,1]],[[2],[8]],True],"Causal rows have different visible counts"),(1,"attention.attend",[[[2,-1]],[[1,0],[0,1]],[[0,1],[1,0]],False],"Positive and negative attention logits"),(2,"translation.encode",[[[1,0],[0,3],[1,1]]],"Mixed source context"),(2,"translation.decode",[[[2,2],[-1,2]],[[1,-1],[-1,1],[1,-1]]],"All source rows are available to the first decoder query")],
    [("Unequal source and target lengths",{"source":[[2,-3],[-2,1],[1,2]],"prefix":[[0,0],[2,1]]}),("Constant source vectors",{"source":[[4,4],[1,1]],"prefix":[[3,-2],[-4,1],[0,0]]}),("Repeated source evidence",{"source":[[0,1],[0,1],[2,0]],"prefix":[[0,0]]})])


gpt_fixture = '''
import math
from tiny_ops import dot, mix, softmax, layer_norm, plus


def decoder_step(rows):
    """One supplied identity-projection, pre-LN causal attention and GELU feed-forward block."""
    normalized = [layer_norm(x) for x in rows]
    hidden = []
    for i,q in enumerate(normalized):
        keys = normalized[:i+1]
        probs = softmax([dot(q,k)/math.sqrt(len(q)) for k in keys])
        hidden.append(plus(rows[i],mix(probs,keys)))
    def gelu(x):
        return x*0.5*(1+math.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3)))
    return [plus(h,[gelu(x) for x in layer_norm(h)]) for h in hidden]
'''
gpt_request=lambda ids,e,p:dict(ids=ids,embedding=e,position=p)
add("gpt2","GPT-2 · score a text continuation","Intermediate","45–65 min",
    "A language-model evaluator reports suspiciously good perplexity. Rebuild its miniature GPT-2 scoring path from token and learned position lookups through a supplied causal block to a tied output head. Score each next token against the preceding position.",
    "ids is a nonempty list of vocabulary indices. embedding is a V×D learned-token table and position is a table of at least len(ids) width-D positional vectors. All weights are fixture values.",
    "Return next_ids, the greedy next token for every input position including the final one, and mean_nll, the average next-token negative log likelihood in nats over len(ids)−1 predictions. A singleton has null mean_nll.",
    "Positions begin at zero. The tied vocabulary head has no bias. Apply the supplied final layer_norm with epsilon 1e−5 before the output projection. Resolve equal logits to the smaller ID. Use a stable log-sum-exp.",
    "One miniature decoder block with learned absolute positions, pre-LN causal self-attention, a GELU branch and tied vocabulary scoring. Matrices inside the supplied decoder use identity projections. BPE and pretraining are omitted.",
    [primary("Radford et al. · Language Models are Unsupervised Multitask Learners","https://openai.com/index/better-language-models/"),primary("OpenAI · GPT-2 model implementation","https://github.com/openai/gpt-2/blob/master/src/model.py")],
    {"tiny_ops.py":ops,"block.py":gpt_fixture,"tied_head.py":'''
from tiny_ops import plus, dot, layer_norm


def embed(ids, table, positions):
    """For T ids return T width-D embeddings, adding positions[i] to table[ids[i]]."""
    return [plus(table[token],positions[i]) for i,token in enumerate(ids)]


def project(hidden, table):
    """Map T width-D states to T×V logits by final layer normalization then dot products against the original V token embeddings."""
    return [[dot(layer_norm(row),entry) for entry in table] for row in hidden]
''',"scoring.py":'''
import math


def evaluate(logits, ids):
    """From T×V logits return greedy next_ids and mean_nll in nats. Row t predicts ids[t+1]. Use all rows for greedy decoding, with smaller ID winning ties."""
    greedy = [max(range(len(row)), key=row.__getitem__) for row in logits]
    losses = []
    for row,target in zip(logits,ids[1:]):
        peak = max(row)
        losses.append(peak - row[target] + math.log(sum(math.exp(x-peak) for x in row)))
    return {"next_ids":greedy,"mean_nll":sum(losses)/len(losses) if losses else None}
''',"gateway.py":'''
from tied_head import embed, project
from block import decoder_step
from scoring import evaluate


def handle(request):
    ids = request["ids"]
    table = request["embedding"]
    rows = embed(ids,table,request["position"])
    return evaluate(project(decoder_step(rows),table),ids)
'''},
    [part("Restore tied embeddings","tied_head.py","embed project","Add learned token and position rows at input. Normalize the final state and reuse the token table for output.","Positions index the sequence, even for repeated token IDs.","A vocabulary logit is a dot product with the corresponding embedding row."),
     part("Align targets and logits","scoring.py","evaluate","Compute the next-token loss in nats, excluding the final score row from the loss and retaining it for greedy decoding.","Pair score rows with ids[1:].","Subtract the maximum inside the exponential and add it back in log space.")],
    [example(gpt_request([0,1],[[2,0],[2,0]],[[0,1],[1,0]]),{"next_ids":[0,0],"mean_nll":math.log(2)},"Both rows of the tied vocabulary head are identical, so each token receives probability 1/2. One prediction contributes log(2) nats and the greedy tie goes to 0."),
     example(gpt_request([1],[[0,1],[1,0]],[[0,0]]),{"next_ids":[1],"mean_nll":None},"The final state favors the second vocabulary vector. There is no observed next token to score.")],
    [(1,"tied_head.embed",[[1,1],[[0,3],[2,1]],[[2,-1],[-1,3]]],"Repeated token, distinct positions"),(1,"tied_head.project",[[[3,1],[1,1]],[[0,1],[-1,2],[2,1]]],"Constant row and non-square vocabulary"),(2,"scoring.evaluate",[[[1000,999],[0,1],[2,-1]],[1,0,1]],"Shifted labels and extreme logits"),(2,"scoring.evaluate",[[[0,0,0]],[2]],"A singleton still decodes")],
    [("Tied weight table",gpt_request([2,0,1],[[2,1],[-1,3],[0,1]],[[3,1],[0,0],[-2,1]])),("Repeated token",gpt_request([0,0,0],[[1,0],[0,1]],[[0,0],[-4,0],[2,1]])),("Width-three hidden state",gpt_request([0,2],[[0,1,3],[1,2,0],[2,1,1]],[[3,0,0],[-2,4,1]]))])

a = 1+1/math.sqrt(0.5+1e-6)
r = a/math.sqrt(a*a/2+1e-6)
add("llama","LLaMA · assemble a rotary decoder block","Advanced","65–90 min",
    "Score a short sequence using a width-two LLaMA-style decoder. Implement rotary queries and keys, a gated feed-forward branch, and the two pre-normalized residual updates. A fixed readout exposes the effect of your block on each next-token score.",
    "states contains T width-two token embeddings in sequence order and offset gives the absolute integer position of the first row. All rows attend within this request. The supplied checkpoint gives the gate, up, and down projections.",
    "Return logits, a T×3 list. The fixed readout maps a final RMS-normalized [a,b] to [a,b,a−b]. Empty input yields an empty list.",
    "Normalize x as x/sqrt(mean(x²)+1e−6) with unit gains and no centering. Q, K and V are the normalized states. Apply adjacent-pair RoPE to Q and K only, with base 10000. Use causal softmax scaled by sqrt(width), identity attention output projection, and a separate RMS normalization before the MLP.",
    "One LLaMA-1-style single-head block with RoPE, RMSNorm and SwiGLU. A small fixture replaces the learned projection matrices and vocabulary. It runs a causal forward pass without a tokenizer or pretrained weights.",
    [primary("Touvron et al. · LLaMA","https://arxiv.org/abs/2302.13971")],
    {"tiny_ops.py":ops,"checkpoint.py":'''
import math

GATE = [[1.0,0.0]]
UP = [[0.0,1.0]]
DOWN = [[1.0],[-1.0]]


def rms(row):
    scale=math.sqrt(sum(v*v for v in row)/len(row)+1e-6)
    return [v/scale for v in row]
''',"rotary.py":'''
import math


def rotate(x, position, base):
    """Rotate each adjacent pair of an even-width vector x. Angle for pair j is position / base**(2*j/len(x)). Return the rotated vector."""
    result=[]
    for i in range(0,len(x),2):
        theta=position/(base**(i/len(x)))
        c,s=math.cos(theta),math.sin(theta)
        a,b=x[i:i+2]
        result.extend([a*c-b*s,a*s+b*c])
    return result
''',"feedforward.py":'''
import math
from tiny_ops import linear


def swiglu(x, gate, up, down):
    """With gate and up shaped H×D and down shaped D×H, return down @ (SiLU(gate @ x) multiplied elementwise by up @ x)."""
    a=linear(gate,x)
    b=linear(up,x)
    activated=[g/(1+math.exp(-g))*u for g,u in zip(a,b)]
    return linear(down,activated)
''',"decoder.py":'''
import math
from tiny_ops import dot,mix,softmax,plus
from checkpoint import GATE,UP,DOWN,rms
from rotary import rotate
from feedforward import swiglu


def forward(states, offset):
    """Run one width-two block on T×2 embeddings. Add causal attention of RMS-normalized states, then add SwiGLU of the new RMS-normalized residual. Return T×2 raw residual states."""
    values=[rms(row) for row in states]
    rotated=[rotate(row,offset+i,10000) for i,row in enumerate(values)]
    out=[]
    for i,q in enumerate(rotated):
        scores=[dot(q,k)/math.sqrt(len(q)) for k in rotated[:i+1]]
        h=plus(states[i],mix(softmax(scores),values[:i+1]))
        out.append(plus(h,swiglu(rms(h),GATE,UP,DOWN)))
    return out
''',"gateway.py":'''
from checkpoint import rms
from decoder import forward


def handle(request):
    rows = forward(request["states"], request["offset"])
    return {"logits":[[a,b,a-b] for a,b in map(rms,rows)]}
'''},
    [part("Rotate position-sensitive vectors","rotary.py","rotate","Preserve each pair’s norm and use a different frequency for each pair in a wider head.","Use absolute positions for angles.","Rotate left component to a cos θ − b sin θ and right to a sin θ + b cos θ."),
     part("Implement a gated MLP","feedforward.py","swiglu","Project an input twice to hidden width, apply SiLU on the gate, multiply elementwise by the up projection, then project back to model width.","SiLU(a) = a/(1+exp(−a)).","The hidden and model widths can differ."),
     part("Wire both residual branches","decoder.py","forward","Add attention to the original input and add the MLP to that updated state. The two branch inputs each use RMSNorm.","Do not rotate values.","The second branch uses the updated residual, before the final readout normalization.")],
    [example({"states":[[1,0]],"offset":0},{"logits":[[r,0,r]]},"With one position, attention copies the normalized value. The zero second coordinate switches off the fixture’s up projection. Final RMSNorm yields about 1.41421332 for both the first and third logits."),
     example({"states":[[0,0]],"offset":7},{"logits":[[0,0,0]]},"The zero vector stays zero under rotation and through both bias-free branches.")],
    [(1,"rotary.rotate",[[1,2,-1,3],4,10000],"Head-pair frequencies"),(1,"rotary.rotate",[[3,4],0,23],"Zero rotation"),(2,"feedforward.swiglu",[[2,-1],[[2,1],[0,1],[1,-3]],[[1,2],[1,0],[0,1]],[[1,-2,0],[0,1,3]]],"Different expansion and model widths"),(2,"feedforward.swiglu",[[1,1],[[-1,-2]],[[3,2]],[[1],[-1]]],"Negative gate"),(3,"decoder.forward",[[[2,0],[0,3]],5],"Only previous values are visible"),(3,"decoder.forward",[[[-1,3]],0],"Both residual branches affect output")],
    [("Three-token dependency",{"states":[[2,1],[-1,0],[0,-1]],"offset":0}),("Nonzero absolute positions",{"states":[[-1,0],[2,-0.5],[1,1]],"offset":29}),("Zero-length prefix",{"states":[],"offset":11})])

def packet(v, q=None, k=None):
    return {"q":q if q is not None else [[0],[0]],"k":k if k is not None else [[0]],"v":v}

def gemma_req(layer,window,packets):
    return dict(layer=layer,window=window,packets=packets)

add("gemma3","Gemma 3 · stream a mixed-window attention layer","Advanced","55–75 min",
    "A text-serving worker must advance a single Gemma 3 attention layer token by token. Five local layers are followed by one global layer. Implement the retention policy and grouped-query read, then verify that head sharing and cache eviction preserve the layer’s causal contexts.",
    "layer is zero-based and window includes the current token. Each packet supplies projected q [Hq,d], k [Hkv,d] and v [Hkv,dv] for one arriving token. Q/K normalization and rotary position have already been applied upstream. The first packet starts a fresh sequence.",
    "Return mode ('local' or 'global'), contexts [T,Hq,dv], and retained, the cache length immediately after each packet. Empty input returns empty histories.",
    "Layers with layer % 6 == 5 use global attention. All others retain the most recent window packets in chronological order. Hq is a positive multiple of Hkv. Contiguous groups of Hq/Hkv query heads share a KV head. Scores use q·k/sqrt(d).",
    "The text decoder’s 5-local/1-global topology, chronological KV eviction and grouped-query attention. This is one layer given projected queries, keys and values. The full multimodal encoder, surrounding residual/MLP stack and training are outside the lab.",
    [primary("Gemma Team · Gemma 3 Technical Report, section 2","https://arxiv.org/html/2503.19786v1")],
    {"tiny_ops.py":ops,"retention.py":'''
def append_packet(entries, keys, values, layer, window):
    """Return a new chronological list of [keys,values] entries after this token. Global layers retain all, local layers only the newest window. Do not mutate entries."""
    new=entries+[[keys,values]]
    return new if layer%6==5 else new[-window:]
''',"grouped_read.py":'''
import math
from tiny_ops import dot, softmax, mix


def read(queries, entries):
    """Return one value-width context per query head. Each cache entry is [K,V], each indexed by KV head. Consecutive equal-sized groups of query heads share one KV head."""
    group_size=len(queries)//len(entries[0][0])
    contexts=[]
    for h,query in enumerate(queries):
        group=h//group_size
        scores=[dot(query,k[group])/math.sqrt(len(query)) for k,v in entries]
        contexts.append(mix(softmax(scores),[v[group] for k,v in entries]))
    return contexts
''',"gateway.py":'''
from retention import append_packet
from grouped_read import read


def handle(request):
    layer,window=request["layer"],request["window"]
    history,retained,contexts=[],[],[]
    for p in request["packets"]:
        history=append_packet(history,p["k"],p["v"],layer,window)
        retained.append(len(history))
        contexts.append(read(p["q"],history))
    return {"mode":"global" if layer%6==5 else "local","contexts":contexts,"retained":retained}
'''},
    [part("Enforce the layer’s memory policy","retention.py","append_packet","Make local cache memory bounded across arrivals while keeping all previous keys and values at global layers.","Count the current packet toward the local window.","The sixth layer, numbered 5, is the first global layer."),
     part("Read shared KV heads","grouped_read.py","read","Implement a separate scaled softmax over retained token positions for each query head. A contiguous group shares keys and values but can produce different contexts.","Compute the KV index using integer division by group size.","A zero query gives the arithmetic mean of that KV head’s retained values.")],
    [example(gemma_req(4,2,[packet([[1]]),packet([[3]]),packet([[11]])]),{"mode":"local","contexts":[[[1],[1]],[[2],[2]],[[7],[7]]],"retained":[1,2,2]},"Identical zero queries average the retained values. At the third arrival the local window holds only 3 and 11, giving 7."),
     example(gemma_req(5,1,[packet([[2]]),packet([[10]])]),{"mode":"global","contexts":[[[2],[2]],[[6],[6]]],"retained":[1,2]},"The sixth layer retains both values despite a local window setting of one. Its second output averages 2 and 10.")],
    [(1,"retention.append_packet",[[[[[0]],[[1]]],[[[1]],[[2]]]],[[3]],[[4]],1,2],"Trim oldest entry"),(1,"retention.append_packet",[[[[[0]],[[1]]]],[[3]],[[4]],11,1],"Global layer in the next cycle"),(2,"grouped_read.read",[[[0],[1],[0],[-1]],[[[[1],[0]],[[3],[7]]],[[[0],[2]],[[2],[9]]]]],"Contiguous head groups and different queries"),(2,"grouped_read.read",[[[1,-1]],[[[[0,1]],[[2,-1]]],[[[1,0]],[[1,4]]]]],"Scale by key width")],
    [("Window of one",gemma_req(6,1,[packet([[1]]),packet([[5]]),packet([[-3]])])),("Shared but unequal queries",gemma_req(11,1,[packet([[1,0]],[[0],[1]],[[1]]),packet([[0,3]],[[1],[-1]],[[0]])])),("Two KV groups",gemma_req(0,3,[packet([[1],[4]],[[0],[1],[1],[0]],[[1],[2]]),packet([[-1],[8]],[[1],[0],[0],[-1]],[[0],[-2]])]))])


def ds_token(h,c,p,q,logits):
    return {"state":h,"latent":c,"pos_key":p,"query":q,"router_logits":logits}

def ds_req(tokens, bias, k=1, scale=1.0):
    return {"tokens":tokens,"bias":bias,"top_k":k,"scale":scale,"experts":[[i+1,1,1] for i in range(len(bias))],"shared":[0,1,1]}

add("deepseek-v3","DeepSeek-V3 · decode from latent KV into sparse experts","Advanced","65–95 min",
    "A decoder worker keeps compressed attention history and a separate position key. After each attention read, route the token across a shared expert and selected routed experts. The load-balancing correction may change which expert runs without changing the weights used to mix its output.",
    "Each token contains a scalar residual state, a scalar KV latent, a pre-rotated scalar pos_key, a query [content,position] and E router_logits. bias has E correction terms. experts contains E triples [gate,up,down] and shared contains one triple. top_k and scale control routing.",
    "Return hidden, one scalar final residual per token, and expert_ids, one list of selected expert IDs per token in rank order.",
    "In this checkpoint a latent c reconstructs content key c and value 2c. Attend over accumulated entries with score (q_content*c + q_pos*pos_key)/sqrt(2). Add that context to state to form h. Select experts by sigmoid(logit)+bias with smaller ID breaking ties. Normalize raw selected sigmoid affinities to sum to scale. Add to h both the shared expert and the weighted routed experts. Each triple computes down × SiLU(gate × h) × (up × h). Inputs have finite magnitudes below 30.",
    "A one-head, rank-one MLA decode and DeepSeekMoE inference path. Position components and router logits come from upstream projections. This single-group lab omits group/device limits, balance-bias training, FP8 and multi-token prediction.",
    [primary("DeepSeek-AI · DeepSeek-V3 Technical Report","https://arxiv.org/abs/2412.19437"),primary("DeepSeek-AI · Inference reference","https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py")],
    {"tiny_ops.py":ops,"latent_read.py":'''
import math
from tiny_ops import softmax


def context(query, entries):
    """For a two-component query and nonempty chronological [latent,pos_key] rows, return the attended scalar value. The model decodes content key c and value 2c from latent c."""
    qc,qp=query
    weights=softmax([(qc*c+qp*p)/math.sqrt(2) for c,p in entries])
    return sum(w*2*c for w,(c,p) in zip(weights,entries))
''',"expert_router.py":'''
import math


def route(logits, bias, k, scale):
    """Return [ids,weights] for k of E experts. Rank by sigmoid(logit)+bias, resolving ties by ID. Mix by the original selected sigmoid affinities normalized to sum to scale."""
    affinity=[1/(1+math.exp(-x)) for x in logits]
    chosen=sorted(range(len(logits)),key=lambda i:(-(affinity[i]+bias[i]),i))[:k]
    total=sum(affinity[i] for i in chosen)
    return [chosen,[scale*affinity[i]/total for i in chosen]]
''',"experts.py":'''
import math


def expert(h, weights):
    gate,up,down=weights
    g=gate*h
    sig=1/(1+math.exp(-g)) if g>=0 else math.exp(g)/(1+math.exp(g))
    return down*(g*sig)*(up*h)
''',"worker_step.py":'''
from latent_read import context
from expert_router import route
from experts import expert


def decode(request):
    """Append each token’s compact KV record, read its context, update its residual, and dispatch shared plus weighted routed experts. Return hidden and expert_ids in arrival order."""
    history,hidden,routes=[],[],[]
    for token in request["tokens"]:
        history.append([token["latent"],token["pos_key"]])
        h=token["state"]+context(token["query"],history)
        ids,weights=route(token["router_logits"],request["bias"],request["top_k"],request["scale"])
        routed=sum(w*expert(h,request["experts"][i]) for i,w in zip(ids,weights))
        hidden.append(h+expert(h,request["shared"])+routed)
        routes.append(ids)
    return {"hidden":hidden,"expert_ids":routes}
''',"gateway.py":'''
from worker_step import decode


def handle(request):
    return decode(request)
'''},
    [part("Read a compressed attention row","latent_read.py","context","Combine the latent-derived content score with the separate position-key score, then read the value reconstructed from the same latent.","The position key influences the weights only.","Subtract the maximum before exponentiating."),
     part("Route with a balancing correction","expert_router.py","route","Use bias-corrected affinities for selecting expert IDs and raw affinities for weights.","Routing can pick a lower-affinity expert if its correction is positive.","A single selected expert gets the full routing scale."),
     part("Compose the sparse decode path","worker_step.py","decode","Use one accumulated compact cache per request. Run the shared expert once per token and sum only selected routed outputs.","The current token participates in its own attention.","Compute every expert from the same post-attention h, before adding any expert outputs.")],
    [example(ds_req([ds_token(0,0.5,2,[3,1],[0,0])],[0,0]),{"hidden":[1+1/(1+math.exp(-1))],"expert_ids":[[0]]},"The only latent contributes value 1. Equal affinities choose expert 0. Its [1,1,1] transform adds SiLU(1) to the residual, and the shared expert’s gate is zero."),
     example(ds_req([],[-1,2]),{"hidden":[],"expert_ids":[]},"An idle request emits no states and dispatches no experts.")],
    [(1,"latent_read.context",[[0,0],[[1,-9],[3,9]]],"Uniform latent value reconstruction"),(1,"latent_read.context",[[0,2],[[2,-1],[-2,3]]],"Position channel can change the cached read"),(2,"expert_router.route",[[2,0,-1],[-1,0,1],2,2],"Bias changes IDs but not affinity weights"),(2,"expert_router.route",[[0,0,0],[0,0,0],2,1],"Stable ties"),(3,"worker_step.decode",[ds_req([ds_token(1,0.1,-1,[1,0],[0,1])],[0,0],2)],"Two routed experts"),(3,"worker_step.decode",[{**ds_req([ds_token(1,0,0,[0,0],[0])],[0]),"shared":[2,1,0.5]}],"Always-on shared expert")],
    [("Balanced routing across arrivals",ds_req([ds_token(0,0.5,2,[2,1],[-1,3,0]),ds_token(-1,-0.5,1,[3,-1],[1,2,0])],[1,0,-1],2,1.5)),("Shared and routed experts both contribute",{**ds_req([ds_token(1,0,-1,[1,0],[0,0])],[0,1]),"shared":[-1,2,1]}),("Content-only versus position-only",ds_req([ds_token(-1,1,3,[1,0],[0]),ds_token(2,-2,1,[0,-1],[1])],[0]))])


def oss_req(layer,window,sink,tokens,experts,k):
    return dict(layer=layer,window=window,sink=sink,tokens=tokens,experts=experts,top_k=k)

def oss_token(h,q,k,v,r):
    return dict(state=h,q=q,k=k,v=v,router=r)

add("gpt-oss","gpt-oss · decode with an attention sink and bounded experts","Advanced","55–85 min",
    "A reference decoder reserves some attention probability for a learned sink. The sink has no value vector. Reconstruct a small token path where this attention output enters a sparse feed-forward block with the gpt-oss gated activation.",
    "Each token supplies a scalar state, projected q, k and v, plus router logits. layer is zero-based, window includes the current token, and sink is the learned head’s scalar sink logit. experts contains one [gate_gain,up_gain,down_gain] triple per expert. top_k is between one and the expert count.",
    "Return hidden (one scalar post-expert residual per token) and sink_mass (the fraction of attention probability spent on the sink at each position).",
    "Use one scalar head with q*k scores. Even layers attend to the most recent window entries, odd layers to the entire prefix. Softmax includes the sink score, whose value is zero. Form h = state + context. Choose top_k experts by router logit with smaller ID winning ties and softmax over just those logits. For each selected expert, gate = min(7, gate_gain*h), up = clip(up_gain*h,−7,7). Its output is down_gain × gate × sigmoid(1.702*gate) × (up+1). Add the weighted sum to h.",
    "One scalar-head attention and sparse expert pass using the auxiliary attention score, alternating windows and bounded gpt-oss SwiGLU rule. Projections are supplied, expert count and active count are configurable, and this reduced fixture omits RMSNorm, RoPE, quantization and biases.",
    [primary("OpenAI · gpt-oss Architecture","https://deploymentsafety.openai.com/gpt-oss/architecture"),primary("OpenAI · gpt-oss PyTorch reference","https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py")],
    {"tiny_ops.py":ops,"sink_head.py":'''
from tiny_ops import softmax


def attend(q, keys, values, sink):
    """Return [context,sink_mass] for one scalar query, equally long key/value arrays and one auxiliary sink logit. Its implicit value is zero."""
    weights=softmax([q*k for k in keys]+[sink])
    return [sum(w*v for w,v in zip(weights,values)),weights[-1]]
''',"bounded_experts.py":'''
import math
from tiny_ops import softmax


def activate(gate, up):
    """Apply the scalar gpt-oss gated activation. Clamp gate from above at 7, clamp up to [-7,7], then return gate*sigmoid(1.702*gate)*(up+1)."""
    gate=min(gate,7)
    up=max(-7,min(7,up))
    t=1.702*gate
    sig=1/(1+math.exp(-t)) if t>=0 else math.exp(t)/(1+math.exp(t))
    return gate*sig*(up+1)


def dispatch(h, logits, experts, k):
    """From scalar h and E router logits, mix the top-k scalar expert outputs by a softmax over selected logits only. Each expert stores gate, up and down gains. Ties favor lower ID."""
    selected=sorted(range(len(logits)),key=lambda i:(-logits[i],i))[:k]
    weights=softmax([logits[i] for i in selected])
    output=0.0
    for i,w in zip(selected,weights):
        gate,up,down=experts[i]
        output+=w*down*activate(gate*h,up*h)
    return output
''',"gateway.py":'''
from sink_head import attend
from bounded_experts import dispatch


def handle(request):
    keys,values,hidden,mass=[],[],[],[]
    for item in request["tokens"]:
        keys.append(item["k"])
        values.append(item["v"])
        start=max(0,len(keys)-request["window"]) if request["layer"]%2==0 else 0
        context,sink=attend(item["q"],keys[start:],values[start:],request["sink"])
        h=item["state"]+context
        hidden.append(h+dispatch(h,item["router"],request["experts"],request["top_k"]))
        mass.append(sink)
    return {"hidden":hidden,"sink_mass":mass}
'''},
    [part("Give the sink a place in the denominator","sink_head.py","attend","Append the learned sink logit during softmax and exclude it from the value sum. Return its normalized mass separately.","With one real value and an equal sink score, exactly half the value survives.","Use one shared maximum across the real scores and the sink."),
     part("Mix bounded expert outputs","bounded_experts.py","activate dispatch","Implement the gate and top-k mixture. Scale gate, up and down independently and apply the offset after clipping the up branch.","The gate is capped only from above. Negative inputs can be below −7.","Normalize only the selected router logits. Multiple selected experts need not have equal weight.")],
    [example(oss_req(0,2,0,[oss_token(0,0,0,2,[0])],[[1,0,1]],1),{"hidden":[1+1/(1+math.exp(-1.702))],"sink_mass":[0.5]},"Two equal scores split the mass between the value 2 and the zero sink. The residual becomes 1. With up gain zero, the expert’s multiplicative up+1 factor is still one."),
     example(oss_req(1,1,0,[oss_token(0,0,0,3,[0]),oss_token(0,0,0,6,[0])],[[0,0,1]],1),{"hidden":[1.5,3],"sink_mass":[0.5,1/3]},"On the global layer the second token shares mass equally between two real values and the sink. The zero gate suppresses the expert.")],
    [(1,"sink_head.attend",[1,[1000,999],[2,4],1000],"Stable softmax includes sink"),(1,"sink_head.attend",[0,[1,2],[0,6],0],"Unassigned attention mass"),(2,"bounded_experts.activate",[-19,30],"Gate only clipped above"),(2,"bounded_experts.activate",[40,-30],"Bound both the up branch and positive gate"),(2,"bounded_experts.dispatch",[2,[0,1,1],[[0,0,0],[1,2,1],[-1,-2,1]],2],"Top-k mixing and tied logits")],
    [("Window includes current token",oss_req(0,1,1,[oss_token(2,1,0,3,[0,1]),oss_token(-1,-1,2,1,[2,1])],[[2,3,1],[-0.5,-1,2]],1)),("Confidence-sensitive sink",oss_req(3,1,-1,[oss_token(1,0,1,4,[1,0]),oss_token(1,-1,3,-2,[2,3])],[[3,2,0.5],[-2,3,1]],2)),("Saturated expert branches",oss_req(2,4,0,[oss_token(4,2,1,5,[0,1,0])],[[3,4,1],[2,-4,1],[-4,-4,2]],2))])
