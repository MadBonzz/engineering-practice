"""Original, small implementations of six recent model mechanisms."""
from textwrap import dedent

PAPERS = []
LLM = "Transformers and LLMs"
SEQ = "Sequence models and embeddings"


def code(s):
    return dedent(s).strip() + "\n"


def part(title, file, funcs, requirement, *hints):
    return dict(title=title, file=file, funcs=funcs, requirement=requirement, hints=list(hints))


def ex(request, output, why):
    return dict(input=request, output=output, why=why)


def check(n, fn, args, title):
    return dict(part=n, fn=fn, args=args, title=title)


def end(n, cases):
    return [check(n, "gateway.handle", [request], title) for request, title in cases]


def src(title, url):
    return dict(title=title, url=url)


def kimi(embedding, queries, gains, block_size, readout):
    return dict(embedding=embedding, layers=[dict(q=q, gain=g) for q,g in zip(queries,gains)], block_size=block_size, readout=readout)


PAPERS.append(dict(
    slug="paper-kimi-k3",title="Kimi K3 · Attention across residual blocks", group=LLM, level="Advanced",duration="55–75 min",
    mission="A single-token execution engine needs to retain information across network depth. Implement Block AttnRes for a stack of small branches. A depth query chooses how much each completed block and the unfinished block contribute to the next branch.",
    inputs="embedding is a width-D vector. layers is an ordered list of objects with width-D q and gain vectors. block_size counts branches per block. readout is the final width-D query.",
    outputs="Return blocks with the sum of branch outputs for each block, excluding the initial embedding, and state with the final readout over the embedding and completed blocks.",
    constraints="D and block_size are positive. All vectors share D. A final partial block is committed. Empty layers return the embedding as state and no blocks. RMS uses epsilon 1e-6 with unit gain. Softmax must tolerate scores with absolute value above 700.",
    scope="Implements the token-wise Block AttnRes depth recurrence from equations 8–10 with diagonal branch operators. KDA, experts and multimodal tokenization are outside this exercise.",
    source=[src("Moonshot AI · Kimi K3 technical report, §2.2","https://arxiv.org/abs/2607.24653")],
    modules={
    "depth_read.py":code('''
        import math

        def mix(query, sources):
            """Map a query [D] and nonempty source matrix [N,D] to one depth read [D]. Score with dot(query, row/RMS(row)), epsilon 1e-6. Apply stable softmax over sources and weight the original rows."""
            scores = []
            for row in sources:
                rms = math.sqrt(sum(x*x for x in row)/len(row) + 1e-6)
                scores.append(sum(a*b for a,b in zip(query,row))/rms)
            peak = max(scores)
            weights = [math.exp(s-peak) for s in scores]
            total = sum(weights)
            return [sum(w*row[j] for w,row in zip(weights,sources))/total for j in range(len(query))]
    '''),
    "block_stack.py":code('''
        from depth_read import mix

        def execute(embedding, layers, block_size, readout):
            """Return {blocks, state}. Start the bank with embedding. A branch reads committed sources plus the current sum, if any. Its output is gain times that read elementwise. Accumulate, commit after each block, then perform the final read."""
            bank = [embedding]
            partial = None
            for index, layer in enumerate(layers, 1):
                sources = bank if partial is None else bank + [partial]
                hidden = mix(layer["q"], sources)
                branch = [g*x for g,x in zip(layer["gain"],hidden)]
                partial = branch if partial is None else [a+b for a,b in zip(partial,branch)]
                if index % block_size == 0:
                    bank.append(partial)
                    partial = None
            if partial is not None:
                bank.append(partial)
            return {"blocks":bank[1:], "state":mix(readout,bank)}
    '''),
    "gateway.py":code('''
        from block_stack import execute
        def handle(request):
            return execute(request["embedding"], request["layers"], request["block_size"], request["readout"])
    ''')},
    parts=[part("Read from a depth bank","depth_read.py",["mix"],"Compute one stable attention read across depth. RMS-normalize keys and retain original vectors as values.","A one-entry bank must return the original source.","Large query coordinates should not overflow exp."),
           part("Commit and reuse block summaries","block_stack.py",["execute"],"Build the depth stack and its final readout. The first branch of each block has no unfinished source.","Use a distinct empty sentinel. A zero partial sum still participates.","Commit the tail when layer count is not divisible by block_size.")],
    examples=[ex(kimi([2,-2],[[0,0],[0,0]],[[1,0],[0,1]],2,[0,0]),{"blocks":[[2,-1]],"state":[2,-1.5]},"The second branch sees the mean of [2,-2] and [2,0], giving [2,-1]. Its gain keeps just the second coordinate. The completed block is [2,-1]."),
              ex(kimi([3],[],[],2,[800]),{"blocks":[],"state":[3]},"The embedding is the only available source, so the readout gives it weight one.")],
    checks=[check(1,"depth_read.mix",[[0,0],[[0,4],[10,0]]],"Original source magnitudes survive"),check(1,"depth_read.mix",[[1200],[[1],[-2]]],"Stable nonuniform scores"),check(2,"block_stack.execute",[[2],[],3,[0]],"Empty stack"),check(2,"block_stack.execute",[[3],[{"q":[0],"gain":[0]},{"q":[0],"gain":[1]}],2,[0]],"Zero partial counts as a source")]+end(2,[(kimi([2,-1],[[1,2],[-1,3],[1,1]],[[2,1],[0,1],[1,0]],2,[1,0]),"Partial final block"),(kimi([2],[[1],[0]],[[2],[3]],1,[-2]),"A new bank after each branch"),(kimi([1,0],[[0,0]],[[1,2]],4,[1234,1234]),"Large readout query")])
))


def arcee(batches, k=1, beta=0, rate=1, sharpness=1, biases=None, momentum=None):
    e=len(batches[0]["logits"][0])
    return dict(batches=batches,k=k,biases=biases or [0]*e,momentum=momentum or [0]*e,rate=rate,beta=beta,sharpness=sharpness)


def batch(logits, values):
    return dict(logits=logits,values=values)


PAPERS.append(dict(
    slug="paper-arcee-trinity",title="Arcee Trinity · Train an expert load balancer",group=LLM,level="Advanced",duration="50–70 min",
    mission="A Trinity-style expert router repeatedly sends tokens to the same few experts. Implement its bias-adjusted top-k routing and SMEBU load controller. Run consecutive batches and inspect how the next batch responds to the previous load.",
    inputs="Each batch has logits and values of shape [tokens, experts]. values contains one supplied scalar expert output per token and expert. biases and momentum describe the initial controller, k sets fan-out, and rate, beta and sharpness set its update.",
    outputs="Return routes and outputs grouped by batch then token, plus the biases and momentum after the last update. Each route lists k expert indices in selection order.",
    constraints="Batches and expert sets are nonempty. k lies between 1 and the expert count. Logits have magnitude at most 50. Rank by descending sigmoid(logit) + bias, resolving ties by smaller expert index. Reweight selected experts using unbiased sigmoid scores normalized to sum to one.",
    scope="Runs the Trinity Large routing and SMEBU equations on small batches with supplied expert outputs. The shared expert, expert MLPs, optimizer and transformer backbone are omitted.",
    source=[src("Arcee AI · Trinity Large technical report, equations 23–28","https://arxiv.org/abs/2602.17004")],
    modules={
    "expert_router.py":code('''
        import math

        def dispatch(logits, values, bias, k):
            """For a single token with E logits, scalar expert values and biases, return [selected_ids, mixture]. Use bias only to select top k and use normalized unbiased sigmoid scores to mix."""
            scores = [1/(1+math.exp(-x)) for x in logits]
            ids = sorted(range(len(scores)), key=lambda i: (-(scores[i]+bias[i]),i))[:k]
            total = sum(scores[i] for i in ids)
            return [ids,sum(scores[i]*values[i] for i in ids)/total]
    '''),
    "load_control.py":code('''
        import math

        def update(counts, bias, momentum, rate, beta, sharpness):
            """Update E expert biases from counts with positive mean n. Compute d_i=rate*tanh(sharpness*(n-count_i)/n), subtract mean(d), set m'=beta*m+(1-beta)*centered_d and b'=b+m'. Return [b',m']."""
            target = sum(counts)/len(counts)
            delta = [rate*math.tanh(sharpness*(target-c)/target) for c in counts]
            center = sum(delta)/len(delta)
            new_m = [beta*m+(1-beta)*(d-center) for m,d in zip(momentum,delta)]
            return [[b+m for b,m in zip(bias,new_m)],new_m]
    '''),
    "gateway.py":code('''
        from expert_router import dispatch
        from load_control import update

        def handle(request):
            bias, momentum = request["biases"], request["momentum"]
            routes, outputs = [], []
            for batch in request["batches"]:
                ids_in_batch, mixed = [], []
                counts = [0]*len(bias)
                for logits, values in zip(batch["logits"],batch["values"]):
                    ids, out = dispatch(logits,values,bias,request["k"])
                    for expert in ids:
                        counts[expert] += 1
                    ids_in_batch.append(ids)
                    mixed.append(out)
                routes.append(ids_in_batch)
                outputs.append(mixed)
                bias,momentum = update(counts,bias,momentum,request["rate"],request["beta"],request["sharpness"])
            return {"routes":routes,"outputs":outputs,"biases":bias,"momentum":momentum}
    ''')},
    parts=[part("Dispatch without biasing the mixture","expert_router.py",["dispatch"],"Select experts with corrected scores, then mix their supplied values using the original sigmoid scores.","Track each expert index through sorting.","The bias changes membership independently of combination weights."),
           part("Carry the controller between batches","load_control.py",["update"],"Implement normalized usage error, tanh clamping, centering and the EMA in that order.","An underused expert needs a positive change.","Center before updating the per-expert momentum.")],
    examples=[ex(arcee([batch([[0,0]],[[5,9]]),batch([[0,0]],[[5,9]])]),{"routes":[[[0]],[[1]]],"outputs":[[5],[9]],"biases":[0,0],"momentum":[0.7615941559557649,-0.7615941559557649]},"The tie selects expert 0. A bias shift of ±tanh(1) makes expert 1 win the next tie. Equal and opposite updates cancel in the final biases."),
              ex(arcee([batch([[0,0]],[[2,10]])],2),{"routes":[[[0,1]]],"outputs":[[6]],"biases":[0,0],"momentum":[0,0]},"Both experts receive one token. Equal unbiased scores average 2 and 10, and balanced usage produces zero controller change.")],
    checks=[check(1,"expert_router.dispatch",[[0,1,-1],[1,20,5],[2,0,1],2],"Biased membership and unbiased weights"),check(1,"expert_router.dispatch",[[0,0,0],[4,2,-3],[0,0,0],1],"Deterministic ties"),check(2,"load_control.update",[[3,0,0],[0,0,0],[0,0,0],0.2,0.5,1.5],"Center unequal load corrections"),check(2,"load_control.update",[[1,1],[0.1,-0.1],[0.2,-0.2],1,0.8,2],"Decay prior momentum")]+end(2,[(arcee([batch([[1,0],[0,1]],[[3,7],[-1,1]])]),"Balanced batch"),(arcee([batch([[1,0,0],[0,1,0]],[[2,4,8],[1,3,5]])],2,0.5,.2,2),"Fractional weights and imbalanced load"),(arcee([batch([[-2,0]],[[4,6]])]*3,1,.5,.25,1,[.4,-.4],[.01,-.01]),"State across three batches")])
))


def position(q,k,v):
    return {"q":q,"k":k,"v":v}


PAPERS.append(dict(
    slug="paper-glm-45", title="GLM-4.5 · Decode with normalized grouped queries",group=LLM,level="Advanced",duration="55–75 min",
    mission="A GLM-4.5 inference worker needs a causal grouped-query attention path. Process preprojected tokens one at a time. Normalize Q and K, rotate a prefix at the token's absolute position, and share each new cached key/value pair across a contiguous group of query heads.",
    inputs="tokens is a nonempty sequence. Each token contains q of shape [H,D] and k and v of shape [G,D]. eps is a positive RMS-normalization constant. H and G can differ and H is divisible by G.",
    outputs="Return heads with shape [time,H,D] in input order and cache_slots counting stored key/value pairs as time × G. Each head's context is in its original value space.",
    constraints="D is at least 2. Normalize every query and key independently across all D coordinates with default unit gain. Rotate only coordinates 0 and 1 by position radians, starting at zero. Values pass through unchanged. Attention uses 1/sqrt(D) and includes the current token.",
    scope="Implements the paper's GQA, query/key normalization, and partial rotary position encoding for a toy head width. QKV projections are supplied. Expert layers, learned normalization gain, language head and full pretraining are outside the lab.",
    source=[src("Z.ai · GLM-4.5 technical report, §2.1","https://arxiv.org/abs/2508.06471"),src("Z.ai · GLM-4.5 model configuration","https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json")],
    modules={
    "head_geometry.py":code('''
        import math

        def prepare(head, pos, eps):
            """For one [D] head, divide by sqrt(mean(head squared)+eps), rotate its first pair by pos radians with x'=x*cos-y*sin and y'=x*sin+y*cos, and return all D coordinates."""
            scale = math.sqrt(sum(x*x for x in head)/len(head)+eps)
            x = [v/scale for v in head]
            c,s = math.cos(pos),math.sin(pos)
            return [x[0]*c-x[1]*s, x[0]*s+x[1]*c] + x[2:]
    '''),
    "group_cache.py":code('''
        import math
        from head_geometry import prepare

        def attend(queries, keys, values):
            """Map prepared queries [H,D] and cached keys/values [T,G,D] to contexts [H,D]. Head h uses group h//(H/G). Apply stable softmax over time to scaled dot products."""
            heads, groups = len(queries),len(keys[0])
            result = []
            for h,q in enumerate(queries):
                g = h // (heads//groups)
                scores = [sum(a*b for a,b in zip(q,row[g]))/math.sqrt(len(q)) for row in keys]
                peak = max(scores)
                exp = [math.exp(s-peak) for s in scores]
                z = sum(exp)
                result.append([sum(w*row[g][j] for w,row in zip(exp,values))/z for j in range(len(q))])
            return result

        def decode(tokens, eps):
            """For nonempty tokens carrying q[H,D], k/v[G,D], append a single prepared key and raw value per group per position. Return {heads: [T,H,D], cache_slots: T*G}."""
            keys,values,outputs = [],[],[]
            for pos,token in enumerate(tokens):
                queries = [prepare(head,pos,eps) for head in token["q"]]
                keys.append([prepare(head,pos,eps) for head in token["k"]])
                values.append(token["v"])
                outputs.append(attend(queries,keys,values))
            return {"heads":outputs,"cache_slots":len(tokens)*len(keys[0])}
    '''),
    "gateway.py":code('''
        from group_cache import decode
        def handle(request):
            return decode(request["tokens"],request["eps"])
    ''')},
    parts=[part("Prepare each attention head","head_geometry.py",["prepare"],"Apply full-width RMS scaling followed by the two-coordinate rotation at the absolute position.","Use the original first pair for both rotated coordinates.","The tail coordinates receive the RMS scale only."),
           part("Share cache entries and decode","group_cache.py",["attend","decode"],"Implement causal softmax reads for contiguous query groups and drive them from an incrementally growing cache.","Prepared keys are cached exactly once.","Normalize over time separately for each query head.")],
    examples=[ex({"eps":1,"tokens":[position([[1,0],[0,1]],[[0,1]],[[4,2]]),position([[0,0],[0,0]],[[1,0]],[[2,6]])]},{"heads":[[[4,2],[4,2]],[[3,4],[3,4]]],"cache_slots":2},"Both heads see the only value at position zero. The zero queries at position one assign equal weight to [4,2] and [2,6]. Two positions occupy two shared key/value slots."),
              ex({"eps":0.01,"tokens":[position([[0,0]]*4,[[1,0],[0,1]],[[2,1],[9,-1]])]},{"heads":[[[2,1],[2,1],[9,-1],[9,-1]]],"cache_slots":2},"Two query heads share each key/value head. With one token, each gets its group's value with probability one.")],
    checks=[check(1,"head_geometry.prepare",[[3,4,5,6],2,.1],"Scaled tail bypasses rotation"),check(1,"head_geometry.prepare",[[0,0],62,.01],"Zero head"),check(2,"group_cache.attend",[[[1,0],[0,1]],[[[1,0]],[[0,1]]],[[[2,3]],[[3,2]]]],"Different queries share K/V"),check(2,"group_cache.decode",[[position([[0,0],[0,0]],[[1,1]],[[0,0]])],1],"Slot count under sharing")]+end(2,[({"eps":.5,"tokens":[position([[2,-1],[3,2]],[[1,0]],[[1,3]]),position([[1,0],[-1,0]],[[0,2]],[[4,1]])]},"Nonuniform temporal weights"),({"eps":1,"tokens":[position([[5,0,2]],[[4,1,-3]],[[1,2,4]])]*3},"Partial rotation across three positions"),({"eps":.1,"tokens":[position([[1,0]]*4,[[1,0],[0,1]],[[1,3],[8,-4]]),position([[2,1]]*4,[[1,3],[0,-2]],[[5,2],[8,2]])]},"Head order with two KV groups")])
))


def qwen(index_keys, keys, values, queries, block, budget):
    return dict(index_keys=index_keys,keys=keys,values=values,queries=queries,block=block,budget=budget)

def at(t, index, query):
    return dict(t=t,index=index,q=query)


PAPERS.append(dict(
    slug="paper-qwen4-max",title="Qwen4-Max companion · micro-block sparse attention",group=LLM,level="Advanced",duration="50–70 min",
    mission="Long-context decoding spends time scoring old tokens. Build a two-stage causal reader from Qwen’s publicly documented architecture preview. A lightweight index selects entire historical micro-blocks. A separate attention head then reads individual tokens within the retained blocks.",
    inputs="index_keys contains one scalar per position. keys and values are token-major matrices with key width D and value width V. Each entry in queries gives a zero-based time t, a scalar index query, and a D-vector q for exact attention. block is a positive block length and budget is a count of historical blocks.",
    outputs="Return positions and contexts, with one ascending token-index list and one V-vector per query. Process queries in request order even when t decreases.",
    constraints="Block boundaries start at token 0. Among blocks strictly before the query's block select up to budget using index × mean(index_keys in block). Earlier block wins a tie. Always add the current block's prefix through t. Exact attention uses stable softmax of dot(q,key)/sqrt(D) over retained positions. Budget may be zero.",
    scope="Qwen describes micro-block selection in its open Qwen4 architecture preview. This exercise uses a scalar mean index, one exact attention head and an explicit block budget. Those are teaching choices. Max checkpoint parameters and exact private serving kernels are outside scope.",
    source=[src("Alibaba · Qwen4-Max launch","https://www.alibabacloud.com/en/press-room/alibaba-unveils-qwen4-max"),src("Qwen · Open Qwen4 architecture preview and QSA","https://huggingface.co/Qwen/Qwen3.8-Flash-Next")],
    modules={
    "block_index.py":code('''
        def retained(index_keys, t, query, block, budget):
            """Return ascending visible token positions. Score each complete block before t's own block by query times its mean scalar index key. Select top budget with earlier block winning ties. Include the visible prefix of the current block."""
            current = t//block
            scores = [query*sum(index_keys[b*block:(b+1)*block])/block for b in range(current)]
            best = sorted(range(current), key=lambda b:(-scores[b],b))[:budget]
            positions = [i for b in sorted(best) for i in range(b*block,(b+1)*block)]
            return positions + list(range(current*block,t+1))
    '''),
    "sparse_read.py":code('''
        import math

        def read(q, keys, values, positions):
            """Given q[D], keys[T,D], values[T,V] and a nonempty selection, return a V-vector. Use softmax(dot(q,key)/sqrt(D)) over selected individual tokens, then combine the original values."""
            scores = [sum(x*y for x,y in zip(q,keys[i]))/math.sqrt(len(q)) for i in positions]
            peak = max(scores)
            exp = [math.exp(x-peak) for x in scores]
            return [sum(w*values[i][j] for w,i in zip(exp,positions))/sum(exp) for j in range(len(values[0]))]
    '''),
    "gateway.py":code('''
        from block_index import retained
        from sparse_read import read

        def handle(request):
            positions,contexts = [],[]
            for query in request["queries"]:
                ids = retained(request["index_keys"],query["t"],query["index"],request["block"],request["budget"])
                positions.append(ids)
                contexts.append(read(query["q"],request["keys"],request["values"],ids))
            return {"positions":positions,"contexts":contexts}
    ''')},
    parts=[part("Choose causal micro-blocks","block_index.py",["retained"],"Score each eligible historical block and combine its tokens with the live prefix. Sort blocks in chronological order after ranking.","Compare blocks using averages over exactly block cells.","Current-block data beyond t is invisible even when present in the input."),
           part("Read the retained KV rows","sparse_read.py",["read"],"Use full-resolution attention on selected individual keys. Key and value widths can differ.","Indexer scores are used only for selection.","Normalize all retained tokens together.")],
    examples=[ex(qwen([0,2,6,4,-1,100],[[1]]*6,[[0],[6],[9],[3],[6],[999]],[at(4,1,[0])],2,1),{"positions":[[2,3,4]],"contexts":[[6]]},"Historical block means are 1 and 5. The larger selects positions 2 and 3. Position 4 is local. Equal exact logits average 9, 3 and 6."),
              ex(qwen([3,9,4],[[0]]*3,[[1],[2],[7]],[at(2,-1,[1])],2,0),{"positions":[[2]],"contexts":[[7]]},"No historical budget remains at the start of block 1. Only token 2 is visible.")],
    checks=[check(1,"block_index.retained",[[1,3,2,2,9],4,1,2,1],"Tie between block means"),check(1,"block_index.retained",[[2,1,-4,8,6],3,-1,1,2],"Negative scalar query"),check(2,"sparse_read.read",[[1000],[[1],[2],[-1]],[[1,8],[2,-1],[8,3]],[0,2]],"Stable softmax on sparse subset"),check(2,"sparse_read.read",[[1,-1],[[1,0],[0,2]],[[2],[9]],[0,1]],"Scale by key width")]+end(2,[(qwen([3,0,2,1],[[2],[0],[-1],[2]],[[1],[4],[-2],[8]],[at(3,-1,[2]),at(1,1,[1])],1,2),"Queries arrive out of time order"),(qwen([1,2,0],[[0,1],[1,2],[2,3]],[[5,2],[3,2],[4,4]],[at(2,1,[1,-1])],5,2),"All tokens in a partial first block"),(qwen([3,3,-1,-1,2],[[0]]*5,[[1],[4],[8],[5],[2]],[at(4,-2,[9])],2,1),"Select negative-scoring keys with negative index query")])
))


def ink(segments, images, text, projection, vocab):
    return dict(segments=segments,images=images,text_table=text,projection=projection,vocab=vocab)


PAPERS.append(dict(
    slug="paper-inkling-2",title="Inkling 2 · Pack visual patches into a text sequence",group=LLM,level="Intermediate",duration="45–60 min",
    mission="A multimodal serving worker receives images between fragments of text. Complete the interface from patch features to the decoder stream. Merge each 2×2 tile, project it to text width, preserve its position among text tokens, and run a supplied tiny decoder readout.",
    inputs="segments contains objects with either text (a list of token IDs) or image (a zero-based image index). images holds even-height, even-width grids of width-C patch features. projection is [D,4C], text_table is [vocabulary,D], and vocab is an output matrix [labels,D]. At least one token survives packing.",
    outputs="Return origins in decoder order as [text,token_id] or [image,image_id,tile_row,tile_col], with the first entries represented as JSON strings. Return logits for the final position, one per row of vocab.",
    constraints="Traverse 2×2 tiles in row-major order. Flatten the four feature vectors within each tile in TL, TR, BL, BR order, then multiply by projection. Text embeddings and projected image tokens share D. Images may repeat. The supplied diagnostic head reads the mean of the visible prefix using uniform attention and projects to logits.",
    scope="Implements the visual/text interface of Inkling 2 Small using precomputed patch features, a teaching merge projection and a deterministic decoder probe. Production vision encoding, attention residuals and sparse decoder layers are outside scope.",
    source=[src("Thinking Machines Lab · Introducing Inkling 2 Small, vision interpretation","https://thinkingmachines.ai/news/inkling-2-small/"),src("Thinking Machines Lab · Inkling 2 Small model card","https://thinkingmachines.ai/model-card/inkling-2-small/")],
    modules={
    "image_bridge.py":code('''
        def visual_tokens(grid, projection):
            """Map an [H,W,C] grid to projected rows [H*W/4,D]. Scan 2x2 tiles in row-major order, concatenate TL,TR,BL,BR patch features within each tile and apply the [D,4C] matrix without a bias."""
            result = []
            for r in range(0,len(grid),2):
                for c in range(0,len(grid[0]),2):
                    flat = grid[r][c]+grid[r][c+1]+grid[r+1][c]+grid[r+1][c+1]
                    result.append([sum(a*b for a,b in zip(row,flat)) for row in projection])
            return result
    '''),
    "prompt_packer.py":code('''
        from image_bridge import visual_tokens

        def pack(segments, images, text_table, projection):
            """Return [rows,origins]. Expand text ID lists in order. Expand each image to projected 2x2 tile tokens in place. Origins are ['text',id] or ['image',image_id,tile_row,tile_col] with zero-based tile coordinates."""
            rows,origins = [],[]
            for seg in segments:
                if "text" in seg:
                    for token in seg["text"]:
                        rows.append(text_table[token])
                        origins.append(["text",token])
                else:
                    image_id=seg["image"]
                    grid=images[image_id]
                    width=len(grid[0])//2
                    for i,vec in enumerate(visual_tokens(grid,projection)):
                        rows.append(vec)
                        origins.append(["image",image_id,i//width,i%width])
            return [rows,origins]
    '''),
    "probe_head.py":code('''
        def last_logits(rows, vocab):
            """A fixed decoder probe with zero query/key projections and identity values."""
            state = [sum(row[j] for row in rows)/len(rows) for j in range(len(rows[0]))]
            return [sum(w*x for w,x in zip(head,state)) for head in vocab]
    '''),
    "gateway.py":code('''
        from prompt_packer import pack
        from probe_head import last_logits
        def handle(request):
            rows,origins=pack(request["segments"],request["images"],request["text_table"],request["projection"])
            return {"origins":origins,"logits":last_logits(rows,request["vocab"])}
    ''')},
    parts=[part("Merge spatially adjacent features","image_bridge.py",["visual_tokens"],"Map each 2×2 neighborhood through the projection. Keep relative position within the tile intact while flattening.","Move two columns between neighboring tile origins.","The first input coordinate of the bottom-left patch follows all top-row channels."),
           part("Interleave image and text tokens","prompt_packer.py",["pack"],"Build one decoder sequence and origin manifest, allowing repeated images, empty text segments and multiple token IDs per segment.","Compute tile column count from the patch grid width.","Expanding an image changes decoder positions for later text.")],
    examples=[ex(ink([{"text":[1]},{"image":0},{"text":[0]}],[[[[1],[2]],[[3],[4]]]],[[0,2],[1,0]],[[1,0,0,1],[0,1,1,0]],[[1,0],[0,1]]),{"origins":[["text",1],["image",0,0,0],["text",0]],"logits":[2,7/3]},"The 2×2 tile maps to [5,5]. The other two embeddings contribute [1,0] and [0,2]. Uniform attention gives their mean [2,7/3]."),
              ex(ink([{"text":[]},{"text":[0]}],[],[[4,-1]],[[1,0,0,0],[0,1,0,0]],[[1,-2]]),{"origins":[["text",0]],"logits":[6]},"An empty fragment contributes zero positions. The sole text embedding yields 4 + 2 = 6.")],
    checks=[check(1,"image_bridge.visual_tokens",[[[[1],[2],[3],[4]],[[5],[6],[7],[8]]],[[1,-1,2,0]]],"Neighbor tiles respect row boundaries"),check(1,"image_bridge.visual_tokens",[[[[1,2],[3,4]],[[5,6],[7,8]]],[[0,1,0,0,0,0,0,-1]]],"Channels are concatenated inside each patch"),check(2,"prompt_packer.pack",[[{"text":[1,0,1]}],[],[[3],[5]],[[1,0,0,0]]],"Repeated text IDs"),check(2,"prompt_packer.pack",[[{"image":0},{"image":0}],[[[[1],[2]],[[3],[4]]]],[[0]],[[1,1,1,1]]],"Repeated image reference")]+end(2,[(ink([{"image":0}],[[[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],[[1]],[[0,1,-1,0]],[[2],[-1]]),"Multiple horizontal image tokens"),(ink([{"text":[0,0]},{"image":0}],[[[[1],[1]],[[2],[2]],[[3],[3]],[[4],[4]]]],[[2]],[[1,2,0,1]],[[1]]),"Tall grid and repeated text"),(ink([{"image":1},{"text":[0]},{"image":0}],[[[[1],[0]],[[2],[0]]],[[[0],[6]],[[8],[0]]]],[[2]],[[1,2,3,4]],[[2]]),"Image order follows the prompt")])
))


def gem(students,teachers,dims,lr):
    return dict(students=students,teachers=teachers,dims=dims,lr=lr)


PAPERS.append(dict(
    slug="paper-embeddinggemma-2",title="EmbeddingGemma 2 · Distill nested embedding widths",group=SEQ,level="Advanced",duration="55–75 min",
    mission="A training worker must align a student's multimodal representations with frozen teacher vectors at several output widths. Implement direct embedding matching and backpropagate through each prefix's normalization. Use the gradients for one update of the supplied student outputs.",
    inputs="students and teachers are equally sized nonempty batches of raw vectors. A teacher vector may be wider than its student counterpart. dims is a nonempty list of distinct positive prefix widths valid for both. lr is a nonnegative step size.",
    outputs="Return loss as the mean squared Euclidean distance between normalized student/teacher prefixes, averaging over examples and widths but summing over coordinates. Return students after one simultaneous gradient descent step on this loss.",
    constraints="Normalize each truncated prefix independently. Every selected prefix has positive norm. A coordinate outside a given prefix receives zero gradient from that term. Treat teachers as constants and evaluate all terms at the original students.",
    scope="Implements the direct distillation and Matryoshka prefix mechanisms of the EmbeddingGemma 2 report for a batch of paired vectors. The exercise uses equal prefix weights and optimizes student vectors directly. Modality encoders, negative sampling and retrieval are outside the lab.",
    source=[src("Google DeepMind · EmbeddingGemma 2 technical report, §2.4 and equation 6","https://arxiv.org/abs/2610.37364"),src("Google · EmbeddingGemma 2 model card","https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2")],
    modules={
    "prefix_distill.py":code('''
        import math

        def match(student, teacher, k):
            """Return [loss,grad] for one width-k prefix pair. For u=x/||x|| and t=y/||y||, loss=sum((u-t)^2) and grad=2*((u-t)-u*dot(u,u-t))/||x||. Return a k-vector gradient."""
            sx=math.sqrt(sum(x*x for x in student[:k]))
            sy=math.sqrt(sum(y*y for y in teacher[:k]))
            u=[x/sx for x in student[:k]]
            diff=[a-b/sy for a,b in zip(u,teacher[:k])]
            along=sum(a*d for a,d in zip(u,diff))
            grad=[2*(d-a*along)/sx for d,a in zip(diff,u)]
            return [sum(d*d for d in diff),grad]
    '''),
    "student_step.py":code('''
        from prefix_distill import match

        def step(students, teachers, dims, lr):
            """Return {loss,students}. Average prefix losses over batch and dims, sum contributions to shared student coordinates, and update all raw vectors once with learning rate lr."""
            count=len(students)*len(dims)
            loss=0
            updated=[]
            for student,teacher in zip(students,teachers):
                total=[0.0]*len(student)
                for k in dims:
                    value,grad=match(student,teacher,k)
                    loss+=value/count
                    for i,g in enumerate(grad):
                        total[i]+=g/count
                updated.append([v-lr*g for v,g in zip(student,total)])
            return {"loss":loss,"students":updated}
    '''),
    "gateway.py":code('''
        from student_step import step
        def handle(request):
            return step(request["students"],request["teachers"],request["dims"],request["lr"])
    ''')},
    parts=[part("Differentiate embedding matching","prefix_distill.py",["match"],"Truncate then normalize both vectors. Compute the distance and the gradient with respect to the unnormalized student prefix.","The derivative of normalization removes the component parallel to the unit vector.","A wider teacher suffix has no influence here."),
           part("Train at overlapping widths","student_step.py",["step"],"Accumulate overlapping prefix gradients in the full-width student, average over batch and widths, then make one simultaneous update.","The short and long prefixes can both affect coordinate zero.","There is no division by the number of embedding coordinates.")],
    examples=[ex(gem([[1,0]],[[0,1,9]],[2],0.5),{"loss":2,"students":[[1,1]]},"The normalized vectors are orthogonal, so their squared distance is 2. Removing the radial component yields gradient [0,-2]. A half-step adds 1 to the second coordinate."),
              ex(gem([[1,0]],[[1,1]],[1,2],1),{"loss":0.2928932188134525,"students":[[1,0.7071067811865476]]},"The width-one vectors agree. At width two the distance is 2−√2. Averaging the two losses and gradients gives loss 1−1/√2 and moves the second coordinate by 1/√2.")],
    checks=[check(1,"prefix_distill.match",[[2,0],[-2,1,8],2],"Tangent-space normalization derivative"),check(1,"prefix_distill.match",[[-3,4],[7,100],1],"Antiparallel scalar has finite loss and zero gradient"),check(2,"student_step.step",[[[1,0],[2,1]],[[1,2],[1,-2]],[2],.2],"Batch gradient scaling"),check(2,"student_step.step",[[[2,1,9]],[[1,3,7,8]],[2],1],"Untrained suffix preserved")]+end(2,[(gem([[2,-1,3]],[[-3,1,2,7]],[2,3],.4),"Multiple overlapping nontrivial gradients"),(gem([[-2,1],[2,1]],[[1,2],[-1,2]],[2,1],.5),"Mixed signs and prefix order"),(gem([[2,3,1]],[[2,4,9]],[3],0),"Zero learning rate retains loss signal")])
))
