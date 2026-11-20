"""Editorial briefs, module boundaries and integration contracts for the problem library."""
from dataclasses import dataclass
from recovery_lab import GATEWAY as RECOVERY_GATEWAY


@dataclass
class Lab:
    mission: str
    inputs: str
    outputs: str
    constraints: str
    modules: dict
    gateway: str
    examples: list
    level: str = "Intermediate"


LABS = {}

LABS["kv-cache"] = Lab(
    "A chat server receives a prompt, then receives one or more continuation chunks for the same request. Its current decoder recomputes the whole prefix every time. Finish the incremental path. The caller must see exactly the same next-token scores as the supplied full-prefix decoder, even when an empty chunk arrives between two tokens.",
    "layers is the positive number of decoder layers. chunks is an ordered list of lists of integer token IDs from a single request. The first chunk is prefill; the remaining chunks continue the same sequence.",
    "Return logits grouped by input chunk. Each token produces three logits in model vocabulary order. Also return tokens_processed, the number of real tokens across all chunks.",
    "Use the provided two-dimensional projections and attention routine. Keep an independent key/value history per layer. Use absolute positions across chunk boundaries. Project each new token once per layer. Numeric tolerances are specified below.",
    {"model_ops":("Fixed projections and attention used by both decoding paths.",["project","attend"]),
     "layer_cache":("Append a token's keys and values, then read that layer's history.",["advance"]),
     "decoder":("Advance one token through all layers and produce its logits.",["decode"]),
     "stream_worker":("Own the cache over the lifetime of a chunked request.",["serve"])},
    """from stream_worker import serve

def handle(request):
    chunks = request["chunks"]
    logits = serve(chunks, request["layers"])
    return {"logits": logits, "tokens_processed": sum(map(len, chunks))}
""",
    [({"layers":1,"chunks":[[0],[]]}, {"logits":[[[1,2,-1]],[]],"tokens_processed":1},
      "Token 0 embeds to [0, 1]. With one cached token, its attention weight is 1. The model adds [1, 1] to the embedding and reads logits from [1, 2]. The empty chunk emits no logits."),
     ({"layers":2,"chunks":[[],[]]}, {"logits":[[],[]],"tokens_processed":0},"Even an empty chunk has a corresponding output list. Neither chunk advances a layer's position.")], "Advanced")

LABS["live-batching"] = Lab(
    "Two clients share one decoding worker. A short request should be able to leave while a longer request continues, and a waiting request should take the free slot on the next tick. Complete the serving loop and return enough information to audit when each client finished.",
    "capacity is the maximum active requests. requests are [arrival_tick, request_id, prompt_token_ids, output_budget], ordered by arrival. The supplied token model defines the greedy output for each state.",
    "timeline contains the [id, token] emissions at each tick, beginning at tick 0. finished_at maps each ID to the zero-based tick of its last emission.",
    "Request IDs are unique. Capacity and budgets are positive. Ties use input order; active requests keep their order. Admit before decoding and release completed slots after decoding.",
    {"token_model":("Prefill prompts and generate one deterministic batch.",["new_state","model_batch"]),
     "batch_executor":("Pair model outputs with the requests that produced them.",["step"]),
     "scheduler":("Decide who runs next and who stays queued.",["tick"]),
     "serving_loop":("Own the arrival clock and advance it through the last emission.",["dispatch"])},
    """from serving_loop import dispatch

def handle(request):
    timeline = dispatch(request["requests"], request["capacity"])
    finished = {}
    for tick, emissions in enumerate(timeline):
        for request_id, _ in emissions:
            finished[request_id] = tick
    return {"timeline": timeline, "finished_at": finished}
""",
    [({"capacity":1,"requests":[[1,"a",[],1],[1,"b",[],1]]},{"timeline":[[],[["a",2]],[["b",2]]],"finished_at":{"a":1,"b":2}},"Tick 0 is idle. Both clients arrive at tick 1. With one slot, a runs first and b starts at tick 2. Both empty prompts start in the same model state."),
     ({"capacity":2,"requests":[]},{"timeline":[],"finished_at":{}},"There is no arrival or emission, so no tick is recorded.")], "Advanced")

LABS["paged-cache"] = Lab(
    "An inference server has a fixed pool of equally sized KV blocks. Several requests grow and finish in interleaved order. Build the allocator and event loop so that an append either reserves every requested slot or leaves the pool exactly as it was.",
    "blocks is the number of physical blocks and block_size is tokens per block. events contains ['append', id, count], ['release', id], or ['free'].",
    "Return replies in event order. A successful append returns [physical_block, slot] for each new token; capacity failure and release return null. A free query returns sorted free block IDs. Also return failed_appends.",
    "Blocks start free and are numbered from zero. Allocate the lowest available IDs. Count zero is valid. Reuse the tail of a request's last block before allocating another block. Releasing an unknown ID is harmless.",
    {"block_pool":("Decide whether a request can grow and reserve its backing blocks.",["reserve"]),
     "page_table":("Translate logical token offsets into physical addresses.",["locate"]),
     "memory_worker":("Maintain per-request tables while processing allocation events.",["replay"])},
    """from memory_worker import replay

def handle(request):
    events = request["events"]
    replies = replay(request["blocks"], request["block_size"], events)
    failed = sum(event[0] == "append" and reply is None for event, reply in zip(events, replies))
    return {"replies": replies, "failed_appends": failed}
""",
    [({"blocks":1,"block_size":2,"events":[["append","a",1],["append","a",2],["append","a",1],["free"]]},
      {"replies":[[[0,0]],None,[[0,1]],[]],"failed_appends":1},"The second append needs two slots when only one remains. It fails atomically. The next one-token append can still claim slot 1."),
     ({"blocks":2,"block_size":1,"events":[["append","a",1],["release","a"],["append","b",1],["free"]]},
      {"replies":[[[0,0]],None,[[0,0]],[1]],"failed_appends":0},"Releasing a returns block 0 to the pool. The lowest-free-ID rule gives it to b.")], "Advanced")

LABS["shared-prefix"] = Lab(
    "Two chat sessions start from the same cached prompt and then diverge. Build the snapshot lookup and session lifecycle so that a continuation can reuse matching model state without changing what another session observes.",
    "events is an ordered stream. save takes [model, tokens, KV_values], open takes [id, model, tokens], append takes [id, value], and read or cancel takes [id]. Each event starts with the operation name.",
    "Return a reads array containing one {id, values} record per read, in read order. Unknown or cancelled requests produce null. An open request with zero reusable tokens produces [].",
    "A snapshot belongs to one model version. Use the longest matching token prefix, including partial matches. On a length tie use the earliest saved entry. Append targets an open session.",
    {"token_prefix":("Compare consecutive prefix tokens.",["prefix_length"]),
     "snapshot_index":("Select compatible saved KV values for a request.",["choose"]),
     "sessions":("Manage open requests, local appends, reads and cancellation.",["run"])},
    """from sessions import run

def handle(request):
    events = request["events"]
    replies = run(events)
    return {"reads": [{"id": event[1], "values": reply} for event, reply in zip(events, replies) if event[0] == "read"]}
""",
    [({"events":[["save","v1",[7,8],[10,20]],["open","a","v1",[7,9]],["open","b","v1",[7]],["append","a",30],["read","a"],["read","b"]]},
      {"reads":[{"id":"a","values":[10,30]},{"id":"b","values":[10]}]},"Both requests reuse the state for token 7. Only a receives the new state 30."),
     ({"events":[["open","x","v2",[3]],["read","x"],["cancel","x"],["read","x"]]},
      {"reads":[{"id":"x","values":[]},{"id":"x","values":None}]},"The first read observes an open request with no cached state. The second read observes a cancelled request.")])

LABS["speculative-decode"] = Lab(
    "A draft model proposes a block of tokens and a target model verifies it. Finish the verifier used by both greedy and sampled generation. Clients must receive a correction at the first rejection, or one bonus token when the entire draft is accepted.",
    "mode selects 'greedy' or 'sample'. Greedy requests contain draft and target token IDs. Sampled requests contain proposals, targets, drafts, uniforms and bonus_u. Probability rows are in vocabulary order; uniforms[i] contains the acceptance draw and the correction draw.",
    "Return {tokens: [...]} in emission order. A rejected position contributes the target correction and ends the block.",
    "The target provides one extra position for a bonus. Probability rows sum to one, draws are in [0,1), and every proposed token has positive draft probability. Inputs are supplied snapshots of model predictions.",
    {"sampling":("Supplied categorical draw over an ordered distribution.",["draw"]),
     "verification":("Check greedy agreement and verify an individual sampled proposal.",["greedy","verify"]),
     "block_decoder":("Decide when to stop, correct, or emit the bonus.",["sample_block"])},
    """from verification import greedy
from block_decoder import sample_block

def handle(request):
    if request["mode"] == "greedy":
        tokens = greedy(request["draft"], request["target"])
    else:
        tokens = sample_block(request["proposals"], request["targets"], request["drafts"], request["uniforms"], request["bonus_u"])
    return {"tokens": tokens}
""",
    [({"mode":"greedy","draft":[4,8,1],"target":[4,3,9,2]},{"tokens":[4,3]},"The first draft token matches. At the second position the target wants 3, so the verifier emits 3 and discards the rest."),
     ({"mode":"sample","proposals":[0],"targets":[[0.2,0.8],[0.5,0.5]],"drafts":[[0.8,0.2]],"uniforms":[[0.9,0.1]],"bonus_u":0.4},{"tokens":[1]},"Token 0 is accepted with probability 0.25. The draw 0.9 rejects it. All positive residual mass lies on token 1, and rejection ends the block.")], "Advanced")

LABS["block-attention"] = Lab(
    "A worker can read only one block of keys and values at a time. Complete its attention reducer so a change in block boundaries has no effect on the answer and a large score does not overflow the softmax.",
    "query is a nonempty vector. blocks is a list of [keys, values] pairs. Each value vector has value_width components. Each key has the same width as the query.",
    "Return {context: [...]} containing the softmax-weighted sum over every key/value pair in all blocks. When there are no keys, return a zero vector of value_width.",
    "Keep working storage proportional to one block plus the output width. Scale dot products by the square root of the query width. Empty blocks can occur anywhere.",
    {"score_kernel":("Score the current block's keys against one query.",["scores"]),
     "online_state":("Maintain the running maximum, mass and weighted values.",["absorb"]),
     "attention_worker":("Consume blocks and finalize the output.",["attention"])},
    """from attention_worker import attention

def handle(request):
    context = attention(request["query"], request["blocks"], request["value_width"])
    return {"context": context}
""",
    [({"query":[0],"value_width":1,"blocks":[[[[2]],[[4]]],[[[8]],[[10]]]]},{"context":[7]},"Both dot products are zero, so both values receive weight 1/2 even though they arrive in separate blocks."),
     ({"query":[1],"value_width":2,"blocks":[[[],[]]]},{"context":[0,0]},"An empty block contributes no mass. The output still has the configured value width.")], "Advanced")

LABS["quantized-linear"] = Lab(
    "A model exporter writes a compact projection and an inference worker evaluates it. Implement both sides of the format. A per-channel scale must mean the same thing to the byte serializer and to the runtime kernel.",
    "weights is a matrix with one row per output channel. x is one input vector and bias has one element per channel.",
    "Return output (the scores), packed_hex (the lowercase serialized weight bytes) and storage_bytes (weights and scales only).",
    "Use symmetric signed int8 codes from -127 to 127 and round ties to even. A zero channel has scale 1. Store each scale as little-endian float32 followed by one signed byte per weight. Compute inference with the unrounded calibration scale.",
    {"calibration":("Choose a scale and encode one output channel.",["quantize"]),
     "inference_kernel":("Evaluate coded rows with independent scales and bias.",["project"]),
     "weight_format":("Write the binary deployment format.",["export"])},
    """from calibration import quantize
from inference_kernel import project
from weight_format import export

def handle(request):
    weights = request["weights"]
    packed = export(weights)
    rows = [quantize(row) for row in weights]
    return {"output": project(rows, request["x"], request["bias"]), "packed_hex": packed, "storage_bytes": len(bytes.fromhex(packed))}
""",
    [({"weights":[[127,-127]],"x":[2,1],"bias":[1]},{"output":[128],"packed_hex":"0000803f7f81","storage_bytes":6},"The scale is 1. Its four bytes are followed by 7f and 81 for +127 and -127. The score is 254 − 127 + 1."),
     ({"weights":[[0]],"x":[5],"bias":[2]},{"output":[2],"packed_hex":"0000803f00","storage_bytes":5},"The zero channel stores a unit scale and a zero code, leaving only the bias.")])

LABS["chat-sft"] = Lab(
    "Your chat training pipeline currently scores every token, including the user's question. Repair the path from tokenized turns through truncation and batch padding. A user token can be visible to attention while contributing nothing to the training loss.",
    "chats is a batch. Each conversation is a list of [role, content_token_ids] turns. limit is the maximum retained tokens per conversation, including framing tokens.",
    "Return three equally shaped grids: input_ids, labels, and attention_mask. labels hold the target ID or -100. attention_mask distinguishes real input positions (1) from padding (0). The trainer handles the next-token shift.",
    "Role IDs are system=1, user=2 and assistant=3. END=4 and PAD=0. Keep assistant content and assistant END as targets. Truncate from the right and pad to the longest retained row in this batch.",
    {"chat_format":("Define the protocol constants and frame a single speaker turn.",["encode_turn"]),
     "conversation":("Flatten turns and enforce the length budget.",["encode_chat"]),
     "batch_collator":("Align input IDs, labels and attention masks across a batch.",["collate"])},
    """from batch_collator import collate

def handle(request):
    return collate(request["chats"], request["limit"])
""",
    [({"chats":[[["user",[9]],["assistant",[8]]],[]],"limit":6},
      {"input_ids":[[2,9,4,3,8,4],[0,0,0,0,0,0]],"labels":[[-100,-100,-100,-100,8,4],[-100,-100,-100,-100,-100,-100]],"attention_mask":[[1,1,1,1,1,1],[0,0,0,0,0,0]]},
      "Only 8 and the assistant's end marker receive loss. All six real tokens are visible to attention. The empty second chat pads to the same width."),
     ({"chats":[[["assistant",[7,8]]]],"limit":2},{"input_ids":[[3,7]],"labels":[[-100,7]],"attention_mask":[[1,1]]},"The cut falls inside an answer. No end marker is inserted after token 7.")])

LABS["lora-layer"] = Lab(
    "A training service adapts a frozen projection for one customer. Its serving tier can load only an ordinary weight matrix. Implement the adapter, the update, and the export so that one training step produces a deployable projection.",
    "weight, a and b are matrices of shapes [out,in], [rank,in] and [out,rank]. x is an input vector, target is an output vector, alpha scales the adapter and lr is the learning rate.",
    "Return before and after (the output vectors before and after one SGD step), plus merged (the post-step standalone weight matrix). The base weight and incoming adapter arrays must stay unchanged.",
    "Rank is positive. Use loss 0.5 × sum of squared output errors. Both adapter gradients use the same pre-update parameters. Scale the adapter by alpha/rank. The projection has no bias.",
    {"linear_ops":("Provided vector and matrix primitives for the frozen model.",["dot","linear"]),
     "adapter":("Compute an adapted projection with frozen weights.",["forward"]),
     "optimizer_step":("Apply a simultaneous SGD update to the two adapter matrices.",["update"]),
     "merge_weights":("Materialize a standalone matrix compatible with the serving model.",["merge"])},
    """from adapter import forward
from optimizer_step import update
from merge_weights import merge
from linear_ops import linear

def handle(request):
    w, a, b = request["weight"], request["a"], request["b"]
    x, alpha = request["x"], request["alpha"]
    before = forward(w, a, b, x, alpha)
    a, b = update(w, a, b, x, request["target"], alpha, request["lr"])
    merged = merge(w, a, b, alpha)
    return {"before": before, "after": linear(merged, x), "merged": merged}
""",
    [({"weight":[[1]],"a":[[1]],"b":[[0]],"x":[2],"target":[4],"alpha":1,"lr":0.25},{"before":[2],"after":[4],"merged":[[2]]},
      "The adapter starts at zero output. The first step changes B from 0 to 1 while A stays at 1. The merged weight is 2 and maps input 2 to 4."),
     ({"weight":[[2]],"a":[[2]],"b":[[1]],"x":[1],"target":[0],"alpha":1,"lr":0},{"before":[4],"after":[4],"merged":[[4]]},"With a zero learning rate the adapter contributes the same value before and after export.")], "Advanced")

LABS["dpo-loss"] = Lab(
    "A post-training job receives precomputed policy and reference log probabilities for pairs of answers. Build the objective that a trainer can call on a whole batch, including batches with very confident preferences.",
    "pairs contains chosen and rejected records. Each has aligned policy and reference token log-probabilities and a binary mask selecting completion tokens. beta is a positive scaling factor.",
    "Return loss, the mean DPO loss, and gradients, one derivative of that mean with respect to each unscaled preference margin, in input order.",
    "Sum completion log probabilities, ignore prompt and padding positions, and use the frozen reference's preference as the baseline. Use a numerically stable loss. For zero pairs return null loss and an empty gradient list.",
    {"sequence_scores":("Mask and aggregate per-token log probabilities.",["completion"]),
     "preferences":("Measure a policy preference relative to the reference.",["margin"]),
     "objective":("Reduce a numerically stable loss and derivatives over a batch.",["batch"])},
    """from objective import batch

def handle(request):
    loss, gradients = batch(request["pairs"], request["beta"])
    return {"loss": loss, "gradients": gradients}
""",
    [({"beta":2,"pairs":[{"chosen":{"policy":[-8,-1],"reference":[-9,-1],"mask":[0,1]},"rejected":{"policy":[-5,-2],"reference":[-6,-2],"mask":[0,1]}}]},
      {"loss":0.6931471805599453,"gradients":[-1.0]},"After masking the first token, both models have the same preference. The margin is zero, the loss is log(2), and the derivative is −beta/2."),
     ({"beta":1,"pairs":[]},{"loss":None,"gradients":[]},"An empty worker batch has no mean loss to report.")])

LABS["data-parallel"] = Lab(
    "A data-parallel trainer gives the last, smaller shard to a second worker. Its update diverges from a single-process run even though every worker started at the same weights. Repair the rank computation and reducer, then drive several synchronized training steps.",
    "weights is the starting vector and lr the learning rate. steps contains global batches. Each step is a list of rank shards, and each shard contains [features, target] examples.",
    "Return history with one post-update weight vector per global step and final with the last vector (or the initial weights when steps is empty).",
    "A step has at least one rank. Rank shards may be empty or unequal. Use half squared error per example, average over all examples globally, and update once per step. The supplied two-process driver runs locally.",
    {"rank_worker":("Predict and compute one rank's sample count and local gradient.",["prediction","local"]),
     "collective":("Combine all rank reports into the global mean gradient.",["reduce_ranks"]),
     "coordinator":("Broadcast a common weight version and apply synchronized steps.",["train"])},
    """from coordinator import train

def handle(request):
    weights = request["weights"]
    history = train(weights, request["steps"], request["lr"])
    return {"history": history, "final": history[-1] if history else weights}
""",
    [({"weights":[0],"lr":1,"steps":[[[[[1],2]],[[[1],0],[[1],0],[[1],0]]]]},{"history":[[0.5]],"final":[0.5]},
      "Rank 0 sees one gradient of −2. Rank 1 sees three zero gradients. Across four examples the mean is −0.5, so SGD changes the weight from 0 to 0.5."),
     ({"weights":[3],"lr":1,"steps":[[[],[]]]},{"history":[[3]],"final":[3]},"Both workers report zero examples. A synchronized step still produces a snapshot.")], "Advanced")

LABS["paired-bootstrap"] = Lab(
    "A release reviewer wants an uncertainty interval for a model improvement. Both models answered the same questions, and several questions can come from one document. Implement the paired statistic and let the reviewer choose the sampling unit.",
    "rows contains [document_id, score_A, score_B]. mode is 'row' or 'document'. repeats is the number of resamples and seed fixes their random draws.",
    "Return delta (mean B minus A across original rows) and ci95 (lower and upper bootstrap percentiles). Use null for undefined statistics.",
    "Sample units with replacement. Each repeated document contributes every one of its rows again. Use the seeded draw order and percentile interpolation specified in the interval contract so reviews can be reproduced.",
    {"paired_metric":("Calculate the observed paired score change.",["delta"]),
     "resampling":("Build one replicate under the chosen sampling unit.",["replicate"]),
     "confidence":("Drive the seeded experiment and interpolate its endpoints.",["interval"])},
    """from confidence import interval

def handle(request):
    delta, low, high = interval(request["rows"], request["mode"], request["repeats"], request["seed"])
    return {"delta": delta, "ci95": [low, high]}
""",
    [({"rows":[["d1",0,1],["d1",1,1]],"mode":"document","repeats":20,"seed":4},{"delta":0.5,"ci95":[0.5,0.5]},
      "The paired changes are 1 and 0. There is just one document to resample, so every replicate includes both rows and has mean 0.5."),
     ({"rows":[],"mode":"row","repeats":5,"seed":7},{"delta":None,"ci95":[None,None]},"There is no question to compare or resample.")])

LABS["judge-validation"] = Lab(
    "Your evaluation team suspects its pairwise judge favors the answer displayed on the left. Each item is judged in both display orders. Normalize the labels, then report where the judge changes its mind and where a stable judgement agrees with the human reviewer.",
    "rows contains ab and ba verdicts ('left', 'right', 'tie', or 'invalid'), a human label ('A', 'B', or 'tie'), task, len_a and len_b. A missing vote counts as invalid.",
    "Return overall statistics and two breakdowns, by_task and by_length. Each includes total, valid, stable, agreement (agreed/stable) and flip_rate ((valid−stable)/valid).",
    "A stable pair requires two valid votes for the same answer after reordering. Keep ties. Use null for rates with zero denominator. Include only observed groups.",
    {"judge_votes":("Translate display positions into answer identities.",["canonical"]),
     "pair_audit":("Audit both presentations against one human label.",["audit"]),
     "slice_report":("Aggregate denominators and rates by task and answer length.",["summarize"])},
    """from slice_report import summarize

def handle(request):
    return summarize(request["rows"])
""",
    [({"rows":[{"ab":"left","ba":"left","human":"A","task":"code","len_a":5,"len_b":5}]},
      {"overall":dict(total=1,valid=1,stable=0,agreement=None,flip_rate=1),"by_task":{"code":dict(total=1,valid=1,stable=0,agreement=None,flip_rate=1)},"by_length":{"equal":dict(total=1,valid=1,stable=0,agreement=None,flip_rate=1)}},
      "Choosing the left answer twice selects A in the first ordering and B in the second. The pair flips. Since it is the only pair, there are no stable votes on which to measure agreement."),
     ({"rows":[]},{"overall":dict(total=0,valid=0,stable=0,agreement=None,flip_rate=None),"by_task":{},"by_length":{}},"No rows means no observed task or length slices.")])

LABS["durable-replay"] = Lab(
    "The interview app has scored a candidate's answer, but the process crashes before recording completion. The client retries, possibly with different text. Recover the original turn across the journal and an external tool that supports idempotency keys.",
    "attempts is a chronological list of [session_id, turn_id, answer, stop]. stop is 'intent', 'tool', or 'finish', indicating where the worker stops. IDs and answers are strings.",
    "Return replies (one result or null per attempt), journal (durable intent and done records), and effects (the external tool's recorded results by key).",
    "Persist the first answer before calling the tool. A finished turn returns its stored result even if the retry requests an earlier stop. Only the journal and external store survive a restart. The key encoding is specified in the identifier component.",
    {"effect_ids":("Encode session and turn identity without collisions.",["effect_key"]),
     "recovery":("Resume work at the first unfinished durable boundary.",["resume"]),
     "turn_worker":("Replay client attempts against shared durable storage.",["replay"])},
    """from turn_worker import replay

def handle(request):
    return replay(request["attempts"])
""",
    [({"attempts":[["s","t","yes","tool"],["s","t","no","finish"]]},
      {"replies":[None,"score:yes"],"journal":[["1:st","intent","yes"],["1:st","done","score:yes"]],"effects":{"1:st":"score:yes"}},
      "The first attempt crashes after the tool stores its result. On retry the original intent still says yes. The worker records that result as done and returns it."),
     ({"attempts":[["s","1","x","intent"]]},{"replies":[None],"journal":[["1:s1","intent","x"]],"effects":{}},"Stopping immediately after intent leaves the remote tool untouched.")], "Advanced")

LABS["release-comparison"] = Lab(
    "An evaluation dashboard must decide whether to promote a candidate model. Two workers label each prompt independently and can correct or withdraw a label. Build the ledger and rollout report so the dashboard always reflects the latest comparable outcomes.",
    "events is a stream of [prompt_id, model, revision, correct] updates in delivery order. model is 'baseline' or 'candidate'; correct is true, false, or null for a withdrawal. min_pairs is the minimum number of comparable prompts.",
    "Return paired, wins, losses, ties, and decision. A candidate-only correct answer is a win. Promote when the paired sample meets min_pairs and wins exceed losses; otherwise hold.",
    "Revision order is per (prompt, model). Repeated deliveries at the same revision agree. A missing or withdrawn label excludes that prompt until both labels are available. A false label is a valid observation.",
    {"events":("Supplied event decoder used at the service boundary.",[]),"ledger":("Apply updates and expose the current outcomes.",[]),"report":("Produce the decision consumed by the dashboard.",[])},
    """from events import read_events
from ledger import OutcomeLedger
from report import summarize

def handle(request):
    ledger = OutcomeLedger()
    for event in read_events(request["events"]):
        ledger.ingest(event)
    return summarize(ledger, request["min_pairs"])
""",
    [({"events":[["p1","baseline",1,False],["p1","candidate",1,True]],"min_pairs":1},{"paired":1,"wins":1,"losses":0,"ties":0,"decision":"promote"},
      "Both labels exist on p1 and only the candidate is correct. The sample meets the threshold and the candidate has more wins than losses."),
     ({"events":[["p1","baseline",1,False],["p1","candidate",1,True],["p1","candidate",2,None]],"min_pairs":1},
      {"paired":0,"wins":0,"losses":0,"ties":0,"decision":"hold"},"Revision 2 withdraws the candidate label. The previously comparable prompt drops out of the current report.")])

# Some labs exercise persistent services or model lifecycle boundaries as well as
# the small unit contracts. Their public examples run through these entry points.
recovery = LABS["durable-replay"]
recovery.inputs += " To restart from an earlier run, supply checkpoint with the previous journal and effects. Without a checkpoint, both stores start empty."
recovery.outputs += " On restart, return replies for the new attempts and the full updated stores. Leave the input checkpoint untouched."
recovery.constraints += " Use the supplied TurnJournal and ScoreClient adapters. An existing remote result is authoritative, including results produced by an older scorer."
recovery.gateway = RECOVERY_GATEWAY
recovery.examples[1] = (
    {"checkpoint":{"journal":[["1:st","intent","yes"]],"effects":{"1:st":"old-scorer:7"}},
     "attempts":[["s","t","different","finish"]]},
    {"replies":["old-scorer:7"],"journal":[["1:st","intent","yes"],["1:st","done","old-scorer:7"]],
     "effects":{"1:st":"old-scorer:7"}},
    "A previous process stopped after the remote write. The new process reuses its stored reply, records completion, and returns that reply even though the client supplied different text.")

LABS["kv-cache"].inputs += " For interleaved traffic, replace chunks with events of [request_id, operation, token_list]; operation is 'push' or 'close'."
LABS["kv-cache"].outputs += " Event mode returns {replies: [...]} with one list of token logits per push and null per close."
LABS["kv-cache"].constraints += " A push creates a missing session. Close discards all layers for that ID; closing an absent session is harmless. A later push to that ID starts at position zero."
LABS["kv-cache"].gateway = """from stream_worker import serve
from session_pool import SessionPool

def handle(request):
    layers = request["layers"]
    if "events" in request:
        pool = SessionPool(layers)
        replies = []
        for session, event, tokens in request["events"]:
            if event == "close":
                pool.close(session)
                replies.append(None)
            else:
                replies.append(pool.push(session, tokens))
        return {"replies": replies}
    chunks = request["chunks"]
    return {"logits": serve(chunks, layers), "tokens_processed": sum(map(len, chunks))}
"""
LABS["kv-cache"].examples[1] = (
    {"layers":1,"events":[["a","push",[0]],["b","push",[0]],["a","close",[]],["a","push",[0]]]},
    {"replies":[[[1,2,-1]],[[1,2,-1]],None,[[1,2,-1]]]},
    "The first token for b starts at position zero even though a is active. Closing a discards its cache; reusing that ID produces the first-token logits again.")

SESSION_POOL = '''"""Runtime boundary used by the gateway when several sessions share a worker."""
from decoder import decode


class SessionPool:
    def __init__(self, layers):
        self.layers = layers
        self.requests = {}

    def push(self, request_id, tokens):
        """Decode a chunk against this request's live layer states; return its logits."""
        if request_id not in self.requests:
            self.requests[request_id] = [[[], []] for _ in range(self.layers)]
        return [decode(token, self.requests[request_id]) for token in tokens]

    def close(self, request_id):
        """Retire this session. Reusing its ID must start a fresh sequence."""
        self.requests.pop(request_id, None)
'''

