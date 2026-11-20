"""Simulated replica delivery and speculative task completion exercises."""
from textwrap import dedent


def module(text):
    return dedent(text).strip() + "\n"


def part(title, file, funcs, requirement, *hints):
    return dict(title=title, file=file, funcs=funcs.split(), requirement=requirement, hints=list(hints))


def check(part_no, fn, args, title, oracle):
    return dict(part=part_no, fn=fn, args=args, title=title, oracle=oracle)


def example(request, expected, why):
    return dict(input=request, output=expected, why=why)


def integrated(part_no, name, request, expected):
    return check(part_no, "gateway.handle", [request], name, expected)


def cluster(names, *events):
    return dict(replicas=names.split(), events=[list(e) for e in events])


def view(reads, **replicas):
    return dict(reads=reads, replicas=replicas)


def state(adds=None, removed=()):
    return dict(adds=adds or {}, removed=list(removed))


orset_modules = {
    "replica.py": module('''
        """Local operations on a replica's tagged set history."""
        def add(state, value, tag):
            """Return fresh state recording value under the previously unused tag. Keep all existing facts."""
            return {"adds": {**state["adds"], tag: value}, "removed": list(state["removed"])}

        def discard(state, value):
            """Return fresh state tombstoning every locally known tag for value. Return sorted, duplicate-free tombstones."""
            removed = set(state["removed"])
            removed.update(tag for tag, item in state["adds"].items() if item == value)
            return {"adds": dict(state["adds"]), "removed": sorted(removed)}

        def visible(state):
            """Return distinct values with at least one surviving tag, in lexicographic order."""
            removed = set(state["removed"])
            return sorted({item for tag, item in state["adds"].items() if tag not in removed})
    '''),
    "anti_entropy.py": module('''
        """The transport calls this join for every delivered snapshot, including duplicates."""
        def join(local, incoming):
            """Return a fresh history containing the union of tag bindings and the sorted union of tombstones in both states."""
            adds = {**local["adds"], **incoming["adds"]}
            removed = set(local["removed"]) | set(incoming["removed"])
            return {"adds": adds, "removed": sorted(removed)}
    '''),
    "gateway.py": module('''
        """Replay local edits and saved network packets in the supplied order."""
        from copy import deepcopy
        from anti_entropy import join
        from replica import add, discard, visible

        def handle(request):
            nodes = {r: {"adds": {}, "removed": []} for r in request["replicas"]}
            counters = {r: 0 for r in nodes}
            packets, reads = {}, []
            for op, *args in request["events"]:
                if op == "deliver":
                    packet, destination = args
                    nodes[destination] = join(nodes[destination], packets[packet])
                    continue
                origin = args[0]
                if op == "add":
                    counters[origin] += 1
                    tag = f"{origin}#{counters[origin]}"
                    nodes[origin] = add(nodes[origin], args[1], tag)
                elif op == "remove":
                    nodes[origin] = discard(nodes[origin], args[1])
                elif op == "send":
                    packets[args[1]] = deepcopy(nodes[origin])
                elif op == "read":
                    reads.append(visible(nodes[origin]))
            return {"reads": reads, "replicas": {r: visible(s) for r, s in nodes.items()}}
    ''')
}

or_normal = cluster("A B", ("add","A","gpu"),("send","A","offer"),("deliver","offer","B"),
    ("read","B"),("remove","B","gpu"),("send","B","cancel"),("deliver","cancel","A"),("read","A"))
or_concurrent = cluster("A B", ("add","A","gpu"),("send","A","old"),("deliver","old","B"),
    ("remove","B","gpu"),("add","A","gpu"),("send","B","rm"),("deliver","rm","A"),
    ("send","A","new"),("deliver","new","B"),("deliver","old","B"),("read","B"))
or_delayed = cluster("A B", ("add","A","x"),("send","A","old"),("remove","A","x"),
    ("send","A","new"),("deliver","new","B"),("deliver","old","B"),("deliver","old","B"))
or_multitag = cluster("A B C",("add","A","x"),("add","B","x"),("send","A","a"),("send","B","b"),
    ("deliver","a","C"),("deliver","b","C"),("remove","C","x"),("send","C","t"),("deliver","t","A"),
    ("read","B"),("deliver","t","B"),("read","B"))
or_readd = cluster("A B",("remove","A","q"),("send","A","empty"),("add","A","q"),
    ("remove","A","q"),("add","A","q"),("send","A","fresh"),("deliver","fresh","B"),("deliver","empty","B"))
or_isolated = cluster("A B",("add","A","z"),("send","A","snapshot"),("add","A","a"),
    ("deliver","snapshot","B"),("read","B"),("send","B","echo"),("deliver","echo","A"))

or_checks = [
    check(1,"replica.discard",[state({"A#1":"x","B#1":"x","C#1":"y"}),"x"],
          "Remove two observed tags and leave the unrelated key",state({"A#1":"x","B#1":"x","C#1":"y"},["A#1","B#1"])),
    check(1,"replica.add",[state({"B#1":"x"},["B#1"]),"x","B#2"],
          "Fresh tag after a local removal",state({"B#1":"x","B#2":"x"},["B#1"])),
    check(1,"replica.visible",[state({"x#2":"a","y#1":"a","p#1":"z","q#1":"b"},["x#2","q#1"])],
          "One surviving tag is sufficient",["a","z"]),
    check(2,"anti_entropy.join",[state({"A#1":"gpu"},["A#1"]),state({"A#1":"gpu","B#1":"gpu"})],
          "Merge a concurrent add across a tombstone",state({"A#1":"gpu","B#1":"gpu"},["A#1"])),
    check(2,"anti_entropy.join",[state({"A#1":"b","C#1":"z"},["C#1"]),state({"A#1":"b"},["A#1"])],
          "Remote removal arrives after unrelated local removal",state({"A#1":"b","C#1":"z"},["A#1","C#1"])),
    integrated(2,"Delayed stale packet cannot resurrect a removed assignment",or_delayed,view([],A=[],B=[])),
    integrated(1,"Remove after seeing both concurrent adds",or_multitag,view([["x"],[]],A=[],B=[],C=[])),
    integrated(2,"Re-add survives a stale empty snapshot",or_readd,view([],A=["q"],B=["q"])),
    integrated(2,"A packet captures its sender at send time",or_isolated,view([["z"]],A=["a","z"],B=["z"])),
]

orset = dict(
    slug="sys-observed-remove-set",title="Reconcile an offline set with an observed-remove CRDT",group="Distributed systems",
    level="Intermediate",duration="60–85 min",
    mission="Two offline consoles edit the same set of machine labels. A console removes a label while a peer adds it again. After reconnecting, the network delivers a current update, then an older copy twice. Implement local edits and replica reconciliation so an old packet cannot resurrect a deleted tag and a concurrent add can survive.",
    scope="Build the tombstone-based, state-based observed-remove set. The provided transport freezes packets at send time and lets the caller replay any delivery order. A replica's view can diverge until it receives relevant state. Tombstone collection, disk recovery, and membership changes are outside this exercise.",
    inputs='A JSON object has replicas (unique replica IDs) and events in execution order. Events are ["add", replica, value], ["remove", replica, value], ["read", replica], ["send", source, packetId], or ["deliver", packetId, destination]. Send saves a full, immutable snapshot under a fresh packet ID. Deliver can replay that snapshot any number of times to any replica.',
    outputs='Return {"reads":[...],"replicas":{...}}. Each read appends only that replica’s visible list at that moment. replicas maps every replica ID to its final visible list. Sort lists lexicographically, suppress duplicate values, and include empty replicas.',
    constraints="All replicas start empty. Replica IDs are nonempty ASCII letters. Values and packet IDs are nonempty strings. All referenced replicas and packets exist. There are up to 12 replicas and 1,000 events. Each local add gets a globally unique replica#counter tag, including repeated adds of the same value. State is {adds:{tag:value}, removed:[tag,...]}. Keep all bindings and all removal tags. Removing an absent value is a no-op.",
    source=[{"title":"Shapiro et al. · Observed-Remove Set (§3.3.5)","url":"https://www.lip6.fr/Marc.Shapiro/papers/2011/Comprehensive-CRDTs-RR7506-2011-01.pdf"},
            {"title":"Bieniusa et al. · An Optimized Conflict-Free Replicated Set","url":"https://arxiv.org/abs/1210.3368"}],
    modules=orset_modules,
    parts=[
        part("Record local add and removal facts","replica.py","add discard visible",
             "Keep tag-level history and derive the visible labels. All updates return fresh state so the network can hold an older version.",
             "Two adds of one label have distinct tags. Removing it covers every tag the caller has seen for it.",
             "A later add of that label carries a new tag."),
        part("Merge out-of-order snapshots","anti_entropy.py","join",
             "Reconcile the two histories. A peer may see the same packet twice or receive a pre-removal snapshot after a post-removal snapshot.",
             "The additions and removal records both grow monotonically.",
             "Try merging states in both orders, and merging a state with itself.")],
    examples=[example(or_normal,view([["gpu"],[]],A=[],B=[]),
        "B first learns tag A#1 for gpu and reads it. Its removal records A#1. A learns about that removal in cancel."),
      example(or_concurrent,view([["gpu"]],A=["gpu"],B=["gpu"]),
        "B removed A#1. A created A#2 before learning of the removal. The new snapshot carries both facts, and the old copy cannot remove A#2 or restore A#1.")],
    checks=or_checks)


def ledger(active=(), retired=(), winner=None):
    return {"active": list(active), "retired": list(retired), "winner": winner}


def job(splits, reducers, *events):
    return {"splits": splits, "reducers": reducers, "events": [list(e) for e in events]}


def report(winners, finishes, reads, partitions):
    return dict(winners=winners, finishes=finishes, reads=reads, partitions=partitions)


mr_modules = {
    "attempts.py": module('''
        """Pure task-level ledger. Attempt IDs are scoped to a logical task."""
        def transition(row, event, attempt):
            """Return [fresh_row, status_or_None]. Row fields are sorted active, sorted retired and winner (a string or None). Follow the attempt lifecycle in the brief."""
            active, retired = set(row["active"]), set(row["retired"])
            winner, status = row["winner"], None
            if event == "finish":
                if winner is not None:
                    status = "replay" if winner == attempt else "superseded"
                elif attempt not in active:
                    status = "stale"
                else:
                    winner, status = attempt, "accepted"
                    active.clear()
            elif winner is None:
                if event == "start" and attempt not in retired:
                    active.add(attempt)
                elif event == "expire" and attempt in active:
                    active.remove(attempt)
                    retired.add(attempt)
            return [{"active": sorted(active), "retired": sorted(retired), "winner": winner}, status]
    '''),
    "shuffle.py": module('''
        """Deterministic intermediate partitioning and reduction."""
        def scatter(shard, count):
            """Map tokens to [token,1] records. Return count bucket lists. Route by the sum of UTF-8 bytes modulo count, preserving shard order within each bucket."""
            buckets = [[] for _ in range(count)]
            for token in shard:
                partition = sum(token.encode("utf-8")) % count
                buckets[partition].append([token, 1])
            return buckets

        def reduce_bucket(map_outputs, index):
            """Aggregate every winning map's bucket at index and return [token,total] rows in lexicographic token order."""
            counts = {}
            for buckets in map_outputs:
                for token, value in buckets[index]:
                    counts[token] = counts.get(token, 0) + value
            return [[token, counts[token]] for token in sorted(counts)]
    '''),
    "coordinator.py": module('''
        """Schedule replay and output manifests shared by all task attempts."""
        from copy import deepcopy
        from attempts import transition
        from shuffle import scatter, reduce_bucket

        def replay(splits, reducers, events):
            """Execute the trace. Enforce the map barrier at reduce start, publish each accepted output once, and reveal partitions only when all reducers have committed."""
            maps = [f"m{i}" for i in range(len(splits))]
            reductions = [f"r{i}" for i in range(reducers)]
            rows = {t: {"active": [], "retired": [], "winner": None} for t in maps + reductions}
            outputs, finishes, reads = {}, [], []

            def published():
                if all(t in outputs for t in reductions):
                    return [outputs[t] for t in reductions]
                return None

            for event in events:
                op = event[0]
                if op == "read":
                    reads.append(deepcopy(published()))
                    continue
                _, task, attempt = event
                is_map = task[0] == "m"
                if op == "start" and not is_map and any(t not in outputs for t in maps):
                    continue
                rows[task], status = transition(rows[task], op, attempt)
                if op != "finish":
                    continue
                finishes.append(status)
                if status == "accepted":
                    index = int(task[1:])
                    if is_map:
                        outputs[task] = scatter(splits[index], reducers)
                    else:
                        outputs[task] = reduce_bucket([outputs[t] for t in maps], index)
            return {"winners": {t: r["winner"] for t, r in rows.items()},
                    "finishes": finishes, "reads": reads, "partitions": published()}
    '''),
    "gateway.py": module('''
        """The bench and command-line runner share this JSON entry point."""
        from coordinator import replay

        def handle(request):
            return replay(request["splits"], request["reducers"], request["events"])
    ''')
}

mr_normal = job([["kiwi","fig","fig"],["fig"]],1,
    ("start","m0","P"),("finish","m0","P"),("start","m1","Q"),("finish","m1","Q"),
    ("start","r0","S"),("finish","r0","S"),("read",))
mr_counts = [[["fig",3],["kiwi",1]]]
mr_good = report({"m0":"P","m1":"Q","r0":"S"}, ["accepted"]*3, [mr_counts], mr_counts)
mr_race = job([["a","a"]],1,("start","r0","early"),("start","m0","slow"),("expire","m0","slow"),
    ("start","m0","new"),("start","m0","backup"),("finish","m0","slow"),("finish","m0","backup"),
    ("finish","m0","new"),("finish","m0","backup"),("finish","r0","early"),("read",),
    ("start","r0","now"),("finish","r0","now"))
mr_raced = report({"m0":"backup","r0":"now"},["stale","accepted","superseded","replay","stale","accepted"],
                   [None],[[["a",2]]])
mr_stuck = job([["x"],[]],2,("start","m0","p"),("finish","m0","p"),("start","m1","q"),
    ("expire","m1","q"),("start","m1","q"),("finish","m1","q"),("start","r0","a"),("finish","r0","a"))
mr_stuck_result = report({"m0":"p","m1":None,"r0":None,"r1":None},["accepted","stale","stale"],[],None)
mr_reduces = job([["a","b","b"]],2,("start","m0","1"),("finish","m0","1"),("start","r0","q"),
    ("start","r0","q2"),("start","r1","q"),("finish","r0","q2"),("read",),("finish","r0","q"),
    ("finish","r1","q"),("finish","r0","q2"))
mr_reduces_result = report({"m0":"1","r0":"q2","r1":"q"},
    ["accepted","accepted","superseded","accepted","replay"],[None],[[["b",2]],[["a",1]]])
mr_zero = job([],2,("start","r1","f"),("finish","r1","f"),("read",),("start","r0","s"),("finish","r0","s"),("read",))
mr_zero_result = report({"r0":"s","r1":"f"},["accepted","accepted"],[None,[[],[]]],[[],[]])
mr_retry = job([["é","ñ","é"]],2,("start","m0","x"),("finish","m0","x"),("expire","m0","x"),
    ("start","r0","z"),("expire","r0","z"),("start","r0","y"),("finish","r0","z"),("finish","r0","y"),
    ("start","r1","z"),("finish","r1","z"))
mr_retry_result = report({"m0":"x","r0":"y","r1":"z"},["accepted","stale","accepted","accepted"],[],
                          [[["é",2],["ñ",1]],[]])

mr_checks = [
    check(1,"attempts.transition",[ledger(["a","b"]),"finish","b"],"Speculative winner closes other attempts",
        [ledger(winner="b"),"accepted"]),
    check(1,"attempts.transition",[ledger(["w"],["old"]),"start","old"],"Expired IDs cannot restart",
        [ledger(["w"],["old"]),None]),
    check(1,"attempts.transition",[ledger(["y","z"]),"expire","y"],"Expiring one competing attempt preserves the other",
        [ledger(["z"],["y"]),None]),
    check(2,"shuffle.scatter",[["a","b","aa","é"],2],"Route UTF-8 bytes and preserve input order",[[["b",1],["aa",1],["é",1]],[["a",1]]]),
    check(2,"shuffle.reduce_bucket",[[[[["y",1],["a",1]],[]],[[["y",1]],[]]],0],
          "Group records across accepted map files and sort keys",[["a",1],["y",2]]),
    check(2,"shuffle.scatter",[[],3],"Produce partition files even for an empty shard",[[],[],[]]),
    check(3,"coordinator.replay",[[["x"]],1,[["start","r0","first"],["finish","r0","first"]]],
          "A reduce start before the barrier creates no live attempt",report({"m0":None,"r0":None},["stale"],[],None)),
    check(3,"coordinator.replay",[[],1,[["start","r0","r"],["finish","r0","r"]]],
          "Job with zero splits can immediately launch reducers", report({"r0":"r"},["accepted"],[],[[]])),
    integrated(3,"An expired map keeps the publication barrier closed",mr_stuck,mr_stuck_result),
    integrated(1,"A speculative reduce commits once across replay and late completion",mr_reduces,mr_reduces_result),
    integrated(3,"Empty job publishes only once all reducers commit",mr_zero,mr_zero_result),
    integrated(2,"Retry a reducer with non-ASCII grouped tokens",mr_retry,mr_retry_result)
]

mapreduce = dict(
    slug="sys-mapreduce-attempts",title="Commit MapReduce output under retries and speculation",group="Distributed systems",
    level="Advanced",duration="100–150 min",
    mission="A slow worker keeps running after the scheduler starts a backup. Both send a completion; another worker reports success after its attempt expired. Build the coordinator for a word-count service so each logical map contributes exactly one output and readers only see a complete job. The trace lets you deliberately pick which worker finishes first.",
    scope="Run a deterministic coordinator simulation for map and reduce attempts. Each worker computes the given word-count function. Logical output files live in an in-memory manifest and the first valid finisher wins. Assume committed files are durable and the coordinator stays up. The original system's worker-local map-file recovery, filesystem atomic renames and speculative-launch heuristics are additional mechanisms.",
    inputs='splits is an array of token arrays (one map task per split); reducers is a positive integer. events contains ["start", task, attempt], ["expire", task, attempt], ["finish", task, attempt], and ["read"]. Task IDs are m0,m1,… for splits and r0,r1,… for reducer partitions. IDs are zero-indexed. Events are processed in list order.',
    outputs='Return winners (task → accepted attempt ID or null); finishes (one status per finish, in order); reads (one publication snapshot per read); and partitions (final snapshot). A snapshot is null until all reducers commit, then an array in reducer-index order. Each partition is an array of [token, count] pairs sorted by token. A finished empty partition is [].',
    constraints="A start launches an attempt unless the task has won, the ID was previously retired on this task, or it is a reduce task with any uncommitted map. An early launch is ignored and can be issued again later. Expire retires an active attempt only. Start and expire on a committed task are no-ops. Before commitment, finish accepts the first active finisher (status accepted), clears active IDs and commits its output; all inactive finishes are stale. After commitment, finish from the winner is replay and any other is superseded. Attempts are scoped per task. There are 0–64 splits, 1–16 reducers and up to 2,000 events. Task IDs are valid. Tokens and attempt IDs are nonempty strings. Route a token by sum of its UTF-8 bytes modulo reducer count.",
    source=[{"title":"Dean & Ghemawat · MapReduce (§3.3 task re-execution, §3.6 backup tasks, §4.1 partitioning)","url":"https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/"}],
    modules=mr_modules,
    parts=[
        part("Arbitrate competing attempts","attempts.py","transition",
             "Maintain a ledger for one task. Keep active attempts, retired attempts and the winner. Preserve retired IDs on commitment. Return None for start/expire status and the specified status for every finish.",
             "An expired attempt can report completion before a replacement finishes.",
             "A duplicate finish from the winner has different diagnostic meaning from a losing worker’s late finish."),
        part("Shuffle and aggregate the winning files","shuffle.py","scatter reduce_bucket",
             "Each map emits one [token,1] record per occurrence into its reducer bucket. Each reducer receives the same bucket index from every winning map and emits key totals in deterministic order.",
             "Python's built-in hash can vary across runs. Use the specified byte checksum.",
             "Empty maps still produce one bucket per reducer."),
        part("Wire the barrier and output manifest","coordinator.py","replay",
             "Track the ledger and finalized output per task. Honor the map barrier when a reducer asks to start. Execute a worker's map or reduce only when transition returns accepted. Serve read and final snapshots from fully committed reducer outputs.",
             "A job with no map tasks starts with its map barrier satisfied.",
             "An output can be an empty bucket. File presence, rather than contents, establishes completion.")],
    examples=[example(mr_normal,mr_good,"Three fig records from two winning maps land in reducer 0. Once S commits, the read sees fig=3 and kiwi=1."),
              example(mr_race,mr_raced,"The early reduce launch is ignored. The expired map is stale, backup wins, new is superseded, and repeating backup is a replay. The reduce must be launched again after the maps finish. The two occurrences of a are counted once each.")],
    checks=mr_checks)

LABS = [orset, mapreduce]
