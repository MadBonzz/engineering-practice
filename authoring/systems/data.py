"""Reproducible systems exercises for placement, persistence, and isolation."""
from textwrap import dedent


def part(title, file, funcs, requirement, *hints):
    return dict(title=title, file=file, funcs=funcs.split(), requirement=requirement, hints=list(hints))


def check(n, fn, args, title, oracle):
    return dict(part=n, fn=fn, args=args, title=title, oracle=oracle)


def sample(request, answer, why):
    return dict(input=request, output=answer, why=why)


def files(**modules):
    return {name + ".py": dedent(code).strip() + "\n" for name, code in modules.items()}


def ring(before, after, rf, keys):
    return dict(before=before, after=after, rf=rf, keys=keys)


def placement(routes, moves):
    return dict(routes=routes, moves=moves)


ring_join = ring({"A": [20], "B": [100]}, {"A": [20], "B": [100], "C": [60]}, 2, [20, 21, 80, 250])
ring_join_result = placement(
    [["A", "C"], ["C", "B"], ["B", "A"], ["A", "C"]],
    [[100, 20, "A", "A", ["C"], ["B"]], [20, 60, "B", "C", ["C"], ["A"]]]
)
ring_leave = ring({"A": [10, 30], "B": [90]}, {"A": [10, 30]}, 3, [31, 90])
ring_leave_result = placement(
    [["A"], ["A"]],
    [[90, 10, "A", "A", [], ["B"]], [10, 30, "A", "A", [], ["B"]],
     [30, 90, "B", "A", [], ["B"]]]
)

HASH = dict(
    slug="sys-consistent-hashing", title="Consistent hashing · plan a live replica handoff",
    group="Distributed systems", level="Intermediate", duration="70–100 min",
    mission="A storage cluster is adding and removing virtual nodes while keys remain live. Build its placement service and a handoff planner. An operator must know exactly which ranges need copying, which replicas can retire, and which machine will answer each read.",
    scope="Plan a transition between two token assignments. The supplied gateway builds both rings and asks your range planner for a complete diff. Hashes and token positions are injected so a failing placement is easy to reproduce by hand.",
    inputs="before and after map each physical node name to its virtual-node positions. rf is the desired number of distinct physical replicas. keys is a list of already-hashed read keys. All positions and keys are integers from 0 through 255.",
    outputs="routes contains one list of post-change owners per input key, in input order. moves contains [left, right, old_primary, new_primary, copy_to, retire_from] for each affected elementary range (left, right]. An empty ring has null primary. Order copy_to by the new preference list and retire_from by the old one.",
    constraints="Sort the union of old and new token positions. Each position ends one elementary range whose left boundary is the previous position in this circular order. A single boundary covers the entire ring. Emit records in ascending right-boundary order if the primary or membership changes. Ignore backup-only reordering and keep adjacent records separate. Equality belongs to the token at that position. Positions are unique within each snapshot. rf is 1–16. A node may own many tokens. Return as many distinct nodes as exist, up to rf.",
    source=[dict(title="DeCandia et al. · Dynamo, sections 4.2 and 4.3",url="https://www.amazon.science/publications/dynamo-amazons-highly-available-key-value-store")],
    modules=files(
        token_ring='''
            """Ring index shared by online routing and the offline transfer planner."""
            from bisect import bisect_left

            def build(members):
                """Return increasing [position, physical_node] rows for the member-to-token mapping."""
                return sorted([position, node] for node, positions in members.items() for position in positions)

            def owners(tokens, key, rf):
                """Return up to rf distinct physical nodes clockwise, starting at the first position >= key and wrapping."""
                if not tokens:
                    return []
                start = bisect_left(tokens, [key])
                selected = []
                seen = set()
                for offset in range(len(tokens)):
                    node = tokens[(start + offset) % len(tokens)][1]
                    if node not in seen:
                        selected.append(node)
                        seen.add(node)
                        if len(selected) == rf:
                            break
                return selected
        ''',
        handoff='''
            """Diff replicas interval by interval across an old and new ring."""
            from token_ring import owners

            def changes(old, new, rf):
                """Diff sorted token rows. Emit [left,right,old_primary,new_primary,copy_to,retire_from] for affected ranges."""
                endpoints = sorted({row[0] for row in old + new})
                moves = []
                for i, right in enumerate(endpoints):
                    previous = owners(old, right, rf)
                    current = owners(new, right, rf)
                    old_primary = previous[0] if previous else None
                    new_primary = current[0] if current else None
                    copy_to = [node for node in current if node not in previous]
                    retire_from = [node for node in previous if node not in current]
                    if old_primary != new_primary or copy_to or retire_from:
                        moves.append([endpoints[i-1], right, old_primary, new_primary, copy_to, retire_from])
                return moves
        ''',
        gateway='''
            """Resolve client keys and prepare the handoff manifest for the same membership change."""
            from token_ring import build, owners
            from handoff import changes

            def handle(request):
                old, new = build(request["before"]), build(request["after"])
                rf = request["rf"]
                return {"routes": [owners(new, key, rf) for key in request["keys"]],
                        "moves": changes(old, new, rf)}
        '''
    ),
    parts=[
        part("Index virtual nodes and route reads", "token_ring.py", "build owners",
             "Build a sorted token index, then walk it clockwise without counting the same physical machine twice.",
             "A read with hash 255 may begin at the smallest token. Stop after one complete revolution even when rf is much larger than the cluster."),
        part("Plan exact replica transfers", "handoff.py", "changes",
             "Partition the whole hash space using both snapshots. Reuse the router once per ring per right endpoint.",
             "A node joining between two tokens can change replicas outside the range for which it becomes primary.")
    ],
    examples=[
        sample(ring_join, ring_join_result, "C takes primary responsibility for (20,60]. It also replaces B as A’s backup on the wrapped range (100,20]. Reads at 20 still go to A. No data moves on (60,100]."),
        sample(ring_leave, ring_leave_result, "A owns two virtual nodes but stores one copy of each key. Removing B leaves A as the sole owner. Only the last range changes primary, and A already has that range.")
    ],
    checks=[
        check(1,"token_ring.owners",[[[8,"A"],[9,"A"],[12,"B"]],9,4],"Equality and distinct physical replicas",["A","B"]),
        check(1,"token_ring.build",[{"C":[240,12],"A":[31,0]}],"Tokens from multiple machines interleave",[[0,"A"],[12,"C"],[31,"A"],[240,"C"]]),
        check(1,"token_ring.owners",[[[0,"x"],[4,"y"],[60,"x"]],255,2],"Wrap around at the top token",["x","y"]),
        check(2,"handoff.changes",[[[10,"A"]],[[10,"A"],[40,"B"]],1],"Cut only the newly acquired primary interval",[[10,40,"A","B",["B"],["A"]]]),
        check(2,"handoff.changes",[[],[[25,"A"]],3],"Bootstrap the entire ring",[[25,25,None,"A",["A"],[]]]),
        check(2,"gateway.handle",[ring({"a":[0],"b":[5]},{"a":[0],"b":[5]},2,[0,1])],"Unchanged ring needs zero transfers",placement([["a","b"],["b","a"]],[])),
        check(2,"gateway.handle",[ring({"a":[42]},{},2,[0,255])],"Drain the final member",placement([[],[]],[[42,42,"a",None,[],["a"]]])),
        check(1,"gateway.handle",[ring({}, {}, 2,[7])],"Both snapshots are empty",placement([[]],[])),
        check(2,"gateway.handle",[ring({"A":[10],"B":[20]},{"B":[10],"A":[20]},2,[0,11])],"Change primary with both replicas already present",placement([["B","A"],["A","B"]],[[20,10,"A","B",[],[]],[10,20,"B","A",[],[]]])),
        check(2,"gateway.handle",[ring({"A":[4,8],"B":[12]},{"A":[4,8],"B":[12],"C":[9]},2,[5,8,10])],"Replica-only changes across adjacent vnodes",placement([["A","C"],["A","C"],["B","A"]],[[12,4,"A","A",["C"],["B"]],[4,8,"A","A",["C"],["B"]],[8,9,"B","C",["C"],["A"]]]))
    ]
)


def trace(*operations):
    return {"ops": list(operations)}


def storage_result(reads, runs, mem):
    return dict(reads=reads, runs=runs, mem=mem)


lsm_replay = trace(["put","b","old"],["flush"],["put","a","logged"],["sync"],["del","b"],
                   ["get","b"],["crash"],["get","a"],["get","b"])
lsm_partial = trace(["put","a","old"],["flush"],["put","b","keep"],["flush"],["del","a"],
                    ["flush"],["compact",1,2],["get","a"])
LSM = dict(
    slug="sys-lsm-engine", title="LSM storage engine · recover writes and compact tombstones",
    group="Database internals", level="Advanced", duration="100–150 min",
    mission="A storage service occasionally resurrects deleted keys after its compactor runs. Implement the read path, log recovery, and range of runs selected for compaction. Replay writes across a crash and inspect the exact disk layout that results.",
    scope="The harness provides atomic flush, sync, and manifest installation at explicit operation boundaries. You implement the mechanics of immutable sorted runs and a write-ahead log. Files, fsync, and power loss are modeled as deterministic data structures.",
    inputs='ops is an ordered array of commands. ["put",key,value] and ["del",key] append to the volatile log and update the memtable. ["get",key] reads. ["sync"] makes the current log prefix durable. ["crash"] discards unsynced records and reconstructs the memtable. ["flush"] installs all buffered writes in one durable sorted run and clears the log. ["compact",start,count] merges count adjacent disk runs starting at zero-based start, oldest run first.',
    outputs="Return reads (values or null from get, in order), runs (oldest to newest, each a sorted list of [key,value] pairs), and mem (the current memtable as sorted pairs). A null value in a run or memtable is a deletion marker. Empty memtable flushes add no run. Remove an empty compacted result from the manifest.",
    constraints="Keys and values are strings, including empty values. Ordering by key is lexicographic. On reads prefer the memtable, then successively older runs, stopping at the first record including a tombstone. Inside a compacted interval the newest record wins. Keep a winning tombstone only if the unselected older runs currently expose a live value for that key. Leave newer runs untouched. Flush retains all memtable tombstones. Compaction requests always use 1 ≤ count and valid bounds. No snapshots are held.",
    source=[dict(title="Google LevelDB · log files, sorted tables, recovery and compaction",url="https://github.com/google/leveldb/blob/main/doc/impl.md")],
    modules=files(
        runs='''
            """Represent immutable runs as lists of [key,value] sorted by key."""
            from bisect import bisect_left

            def freeze(mem):
                """Return an immutable-run representation (sorted [key,value] rows) of the memtable, keeping null tombstones."""
                return [[key, mem[key]] for key in sorted(mem)]

            def lookup(mem, tables, key):
                """Return the newest visible value or None, checking mem first, then immutable tables newest first."""
                if key in mem:
                    return mem[key]
                for table in reversed(tables):
                    pos = bisect_left(table, [key])
                    if pos < len(table) and table[pos][0] == key:
                        return table[pos][1]
                return None
        ''',
        recovery='''
            """Reconstruct in-memory state after losing the volatile WAL suffix."""
            def replay(log, durable):
                """Replay the first durable records of the [key,value] log into a dict retaining the last value or tombstone per key."""
                mem = {}
                for key, value in log[:durable]:
                    mem[key] = value
                return mem
        ''',
        compaction='''
            """Install a compacted run into one contiguous manifest interval."""
            from runs import freeze, lookup

            def merge(tables, start, count):
                """Replace tables[start:start+count] with one sorted run (or no run if empty), preserving visible values."""
                older = tables[:start]
                merged = {}
                for table in tables[start:start+count]:
                    merged.update(table)
                kept = {}
                for key, value in merged.items():
                    if value is not None or lookup({}, older, key) is not None:
                        kept[key] = value
                replacement = [freeze(kept)] if kept else []
                return older + replacement + tables[start+count:]
        ''',
        gateway='''
            """Replay write acknowledgments, durability boundaries, reads and manifest changes."""
            from runs import freeze, lookup
            from recovery import replay
            from compaction import merge

            def handle(request):
                tables, log, mem, reads = [], [], {}, []
                durable = 0
                for command, *args in request["ops"]:
                    if command in ("put", "del"):
                        key = args[0]
                        value = args[1] if command == "put" else None
                        log.append([key, value])
                        mem[key] = value
                    elif command == "get":
                        reads.append(lookup(mem, tables, args[0]))
                    elif command == "sync":
                        durable = len(log)
                    elif command == "crash":
                        mem = replay(log, durable)
                        log = log[:durable]
                    elif command == "flush":
                        if mem:
                            tables.append(freeze(mem))
                        log, mem, durable = [], {}, 0
                    elif command == "compact":
                        tables = merge(tables, *args)
                return {"reads": reads, "runs": tables, "mem": freeze(mem)}
        '''
    ),
    parts=[
        part("Read and freeze sorted runs","runs.py","freeze lookup","Expose the newest visible value across memory and the disk manifest.","Checking membership separately from the record value distinguishes a tombstone in memory from an absent key."),
        part("Recover a synchronized WAL prefix","recovery.py","replay","Rebuild a memtable after the emulated power cut. Preserve repeated writes and deletes in their original order.","durable is a record count. A value overwritten in the lost suffix can reappear after restart."),
        part("Compact without reviving deleted keys","compaction.py","merge","Merge the selected manifest interval and install the result, preserving shadowing of older values.","Inspect the state you would expose below the selected interval if the winning tombstone disappeared.")
    ],
    examples=[
        sample(lsm_replay,storage_result([None,"logged","old"],[[["b","old"]]],[["a","logged"]]),
               "The buffered delete hides b immediately. A crash loses that delete while preserving the earlier sync of a. The disk run still contains b’s original value."),
        sample(lsm_partial,storage_result([None],[[["a","old"]],[["a",None],["b","keep"]]],[]),
               "Only runs 1 and 2 merge. Removing a’s tombstone would make the old value in run 0 visible. The new run therefore retains both a’s marker and b.")
    ],
    checks=[
        check(1,"runs.lookup",[{},[[["k","first"]],[["k",None]]],"k"],"A disk tombstone shadows an old value",None),
        check(1,"runs.freeze",[{"z":None,"a":""}],"Flush emits sorted live and deleted keys",[["a",""],["z",None]]),
        check(2,"recovery.replay",[[["a","1"],["a",None],["a","2"]],2],"Stop WAL replay at the sync boundary",{"a":None}),
        check(2,"recovery.replay",[[["a","1"]],0],"Drop a wholly unsynced log",{}),
        check(3,"compaction.merge",[[[["k",""]],[["k",None]]],1,1],"The empty string is a live older value",[[["k",""]],[["k",None]]]),
        check(3,"compaction.merge",[[[["k","old"]],[["k",None]],[["k","new"]]],0,2],"Remove an obsolete bottom tombstone below a newer value",[[["k","new"]]]),
        check(3,"gateway.handle",[trace(["put","x","v"],["flush"],["del","x"],["flush"],["compact",0,2],["get","x"])],"Full compaction drops last deleted key",storage_result([None],[],[])),
        check(2,"gateway.handle",[trace(["put","x","v"],["sync"],["del","x"],["crash"],["crash"],["get","x"],["flush"],["crash"])],"Repeated recovery and flush do not duplicate records",storage_result(["v"],[[["x","v"]]],[])),
        check(3,"gateway.handle",[trace(["del","z"],["flush"],["put","b","v"],["flush"],["del","z"],["flush"],["compact",1,2])],"An older tombstone already hides the absent key",storage_result([],[[["z",None]],[["b","v"]]],[])),
        check(1,"gateway.handle",[trace(["flush"],["put","z","0"],["put","a",""],["flush"],["del","a"],["get","a"],["get","z"],["get","other"])],"Reads combine memory and sorted durable data",storage_result([None,"0",None],[[["a",""],["z","0"]]],[["a",None]]))
    ]
)


def tx(initial, *commands):
    return dict(initial=initial, ops=list(commands))


def tx_result(reads, outcomes, live):
    return dict(reads=reads, outcomes=outcomes, live=live)


overdraft = tx({"balance":100},["begin","a"],["begin","b"],["put","a","balance",90],
               ["get","b","balance"],["commit","a"],["get","b","balance"],["put","b","balance",80],["commit","b"])
on_call = tx({"alice":1,"bob":1},["begin","t1"],["begin","t2"],["get","t1","bob"],["get","t2","alice"],
             ["put","t1","alice",0],["put","t2","bob",0],["commit","t1"],["commit","t2"])
MVCC = dict(
    slug="sys-mvcc-snapshots", title="MVCC · implement snapshot isolation and reproduce write skew",
    group="Database internals", level="Advanced", duration="90–130 min",
    mission="Two on-call engineers each see the other person on duty and take themselves off call. Your transaction engine should reproduce that outcome under snapshot isolation, while rejecting two concurrent writers to the same key. Implement version selection and atomic commit, then use the replay trace to explain the difference.",
    scope="The engine simulates interleaved point reads, writes, and deletes with one logical commit clock. Transactions capture their snapshot at begin. The harness retains history for all snapshots. This version uses optimistic first-committer-wins validation.",
    inputs='initial maps string keys to integer values at version 0. ops contains ["begin",id], ["get",id,key], ["put",id,key,value], ["del",id,key], ["commit",id], or ["abort",id]. Each id begins once. All subsequent operations reference an open transaction. A transaction can issue repeated reads and writes.',
    outputs='Return reads as [id,key,value_or_null] in get order, outcomes as [id,"commit",timestamp] or [id,"abort",null] in completion order, and live as a map of the latest non-deleted keys. Only a successful transaction with writes increments the clock. A successful read-only transaction reports the current clock.',
    constraints="Read the transaction’s private writes first, including deletes. Otherwise pick the committed version with the largest timestamp ≤ the transaction’s snapshot. On commit, check every written key for a committed version newer than the snapshot, including tombstones. One conflict aborts the entire transaction. Install all uncontended writes at a single fresh timestamp. A delete of a missing key still writes a version. Reads of keys written by others do not themselves cause an abort. Explicit aborts publish nothing.",
    source=[dict(title="PostgreSQL · transaction isolation and serialization anomalies",url="https://www.postgresql.org/docs/current/transaction-iso.html")],
    modules=files(
        version_store='''
            """Look up committed and transaction-local values."""
            def read(chains, staged, snapshot, key):
                """Return the value in staged or at the latest committed version <= snapshot. Missing and deleted keys read as None."""
                if key in staged:
                    return staged[key]
                for timestamp, value in reversed(chains.get(key, [])):
                    if timestamp <= snapshot:
                        return value
                return None
        ''',
        commit_log='''
            """Validate and publish a write set as a single atomic commit."""
            def publish(chains, staged, snapshot, clock):
                """Return [status,new_clock,new_chains]. Each chain holds ascending [timestamp,value] versions; never partially publish."""
                for key in staged:
                    history = chains.get(key, [])
                    if history and history[-1][0] > snapshot:
                        return ["abort", clock, chains]
                if not staged:
                    return ["commit", clock, chains]
                clock += 1
                result = chains.copy()
                for key, value in staged.items():
                    result[key] = chains.get(key, []) + [[clock, value]]
                return ["commit", clock, result]
        ''',
        gateway='''
            """Own transaction lifetimes and expose operations in trace order."""
            from version_store import read
            from commit_log import publish

            def handle(request):
                chains = {key: [[0, value]] for key, value in request["initial"].items()}
                clock = 0
                active, reads, outcomes = {}, [], []
                for action, tx_id, *args in request["ops"]:
                    if action == "begin":
                        active[tx_id] = [clock, {}]
                        continue
                    snapshot, staged = active[tx_id]
                    if action == "get":
                        key = args[0]
                        reads.append([tx_id, key, read(chains, staged, snapshot, key)])
                    elif action in ("put", "del"):
                        staged[args[0]] = args[1] if action == "put" else None
                    elif action in ("commit", "abort"):
                        status = "abort"
                        if action == "commit":
                            status, clock, chains = publish(chains, staged, snapshot, clock)
                        outcomes.append([tx_id, status, clock if status == "commit" else None])
                        del active[tx_id]
                live = {key: history[-1][1] for key, history in chains.items() if history[-1][1] is not None}
                return {"reads": reads, "outcomes": outcomes, "live": live}
        '''
    ),
    parts=[
        part("Serve reads from the correct snapshot","version_store.py","read","Resolve the transaction’s private overlay and version chain, including historical reads after a delete.","Timestamps may be equal to the snapshot. Check the local overlay for the presence of a key even when its value is null."),
        part("Validate and publish a write set atomically","commit_log.py","publish","Apply first-committer-wins to all written keys before installing any new version.","A committed delete is evidence of a conflicting write. A successful transaction that writes multiple keys advances the clock once.")
    ],
    examples=[
        sample(overdraft,tx_result([["b","balance",100],["b","balance",100]],[["a","commit",1],["b","abort",None]],{"balance":90}),
               "Both transactions begin at 0. B reads the original balance before and after A commits. When B tries to publish, A’s version 1 is newer than B’s snapshot, so B aborts."),
        sample(on_call,tx_result([["t1","bob",1],["t2","alice",1]],[["t1","commit",1],["t2","commit",2]],{"alice":0,"bob":0}),
               "Each transaction reads its colleague’s original value and writes only its own row. The write sets are disjoint, so both commits succeed and the on-call invariant fails. This is write skew.")
    ],
    checks=[
        check(1,"version_store.read",[{"x":[[0,3],[3,None],[5,8]]},{},4,"x"],"A snapshot between deletion and reinsertion",None),
        check(1,"version_store.read",[{"x":[[0,3],[3,None]]},{"x":0},1,"x"],"Private zero shadows the historical version",0),
        check(1,"version_store.read",[{"x":[[2,3]]},{},1,"x"],"Future insertion is invisible",None),
        check(2,"commit_log.publish",[{"x":[[0,1],[2,None]]},{"new":8,"x":4},0,2],"Abort every write when a touched key has a newer tombstone",["abort",2,{"x":[[0,1],[2,None]]}]),
        check(2,"commit_log.publish",[{"x":[[1,2]]},{"x":3,"y":None},1,1],"Publish multiple keys with a shared timestamp",["commit",2,{"x":[[1,2],[2,3]],"y":[[2,None]]}]),
        check(1,"gateway.handle",[tx({"k":5},["begin","a"],["del","a","k"],["get","a","k"],["begin","b"],["get","b","k"],["abort","a"],["commit","b"])],"Private deletion and explicit rollback",tx_result([["a","k",None],["b","k",5]],[["a","abort",None],["b","commit",0]],{"k":5})),
        check(2,"gateway.handle",[tx({},["begin","a"],["begin","b"],["del","a","x"],["commit","a"],["put","b","y",8],["put","b","x",2],["commit","b"])],"Deleting a missing key conflicts with a concurrent insertion",tx_result([],[["a","commit",1],["b","abort",None]],{})),
        check(1,"gateway.handle",[tx({},["begin","r"],["begin","w"],["put","w","new",6],["commit","w"],["get","r","new"],["commit","r"],["begin","next"],["get","next","new"],["commit","next"])],"An old reader can commit after a concurrent insertion",tx_result([["r","new",None],["next","new",6]],[["w","commit",1],["r","commit",1],["next","commit",1]],{"new":6})),
        check(2,"gateway.handle",[tx({"l":1,"r":1},["begin","x"],["begin","y"],["get","x","r"],["get","y","l"],["del","y","r"],["commit","y"],["del","x","l"],["commit","x"])],"Disjoint deletes both commit despite cross reads",tx_result([["x","r",1],["y","l",1]],[["y","commit",1],["x","commit",2]],{}))
    ]
)

LABS = [HASH, LSM, MVCC]
