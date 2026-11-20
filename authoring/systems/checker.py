"""A bounded search engine and a pair of executable credit-handler models."""


def state(pending=(), remaining=0, credits=None):
    return {"pending": list(pending), "copies_left": remaining, "credits": credits or {}}


def query(handler, requests, copies, depth, **kwargs):
    return dict(handler=handler, requests=requests, copies=copies, depth=depth, **kwargs)


def answer(trace=None, account=None):
    return {"result": "violation" if trace is not None else "clear_within_bound",
            "trace": trace or [], "offender": account}


def test(part, function, args, title, expected):
    return dict(part=part, fn=function, args=args, title=title, oracle=expected)


naive = query("naive", ["r"], 1, 3)
protected = query("deduplicate", ["r"], 1, 3)
duplicate_trace = [["copy", "r"], ["deliver", "r"], ["deliver", "r"]]
LABS = [dict(
    slug="sys-model-checker", title="Build a bounded message-schedule checker",
    group="Distributed systems", level="Advanced", duration="75–110 min",
    mission="A payment handler occasionally credits the same request twice. The retry that triggers it never appears in sequential tests. Build the event explorer behind a distributed-systems test harness and return a shortest schedule that reproduces the double credit.",
    scope="The supplied handlers model one replica and one unit of credit per request ID. You implement the network event choices, a canonical state key, and bounded breadth-first search. The optional replay mode lets you inspect the state reached by a witness.",
    inputs='A request has handler ("naive" or "deduplicate"), requests (an initial bag of 0–4 nonempty ASCII request IDs, possibly repeated), copies (0–2), and depth (0–8). Optional drop is a boolean, default false. Omit replay to search, or provide replay as an array of legal [action, id] pairs to execute exactly those actions.',
    outputs='Search returns {result, trace, offender}. A violation gives the first shortest schedule that credits an ID twice and the offending ID. Exhausting the inclusive bound gives {"result":"clear_within_bound","trace":[],"offender":null}. Replay returns {pending, credits, copies_left} after the supplied actions; pending is sorted and credits omits zero counts.',
    constraints="Count each delivery, copy, or drop as one step. For each state, enumerate distinct IDs lexicographically within each action, in action order deliver, copy, drop. Delivery consumes one pending occurrence. A copy leaves the original pending, enqueues one more occurrence, and spends one global copy. Drop consumes one pending occurrence when enabled. The search checks safety at depths zero through depth inclusive. Replay uses its own explicit schedule independently of depth. If more than one ID violates at a supplied initial state, return the lexicographically smallest.",
    source=[dict(title="Michael et al. · Teaching Rigorous Distributed Systems With Efficient Model Checking",url="https://ellismichael.com/papers/dslabs-eurosys19.pdf")],
    modules={
        "handlers.py": '''
            """Protocol under test. Inspect both handlers to understand the invariant."""
            def receive(credits, message, handler):
                if handler == "deduplicate":
                    return {**credits, message: 1}
                return {**credits, message: credits.get(message, 0) + 1}

            def offender(state):
                return min((key for key, count in state["credits"].items() if count > 1), default=None)
        ''',
        "network.py": '''
            """Generate legal message events from a network state."""
            def choices(state, drop):
                """Return unique [action, id] events, ordered by action then ID. Copy requires a pending ID and an unspent copy. Drop requires drop=True."""
                ids = sorted(set(state["pending"]))
                actions = ["deliver"]
                if state["copies_left"] > 0:
                    actions.append("copy")
                if drop:
                    actions.append("drop")
                return [[action, item] for action in actions for item in ids]

            def state_key(state):
                """Return a hashable triple (sorted pending tuple with repetitions, copies_left, sorted tuple of (id,count) credit pairs)."""
                return tuple(sorted(state["pending"])), state["copies_left"], tuple(sorted(state["credits"].items()))
        ''',
        "transition.py": '''
            """One deterministic event delivery. Branches receive independent state."""
            from handlers import receive

            def apply(state, event, handler):
                action, item = event
                pending = list(state["pending"])
                credits = dict(state["credits"])
                copies = state["copies_left"]
                if action == "copy":
                    pending.append(item)
                    copies -= 1
                else:
                    pending.remove(item)
                    if action == "deliver":
                        credits = receive(credits, item, handler)
                return {"pending": sorted(pending), "copies_left": copies, "credits": credits}
        ''',
        "explorer.py": '''
            """Search an asynchronous state space and reconstruct a counterexample."""
            from collections import deque
            from handlers import offender
            from network import choices, state_key
            from transition import apply

            def search(initial, handler, drop, depth):
                """Search inclusive depth with FIFO expansion and choices() ordering. Return result, trace and offender. Expand each canonical state at most once."""
                first = state_key(initial)
                queue = deque([(initial, first, 0)])
                parent = {first: None}
                while queue:
                    state, key, distance = queue.popleft()
                    bad = offender(state)
                    if bad is not None:
                        trace = []
                        while parent[key] is not None:
                            key, event = parent[key]
                            trace.append(event)
                        return {"result": "violation", "trace": trace[::-1], "offender": bad}
                    if distance == depth:
                        continue
                    for event in choices(state, drop):
                        child = apply(state, event, handler)
                        child_key = state_key(child)
                        if child_key not in parent:
                            parent[child_key] = (key, event)
                            queue.append((child, child_key, distance + 1))
                return {"result": "clear_within_bound", "trace": [], "offender": None}
        ''',
        "gateway.py": '''
            """Construct the initial state, then search it or replay a supplied witness."""
            from explorer import search
            from network import choices
            from transition import apply

            def handle(request):
                state = {"pending": sorted(request["requests"]), "credits": {}, "copies_left": request["copies"]}
                drop = request.get("drop", False)
                if "replay" not in request:
                    return search(state, request["handler"], drop, request["depth"])
                for event in request["replay"]:
                    if event not in choices(state, drop):
                        raise ValueError("Illegal replay event")
                    state = apply(state, event, request["handler"])
                return state
        '''},
    parts=[
        dict(title="Enumerate deliveries and identify equivalent states",file="network.py",funcs=["choices","state_key"],
             requirement="Treat pending messages as a multiset. Two identical messages enable one distinct event choice yet have a different future from one message.",
             hints=["A set of pending IDs suffices for choices(). Retain multiplicity in the state key.","Delivery and drop both consume a pending item. Copying needs a pending item as well as an available copy."]),
        dict(title="Find and reconstruct a shortest failing schedule",file="explorer.py",funcs=["search"],
             requirement="Use the supplied transition function, protocol invariant and your event enumerator. Preserve deterministic tie order and return a replayable witness.",
             hints=["Enqueue the initial state with distance zero. Record the first predecessor and event when a state is enqueued.","Check the invariant before cutting off expansion at the depth bound. Walk the predecessor chain backward and reverse the events."])],
    examples=[dict(input=naive,output=answer(duplicate_trace,"r"),
                   why="Delivering the sole request first empties the network. Copying it first leaves two pending deliveries. The second delivery raises its credit count to two in exactly three steps."),
              dict(input=protected,output=answer(),
                   why="Even with two copies in flight, the idempotent handler records one credit for r. Every state reachable through three events satisfies the invariant.")],
    checks=[
        test(1,"network.choices",[state(["b","a","a"],1),True],"Distinct IDs per event kind",
             [[op,i] for op in ["deliver","copy","drop"] for i in ["a","b"]]),
        test(1,"network.state_key",[state(["b","a","a"],1,{"z":1,"b":2})],"Canonical network key preserves counts",
             [["a","a","b"],1,[["b",2],["z",1]]]),
        test(1,"network.choices",[state([],2),True],"An empty network has no enabled action",[]),
        test(2,"explorer.search",[state(["x","x"]), "naive", False, 2],"Two copies already in flight",
             answer([["deliver","x"],["deliver","x"]],"x")),
        test(2,"explorer.search",[state([],0,{"b":3,"a":2}),"naive",False,0],"Initial violation at depth zero",answer([],"a")),
        test(2,"gateway.handle",[query("naive",["k"],1,2)],"Bound stops immediately before the violating delivery",answer()),
        test(2,"gateway.handle",[query("naive",["z","a"],1,3,drop=True)],"Lexicographic tie between shortest witnesses",
             answer([["copy","a"],["deliver","a"],["deliver","a"]],"a")),
        test(2,"gateway.handle",[query("deduplicate",["p","q","p"],2,7,drop=True)],"Deduplication survives drops and copies",answer()),
        test(2,"gateway.handle",[query("naive",["q","p"],0,4,drop=True)],"No possible duplicate delivery without copies",answer()),
        test(2,"gateway.handle",[dict(naive,replay=duplicate_trace)],"Clock independent reproduction of the bug",state([],0,{"r":2})),
        test(1,"gateway.handle",[query("naive",["b","a","b"],1,0,drop=True,replay=[["drop","a"],["copy","b"],["deliver","b"]])],
             "Replay preserves two identical undelivered messages",state(["b","b"],0,{"b":1})),
    ])]
