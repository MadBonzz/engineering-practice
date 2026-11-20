"""Trace-driven distributed protocols. Each gateway starts with pristine state."""
from textwrap import dedent


def module(code):
    return dedent(code).strip() + "\n"


def part(title, file, functions, goal, hint):
    return dict(title=title, file=file, funcs=functions.split(), requirement=goal, hints=[hint])


def check(n, name, args, title, expected):
    return dict(part=n, fn=name, args=args, title=title, oracle=expected)


def example(request, output, why):
    return dict(input=request, output=output, why=why)


def source(title, url):
    return dict(title=title, url=url)


LABS = []


def tx(members, *events):
    return dict(members=members.split(), events=list(events))


def outcome(coordinator, cohorts, *recoveries):
    return dict(coordinator=coordinator, cohorts=cohorts, recoveries=list(recoveries))


def tx_case(title, request, expected):
    return check(3, "gateway.handle", [request], title, expected)


vote_a = ["vote", "A", "YES"]
vote_b = ["vote", "B", "YES"]
partial = tx("A B", vote_a, vote_b, ["precommit", "A"], ["crash"], ["recover", ["A", "B"]])
uncertain = tx("A B", vote_a, vote_b, ["crash"], ["recover", ["A"]], ["recover", ["A", "B"]])
LABS.append(dict(
    slug="sys-three-phase-commit", title="Three-phase commit · recover an interrupted transfer",
    group="Distributed systems", level="Advanced", duration="75–110 min",
    mission="A transfer spans two or more ledger shards. The coordinator can disappear between any two message deliveries. Implement the coordinator, the durable participant transitions and a replacement's termination procedure. Your replay must give the same final decision to every shard whenever the recovery assumptions hold.",
    inputs="members lists distinct shard IDs. events is an ordered list. [vote,id,YES|NO] records a first vote and its reply atomically. [precommit,id], [commit,id] and [abort,id] deliver a message and its acknowledgement atomically. [crash] fences the original coordinator. [recover,[ids...]] runs the replacement's recovery round using exactly the listed respondents. All tags are lowercase JSON strings except YES and NO.",
    outputs="Return coordinator (VOTE, PRECOMMIT, COMMIT, ABORT or DOWN), cohorts (ID to durable INIT, READY, PREPARED, COMMITTED or ABORTED), and recoveries (one WAIT, COMMIT or ABORT for each recovery attempted after a crash). A fenced coordinator stays DOWN even after recovery.",
    constraints="Only a live coordinator processes a vote or delivery. Start in VOTE with every shard INIT. Accept the first vote from each INIT shard in VOTE. Any NO selects ABORT; unanimous YES selects PRECOMMIT. PRECOMMIT sends prepare messages; unanimous durable PREPARED acknowledgements select COMMIT. Deliveries outside their matching coordinator phase leave state unchanged. Ignore repeats and stale messages. Recovery applies its decision to all shards only with a complete report; WAIT leaves every state unchanged. Transaction IDs and shard IDs are valid, and traces contain at most one coordinator crash.",
    scope="One transaction with durable shard state and a single fenced coordinator. The supported recovery model has bounded delay, no partition and every participant available to the replacement. An incomplete survey explicitly returns WAIT. A successful recovery event simulates reliable prepare/ack and final-decision rounds without intermediate failures; a slow link alone never authorizes commit or abort.",
    source=[source("Dale Skeen · Nonblocking commit protocols","https://doi.org/10.1145/582318.582339"),source("Rutgers CS 417 · Distributed transactions and 3PC failure assumptions","https://people.cs.rutgers.edu/pxk/classes/417/notes/transactions.html")],
    modules={
        "cohort.py":module('''
            def transition(state, message):
                """Return the durable successor for YES, NO, precommit, commit or abort. Terminal states absorb all messages. A READY voter can abort, and a PREPARED voter can only commit."""
                if state == "INIT":
                    return {"YES": "READY", "NO": "ABORTED", "abort": "ABORTED"}.get(message, state)
                if state == "READY":
                    return {"precommit": "PREPARED", "abort": "ABORTED"}.get(message, state)
                if state == "PREPARED" and message == "commit":
                    return "COMMITTED"
                return state
        '''),
        "coordinator.py":module('''
            from cohort import transition

            def deliver(phase, states, event):
                """Return [next_phase,new_states]. Gate votes and delivery by phase; advance after the last required acknowledgement. Each trace event includes its reply."""
                states = states.copy()
                kind, node = event[:2]
                if kind == "vote" and phase == "VOTE" and states[node] == "INIT":
                    states[node] = transition(states[node], event[2])
                    if states[node] == "ABORTED":
                        phase = "ABORT"
                    elif all(value == "READY" for value in states.values()):
                        phase = "PRECOMMIT"
                elif kind in {"precommit", "commit", "abort"} and kind.upper() == phase:
                    states[node] = transition(states[node], kind)
                    if phase == "PRECOMMIT" and all(value == "PREPARED" for value in states.values()):
                        phase = "COMMIT"
                return [phase, states]
        '''),
        "termination.py":module('''
            from cohort import transition

            def recover(states, visible):
                """Return [verdict,new_states]. A complete survey with PREPARED or COMMITTED commits, completing the prepare round first. All other reachable complete surveys abort. Incomplete evidence returns WAIT."""
                if set(visible) != set(states):
                    return ["WAIT", states.copy()]
                if any(value in {"PREPARED", "COMMITTED"} for value in states.values()):
                    restored = {n: transition(s, "precommit") for n, s in states.items()}
                    return ["COMMIT", {n: transition(s, "commit") for n, s in restored.items()}]
                return ["ABORT", {n: transition(s, "abort") for n, s in states.items()}]
        '''),
        "gateway.py":module('''
            """Replay coordinator delivery and recovery events for one transaction."""
            from coordinator import deliver
            from termination import recover

            def handle(request):
                phase = "VOTE"
                states = dict.fromkeys(request["members"], "INIT")
                recoveries = []
                for event in request["events"]:
                    kind = event[0]
                    if kind == "crash":
                        phase = "DOWN"
                    elif kind == "recover":
                        if phase == "DOWN":
                            verdict, states = recover(states, event[1])
                            recoveries.append(verdict)
                    elif phase != "DOWN":
                        phase, states = deliver(phase, states, event)
                return {"coordinator": phase, "cohorts": states, "recoveries": recoveries}
        ''')},
    parts=[
        part("Persist votes and decisions","cohort.py","transition","Encode valid local state edges; a duplicate or an old message must have no effect.","A PREPARED participant carries durable evidence that all shards voted YES."),
        part("Advance after acknowledgements","coordinator.py","deliver","Gate messages on the active phase and count distinct shard acknowledgements.","Inspect the map by shard; duplicate delivery to A cannot substitute for B's acknowledgement."),
        part("Terminate after coordinator loss","termination.py","recover","Recover from a full durable survey and defer an incomplete one. Complete missing precommits before final commit.","Compare complete reports of [READY, READY] and [READY, PREPARED].")],
    examples=[
        example(partial,outcome("DOWN",{"A":"COMMITTED","B":"COMMITTED"},"COMMIT"),"A reached PREPARED, which proves both shards voted YES. The replacement prepares B before committing both."),
        example(uncertain,outcome("DOWN",{"A":"ABORTED","B":"ABORTED"},"WAIT","ABORT"),"The first survey cannot rule out a PREPARED shard. The full survey finds two READY shards. No shard saw a precommit, so the replacement aborts.")],
    checks=[
        check(1,"cohort.transition",["READY","commit"],"A final decision cannot skip precommit","READY"),
        check(1,"cohort.transition",["COMMITTED","abort"],"An old abort cannot reverse a terminal decision","COMMITTED"),
        check(2,"coordinator.deliver",["VOTE",{"A":"READY","B":"INIT"},vote_b],"Last positive vote opens precommit",["PRECOMMIT",{"A":"READY","B":"READY"}]),
        check(2,"coordinator.deliver",["PRECOMMIT",{"A":"PREPARED","B":"READY"},["precommit","A"]],"Duplicate prepare acknowledgement",["PRECOMMIT",{"A":"PREPARED","B":"READY"}]),
        check(3,"termination.recover",[{"a":"PREPARED","b":"READY"},["a"]],"Wait for the missing respondent",["WAIT",{"a":"PREPARED","b":"READY"}]),
        check(3,"termination.recover",[{"a":"PREPARED","b":"COMMITTED"},["b","a"]],"Commit evidence survives coordinator loss",["COMMIT",{"a":"COMMITTED","b":"COMMITTED"}]),
        tx_case("Delayed final delivery and client replay",tx("A B",vote_a,["commit","A"],vote_a,vote_b,["precommit","B"],["precommit","B"],["precommit","A"],["commit","B"]),outcome("COMMIT",{"A":"PREPARED","B":"COMMITTED"})),
        tx_case("Negative vote aborts a shard that did not vote",tx("A B C",vote_a,["vote","B","NO"],["abort","A"],["abort","C"],["crash"],["recover",["C","A","B"]]),outcome("DOWN",dict.fromkeys("ABC","ABORTED"),"ABORT")),
        tx_case("Partial precommit in a three-shard group",tx("A B C",vote_a,vote_b,["vote","C","YES"],["precommit","C"],["crash"],["recover",["A","B"]],["recover",["C","B","A"]]),outcome("DOWN",dict.fromkeys("ABC","COMMITTED"),"WAIT","COMMIT")),
        tx_case("Crash before all votes ignores the later YES",tx("A B",vote_a,["crash"],vote_b,["recover",[]],["recover",["A","B"]]),outcome("DOWN",dict.fromkeys("AB","ABORTED"),"WAIT","ABORT")),
        tx_case("Repeated termination after a delivered commit",tx("A",vote_a,["precommit","A"],["commit","A"],["crash"],["recover",["A"]],["recover",["A"]]),outcome("DOWN",{"A":"COMMITTED"},"COMMIT","COMMIT"))
    ]))


def swim(*events, helpers=1, members="A B C", observer="A"):
    return dict(members=members.split(), observer=observer, helpers=helpers, events=list(events))


def view(records, *sent):
    table = {node: [0, "alive"] for node in records} if isinstance(records, str) else records
    return dict(table=table, sent=list(sent))


def swim_case(title, request, expected):
    return check(2, "gateway.handle", [request], title, expected)


probe = ["probe","p","B"]
direct = ["direct_timeout","p"]
indirect = ["indirect_timeout","p"]
ping = ["PING","p","B"]
relay = ["PING_REQ","p","C","B"]
LABS.append(dict(
    slug="sys-swim-membership", title="SWIM membership · investigate a false suspicion",
    group="Distributed systems", level="Intermediate", duration="55–85 min",
    mission="A slow service answers a relayed ping just as a timeout fires. Later, stale gossip accuses the recovered process again. Implement the failure detector and incarnation-aware membership table at a single observer. Return the outgoing messages so an operator can distinguish a relayed success from a declaration of failure.",
    inputs="members contains distinct node IDs, observer names the local member, and helpers is a nonnegative limit. All members start [0,alive]. Replay events in order: [probe,token,target], [direct_timeout,token], [indirect_timeout,token], [ack,token], [gossip,node,inc,status], [expire,node,inc]. inc is a nonnegative integer; status is alive, suspect or dead. Each probe uses a fresh token and a remote target.",
    outputs="Return table mapping each member to [incarnation,status], plus sent as ordered message arrays: [PING,token,target], [PING_REQ,token,helper,target], [SUSPECT,node,inc], [ALIVE,observer,inc] or [DEAD,node,inc]. A received remote rumor updates only the table. Announce locally generated suspicion, refutation or death once when it changes state.",
    constraints="Probe only an alive target. On direct timeout, send PING_REQ to at most helpers currently alive third parties in ID order. Any matching ACK closes the pending probe. Indirect timeout suspects only a still-alive target at the probed incarnation. Unknown, late and wrong-phase events are ignored. Remote gossip orders records by incarnation, then alive < suspect < dead. An accepted higher incarnation can rejoin. Expire confirms only a matching suspect. A suspect rumor about self at or above the local incarnation emits ALIVE with rumor incarnation + 1; other self rumors are ignored.",
    scope="A deterministic observer-level SWIM-style replay of probing, suspicion and refutation. The trace supplies rumor delivery, expiration and scheduling. Helper selection is deterministic for grading; random peer sampling, gossip retransmission budgets and full-cluster convergence are outside the exercise.",
    source=[source("Das, Gupta and Motivala · SWIM group membership","https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SWIM.pdf")],
    modules={
        "membership.py":module('''
            PRECEDENCE = {"alive": 0, "suspect": 1, "dead": 2}

            def merge(record, rumor):
                """Return the winning [incarnation,status]. Compare incarnation first; break ties by alive < suspect < dead."""
                def priority(value):
                    return value[0], PRECEDENCE[value[1]]
                return list(max([record, rumor], key=priority))

            def hear(table, observer, node, inc, status):
                """Merge incoming gossip into table in place. Return outgoing ALIVE messages when the observer refutes a current or newer self-suspicion."""
                if node == observer:
                    if status == "suspect" and inc >= table[node][0]:
                        table[node] = [inc + 1, "alive"]
                        return [["ALIVE", node, inc + 1]]
                else:
                    table[node] = merge(table[node], [inc, status])
                return []
        '''),
        "detector.py":module('''
            from membership import merge

            def handle_probe(table, pending, observer, limit, event):
                """Mutate table/pending and return outbound messages for one probe, timeout or ACK. Keep target, original incarnation and stage for each pending token."""
                kind, token = event[:2]
                if kind == "probe":
                    target = event[2]
                    inc, status = table[target]
                    if status == "alive":
                        pending[token] = [target, inc, "direct"]
                        return [["PING", token, target]]
                    return []
                if token not in pending:
                    return []
                target, inc, stage = pending[token]
                if kind == "ack":
                    del pending[token]
                elif kind == "direct_timeout" and stage == "direct":
                    pending[token][2] = "indirect"
                    peers = [n for n in sorted(table) if n not in {observer, target} and table[n][1] == "alive"]
                    return [["PING_REQ", token, peer, target] for peer in peers[:limit]]
                elif kind == "indirect_timeout" and stage == "indirect":
                    del pending[token]
                    if table[target] == [inc, "alive"]:
                        table[target] = merge(table[target], [inc, "suspect"])
                        return [["SUSPECT", target, inc]]
                return []

            def expire(table, node, inc):
                """Confirm a suspect whose current incarnation equals inc. Update in place and return the emitted messages."""
                if table[node] != [inc, "suspect"]:
                    return []
                table[node] = merge(table[node], [inc, "dead"])
                return [["DEAD", node, inc]]
        '''),
        "gateway.py":module('''
            """Replay wire and timer events at one membership observer."""
            from membership import hear
            from detector import handle_probe, expire

            def handle(request):
                table = {n: [0, "alive"] for n in request["members"]}
                observer = request["observer"]
                pending, sent = {}, []
                for event in request["events"]:
                    if event[0] == "gossip":
                        sent.extend(hear(table, observer, *event[1:]))
                    elif event[0] == "expire":
                        sent.extend(expire(table, *event[1:]))
                    else:
                        sent.extend(handle_probe(table, pending, observer, request["helpers"], event))
                return {"table": table, "sent": sent}
        ''')},
    parts=[
        part("Merge rumors and refute suspicion","membership.py","merge hear","Define the partial order on versions and issue a higher incarnation when this node is suspected.","For equal versions, a remote alive announcement cannot undo an earlier suspicion."),
        part("Drive direct and indirect probes","detector.py","handle_probe expire","Correlate acknowledgements with tokens and fence failure timers by incarnation.","A delayed indirect timeout for an older process incarnation must leave its replacement alive.")],
    examples=[
        example(swim(probe,direct,["ack","p"],indirect),view("ABC",ping,relay),"A asks C to relay a ping to B. The matching acknowledgement closes p. Its late timeout emits nothing."),
        example(swim(probe,direct,indirect,["gossip","B",1,"alive"],["expire","B",0]),view({"A":[0,"alive"],"B":[1,"alive"],"C":[0,"alive"]},ping,relay,["SUSPECT","B",0]),"B's later incarnation supersedes the suspicion. Expiring the old timer cannot mark B dead.")],
    checks=[
        check(1,"membership.merge",[[3,"suspect"],[3,"alive"]],"Same-incarnation gossip preserves suspicion",[3,"suspect"]),
        check(1,"membership.merge",[[5,"suspect"],[4,"dead"]],"Old death report loses",[5,"suspect"]),
        check(1,"membership.hear",[{"A":[2,"alive"]},"A","A",4,"suspect"],"Refute own newer suspicion",[["ALIVE","A",5]]),
        check(2,"detector.handle_probe",[{"A":[0,"alive"],"B":[0,"alive"],"C":[1,"alive"]},{"p":["B",0,"direct"]},"A",1,direct],"A direct timeout delegates to C",[relay]),
        check(2,"detector.expire",[{"B":[3,"suspect"]},"B",3],"A matching suspicion expires",[["DEAD","B",3]]),
        swim_case("Ordered healthy relays skip dead peers",swim(["gossip","B",0,"dead"],["probe","z","E"],["direct_timeout","z"],members="E D C B A",helpers=2),view({"A":[0,"alive"],"B":[0,"dead"],"C":[0,"alive"],"D":[0,"alive"],"E":[0,"alive"]},["PING","z","E"],["PING_REQ","z","C","E"],["PING_REQ","z","D","E"])),
        swim_case("Late acknowledgement after suspicion cannot refute B",swim(probe,direct,indirect,["ack","p"],["expire","B",0],["expire","B",0]),view({"A":[0,"alive"],"B":[0,"dead"],"C":[0,"alive"]},ping,relay,["SUSPECT","B",0],["DEAD","B",0])),
        swim_case("A new incarnation appears during an active probe",swim(probe,["gossip","B",3,"alive"],direct,indirect),view({"A":[0,"alive"],"B":[3,"alive"],"C":[0,"alive"]},ping,relay)),
        swim_case("Self-refutation races an older accusation",swim(["gossip","A",2,"suspect"],["gossip","A",2,"suspect"],["expire","A",2]),view({"A":[3,"alive"],"B":[0,"alive"],"C":[0,"alive"]},["ALIVE","A",3])),
        swim_case("Two outstanding probes share a target",swim(probe,["probe","q","B"],["direct_timeout","q"],direct,indirect,["indirect_timeout","q"],members="A B",helpers=0),view({"A":[0,"alive"],"B":[0,"suspect"]},ping,["PING","q","B"],["SUSPECT","B",0]))
    ]))


def cut(balances, edges, *events):
    return dict(balances=balances, edges=[list(link) for link in edges.split()], events=list(events))


def image(local, channels, complete, total):
    return dict(local=local, channels=channels, complete=complete, total=total)


def cut_case(title, request, expected):
    return check(2, "gateway.handle", [request], title, expected)


bank = {"A":10, "B":10}
two = "AB BA"
start = ["start","A"]
ab = ["deliver","A","B"]
ba = ["deliver","B","A"]
first = cut(bank,two,["send","B","A","x",4],start,ab,ba,ba)
second = cut(bank,two,["send","A","B","x",3],start,ab,ab,ba)
empty_two = {"A>B":[],"B>A":[]}
LABS.append(dict(
    slug="sys-distributed-snapshot", title="Distributed snapshot · account for money in flight",
    group="Distributed systems", level="Intermediate", duration="60–90 min",
    mission="Two ledgers transfer credits over FIFO links. Recording both balances while payments are moving can make credits seem to disappear. Implement the Chandy–Lamport marker rules and channel recorder in the supplied transport simulator. Produce a consistent cut that an auditor can add up.",
    inputs="balances maps node IDs to initial integer credits. edges lists distinct directed [source,destination] links. Replay [send,source,destination,id,amount], [deliver,source,destination] and [start,node]. Each send debits immediately and queues the identified positive transfer. Each deliver consumes one oldest message (payment or marker) from that link. Delivering from an empty queue does nothing.",
    outputs="Return local mapping every ID to its frozen balance or null, channels mapping every 'source>destination' edge to [[id,amount],...] in FIFO order (empty until anything is recorded), complete indicating every node was recorded and every incoming channel closed, and total equal to the sum of frozen balances plus recorded transfers only when complete; otherwise total is null.",
    constraints="Only one snapshot instance. start may repeat at the same initiator; record and broadcast once per node. A first marker records the receiver before it does anything else and closes the arriving channel. Upon first recording, enqueue a marker on every outgoing link before processing the next event. Subsequent markers close only their arriving links. An incoming payment always credits the live balance; append it to the snapshot only if that receiver has recorded and the arriving link is still open. Nodes are reachable from the initiator; IDs exclude '>', and traces use valid transfers with unique IDs and sufficient funds.",
    scope="One asynchronous snapshot with reliable FIFO channels, arbitrary delivery interleaving and no process failures. A trace may intentionally stop mid-snapshot. The scheduler, live account debit/credit and per-link queues are provided; the snapshot logic is yours.",
    source=[source("Chandy and Lamport · Distributed Snapshots","https://lamport.azurewebsites.net/pubs/chandy.pdf")],
    modules={
        "markers.py":module('''
            def mark(cut, balance, incoming, sender):
                """Return [next_cut,broadcast]. cut is None or {balance,open,messages}. Open uses sorted sender IDs. sender=None starts locally; a marker closes its sender. Broadcast only on first recording."""
                if cut is None:
                    return [{
                        "balance": balance,
                        "open": sorted(node for node in incoming if node != sender),
                        "messages": {node: [] for node in incoming},
                    }, True]
                if sender not in cut["open"]:
                    return [cut, False]
                return [{**cut, "open": [node for node in cut["open"] if node != sender]}, False]
        '''),
        "channels.py":module('''
            def record_payment(cut, sender, payment):
                """Return next_cut, copying a payment [id,amount] into this sender's log only while its channel is open. Preserve other channels and the frozen balance."""
                if cut is None or sender not in cut["open"]:
                    return cut
                messages = {**cut["messages"], sender: cut["messages"][sender] + [payment]}
                return {**cut, "messages": messages}

            def assemble(cuts, edges):
                """Format a partial or complete cut, exposing all directed channel logs and all local balances. Sum credits only when every node recorded and all channels closed."""
                local = {node: None if cut is None else cut["balance"] for node, cut in cuts.items()}
                channels = {}
                for sender, receiver in edges:
                    cut = cuts[receiver]
                    channels[sender + ">" + receiver] = [] if cut is None else cut["messages"][sender]
                complete = all(cut is not None and not cut["open"] for cut in cuts.values())
                total = None
                if complete:
                    in_flight = sum(amount for payments in channels.values() for _, amount in payments)
                    total = sum(local.values()) + in_flight
                return {"local": local, "channels": channels, "complete": complete, "total": total}
        '''),
        "gateway.py":module('''
            """Deterministic per-link FIFO transport with live balances and one snapshot instance."""
            from collections import deque
            from markers import mark
            from channels import record_payment, assemble

            def handle(request):
                balances = request["balances"].copy()
                edges = request["edges"]
                queues = {(a, b): deque() for a, b in edges}
                cuts = dict.fromkeys(balances)

                def receive_marker(node, sender):
                    incoming = [a for a, b in edges if b == node]
                    cuts[node], broadcast = mark(cuts[node], balances[node], incoming, sender)
                    if broadcast:
                        for a, b in edges:
                            if a == node:
                                queues[a, b].append(None)

                for event in request["events"]:
                    if event[0] == "start":
                        receive_marker(event[1], None)
                    else:
                        a, b = event[1:3]
                        queue = queues[a, b]
                        if event[0] == "send":
                            payment = event[3:]
                            balances[a] -= payment[1]
                            queue.append(payment)
                        elif queue:
                            payment = queue.popleft()
                            if payment is None:
                                receive_marker(b, a)
                            else:
                                balances[b] += payment[1]
                                cuts[b] = record_payment(cuts[b], a, payment)
                return assemble(cuts, edges)
        ''')},
    parts=[
        part("Freeze the receiver on its first marker","markers.py","mark","Implement local initiation and the first-versus-subsequent marker rules. Make repeated starts idempotent.","Local state and incoming-channel state have separate recording boundaries."),
        part("Record payments crossing the cut","channels.py","record_payment assemble","Preserve per-link arrival order and assemble partial and complete snapshots from the same structures.","A payment already reflected in the receiver's frozen balance must never appear in the channel log.")],
    examples=[
        example(first,image({"A":10,"B":6},{"A>B":[],"B>A":[["x",4]]},True,20),"B sent four credits before its snapshot. A received them after its own snapshot but before the marker on B>A. The audit counts 10 + 6 + 4."),
        example(second,image({"A":7,"B":13},empty_two,True,20),"The A>B queue delivers x before A's marker. B already includes those three credits when it freezes; both recorded channels are empty.")],
    checks=[
        check(1,"markers.mark",[None,6,["D","B"],"B"],"First marker closes just its incoming channel",[{"balance":6,"open":["D"],"messages":{"D":[],"B":[]}},True]),
        check(1,"markers.mark",[{"balance":2,"open":["C"],"messages":{"B":[],"C":[["p",1]]}},9,["B","C"],"C"],"Last marker preserves recorded money",[{"balance":2,"open":[],"messages":{"B":[],"C":[["p",1]]}},False]),
        check(2,"channels.record_payment",[{"balance":5,"open":["B"],"messages":{"B":[]}},"B",["p",2]],"Open channel records a payment",{"balance":5,"open":["B"],"messages":{"B":[["p",2]]}}),
        check(2,"channels.record_payment",[{"balance":5,"open":[],"messages":{"B":[]}},"B",["p",2]],"Closed channel ignores later receipts",{"balance":5,"open":[],"messages":{"B":[]}}),
        cut_case("Partial trace before a marker reaches B",cut(bank,two,start),image({"A":10,"B":None},empty_two,False,None)),
        cut_case("Multiple ordered in-flight transfers survive repeated start",cut(bank,two,["send","B","A","p",3],["send","B","A","q",2],start,start,ba,ab,ba,ba),image({"A":10,"B":5},{"A>B":[],"B>A":[["p",3],["q",2]]},True,20)),
        cut_case("Post-marker payment does not enter a completed channel",cut(bank,two,start,ab,["send","B","A","late",7],ba,ba),image(bank,empty_two,True,20)),
        cut_case("Independent incoming channels close at different times",cut({"A":5,"B":4,"C":6},"AB AC BA CA",["send","C","A","c",2],start,ab,ba,["deliver","A","C"],["deliver","C","A"],["deliver","C","A"]),image({"A":5,"B":4,"C":4},{"A>B":[],"A>C":[],"B>A":[],"C>A":[["c",2]]},True,15)),
        cut_case("A source with no incoming edges and a sink",cut({"A":2,"B":0},"AB",["deliver","A","B"],start,ab),image({"A":2,"B":0},{"A>B":[]},True,2)),
        cut_case("A partial cut can expose in-flight messages",cut(bank,two,["send","B","A","x",5],start,ba),image({"A":10,"B":None},{"A>B":[],"B>A":[["x",5]]},False,None))
    ]))
