"""Operating system and network implementation exercises using deterministic trace replay."""
from textwrap import dedent


def part(title, file, funcs, requirement, *hints):
    return dict(title=title, file=file+".py", funcs=funcs.split(), requirement=requirement, hints=list(hints))


def check(number, target, args, title, **kwargs):
    return dict(part=number, fn=target, args=args, title=title, **kwargs)


def example(data, answer, why):
    return dict(input=data, output=answer, why=why)


def link(title, url):
    return dict(title=title, url=url)


def lab(slug, title, group, duration, level, mission, scope, inputs, outputs, constraints, source, modules, parts, examples, checks):
    return dict(slug="sys-"+slug, title=title, group=group, duration=duration, level=level, mission=mission,
                scope=scope, inputs=inputs, outputs=outputs, constraints=constraints, source=source,
                modules={k+".py":dedent(v).strip()+"\n" for k,v in modules.items()},
                parts=parts, examples=examples, checks=checks)


def replay(n, request, title, **kwargs):
    return check(n, "gateway.handle", [request], title, **kwargs)


LABS = []


def vm(pages, ops):
    return dict(pages=pages, ops=ops)


cow_a=vm([["p",7,"W"]],[["fork","init","a"],["write","a","p",9],["read","init","p"],["exit","a"],["write","init","p",5]])
cow_b=vm([["code",42,"R"]],[["fork","init","b"],["write","b","code",0],["exit","init"],["read","b","code"]])
LABS.append(lab("cow-fork","Implement copy-on-write fork and page faults","Operating systems","75–110 min","Advanced",
    "A worker forks immediately before reading a shared dataset. Eager copying doubles memory use, but sharing a writable page lets the child corrupt its parent. Finish the page-table clone and write-fault paths, then replay forks, writes, and exits against a reference-counted physical allocator.",
    "A userspace simulator represents one integer per page. The supplied allocator, page-table trace driver, and read/exit paths stand in for kernel machinery. The lab models mapping permissions, allocation, and reclamation without emulating a CPU or TLB.",
    "pages lists [virtualPage, initialValue, permission] for process init. Each distinct virtualPage is a string. Permission is W or R. ops contains [fork,parent,child], [write,pid,page,value], [read,pid,page], or [exit,pid] in execution order.",
    "Return events (only read and write outcomes, in order), mappings (each surviving pid maps page names to [frameID, permission]), and frames (live frame IDs as string keys, each mapped to [value, referenceCount]). Reads emit [read,value]. Writes emit [write,ok], [write,copy], [write,upgrade], or [write,protection].",
    "Frame IDs are integers allocated from zero in input order and never reused. W is writable/private, C is shared-by-fork copy-on-write, R is permanently read-only. A denied write leaves the process alive. Every read/write names a mapped page, every process exists, and fork creates a fresh PID. Exit frees a frame only after the last reference disappears.",
    [link("MIT 6.1810 · Copy-on-write fork for xv6","https://pdos.csail.mit.edu/6.1810/2024/labs/cow.html")],
    dict(allocator="""
        '''Physical frame allocator. Tables use string keys so traces round-trip through JSON.'''
        def alloc(memory, value):
            frame = memory["next"]
            memory["next"] += 1
            memory["frames"][str(frame)] = [value, 1]
            return frame

        def release(memory, frame):
            key = str(frame)
            memory["frames"][key][1] -= 1
            if memory["frames"][key][1] == 0:
                del memory["frames"][key]
    """, vm_fork="""
        def clone_table(parent, memory):
            '''Return a distinct child page table. Change writable mappings in both tables to C, retain R/C permissions, and add one frame reference per new mapping.'''
            child = {}
            for page, (frame, permission) in parent.items():
                if permission == "W":
                    permission = "C"
                parent[page] = [frame, permission]
                child[page] = [frame, permission]
                memory["frames"][str(frame)][1] += 1
            return child
    """, vm_fault="""
        from allocator import alloc, release

        def store(table, memory, page, value):
            '''Mutate this mapping and the allocator. Return protection for R, ok for W, upgrade for sole-owner C, and copy for multi-owner C. After a successful write the mapping is W.'''
            frame, permission = table[page]
            old = memory["frames"][str(frame)]
            if permission == "R":
                return "protection"
            outcome = "ok"
            if permission == "C":
                if old[1] == 1:
                    outcome = "upgrade"
                else:
                    outcome = "copy"
                    release(memory, frame)
                    frame = alloc(memory, old[0])
                table[page] = [frame, "W"]
            memory["frames"][str(frame)][0] = value
            return outcome
    """, gateway="""
        '''Replay process operations and expose the final physical and virtual memory.'''
        from allocator import alloc, release
        from vm_fork import clone_table
        from vm_fault import store

        def handle(request):
            memory = {"next": 0, "frames": {}}
            processes = {"init": {}}
            for page, value, permission in request["pages"]:
                processes["init"][page] = [alloc(memory,value), permission]
            events = []
            for op, pid, *args in request["ops"]:
                space = processes[pid]
                if op == "fork":
                    processes[args[0]] = clone_table(space, memory)
                elif op == "write":
                    events.append(["write", store(space,memory,*args)])
                elif op == "read":
                    frame = space[args[0]][0]
                    events.append(["read", memory["frames"][str(frame)][0]])
                elif op == "exit":
                    for frame, _ in space.values():
                        release(memory,frame)
                    del processes[pid]
            return {"events": events, "mappings": processes, "frames": memory["frames"]}
    """),
    [part("Share frames during fork","vm_fork","clone_table","Mark parent and child mappings appropriately, build independently editable entries, and update physical reference counts.","A nested fork can inherit a C page. Keep that protection and count the additional reference."),
     part("Handle writes without leaking frames","vm_fault","store","Distinguish true protection faults from COW. Allocate only when another mapping still owns the frame.","Exiting siblings can leave a C mapping as the only owner. Test that path without allocating.")],
    [example(cow_a,dict(events=[["write","copy"],["read",7],["write","upgrade"]],mappings={"init":{"p":[0,"W"]}},frames={"0":[5,1]}),
             "The child's write creates frame 1. The parent still reads 7 from frame 0. Exiting the child frees frame 1; the parent can now write in place."),
     example(cow_b,dict(events=[["write","protection"],["read",42]],mappings={"b":{"code":[0,"R"]}},frames={"0":[42,1]}),
             "Fork shares the code page as R. A denied write changes no bytes, and the remaining child keeps frame 0 alive after its parent exits.")],
    [check(1,"vm_fork.clone_table",[{"x":[0,"W"],"text":[1,"R"]},{"next":2,"frames":{"0":[9,1],"1":[4,1]}}],"Protect both mapping kinds",oracle={"x":[0,"C"],"text":[1,"R"]}),
     check(1,"vm_fork.clone_table",[{"x":[0,"C"]},{"next":1,"frames":{"0":[4,2]}}],"Nested shared mapping",oracle={"x":[0,"C"]}),
     check(2,"vm_fault.store",[{"x":[0,"C"]},{"next":1,"frames":{"0":[6,1]}},"x",13],"Sole reference upgrades in place",oracle="upgrade"),
     check(2,"vm_fault.store",[{"x":[0,"C"]},{"next":1,"frames":{"0":[6,3]}},"x",13],"Three references require a copy",oracle="copy"),
     replay(1,vm([["x",3,"W"]],[["fork","init","a"],["fork","a","b"]]),"Nested fork accounts for all mappings",oracle={"events":[],"mappings":{p:{"x":[0,"C"]} for p in ["init","a","b"]},"frames":{"0":[3,3]}}),
     replay(2,vm([["x",2,"W"]],[["fork","init","a"],["fork","a","b"],["write","a","x",7],["write","init","x",8],["write","b","x",9],["read","a","x"]]),"Three writers separate and reuse the last original frame"),
     replay(2,vm([["code",1,"R"],["heap",-1,"W"]],[["fork","init","a"],["write","init","code",9],["write","init","heap",0],["exit","init"],["exit","a"]]),"Last unmap reclaims shared and private frames",oracle={"events":[["write","protection"],["write","copy"]],"mappings":{},"frames":{}}),
     replay(2,vm([["x",7,"W"]],[["write","init","x",8],["fork","init","a"],["exit","init"],["write","a","x",6],["fork","a","b"],["read","b","x"]]),"Exit and refork after an in-place upgrade")]))


def sched(quanta, jobs, boost=0):
    return dict(quanta=quanta, boost=boost, jobs=[dict(id=p,arrive=t,bursts=b) for p,t,b in jobs])


sched_a=sched([2,4],[("A",0,[6]),("B",3,[1])])
sched_b=sched([2,3],[("X",0,[1,2,3]),("Y",0,[5])],4)
LABS.append(lab("mlfq-scheduler","Schedule interactive jobs with multilevel feedback queues","Operating systems","90–130 min","Advanced",
    "A long batch job owns the CPU when an interactive job wakes up. Build an event-driven scheduler that reacts immediately, preserves CPU usage across interruptions, and occasionally restores priority. Its dispatch trace is the only evidence available when a job finishes late.",
    "One CPU, FIFO queues at each priority, optional global boosts, and alternating CPU/I/O bursts. Time, interrupt handling, and context switches are deterministic. I/O devices can serve any number of sleeping jobs.",
    "quanta is a nonempty array of positive CPU allotments from highest priority (level 0) to lowest. boost is 0 (disabled) or a positive period. Each job has a distinct string id, nonnegative arrive time, and an odd-length list of positive bursts alternating CPU, I/O duration, CPU. Jobs may be absent.",
    "Return {timeline, finished}. Each timeline interval [start,end,id,level] represents CPU execution over [start,end). Merge adjacent intervals with the same id and level. Omit idle time. finished maps each id to its completion time.",
    "Charge CPU at the old level first. Exhausting a level budget demotes one level (capped at the bottom) and resets its budget, even if a burst ends then. A continuing job goes to the tail on exhaustion and to its own queue's head on interruption. A sleeping job keeps unused budget and wakes at its queue's tail. At each positive boost multiple, after charging/requeueing and before arrivals/wakes, reset budgets of all unfinished jobs and concatenate ready queues in old priority/FIFO order into level 0. Then enqueue arrivals/wakes sorted by id, and dispatch the highest nonempty queue. New jobs start at 0. Traces end by time 100,000.",
    [link("Arpaci-Dusseau and Arpaci-Dusseau · Scheduling with multilevel feedback queues","https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-sched-mlfq.pdf")],
    dict(dispatch_policy="""
        def charge(level, budget, elapsed, quanta):
            '''Given 0 < elapsed <= budget, return [new_level,new_budget]. A consumed allotment rolls over even at the bottom.'''
            if elapsed < budget:
                return [level, budget - elapsed]
            level = min(level + 1, len(quanta) - 1)
            return [level, quanta[level]]

        def take(queues):
            '''Remove and return the first job ID in the highest nonempty queue, or None if idle.'''
            for queue in queues:
                if queue:
                    return queue.pop(0)
            return None
    """, timeline="""
        '''Provided rendering helper. Retain level changes, merge identical adjacent spans.'''
        def record(trace, start, end, pid, level):
            if trace and trace[-1][1] == start and trace[-1][2:] == [pid,level]:
                trace[-1][1] = end
            else:
                trace.append([start,end,pid,level])
    """, event_loop="""
        from dispatch_policy import charge, take
        from timeline import record

        def schedule(jobs, quanta, boost):
            '''Return the compressed timeline and completion map. Jump to the next arrival/wake, boost, burst end or budget exhaustion and resolve ties per the contract.'''
            queues = [[] for _ in quanta]
            waiting = {job["id"]: job["arrive"] for job in jobs}
            state = {job["id"]: {"bursts": job["bursts"], "index": 0, "left": job["bursts"][0],
                                  "level": 0, "budget": quanta[0]} for job in jobs}
            finished = {}
            trace = []
            now = 0
            while len(finished) < len(jobs):
                if boost and now > 0 and now % boost == 0:
                    queues = [[pid for queue in queues for pid in queue]] + [[] for _ in quanta[1:]]
                    for pid, job in state.items():
                        if pid not in finished:
                            job["level"], job["budget"] = 0, quanta[0]
                for pid in sorted(pid for pid, ready in waiting.items() if ready == now):
                    del waiting[pid]
                    queues[state[pid]["level"]].append(pid)

                pid = take(queues)
                deadlines = list(waiting.values())
                if boost:
                    deadlines.append((now // boost + 1) * boost)
                if pid is None:
                    now = min(deadlines)
                    continue

                job = state[pid]
                old_level, old_budget = job["level"], job["budget"]
                until = min([now + job["left"], now + old_budget] + deadlines)
                elapsed = until - now
                record(trace,now,until,pid,old_level)
                job["left"] -= elapsed
                job["level"], job["budget"] = charge(old_level,old_budget,elapsed,quanta)
                if job["left"] == 0:
                    index = job["index"]
                    if index + 1 == len(job["bursts"]):
                        finished[pid] = until
                    else:
                        waiting[pid] = until + job["bursts"][index+1]
                        job["index"] += 2
                        job["left"] = job["bursts"][index+2]
                elif elapsed == old_budget:
                    queues[job["level"]].append(pid)
                else:
                    queues[old_level].insert(0,pid)
                now = until
            return {"timeline": trace, "finished": finished}
    """, gateway="""
        '''Kernel trace entry point. The input order does not break simultaneous ties.'''
        from event_loop import schedule

        def handle(request):
            return schedule(request["jobs"], request["quanta"], request["boost"])
    """),
    [part("Account for CPU and dispatch by priority","dispatch_policy","charge take","Make accounting independent of why a run stopped; pick FIFO within the highest nonempty priority.","Elapsed time can equal the budget on a sleep boundary. A lower-level job keeps its remaining budget when preempted."),
     part("Drive the scheduler from events","event_loop","schedule","Maintain ready queues and wake times. Advance by the shortest competing deadline and emit half-open CPU intervals.","At a timestamp shared by a boost and a wakeup, first restore the existing ready set, then admit the waking job. The supplied timeline helper merges only adjacent equivalent intervals.")],
    [example(sched_a,dict(timeline=[[0,2,"A",0],[2,3,"A",1],[3,4,"B",0],[4,7,"A",1]],finished={"A":7,"B":4}),
             "A consumes two ticks at level 0. B arrives during A's lower-priority slice and finishes immediately. A resumes with three ticks of its level-1 budget."),
     example(sched_b,dict(timeline=[[0,1,"X",0],[1,3,"Y",0],[3,4,"X",0],[4,6,"Y",0],[6,8,"X",0],[8,9,"Y",0]],finished={"X":8,"Y":9}),
             "X's two level-0 ticks straddle its I/O. At time 4 the boost promotes queued Y then X. At time 8 X finishes, and the next boost brings Y back to level 0.")],
    [check(1,"dispatch_policy.charge",[0,2,1,[2,5]],"Retain a partially consumed level budget",oracle=[0,1]),
     check(1,"dispatch_policy.charge",[1,1,1,[2,5]],"Exhaust bottom budget",oracle=[1,5]),
     check(1,"dispatch_policy.take",[[[],["x","a"],["p"]]],"First eligible FIFO job",oracle="x"),
     check(2,"event_loop.schedule",[[dict(id="a",arrive=3,bursts=[2])],[1,2],0],"Skip idle time before first arrival",oracle={"timeline":[[3,4,"a",0],[4,5,"a",1]],"finished":{"a":5}}),
     check(2,"event_loop.schedule",[[],[2],3],"Empty ready set",oracle={"timeline":[],"finished":{}}),
     replay(2,sched([1],[("b",2,[2]),("a",2,[2])]),"Bottom-level round robin with simultaneous arrivals",oracle={"timeline":[[2,3,"a",0],[3,4,"b",0],[4,5,"a",0],[5,6,"b",0]],"finished":{"a":5,"b":6}}),
     replay(2,sched([2,3],[("s",1,[1,1,1,1,2]),("x",0,[6])]),"Short repeated bursts cannot discard prior CPU usage"),
     replay(2,sched([2,4,8],[("a",0,[8]),("b",4,[1]),("c",4,[2]),("d",5,[1])],4),"Boost and multiple arrivals tied to one timestamp"),
     replay(2,sched([3,7],[("z",5,[3,9,2]),("a",18,[2])],6),"Sleeping job regains priority across an idle boost"),
     replay(2,sched([1,2,4],[("cpu",0,[9]),("hi",4,[1,1,1])]),"Interrupted lower-level head retains the residual slice")]))


def stream(capacity, segments):
    return {"capacity":capacity,"segments":segments}


stream_a=stream(5,[[2,"4344",True],[1,"62",False],[0,"414243",False]])
stream_b=stream(3,[[2,"00ff",True],[0,"abcd",False],[0,"abcd",False],[3,"ff",False]])
LABS.append(lab("stream-reassembly","Reassemble an out-of-order byte stream","Networking","55–85 min","Intermediate",
    "A file transfer delivers some bytes twice, some in the wrong order, and its end marker before a gap is filled. Implement the receive buffer that reconstructs exactly the byte sequence an application can read after each delivery.",
    "This laboratory isolates stream reassembly after converting transport sequence numbers to absolute offsets. It uses an eagerly reading application, a bounded byte window, and an explicit reliable end-of-stream announcement. Handshaking, congestion control, wire-level FIN acknowledgment, and sequence-number wraparound belong to other transport layers.",
    "capacity is a positive receive window in bytes. segments is an arrival-ordered list of [offset, hexData, end]. Offsets count bytes starting at zero; hexData has two hex digits per byte. A true end announces that the exclusive stream length is offset plus payload length. Empty payloads are valid.",
    "Return emitted, a list of lowercase hex strings (one per arrival, empty when the gap stays open); next, the first undelivered byte offset; pending, sorted maximal half-open ranges [start,end) of retained bytes; and closed, true exactly when the announced length equals next.",
    "Freeze the receive window [next,next+capacity) at the start of each arrival. Insert only bytes in it. Previously retained bytes win conflicting overlaps. Then emit and discard the longest contiguous prefix starting at next; future bytes clipped out must be retransmitted. Remember an end announcement even if some payload lies outside the window. End announcements agree; no segment extends beyond the final length. At most 256 segments with 4,096 bytes each.",
    [link("RFC 9293 · Transmission Control Protocol, segment processing and receive sequence space","https://www.rfc-editor.org/rfc/rfc9293")],
    dict(window="""
        def accept(pending, cursor, capacity, offset, payload):
            '''Return an updated offset-to-byte dict. Keys are decimal offset strings; values are byte integers. Decode hex payload, filter to the window fixed at cursor, and retain the first copy.'''
            result = dict(pending)
            for index, value in enumerate(bytes.fromhex(payload), offset):
                if cursor <= index < cursor + capacity:
                    result.setdefault(str(index), value)
            return result
    """, delivery="""
        def drain(pending, cursor):
            '''Return [emittedHex, nextCursor, remainingDict], removing consecutive bytes from cursor up to the first gap. Preserve the input dict.'''
            remaining = dict(pending)
            ready = bytearray()
            while str(cursor) in remaining:
                ready.append(remaining.pop(str(cursor)))
                cursor += 1
            return [ready.hex(),cursor,remaining]
    """, inspection="""
        '''Render sparse buffered offsets as sorted, non-overlapping half-open ranges.'''
        def ranges(pending):
            out = []
            for pos in sorted(int(key) for key in pending):
                if out and pos == out[-1][1]:
                    out[-1][1] += 1
                else:
                    out.append([pos,pos+1])
            return out
    """, gateway="""
        '''Replay real byte strings, remembering end-of-stream independently of byte delivery.'''
        from window import accept
        from delivery import drain
        from inspection import ranges

        def handle(request):
            pending = {}
            cursor = 0
            final = None
            emitted = []
            for offset, payload, end in request["segments"]:
                if end:
                    final = offset + len(payload) // 2
                pending = accept(pending,cursor,request["capacity"],offset,payload)
                batch, cursor, pending = drain(pending,cursor)
                emitted.append(batch)
            return {"emitted":emitted, "next":cursor, "pending":ranges(pending), "closed":cursor == final}
    """),
    [part("Clip payloads to the receive window","window","accept","Merge arbitrary binary segments at their byte offsets and preserve the earliest accepted value at each offset.","Clipping occurs before delivery advances next. Never use character count as byte count for hex data."),
     part("Deliver only a gap-free prefix","delivery","drain","Remove contiguous bytes starting at the cursor. Leave every byte beyond the first gap available for a later packet.","Input dictionaries use string keys. Accumulate numeric bytes before converting the emitted prefix to hex.")],
    [example(stream_a,dict(emitted=["","","41624344"],next=4,pending=[],closed=True),
             "CD arrives at offsets 2 and 3 with the end marker. Lowercase b fills offset 1. The final ABC supplies offset 0, while b and C already occupy the overlapping positions."),
     example(stream_b,dict(emitted=["","abcd00","","ff"],next=4,pending=[],closed=True),
             "The initial window [0,3) retains only byte 00 at offset 2 but records length 4. Filling the gap delivers three bytes. Duplicates before next are ignored; retransmitting ff at offset 3 closes the stream.")],
    [check(1,"window.accept",[{"2":255},1,3,0,"0011223344"],"Clip the left and right edges and preserve an overlap",oracle={"1":17,"2":255,"3":51}),
     check(1,"window.accept",[{},9,1,10,"42"],"Right boundary is exclusive",oracle={}),
     check(2,"delivery.drain",[{"0":0,"1":255,"3":2},0],"Deliver binary prefix but keep far data",oracle=["00ff",2,{"3":2}]),
     check(2,"delivery.drain",[{"11":5},10],"Missing first byte",oracle=["",10,{"11":5}]),
     replay(1,stream(8,[[6,"aa",False],[3,"ff00",False],[7,"ee",False]]),"Display disjoint pending runs",oracle={"emitted":["","",""],"next":0,"pending":[[3,5],[6,8]],"closed":False}),
     replay(2,stream(1,[[0,"",True]]),"Empty stream closes at offset zero",oracle={"emitted":[""],"next":0,"pending":[],"closed":True}),
     replay(2,stream(2,[[2,"",True],[0,"aabb",False],[1,"bb",False],[2,"",True]]),"Duplicate final marker and delivery"),
     replay(2,stream(4,[[3,"81",False],[1,"fe0203",True],[0,"ffff",False]]),"First observed conflicting byte survives multiple overlaps"),
     replay(2,stream(6,[[0,"fafa",False],[4,"00ff",False],[3,"aa",False]]),"Unannounced stream ends with a gap")]))


def route(*events):
    return {"events":list(events)}


router_a=route(["add","0.0.0.0/0","wan"],["add","10.0.0.0/8","lan"],
               ["send","10.1.2.3",4,17,4660],["del","10.0.0.0/8"],["send","10.1.2.3",4,17,4660])
router_b=route(["add","198.51.100.7/32","host"],["send","198.51.100.7",1,6,123],
               ["send","198.51.100.8",64,6,123],["send","198.51.100.7",2,17,65279])
LABS.append(lab("ipv4-forwarding","Build a forwarding trie and update IPv4 headers","Networking","65–95 min","Intermediate",
    "A router learns a specific subnet and later loses it. Packets should switch to the remaining covering route, leave with one fewer hop, and carry a consistent header checksum. Build the forwarding information base and connect it to the packet path.",
    "An IPv4 unicast forwarding core with a dynamic binary prefix trie and an incremental Internet checksum. The trace supplies validated headers and names the egress next hop. Drop reasons stand in for generating ICMP messages. ARP, local delivery, options, fragmentation, and routing-protocol convergence are outside the model.",
    "events is an ordered list. [add,cidr,hop] installs or replaces one route to a string next hop. [del,cidr] withdraws that exact prefix if present. [send,dst,ttl,proto,checksum] supplies an IPv4 destination, TTL (1–255), protocol (0–255), and an existing 16-bit header checksum. CIDRs have canonical network addresses; all IP addresses are ordinary dotted IPv4 strings.",
    "Return a list in send order. Each result is [forward,hop,newTTL,newChecksum], [drop,time-exceeded], or [drop,no-route]. Route updates do not emit output. In this lab a TTL of one yields time-exceeded even if the route table is empty.",
    "Walk address bits from most significant to least. Prefer the matching route with the longest prefix length; /0 is a valid default and /32 an exact host. A withdrawal keeps both more-specific descendants and shorter alternatives. Decrement TTL by one and update only the combined TTL/protocol 16-bit word using RFC 1624 equation 3, folding end-around carry. Keep the canonical zero checksum from that equation.",
    [link("RFC 1812 · IPv4 forwarding and longest-match route selection","https://www.rfc-editor.org/rfc/rfc1812"),
     link("RFC 1624 · Incremental update of the Internet checksum","https://www.rfc-editor.org/rfc/rfc1624")],
    dict(address_bits="""
        '''Provided address parser. The high-order bit of an IPv4 address is bit 0.'''
        from ipaddress import IPv4Address, IPv4Network

        def path(value, length=32):
            address = int(IPv4Address(value))
            return [str((address >> (31-i)) & 1) for i in range(length)]

        def prefix_path(cidr):
            network = IPv4Network(cidr)
            return path(network.network_address,network.prefixlen)
    """, fib="""
        from address_bits import path, prefix_path

        def install(root, cidr, hop):
            '''Mutate and return root. Node keys "0"/"1" are child dictionaries, and "hop" is optional. A None hop removes the route and prunes empty nodes, preserving the root.'''
            node = root
            parents = []
            for bit in prefix_path(cidr):
                if bit not in node:
                    if hop is None:
                        return root
                    node[bit] = {}
                parents.append((node,bit))
                node = node[bit]
            if hop is not None:
                node["hop"] = hop
            else:
                node.pop("hop",None)
                for parent, bit in reversed(parents):
                    if parent[bit]:
                        break
                    del parent[bit]
            return root

        def lookup(root, address):
            '''Return the most specific next hop for the IPv4 address, or None if no prefix matches. Inspect at most 32 child edges.'''
            node = root
            best = node.get("hop")
            for bit in path(address):
                if bit not in node:
                    break
                node = node[bit]
                best = node.get("hop",best)
            return best
    """, forwarding="""
        from fib import lookup

        def change_word(checksum, old, new):
            '''Apply RFC 1624 Eq. 3 to a 16-bit header checksum and one changed 16-bit network-order word. Return an unsigned checksum.'''
            value = ((~checksum) & 0xffff) + ((~old) & 0xffff) + new
            while value >> 16:
                value = (value & 0xffff) + (value >> 16)
            return (~value) & 0xffff

        def forward(root, dst, ttl, proto, checksum):
            '''Return a drop reason or [forward,hop,adjusted TTL,adjusted checksum] for one validated transit packet.'''
            if ttl <= 1:
                return ["drop","time-exceeded"]
            hop = lookup(root,dst)
            if hop is None:
                return ["drop","no-route"]
            old = (ttl << 8) | proto
            new = ((ttl-1) << 8) | proto
            return ["forward",hop,ttl-1,change_word(checksum,old,new)]
    """, gateway="""
        '''Feed control-plane updates and transit packets to the same forwarding table.'''
        from fib import install
        from forwarding import forward

        def handle(request):
            root = {}
            packets = []
            for op, *fields in request["events"]:
                if op == "add":
                    install(root,*fields)
                elif op == "del":
                    install(root,fields[0],None)
                else:
                    packets.append(forward(root,*fields))
            return packets
    """),
    [part("Maintain the forwarding information base","fib","install lookup","Insert, replace, and withdraw exact CIDRs without damaging neighboring routes. Resolve destinations by remembering the last matching hop.","Store /0 on the root. During deletion a node carrying children or a hop must stay in the trie."),
     part("Decrement TTL and repair the checksum","forwarding","change_word forward","Dispatch or drop each incoming packet and incrementally replace its TTL/protocol header word using one's-complement arithmetic.","RFC 1624 uses HC′ = ~(~HC + ~m + m′), with 16-bit complements and end-around carries. A shortcut based on ordinary addition mishandles the zero representation.")],
    [example(router_a,[["forward","lan",3,4916],["forward","wan",3,4916]],
             "10.1.2.3 matches both routes at first; withdrawing /8 reveals /0. In each packet 0x0411 becomes 0x0311, changing checksum 0x1234 to 0x1334."),
     example(router_b,[["drop","time-exceeded"],["drop","no-route"],["forward","host",1,0]],
             "The first TTL expires. The next address misses the host route. The final packet's checksum 0xFEFF and TTL/protocol update 0x0211 → 0x0111 produce 0x0000 by one's-complement arithmetic.")],
    [check(1,"fib.install",[{"hop":"d"},"128.0.0.0/2","a"],"Attach a route below the default",oracle={"hop":"d","1":{"0":{"hop":"a"}}}),
     check(1,"fib.lookup",[{"hop":"wan","1":{"hop":"east","1":{"0":{"hop":"lan"}}}},"193.0.0.1"],"Most specific subtree",oracle="lan"),
     check(1,"fib.install",[{"1":{"0":{"hop":"h"}}},"128.0.0.0/2",None],"Prune an empty branch",oracle={}),
     check(2,"forwarding.change_word",[65279,529,273],"Ones-complement boundary yields zero",oracle=0),
     check(2,"forwarding.forward",[{"hop":"n"},"8.8.8.8",32,6,65520],"Carry wraps into low bit",oracle=["forward","n",31,241]),
     replay(1,route(["add","172.16.0.0/12","r"],["add","172.17.0.0/16","x"],["del","172.16.0.0/12"],["send","172.17.1.2",4,6,100],["send","172.18.1.2",4,6,100]),"Removing an ancestor retains its child",oracle=[["forward","x",3,356],["drop","no-route"]]),
     replay(2,route(["add","0.0.0.0/0","old"],["add","0.0.0.0/0","new"],["send","203.0.113.8",2,1,0],["del","0.0.0.0/0"],["send","203.0.113.8",1,1,7]),"Replace and remove the default; TTL precedence"),
     replay(2,route(["add","10.0.0.0/8","base"],["add","10.128.0.0/9","upper"],["del","10.192.0.0/10"],["send","10.127.255.255",255,17,999],["send","10.128.0.0",254,17,999]),"Non-octet boundary and idempotent delete"),
     replay(2,route(["add","255.255.255.254/32","edge"],["send","255.255.255.254",8,253,65280],["send","255.255.255.253",8,253,65280]),"Host-bit comparison reaches the last trie node")]))
