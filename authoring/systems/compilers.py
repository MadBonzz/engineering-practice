"""Two self-contained compiler and runtime labs for Practice Bench."""
from textwrap import dedent


def check(part, fn, args, title, oracle=None):
    row = dict(part=part, fn=fn, args=args, title=title)
    if oracle is not None:
        row["oracle"] = oracle
    return row


def part(title, file, func, requirement, *hints):
    return dict(title=title, file=file + ".py", funcs=[func], requirement=requirement, hints=list(hints))


def example(request, answer, why):
    return dict(input=request, output=answer, why=why)


def modules(**files):
    return {name + ".py": dedent(source).strip() + "\n" for name, source in files.items()}


def source(title, url):
    return dict(title=title, url=url)


def run(value, reads, error=None):
    return dict(value=value, reads=reads, error=error)


def expr(tree, **bindings):
    return dict(expr=tree, env=bindings)


def read(name):
    return ["read", name]


def program(*instructions):
    return list(instructions) + [["HALT"]]


lazy_or = expr(["or", read("cache"), read("backend")], cache=7)
lazy_if = expr(["if", read("ready"), ["add", 2, read("x")], ["and", read("quota"), read("cost")]],
               ready=0, quota=-2, cost=4)

LABS = [dict(
    slug="sys-bytecode-vm", title="Compile short-circuit expressions to a stack VM",
    group="Compilers and runtimes", level="Intermediate", duration="90–120 min",
    mission="A rule engine charges for every variable lookup. A cached result should skip a backend lookup, yet an eager compiler still reads it and can crash on an unbound name. Implement the bytecode compiler and virtual machine. The access log must show exactly which reads ran.",
    inputs="A request has expr and env. A tree is an integer, [\"read\", name], [\"add\", a, b], [\"and\", a, b], [\"or\", a, b], or [\"if\", condition, then_expr, else_expr]. Each operand is another tree. env maps strings to integer values. env may omit names in the tree.",
    outputs="Return {value, reads, error}. On success value is the integer result and error is null. Append each name to reads at the instant it is looked up, including repeated and missing reads. On the first missing lookup, stop and return null for value and the string 'unbound:' followed immediately by the name for error.",
    constraints="Zero is false and every other integer is true. and/or return the selected operand's integer, including its sign. add evaluates left first. if evaluates only its selected arm. Inputs have at most 256 AST nodes and depth at most 40. All bytecode supplied directly to the VM follows the instruction contract in isa.py, with valid forward targets, balanced stacks, and one HALT. Only a missing READ can fail.",
    scope="Implement the backend of a small expression language. A caller supplies a parsed JSON tree. The debugger and unit tests use the canonical instruction layout in isa.py, so branch targets and the placement of DROP are part of the compiler interface.",
    source=[source("Robert Nystrom · Jumping Back and Forth", "https://craftinginterpreters.com/jumping-back-and-forth.html"),
            source("LLVM tutorial · Extending the Language with Control Flow", "https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl05.html")],
    parts=[
        part("Lower expressions and patch jumps", "compiler", "compile_expr",
             "Emit one flat sequence using the canonical lowering table in isa.py. Use absolute instruction indices. Every expression leaves one result and the entire program ends with one HALT.",
             "Emit a placeholder before compiling a nested arm. Its destination is known when the arm is complete.",
             "The conditional jumps for and/or leave the left operand on the stack. Emit DROP only along the fallthrough path."),
        part("Execute the bytecode and record reads", "machine", "execute",
             "Run the independent bytecode stream from instruction zero. Apply the stack and branch effects in isa.py, and preserve the read log even when an unbound lookup stops execution.",
             "The if opcode consumes its condition on both paths. The and/or jump opcodes only peek.",
             "A branch target is an instruction index. Account for any automatic program-counter increment before assigning it.")],
    modules=modules(
        isa="""
            '''Instruction set and canonical lowering used by the debugger.

            Instructions are lists. Targets are absolute zero-based instruction indices. The
            first instruction is index 0. Unless a jump is taken, advance by one.
              PUSH n           push n
              READ name        log name, then push env[name] or stop with an error
              ADD              pop the right, then the left integer; push their sum
              DROP             pop and discard the top integer
              JZ_KEEP target   jump if top == 0; leave top unchanged
              JNZ_KEEP target  jump if top != 0; leave top unchanged
              JZ_POP target    pop and jump if the popped value is zero
              JMP target       jump unconditionally
              HALT             return the only stack item

            Emit PUSH for an integer and READ for a variable. Lower add as L, R, ADD.
            Lower and as L, JZ_KEEP end, DROP, R, end. Lower or identically with JNZ_KEEP.
            Lower if as C, JZ_POP otherwise, T, JMP end, otherwise: F, end.
            Labels mark boundaries; never emit a label instruction. A single HALT follows
            the outermost expression. Do not fold constants or remove unreachable branches:
            the debugger indexes every source arm.
            '''
        """,
        compiler="""
            def compile_expr(tree):
                '''Return the canonical instruction list for one AST, with absolute jump targets and a final HALT. See isa.py.'''
                code = []

                def emit(node):
                    if isinstance(node, int):
                        code.append(["PUSH", node])
                        return
                    op = node[0]
                    if op == "read":
                        code.append(["READ", node[1]])
                        return
                    emit(node[1])
                    if op == "add":
                        emit(node[2])
                        code.append(["ADD"])
                    elif op in ("and", "or"):
                        branch = len(code)
                        code.append(["JZ_KEEP" if op == "and" else "JNZ_KEEP", None])
                        code.append(["DROP"])
                        emit(node[2])
                        code[branch][1] = len(code)
                    else:
                        skip_then = len(code)
                        code.append(["JZ_POP", None])
                        emit(node[2])
                        skip_else = len(code)
                        code.append(["JMP", None])
                        code[skip_then][1] = len(code)
                        emit(node[3])
                        code[skip_else][1] = len(code)

                emit(tree)
                code.append(["HALT"])
                return code
        """,
        machine="""
            def execute(code, env):
                '''Execute the instruction stream. Return value, ordered reads, and error as specified, including failed reads.'''
                stack = []
                reads = []
                pc = 0
                while True:
                    op, *args = code[pc]
                    pc += 1
                    if op == "PUSH":
                        stack.append(args[0])
                    elif op == "READ":
                        name = args[0]
                        reads.append(name)
                        if name not in env:
                            return {"value": None, "reads": reads, "error": "unbound:" + name}
                        stack.append(env[name])
                    elif op == "ADD":
                        right = stack.pop()
                        stack[-1] += right
                    elif op == "DROP":
                        stack.pop()
                    elif op == "JMP":
                        pc = args[0]
                    elif op == "JZ_POP":
                        if stack.pop() == 0:
                            pc = args[0]
                    elif op == "JZ_KEEP":
                        if stack[-1] == 0:
                            pc = args[0]
                    elif op == "JNZ_KEEP":
                        if stack[-1] != 0:
                            pc = args[0]
                    elif op == "HALT":
                        return {"value": stack.pop(), "reads": reads, "error": None}
        """,
        gateway="""
            '''Compile a rule once, then evaluate its bytecode against the caller's environment.'''
            from compiler import compile_expr
            from machine import execute

            def handle(request):
                return execute(compile_expr(request["expr"]), request["env"])
        """),
    examples=[
        example(lazy_or, run(7, ["cache"]),
                "The first lookup yields 7. The VM's conditional jump goes straight to HALT, preserving 7. The missing backend binding is never read."),
        example(lazy_if, run(4, ["ready", "quota", "cost"]),
                "ready is zero, so the VM jumps past the add expression. The else arm reads quota. Its negative value is truthy, so and discards it and evaluates cost.")],
    checks=[
        check(1, "compiler.compile_expr", [["and", read("a"), ["add", 1, read("b")]]], "Patch a jump over an addition",
              program(["READ","a"], ["JZ_KEEP",6], ["DROP"], ["PUSH",1], ["READ","b"], ["ADD"])),
        check(1, "compiler.compile_expr", [["if", 0, ["and", 0, 3], ["or", 1, 2]]], "Nested branch ends at the outer jump",
              program(["PUSH",0], ["JZ_POP",7], ["PUSH",0], ["JZ_KEEP",6], ["DROP"], ["PUSH",3], ["JMP",11],
                      ["PUSH",1], ["JNZ_KEEP",11], ["DROP"], ["PUSH",2])),
        check(2, "machine.execute", [program(["READ","q"], ["JNZ_KEEP",4], ["DROP"], ["READ","r"]), {"q":-8}],
              "A taken branch preserves the original signed operand", run(-8, ["q"])),
        check(2, "machine.execute", [program(["PUSH",0],["JZ_POP",4],["PUSH",9],["JMP",5],["READ","absent"]), {}],
              "A missing read on a taken branch stops immediately",run(None,["absent"],"unbound:absent")),
        check(2, "gateway.handle", [expr(["and", read("x"), read("missing")], x=0)], "Falsy cache suppresses missing read",run(0,["x"])),
        check(2, "gateway.handle", [expr(["add",read("x"),["if",read("x"),read("x"),read("y")]], x=-3)], "Only selected arm can repeat the read",run(-6,["x","x","x"])),
        check(2, "gateway.handle", [expr(["or",0,["if",read("q"),read("bad"),read("safe")]], q=1,safe=1)], "First reachable fault captures partial trace",run(None,["q","bad"],"unbound:bad")),
        check(2, "gateway.handle", [expr(["if",["and",0,read("z")],read("x"),["add",2,["or",0,4]]])], "Nested branches landing on ADD",run(6,[])),
        check(2, "gateway.handle", [expr(["add", read("first"), read("last")])], "Evaluation stops before the right operand", run(None,["first"],"unbound:first"))
    ])]


def block(body, *end):
    return dict(body=body, end=list(end))


def cfg(blocks, env=None, entry="entry", budget=32):
    return dict(blocks=blocks, env=env or {}, entry=entry, budget=budget)


def observation(value, trace, status="return"):
    return dict(status=status, value=value, trace=trace)


def report(kept, live_in, value, trace, status="return"):
    return dict(kept=kept, live_in=live_in, result=observation(value, trace, status))


dead_chain = cfg({
    "entry":block([["const","tmp",2], ["add","stale","tmp","p"], ["const","x",3], ["emit","x"]], "jump","exit"),
    "exit":block([["copy","unused","stale"]], "ret","x")
}, {"p":7})
countdown = cfg({
    "entry":block([["const","n",2],["const","step",-1],["const","z",0]], "jump","loop"),
    "loop":block([["emit","n"],["add","n","n","step"]], "br","n","loop","exit"),
    "exit":block([], "ret","n")
})
trap = cfg({"entry":block([["emit","x"], ["div","ignored","x","y"], ["emit","y"]], "ret","x")}, {"x":4,"y":0})
diamond = cfg({
    "entry":block([["const","x",6],["const","y",9]],"br","flag","a","b"),
    "a":block([["copy","v","x"]],"jump","join"),
    "b":block([["copy","v","y"]],"jump","join"),
    "join":block([["emit","v"]],"ret","v")
}, {"flag":0})
overwrites = cfg({"entry":block([["const","x",1], ["const","x",2], ["copy","x","x"], ["emit","x"]], "ret","x")})
bounded = cfg({"entry":block([["emit","p"], ["const","unused",2]], "jump","entry")}, {"p":5}, budget=3)


LABS.append(dict(
    slug="sys-live-variable-optimizer", title="Remove dead assignments across a control-flow graph",
    group="Compilers and runtimes", level="Advanced", duration="110–150 min",
    mission="A compiler deletes an accumulator that a loop reads on its next iteration. Its cleanup pass also keeps whole chains that feed a single unused value. Implement backward liveness and an optimizer over a small imperative IR, then run the optimized program in the supplied interpreter. An emitted event or a division error is observable even if the destination is never read.",
    inputs="blocks maps labels to {body, end}. A body is an ordered list of instructions [\"const\", dst, int], [\"copy\", dst, src], [\"add\", dst, a, b], [\"div\", dst, a, b], or [\"emit\", src]. A terminator is [\"ret\", src], [\"jump\", label], or [\"br\", cond, true_label, false_label]. entry names the first block, env maps incoming variables to integers, and budget is the maximum number of block visits.",
    outputs="Return kept, live_in, and result. kept maps every block to the original zero-based body indices retained in original order. live_in maps every block of the final program to its sorted list of live variables at entry. result is {status, value, trace}; status is 'return', 'division_by_zero', or 'budget'. value is null unless the program returns. trace lists the integer values emitted before stopping.",
    constraints="All labels and reachable reads are defined. Inputs have 1–40 blocks, 0–40 instructions per block, and a visit budget of 1–256. Arithmetic is integer arithmetic. Division floors toward negative infinity and traps on zero. br takes its true edge for a nonzero value. On budget exhaustion stop before entering the next block. Analyse every given block and both edges of br, including untaken edges.",
    scope="Implement the classical least fixed point for liveness. At each optimization round, take block live-outs from that round's unmodified CFG, scan bodies backward, and drop pure writes to dead variables without adding their operands to liveness. Recompute after pruning until no instruction is removed. Keep every div, emit, and terminator. This defines the required output even for dead cycles of mutually dependent assignments.",
    source=[source("Cornell CS 4120 · Live Variable Analysis","https://www.cs.cornell.edu/courses/cs4120/2023sp/notes/livevar/"),
            source("Cornell CS 6120 · Data Flow","https://www.cs.cornell.edu/courses/cs6120/2020fa/lesson/4/")],
    parts=[
        part("Solve backward liveness through cycles","liveness","solve",
             "Compute live-in on the given CFG before any pruning. At a block exit take the union of successor live-ins. Include terminator reads, then process each instruction backward, killing its destination before adding its reads. Return every block's live-in as a sorted list.",
             "Use the supplied successor, terminator-read, and operand helpers in ir.py. Start each set empty.",
             "At x = x + step the old x is read before the new x is written. A single reverse pass across blocks misses some back edges."),
        part("Prune across blocks without changing observable behavior","optimizer","retained",
             "Return the surviving original instruction indices by block. For each round solve liveness on the current program, prune backward from each block's live-out and terminator reads, then repeat after any removal. Effectful operations must survive even when their destination is dead.",
             "Keep instruction indices from the initial program. Index 0 after a rewrite may have been index 4 originally.",
             "If a pure operation disappears, its operands no longer need to be read. Removing a use in a successor can expose another deletion in a predecessor.")],
    modules=modules(
        ir="""
            '''Utilities shared by analysis, optimizer and execution.'''
            def successors(end):
                if end[0] == "jump":
                    return end[1:]
                if end[0] == "br":
                    return end[2:]
                return []

            def end_reads(end):
                return {end[1]} if end[0] in ("br", "ret") else set()

            def operands(inst):
                '''Return the defined name (or None) and the set of names read.'''
                op = inst[0]
                if op == "emit":
                    return None, {inst[1]}
                if op == "const":
                    return inst[1], set()
                return inst[1], set(inst[2:])

            def select(blocks, kept):
                return {label: {"body": [block["body"][i] for i in kept[label]], "end": block["end"]}
                        for label, block in blocks.items()}
        """,
        liveness="""
            from ir import end_reads, operands, successors

            def solve(blocks):
                '''Return sorted live-in variable lists by block, at the least fixed point for the given bodies.'''
                incoming = {label: set() for label in blocks}
                changed = True
                while changed:
                    changed = False
                    for label, block in blocks.items():
                        live = end_reads(block["end"])
                        for following in successors(block["end"]):
                            live.update(incoming[following])
                        for inst in reversed(block["body"]):
                            name, reads = operands(inst)
                            live.discard(name)
                            live.update(reads)
                        if live != incoming[label]:
                            incoming[label] = live
                            changed = True
                return {label: sorted(values) for label, values in incoming.items()}
        """,
        optimizer="""
            from ir import end_reads, operands, select, successors
            from liveness import solve

            def retained(blocks):
                '''Return original zero-based body indices remaining after repeated liveness-based dead assignment elimination.'''
                kept = {label: list(range(len(block["body"]))) for label, block in blocks.items()}
                while True:
                    current = select(blocks, kept)
                    incoming = solve(current)
                    next_kept = {}
                    for label, block in current.items():
                        live = end_reads(block["end"])
                        for following in successors(block["end"]):
                            live.update(incoming[following])
                        surviving = []
                        for index, inst in reversed(list(zip(kept[label], block["body"]))):
                            name, reads = operands(inst)
                            if inst[0] in ("const", "copy", "add") and name not in live:
                                continue
                            surviving.append(index)
                            live.discard(name)
                            live.update(reads)
                        next_kept[label] = list(reversed(surviving))
                    if next_kept == kept:
                        return kept
                    kept = next_kept
        """,
        interpreter="""
            '''Executable semantics. The budget counts block entries, independent of body lengths.'''
            def run(blocks, entry, initial, budget):
                env = dict(initial)
                trace = []
                label = entry
                for _ in range(budget):
                    block = blocks[label]
                    for inst in block["body"]:
                        op, *args = inst
                        if op == "emit":
                            trace.append(env[args[0]])
                        elif op == "const":
                            env[args[0]] = args[1]
                        elif op == "copy":
                            env[args[0]] = env[args[1]]
                        elif op == "add":
                            env[args[0]] = env[args[1]] + env[args[2]]
                        elif op == "div":
                            if env[args[2]] == 0:
                                return {"status": "division_by_zero", "value": None, "trace": trace}
                            env[args[0]] = env[args[1]] // env[args[2]]
                    op, *args = block["end"]
                    if op == "ret":
                        return {"status": "return", "value": env[args[0]], "trace": trace}
                    if op == "jump":
                        label = args[0]
                    else:
                        label = args[1] if env[args[0]] != 0 else args[2]
                return {"status": "budget", "value": None, "trace": trace}
        """,
        gateway="""
            '''Run the optimizer, attach its analysis and execute the retained IR.'''
            from ir import select
            from interpreter import run
            from liveness import solve
            from optimizer import retained

            def handle(request):
                kept = retained(request["blocks"])
                optimized = select(request["blocks"], kept)
                result = run(optimized, request["entry"], request["env"], request["budget"])
                return {"kept": kept, "live_in": solve(optimized), "result": result}
        """),
    examples=[
        example(dead_chain,report({"entry":[2,3],"exit":[]},{"entry":[],"exit":["x"]},3,[3]),
                "Removing exit's unused copy makes stale dead in the predecessor. On a later pass the add disappears, and hence so does the constant that feeds it. The emit and return still see 3."),
        example(countdown,report({"entry":[0,1],"loop":[0,1],"exit":[]},{"entry":[],"loop":["n","step"],"exit":["n"]},0,[2,1]),
                "On each loop visit emit reads the current n before add writes its replacement. The back edge carries both n and step. The unused z initialization disappears. After emitting 2 and 1 the branch exits with n equal to zero.")],
    checks=[
        check(1,"liveness.solve",[{"entry":block([["add","x","x","s"]],"br","x","entry","done"),"done":block([],"ret","x")}],
              "A loop-carried variable reads its old value",{"entry":["s","x"],"done":["x"]}),
        check(1,"liveness.solve",[diamond["blocks"]],"Merge both possible successors",{"entry":["flag"],"a":["x"],"b":["y"],"join":["v"]}),
        check(1,"liveness.solve",[dead_chain["blocks"]],"Standard liveness still counts operands of an unpruned dead assignment",
              {"entry":["p"],"exit":["stale","x"]}),
        check(2,"optimizer.retained",[dead_chain["blocks"]],"Eliminate transitive cross-block dead uses",{"entry":[2,3],"exit":[]}),
        check(2,"optimizer.retained",[trap["blocks"]],"Discarding an unused quotient must not erase its trap",{"entry":[0,1,2]}),
        check(2,"gateway.handle",[trap],"Preserve the emitted prefix before a trap", report({"entry":[0,1,2]},{"entry":["x","y"]},None,[4],"division_by_zero")),
        check(2,"gateway.handle",[diamond],"Untaken branch still requires a reaching definition",
              report({"entry":[0,1],"a":[0],"b":[0],"join":[0]}, {"entry":["flag"],"a":["x"],"b":["y"],"join":["v"]},9,[9])),
        check(2,"gateway.handle",[overwrites],"An overwritten initialization drops out before a self-copy",
              report({"entry":[1,2,3]},{"entry":[]},2,[2])),
        check(2,"gateway.handle",[bounded],"Infinite control flow keeps its effects at the visit budget",
              report({"entry":[0]},{"entry":["p"]},None,[5,5,5],"budget")),
        check(2,"gateway.handle",[cfg({"entry":block([["div","q","x","y"],["add","unused","q","x"]],"ret","q")},{"x":-5,"y":2})],
              "Floor division and dead suffix",report({"entry":[0]},{"entry":["x","y"]},-3,[]))
    ]))
