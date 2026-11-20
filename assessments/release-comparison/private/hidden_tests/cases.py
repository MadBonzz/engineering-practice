from random import Random

def workloads():
    yield 1, "Missing observations", [["get","ghost","candidate"], ["prompts"]]
    for revs in ([2, 0, 1], [0, 2, 2, 1, 0], [1, 0, 2, 1]):
        ops = []
        for rev in revs:
            ops.extend([["write","a","baseline",rev,[True,False,None][rev]],["get","a","baseline"],["prompts"]])
        yield 1, "Updates and withdrawals", ops
    for i in range(18):
        rng = Random(118+i)
        stage = 1 if i < 6 else 2 if i < 12 else 3
        prompts = ["a", "b", "c", "d", "e"]
        ops = []
        for step in range(18 + i):
            key = rng.choice(prompts)
            model = rng.choice(["baseline", "candidate"])
            rev = rng.randrange(4)
            label = [False, True, None][(prompts.index(key) + (model == "baseline") + rev)%3]
            ops.append(["write",key,model,rev,label])
            ops.append(["get",key,model] if stage == 1 else ["pair",key] if stage == 2 else ["report",1 + step%4])
        yield stage, ["","Updates in delivery order","Comparable outcomes","Release decision over time"][stage], ops
    for baseline in (False,True,None):
        for candidate in (False,True,None):
            ops = [["write","x","baseline",0,baseline],["write","x","candidate",1,candidate]]
            yield 2, "Model order and missing labels", ops + [["pair","x"],["pair","unknown"]]
            yield 3, "Decision threshold", ops + [["report",1],["report",2]]
    yield 3,"Tied decision", [["write","a","baseline",0,False],["write","a","candidate",0,True],["write","b","baseline",0,True],["write","b","candidate",0,False],["report",2],["write","b","candidate",1,True],["report",2]]
