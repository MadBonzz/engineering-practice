def prefix_length(a,b):
    """Return the number of consecutive matching token IDs from position zero."""
    count = 0
    for left,right in zip(a,b):
        if left != right:
            break
        count += 1
    return count


def choose(entries, model, tokens):
    """Saved entries are [model_version, token_ids, kv_values], with one opaque KV value per token. Return the longest reusable KV prefix from a matching model. Partial entry reuse is valid. Earlier entry wins a length tie. Return [] on a miss; edits to the returned list must not change an entry."""
    import copy
    best = []
    for version, ids, values in entries:
        n = prefix_length(ids,tokens)
        if version == model and n > len(best):
            best = copy.deepcopy(values[:n])
    return best


def run(events):
    """Events: ['save',model,tokens,values] adds an immutable snapshot; ['open',id,model,tokens] initializes this request to choose()'s reusable values only. ['append',id,value] adds a locally computed KV value; ['cancel',id] drops a request. These return null. ['read',id] returns its current values or null if absent. Reads are snapshots. Return one result per event; appends target open requests."""
    entries, active, out = [], {}, []
    for op,*args in events:
        value = None
        if op == "save":
            entries.append(args)
        elif op == "open":
            rid,model,ids = args
            active[rid] = choose(entries,model,ids)
        elif op == "append":
            active[args[0]].append(args[1])
        elif op == "cancel":
            active.pop(args[0],None)
        elif op == "read":
            value = active[args[0]][:] if args[0] in active else None
        out.append(value)
    return out

# AUTHORING
TITLE = "Shared-prefix caching"
GROUP = "Inference"
INTRO = "Several requests start with the same system prompt. Reuse compatible KV states while keeping each continuation and saved prefix independent."
PARTS = ["prefix_length","choose","run"]
LABELS = ["Measure reusable tokens","Select a compatible snapshot","Isolate concurrent continuations"]
HINTS = ["Only the initial run of equal tokens is reusable.|Try a mismatch followed by equal suffixes.", "A matching token sequence can belong to a different model.|Track the best prefix, including ties.", "Read before and after an append.|One request ending must leave the others untouched."]
def cases(n):
    ids = [n,2,3,5]
    q = [n,2,7,5] if n%2==0 else [n,2,3,5,4]
    entries = [["v2",q,[0]*len(q)],["v1",ids,[1,2,3,4]],["v1",ids,[1,2,3,4]]]
    events = [["save",*row] for row in entries]+[["open","a","v1",q],["open","b","v1",q],["read","a"],["append","a",9],["read","b"],["cancel","a"],["read","a"],["read","b"],["open","c","v3",q],["read","c"]]
    return [[ids,q],[entries,"v1" if n%4 else "v3",q],[events]]
MUTANTS = [("break","continue"),("version == model and ",""),("values[:n]","values[:]"),("active.pop(args[0],None)","active.clear()")]

