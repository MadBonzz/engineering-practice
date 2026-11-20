def reserve(table, blocks, length, extra, block_size):
    """Reserve room for extra tokens of this request. table is its ordered list of physical block IDs, blocks the sorted free IDs, length its used token count. Return [new_table, new_free, ok]. Keep old blocks; allocate lowest free IDs first, only as needed. Failure leaves both lists unchanged. Return copies or equivalent values."""
    target = (length+extra+block_size-1)//block_size
    need = max(0, target-len(table))
    if need > len(blocks):
        return [table,blocks,False]
    return [table+blocks[:need],blocks[need:],True]


def locate(table, offset, block_size):
    """Return [physical_block, slot] for an allocated zero-based logical token offset. Caller supplies a valid offset."""
    return [table[offset//block_size],offset%block_size]


def replay(block_count, block_size, events):
    """Initially all numbered blocks are free. 'append', id, n reserves atomically and returns the physical addresses of n new tokens; return null on capacity failure. 'release', id frees the entire request and returns null (unknown is a no-op). 'free' returns sorted free IDs. Return one output per event. A released ID can start again at length zero."""
    free = list(range(block_count))
    state, replies = {}, []
    for op,*args in events:
        if op == "free":
            replies.append(free[:])
        elif op == "release":
            table,_ = state.pop(args[0],([],0))
            free = sorted(free+table)
            replies.append(None)
        else:
            rid,n = args
            table,length = state.get(rid, ([],0))
            table,free,ok = reserve(table,free,length,n,block_size)
            if ok:
                state[rid] = (table,length+n)
            replies.append([locate(table,i,block_size) for i in range(length,length+n)] if ok else None)
    return replies

# AUTHORING
TITLE = "Paged KV-cache allocator"
GROUP = "Inference"
INTRO = "Requests grow one token at a time in a fixed physical memory pool. Implement reservation, logical addressing, and reuse when requests finish out of order."
PARTS = ["reserve","locate","replay"]
LABELS = ["Reserve capacity atomically","Translate a token address","Replay allocation traffic"]
HINTS = ["Can the tail block already hold the growth?|Compute total capacity before changing the free list.", "Logical and physical addresses can have different order.|Think about the first token of the next page.", "Probe free space after a failed append.|Trace two requests sharing the same free pool."]
def cases(n):
    size = 1+n%4
    return [[[4],[0,2,5],size,n%9,size],[[3,0,2], n%(3*size),size],
      [3,size,[["append","a",1],["append","b",size],["append","a",size],["append","c",4*size],["free"],["release","b"],["append","c",size],["release","a"],["release","a"],["append","a",n%4],["release","c"],["free"]]]]
MUTANTS = [("length+extra+block_size-1","length+extra"),("need > len(blocks)","need >= len(blocks)"),("offset%block_size","offset//block_size"),("sorted(free+table)","free+table")]

