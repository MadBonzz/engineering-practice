def effect_key(session,turn):
    """Return an unambiguous stable tool idempotency key for string session and turn IDs, even if either contains ':'. Format: decimal character length of session, ':', session, then turn (no second separator)."""
    return f"{len(session)}:{session}{turn}"


def resume(log, external, session, turn, answer, stop):
    """One retryable evaluation turn. log is a mutable write-ahead journal of [key, 'intent'/'done', value]. external is a mutable idempotent tool store: key -> recorded result. Existing done returns its result. Otherwise persist intent with original answer once (first answer wins), then use the key at the tool boundary. The deterministic tool result is 'score:' plus that original answer. Persist done once. stop='intent' returns None after ensuring intent; stop='tool' returns None after the effect but before done; stop='finish' completes. Existing done always wins. Preserve unrelated records; never repeat or replace existing effects."""
    key = effect_key(session,turn)
    records = [row for row in log if row[0]==key]
    for _,status,value in records:
        if status=="done":
            return value
    if records:
        answer = records[0][2]
    else:
        log.append([key,"intent",answer])
    if stop=="intent":
        return None
    if key not in external:
        external[key] = "score:"+answer
    if stop=="tool":
        return None
    log.append([key,"done",external[key]])
    return external[key]


def replay(attempts):
    """Start with an empty durable journal and external tool. Each attempt is [session,turn,answer,stop]; only those two stores survive between attempts. Return {replies, journal, effects} in execution order. Keyed effect storage stands in for an external tool that supports idempotency keys."""
    log,external,replies = [],{},[]
    for attempt in attempts:
        replies.append(resume(log,external,*attempt))
    return {"replies":replies,"journal":log,"effects":external}

# AUTHORING
TITLE = "Durable tool execution and replay"
GROUP = "Agent engineering"
INTRO = "A mock-interview evaluator can crash after the remote tool finishes but before the turn is saved. Restore interrupted turns, preserve the first answer, and make retries safe across sessions."
PARTS = ["effect_key","resume","replay"]
LABELS = ["Identify an effect","Recover across crash points","Replay interleaved turns"]
HINTS = ["Session and turn boundaries must be recoverable.|Try two different splits of the same concatenation.", "What survives a crash after the remote write?|Look for a durable intent before accepting a new answer.", "Interleave two turns that look similar.|A finished turn must remain finished on a retry."]
def cases(n):
    s,t = ["a:","a","", "α:"][n%4],str(n)
    k = effect_key(s,t)
    log = [] if n%3==0 else [[k,"intent","original"]]
    ext = {k:"score:original"} if n%3==2 else {}
    history = [[s,t,"one","intent"],[s+"x",t,"other","finish"],[s,t,"two","tool"],[s,t,"three","finish"],[s,t,"four","tool"]]
    return [[s,t],[log,ext,s,t,"new",["finish","intent","tool"][n%3]],[history]]
MUTANTS = [('f"{len(session)}:{session}{turn}"','session+":"+turn'),('answer = records[0][2]','answer = answer'),('if stop=="tool":','if stop=="never":'),('if status=="done":','if status=="intent":')]

