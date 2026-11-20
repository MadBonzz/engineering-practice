def new_state(prompt):
    """Prefill the supplied recurrent token model."""
    state = 0
    for token in prompt:
        state = (state*3 + token + 1) % 11
    return state


def model_batch(states):
    """One real batched greedy step of a tiny deterministic model. Returns (token, new_state) pairs."""
    result = []
    for state in states:
        token = max(range(5), key=lambda t: ((state + 2)*(t + 1) + t*t) % 13)
        result.append((token, (state*3 + token + 1) % 11))
    return result


def step(rows):
    """Rows are [request_id, state, remaining]. Make one model_batch call for the whole nonempty batch. Return [id, token, new_state, remaining_after] in the input order. Empty input returns []."""
    if not rows:
        return []
    outputs = model_batch([row[1] for row in rows])
    return [[rid, token, state, remaining-1] for (rid, _, remaining), (token, state) in zip(rows, outputs)]


def tick(active, waiting, capacity):
    """Active and waiting rows have the same shape as step(). Fill vacant slots FIFO behind existing active rows, then advance all admitted rows once. Return [surviving_rows, still_waiting_rows, emitted_pairs]. Surviving rows contain the updated state and remaining count. Pairs are [id, token] in batch order. Finished slots become available next tick."""
    count = capacity-len(active)
    advanced = step(active + waiting[:count])
    survivors = [[rid, state, remaining] for rid, _, state, remaining in advanced if remaining > 0]
    return [survivors, waiting[count:], [[rid, token] for rid, token, _, _ in advanced]]


def dispatch(requests, capacity):
    """Requests are [arrival_tick, id, prompt_tokens, positive_budget], in arrival order. Starting at tick zero, append arrivals before admission. Prefill each once using new_state. Return the list of emitted-pair lists per tick including idle ticks, ending at final emission. Empty input returns []. Capacity is positive."""
    active, waiting, timeline = [], [], []
    cursor = 0
    while cursor < len(requests) or waiting or active:
        while cursor < len(requests) and requests[cursor][0] == len(timeline):
            _, rid, prompt, budget = requests[cursor]
            waiting.append([rid, new_state(prompt), budget])
            cursor += 1
        active, waiting, emitted = tick(active, waiting, capacity)
        timeline.append(emitted)
    return timeline

# AUTHORING
TITLE = "Live continuous batching"
GROUP = "Inference"
INTRO = "Your serving worker receives requests at different times. Wire the supplied model to a scheduler that admits new work while longer generations are still running."
PARTS = ["step", "tick", "dispatch"]
LABELS = ["Execute one model batch", "Admit and retire requests", "Serve the arrival stream"]
HINTS = ["Match outputs back to the original rows.|The step is shared by all admitted requests.", "Which tick can reuse a finished slot?|Keep the waiting order when capacity is exhausted.", "Include ticks before the first arrival.|A later arrival must see the queue left by the previous tick."]
def cases(n):
    rows = [[str(i), (3*i+n)%11, 1+(n+i)%3] for i in range(n%4+1)]
    return [[rows if n!=4 else []], [rows[:1],rows[1:]+[["z",7,1]],1+n%3],
            [[[n%3,"a",[n,2],1+n%4],[n%3,"b",[1],2],[n%3+1,"c",[],2],[n%3+4,"d",[4],1]] if n!=5 else [], 1+n%3]]
MUTANTS = [("remaining-1","remaining"),("active + waiting[:count]","waiting[:count] + active"),("remaining > 0","remaining > 1"),("new_state(prompt), budget","0, budget")]

