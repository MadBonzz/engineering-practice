ROLE = {"system": 1, "user": 2, "assistant": 3}
END = 4
PAD = 0
IGNORE = -100


def encode_turn(role, tokens):
    """Frame tokenized content as [role_id, *tokens, END]. Return [ids, labels] of equal length. Ignore every role marker. Only assistant content and its END are prediction targets. Constants are supplied."""
    ids = [ROLE[role], *tokens, END]
    labels = [IGNORE, *tokens, END] if role == "assistant" else [IGNORE]*len(ids)
    return [ids, labels]


def encode_chat(turns, limit):
    """Concatenate encoded [role, tokens] turns and keep the first limit tokens (limit >= 0). Return [ids, labels] without padding or shifting; truncation must not invent an end marker."""
    ids, labels = [], []
    for role, tokens in turns:
        seq, targets = encode_turn(role, tokens)
        ids.extend(seq)
        labels.extend(targets)
    return [ids[:limit], labels[:limit]]


def collate(chats, limit):
    """Right-pad encoded conversations to the longest retained length, returning a dict with input_ids, labels, attention_mask. Masks use 1 for real tokens and 0 for padding. Padding has IGNORE labels. Empty batch has three empty lists. Preserve row order."""
    rows = [encode_chat(chat, limit) for chat in chats]
    width = max((len(ids) for ids, _ in rows), default=0)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for ids, labels in rows:
        gap = width-len(ids)
        batch["input_ids"].append(ids+[PAD]*gap)
        batch["labels"].append(labels+[IGNORE]*gap)
        batch["attention_mask"].append([1]*len(ids)+[0]*gap)
    return batch

# AUTHORING
TITLE = "Chat supervised fine-tuning"
GROUP = "Training"
INTRO = "A fine-tuning job is learning to echo user prompts. Build the data path from tokenized chat turns to a labeled training batch, and inspect exactly which tokens receive loss."
PARTS = ["encode_turn", "encode_chat", "collate"]
LABELS = ["Label a single turn", "Frame a conversation", "Collate a training batch"]
HINTS = ["Which speaker owns the end marker?|A role marker has context but no loss.", "Try truncating inside an answer.|IDs and targets must remain aligned.", "An ignored target can still be visible to attention.|Use the retained lengths, including empty rows."]
def cases(n):
    role = ["assistant","user","system"][n%3]
    a = [["user",[9,8]],["assistant",[11+n,12]],["user",[]],["assistant",[9]]]
    b = [["assistant",[]],["user",[12]]]
    limit = [6,12,0,3,4,1,7,9,20][n%9]
    return [[role,[10+n]* (n%4)], [a,limit], [[a,b,[]] if n!=7 else [],limit]]
MUTANTS = [('role == "assistant"','role != "user"'),("[IGNORE, *tokens, END]","[IGNORE, *tokens, IGNORE]"),("ids[:limit], labels[:limit]","ids[:limit+1], labels[:limit+1]"),("labels+[IGNORE]*gap","labels+[PAD]*gap")]

