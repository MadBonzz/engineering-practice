def replay_scenario(actions):
    from ledger import OutcomeLedger
    from events import Event
    from report import summarize
    ledger = OutcomeLedger()
    output = []
    for op, *args in actions:
        if op == "write":
            ledger.ingest(Event(*args))
        elif op == "get":
            output.append(ledger.latest(*args))
        elif op == "pair":
            value = ledger.pair(*args)
            if value is not None and not isinstance(value, tuple):
                raise TypeError("pair must return a tuple or None")
            output.append(None if value is None else list(value))
        elif op == "prompts":
            output.append(sorted(ledger.prompts()))
        elif op == "report":
            output.append(summarize(ledger, *args))
    return output
