from events import Event

class OutcomeLedger:
    def __init__(self):
        self._entries = {}
        self._prompts = set()

    def ingest(self, event: Event):
        self._prompts.add(event.prompt)
        key = (event.prompt, event.model)
        previous = self._entries.get(key)
        if previous is None or event.revision > previous.revision:
            self._entries[key] = event

    def latest(self, prompt, model):
        event = self._entries.get((prompt, model))
        return event.correct if event else None

    def prompts(self):
        return frozenset(self._prompts)

    def pair(self, prompt):
        values = (self.latest(prompt, "baseline"), self.latest(prompt, "candidate"))
        return None if None in values else values
