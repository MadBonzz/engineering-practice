"""Accumulate evaluation outcomes as worker updates arrive."""
from events import Event


class OutcomeLedger:
    def __init__(self):
        self._prompts = set()
        # Add any state needed by the public methods below.

    def ingest(self, event: Event) -> None:
        self._prompts.add(event.prompt)
        # TODO: Record the update according to the delivery contract.
        raise NotImplementedError("OutcomeLedger.ingest")

    def latest(self, prompt: str, model: str):
        # TODO: Return the current outcome for this prompt and model.
        raise NotImplementedError("OutcomeLedger.latest")

    def prompts(self):
        """Return a read-only snapshot of all observed prompt IDs."""
        return frozenset(self._prompts)

    def pair(self, prompt: str):
        # TODO: Return the two comparable outcomes when available.
        raise NotImplementedError("OutcomeLedger.pair")
