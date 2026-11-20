"""Public input contract and fixture loader for evaluation workers."""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Event:
    prompt: str
    model: str
    revision: int
    correct: Optional[bool]

def read_events(rows):
    """Rows are [prompt, model, revision, correct] in delivery order."""
    return [Event(*row) for row in rows]
