"""Repository components and restart exercise for the durable tool lab."""

JOURNAL = '''"""Adapter over the append-only records that survive a worker restart."""


class TurnJournal:
    def __init__(self, rows):
        self.rows = rows

    def read(self, key):
        """Return {event_type: value} for the given turn, including empty strings."""
        return {event: value for stored, event, value in self.rows if stored == key}

    def append(self, key, event, value):
        """Append one durable record. The caller decides whether a write is due."""
        self.rows.append([key, event, value])
'''

TOOL = '''"""Supplied remote scoring service. Its records outlive the caller."""


class ScoreClient:
    def __init__(self, effects):
        self.effects = effects

    def evaluate(self, key, answer):
        """An existing key returns the first recorded reply verbatim."""
        if key not in self.effects:
            self.effects[key] = "score:" + answer
        return self.effects[key]
'''

RECOVER = '''from effect_ids import effect_key
from journal import TurnJournal
from score_client import ScoreClient


def resume(log, external, session, turn, answer, stop):
    """Run one attempt using the supplied journal and idempotent scoring client."""
    key = effect_key(session, turn)
    journal = TurnJournal(log)
    prior = journal.read(key)
    if "done" in prior:
        return prior["done"]
    if "intent" in prior:
        answer = prior["intent"]
    else:
        journal.append(key, "intent", answer)
    if stop == "intent":
        return None
    reply = ScoreClient(external).evaluate(key, answer)
    if stop == "tool":
        return None
    journal.append(key, "done", reply)
    return reply
'''

RESTORE_STUB = '''

def continue_from(checkpoint, attempts):
    """Restore copies of journal and effects, replay attempts, and report the new state."""
    raise NotImplementedError("turn_worker.continue_from")
'''

RESTORE_REF = '''

def continue_from(checkpoint, attempts):
    """Continue from a detached snapshot of the two durable stores."""
    log = [row.copy() for row in checkpoint["journal"]]
    effects = checkpoint["effects"].copy()
    replies = [resume(log, effects, *attempt) for attempt in attempts]
    return {"replies": replies, "journal": log, "effects": effects}
'''

GATEWAY = '''from turn_worker import replay, continue_from


def handle(request):
    if "checkpoint" in request:
        return continue_from(request["checkpoint"], request["attempts"])
    return replay(request["attempts"])
'''

DEMO = '''"""Run locally: python restart_demo.py. Each invocation starts a fresh Python process."""
import json
from pathlib import Path
import subprocess
import sys


def launch(request):
    runner = Path(__file__).with_name("run_lab.py")
    output = subprocess.check_output(
        [sys.executable, str(runner)], input=json.dumps(request), text=True
    )
    return json.loads(output)


if __name__ == "__main__":
    first = launch({"attempts": [["chat", "7", "first", "tool"]]})
    assert first["replies"] == [None]
    checkpoint = {name: first[name] for name in ("journal", "effects")}
    restarted = launch({"checkpoint": checkpoint, "attempts": [["chat", "7", "retry", "finish"]]})
    assert restarted["replies"] == ["score:first"]
    assert len(restarted["journal"]) == 2 and len(restarted["effects"]) == 1
    print(json.dumps(restarted, indent=2))
'''

RESTART_TEST = '''import copy
import unittest
from gateway import handle


class RecoveryAfterRestart(unittest.TestCase):
    def test_resume_keeps_recorded_remote_response_and_preserves_snapshot(self):
        checkpoint = {"journal": [["1:st", "intent", "old"]], "effects": {"1:st": "recorded-before-crash"}}
        original = copy.deepcopy(checkpoint)
        result = handle({"checkpoint": checkpoint, "attempts": [["s", "t", "new", "finish"]]})
        self.assertEqual(result["replies"], ["recorded-before-crash"])
        self.assertEqual(checkpoint, original)
        self.assertEqual(result["journal"][-1], ["1:st", "done", "recorded-before-crash"])
'''
