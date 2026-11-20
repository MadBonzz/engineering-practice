import unittest
from events import read_events
from ledger import OutcomeLedger
from report import summarize


def loaded(rows):
    ledger = OutcomeLedger()
    for event in read_events(rows):
        ledger.ingest(event)
    return ledger


class Part1(unittest.TestCase):
    def test_no_observation(self):
        self.assertIsNone(OutcomeLedger().latest("p1", "baseline"))

    def test_newer_outcome(self):
        ledger = loaded([["p1", "baseline", 0, False], ["p1", "baseline", 1, True]])
        self.assertIs(ledger.latest("p1", "baseline"), True)


class Part2(unittest.TestCase):
    def test_incomplete_pair(self):
        ledger = loaded([["p1", "candidate", 1, True]])
        self.assertIsNone(ledger.pair("p1"))

    def test_model_order(self):
        ledger = loaded([["p1", "candidate", 0, False], ["p1", "baseline", 2, True]])
        self.assertEqual(ledger.pair("p1"), (True, False))


class Part3(unittest.TestCase):
    def test_empty_report(self):
        self.assertEqual(summarize(OutcomeLedger(), 1), {
            "paired": 0, "wins": 0, "losses": 0, "ties": 0, "decision": "hold"})

    def test_candidate_gain(self):
        ledger = loaded([["a", "baseline", 2, False], ["a", "candidate", 0, True]])
        self.assertEqual(summarize(ledger, 1), {
            "paired": 1, "wins": 1, "losses": 0, "ties": 0, "decision": "promote"})
