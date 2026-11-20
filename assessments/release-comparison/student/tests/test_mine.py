"""Add your own tests here. Run them with 'My tests' or 'All public tests'."""
import unittest
from events import Event
from ledger import OutcomeLedger
from report import summarize


class MyTests(unittest.TestCase):
    def test_empty_prompt_inventory(self):
        self.assertEqual(OutcomeLedger().prompts(), frozenset())
