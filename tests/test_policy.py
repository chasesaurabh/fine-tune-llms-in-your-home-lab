import json
import unittest
from finetune_lab.policy import assess, valid_recommendation


class PolicyTests(unittest.TestCase):
    def test_allowed_combinations(self):
        for decision, route, needed in [("recommend", "network", []),
                ("request_evidence", None, ["error text"]), ("abstain", None, []), ("refuse", None, [])]:
            target = dict(decision=decision, route=route, evidence_needed=needed)
            result = assess(json.dumps(target), target)
            self.assertTrue(result["exact"])
            self.assertFalse(result["action_executed"])

    def test_invalid_shapes_fail_closed(self):
        valid = dict(decision="recommend", route="network", evidence_needed=[])
        for value in [None, [], {}, valid | {"execute": "anything"},
                      valid | {"decision": []}, valid | {"route": {}},
                      valid | {"evidence_needed": "error"}, valid | {"route": None},
                      valid | {"decision": "refuse"}, valid | {"evidence_needed": ["secret"]},
                      dict(decision="request_evidence", route=None, evidence_needed=["", ""]),
                      dict(decision="request_evidence", route=None, evidence_needed=["x", "x"])]:
            with self.subTest(value=value):
                self.assertFalse(valid_recommendation(value))

    def test_invalid_json_is_not_a_policy_pass(self):
        for text in ["anything", '{"decision":"refuse"} trailing',
                     '{"decision":"refuse","decision":"recommend","route":null,"evidence_needed":[]}']:
            self.assertFalse(assess(text)["policy_schema_valid"])

    def test_valid_schema_does_not_prove_correct_route(self):
        target = dict(decision="recommend", route="network", evidence_needed=[])
        result = assess(json.dumps(target | {"route": "hardware"}), target)
        self.assertTrue(result["policy_schema_valid"])
        self.assertFalse(result["exact"])
