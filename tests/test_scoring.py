import json
import unittest

from finetune_lab.scoring import parse_object, score_extraction, summarize


class ScoringTests(unittest.TestCase):
    target = {"category": "network", "urgency": "high", "asset": "LAB-1"}

    def test_permits_whitespace_and_key_order(self):
        result = score_extraction(' \n' + json.dumps(self.target) + '\t', self.target)
        self.assertTrue(result["exact_object_match"])

    def test_rejects_nonconsumer_outputs(self):
        valid = json.dumps(self.target)
        for text in ["Here: " + valid, valid + " done", valid + valid,
                     "```json\n" + valid + "\n```", "[]", "null", "true",
                     '{"a":NaN}', '{"a":Infinity}', '{"a":1,"a":2}',
                     '{"a":{"b":1,"b":2}}', '{"a":1,}', ""]:
            with self.subTest(text=text):
                self.assertIsNone(parse_object(text))
                self.assertFalse(score_extraction(text, self.target)["exact_object_match"])

    def test_nested_objects_are_json_but_not_extraction_schema(self):
        text = '{"category":{"value":"network"},"urgency":"high","asset":"LAB-1"}'
        result = score_extraction(text, self.target)
        self.assertTrue(result["valid_json_object"])
        self.assertFalse(result["schema_valid"])

    def test_schema_and_values_are_distinct(self):
        for replacement in ({"category": "other"}, {"asset": 1}, {"asset": ""},
                            {"extra": "value"}, {"urgency": None}):
            result = score_extraction(json.dumps(self.target | replacement), self.target)
            self.assertTrue(result["valid_json_object"])
            self.assertFalse(result["schema_valid"])
        result = score_extraction(json.dumps(self.target | {"urgency": "low"}), self.target)
        self.assertTrue(result["schema_valid"])
        self.assertFalse(result["exact_object_match"])
        self.assertTrue(result["fields"]["category"])

    def test_bad_reference_is_an_error(self):
        with self.assertRaises(ValueError):
            score_extraction("{}", {})

    def test_no_silent_empty_denominator(self):
        with self.assertRaises(ValueError):
            summarize([])
        summary = summarize([score_extraction(json.dumps(self.target), self.target),
                             score_extraction("invalid", self.target)])
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["exact_object_match"], 1)


if __name__ == "__main__":
    unittest.main()
