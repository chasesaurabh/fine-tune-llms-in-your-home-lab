import json
import unittest
from finetune_lab.wrapper_diagnostic import unwrap,analyze


class WrapperTests(unittest.TestCase):
    def test_only_a_complete_single_json_fence_is_accepted(self):
        body='{"asset":"LAB-1234ABCD","category":"network","urgency":"low"}'
        self.assertEqual(unwrap('```json\n'+body+'\n```'),(body,True))
        for text in ('Here: ```json\n'+body+'\n```','```\n'+body+'\n```',
                     '```json\n'+body+'\n```\nMore','```json\n```json\n'+body+'\n```\n```'):
            self.assertEqual(unwrap(text),(text,False))

    def test_primary_failure_is_preserved_and_values_still_scored(self):
        target={'asset':'LAB-1234ABCD','category':'network','urgency':'low'}
        report=analyze([{'id':'a','target':target,'raw_output':'```json\n'+json.dumps(target)+'\n```'}],
                       [{'id':'a','target':target}])
        self.assertEqual(report['primary']['exact_object_match'],0)
        self.assertEqual(report['alternate_consumer']['exact_object_match'],1)
        wrong={**target,'urgency':'high'}
        report=analyze([{'id':'a','target':target,'raw_output':'```json\n'+json.dumps(wrong)+'\n```'}],
                       [{'id':'a','target':target}])
        self.assertEqual(report['alternate_consumer']['schema_valid'],1)
        self.assertEqual(report['alternate_consumer']['exact_object_match'],0)


if __name__=='__main__':unittest.main()
