import copy
import json
import unittest
from finetune_lab.paired import compare


class PairedTests(unittest.TestCase):
    def setUp(self):
        target={'category':'network','urgency':'high','asset':'LAB-12345678'}
        self.data=[{'id':str(i),'group_id':'g','target':target} for i in range(2)]
        self.base=[{'id':str(i),'target':target,'prompt_sha256':'p','raw_output':json.dumps(target)} for i in range(2)]
        self.adapter=copy.deepcopy(self.base)

    def test_raw_rescoring_exposes_equal_aggregate_tradeoff(self):
        self.base[0]['raw_output']='Here is JSON: '+self.base[0]['raw_output']
        self.adapter[1]['raw_output']='{}'
        # Contradictory cached fields must not override consumer-side parsing.
        self.adapter[1]['exact_object_match']=True
        result=compare(self.base,self.adapter,self.data)
        self.assertEqual(result['overall']['gains'],1)
        self.assertEqual(result['overall']['regressions'],1)
        self.assertEqual(result['overall']['base']['exact_object_match'],1)
        self.assertEqual(result['overall']['adapter']['exact_object_match'],1)
        self.assertEqual(result['slices']['group']['g']['cases'],2)

    def test_changed_prompt_or_missing_case_is_not_matched(self):
        with self.assertRaisesRegex(ValueError,'case IDs differ'):
            compare(self.base,self.adapter[:1],self.data)
        self.adapter[0]['prompt_sha256']='different'
        with self.assertRaisesRegex(ValueError,'prompts differ'):
            compare(self.base,self.adapter,self.data)


if __name__=='__main__':unittest.main()
