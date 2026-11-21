import copy
import unittest
from finetune_lab.compare_preferences import compare


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.policy={'model':'m','revision':'r','method':'lora','dataset':{'train':'h'},
            'split':'test','eos_scored':True,'prompt_scored':False,'cases':1,
            'algorithm':'DPO sigmoid','mode':'adapter','parent_adapter_hashes':{'weights':'a'}}
        self.reference={**self.policy,'algorithm':'SFT','adapter_hashes':{'weights':'a'}}
        self.rows=[{'id':'one','rule':'format','prompt_sha256':'p',
            'chosen':{'tokens':3,'sum_logprob':-2,'token_ids_sha256':'c'},
            'rejected':{'tokens':4,'sum_logprob':-8,'token_ids_sha256':'r'}}]

    def test_reference_adjusted_pair(self):
        reference=copy.deepcopy(self.rows)
        reference[0]['chosen']['sum_logprob']=-5
        reference[0]['rejected']['sum_logprob']=-6
        self.assertEqual(compare(self.policy,self.reference,self.rows,reference)[0]['reference_adjusted_sum_margin'],5)

    def test_wrong_reference_and_same_length_wrong_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError,'exact SFT parent'):
            compare(self.policy,{**self.reference,'adapter_hashes':{'weights':'other'}},self.rows,self.rows)
        changed=copy.deepcopy(self.rows)
        changed[0]['chosen']['token_ids_sha256']='different-same-length'
        with self.assertRaisesRegex(ValueError,'response tokens'):
            compare(self.policy,self.reference,self.rows,changed)

    def test_duplicate_ids_and_different_prompts_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            compare(self.policy,self.reference,self.rows,self.rows*2)
        changed=copy.deepcopy(self.rows);changed[0]['prompt_sha256']='different'
        with self.assertRaisesRegex(ValueError,'prompt differs'):
            compare(self.policy,self.reference,self.rows,changed)


if __name__=='__main__':unittest.main()
