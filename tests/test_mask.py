import unittest
from finetune_lab.training import check_mask


class MaskTests(unittest.TestCase):
    def test_completion_and_eos_only(self):
        check_mask([10,11,20,99,0],[-100,-100,20,99,-100],[1,1,1,1,0],2,99)

    def test_prompt_padding_or_missing_eos_fails(self):
        cases=[([10,11,20,99,0],[10,-100,20,99,-100],[1,1,1,1,0]),
               ([10,11,20,99,0],[-100,-100,20,99,0],[1,1,1,1,0]),
               ([10,11,20,21,0],[-100,-100,20,21,-100],[1,1,1,1,0]),
               ([10,11,20,99,0],[-100,-100,-100,-100,-100],[1,1,1,1,0])]
        for ids,labels,attention in cases:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError): check_mask(ids,labels,attention,2,99)
