import unittest
from finetune_lab.preference import pair_metrics

class PairTests(unittest.TestCase):
    def test_length_confound_is_visible(self):
        result=pair_metrics({'tokens':10,'sum_logprob':-5},{'tokens':2,'sum_logprob':-2})
        self.assertFalse(result['chosen_higher_sum'])
        self.assertTrue(result['chosen_higher_mean'])
        self.assertEqual(result['length_delta'],8)

    def test_reference_adjustment(self):
        result=pair_metrics({'tokens':3,'sum_logprob':-2},{'tokens':4,'sum_logprob':-8},
            {'tokens':3,'sum_logprob':-5},{'tokens':4,'sum_logprob':-6})
        self.assertEqual(result['reference_adjusted_sum_margin'],5)

    def test_rejects_invalid_comparisons(self):
        good={'tokens':3,'sum_logprob':-2}
        for bad in [{'tokens':0,'sum_logprob':0},{'tokens':3,'sum_logprob':float('nan')}]:
            with self.assertRaises(ValueError): pair_metrics(good,bad)
        with self.assertRaises(ValueError): pair_metrics(good,good,good)
        with self.assertRaises(ValueError): pair_metrics(good,good,{'tokens':4,'sum_logprob':-2},good)

if __name__=='__main__': unittest.main()
