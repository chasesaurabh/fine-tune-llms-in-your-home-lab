import unittest
from finetune_lab.study_report import literal_comparison,reconstruction,decision


class StudyTests(unittest.TestCase):
    def test_equal_probe_totals_do_not_hide_a_lost_case(self):
        targets=[{'id':'a','target':'yes'},{'id':'b','target':'no'}]
        base=[dict(t,raw_output='yes',prompt_sha256=t['id'],exact_text_contract=True) for t in targets]
        after=[dict(t,raw_output='no',prompt_sha256=t['id'],exact_text_contract=True) for t in targets]
        result=literal_comparison(base,after,targets)
        self.assertEqual((result['base_correct'],result['adapter_correct']),(1,1))
        self.assertEqual(result['lost_ids'],['a'])
        after[0]['prompt_sha256']='changed'
        with self.assertRaises(ValueError):literal_comparison(base,after,targets)

    def test_reconstruction_uses_raw_text_not_equal_scores(self):
        targets=[{'id':'a','target':{'answer':1}}]
        first=[dict(targets[0],raw_output='{"answer":1}',prompt_sha256='same')]
        second=[dict(targets[0],raw_output=' {"answer":1}',prompt_sha256='same')]
        self.assertFalse(reconstruction(first,second,targets)['raw_identical'])
        with self.assertRaises(ValueError):reconstruction(first,[],targets)

    def test_frozen_decision_preserves_base_and_probe_gates(self):
        paired={'overall':{'cases':144,'base':{'exact_object_match':126,'schema_valid':144},
            'adapter':{'exact_object_match':139,'schema_valid':144},'gains':18,'regressions':5}}
        probes={'cases':16,'lost_ids':[]};reload={'cases':96,'raw_identical':True}
        self.assertTrue(decision(paired,probes,reload)['accepted_for_stated_synthetic_task'])
        probes['lost_ids']=['one']
        self.assertFalse(decision(paired,probes,reload)['accepted_for_stated_synthetic_task'])
        probes['lost_ids']=[];paired['overall']['base']['exact_object_match']=137
        self.assertFalse(decision(paired,probes,reload)['accepted_for_stated_synthetic_task'])
        paired['overall']['cases']=143
        with self.assertRaises(ValueError):decision(paired,probes,reload)


if __name__=='__main__':unittest.main()
