import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from finetune_lab.study_report import (literal_comparison,reconstruction,decision,
    validate_run,validate_inputs,EXPECTED_RUN,EXPECTED_ACCEPTANCE)


class StudyTests(unittest.TestCase):
    def test_modified_budget_or_instruction_is_not_the_frozen_run(self):
        policy='Shared task policy.\n'
        config={**EXPECTED_RUN,'instruction':policy.strip(),
            'instruction_sha256':hashlib.sha256(policy.strip().encode()).hexdigest()}
        validate_run(config,policy)
        for key,value in [('epochs',1),('learning_rate',0.0001),('rank',16),
                          ('warmup_steps',3),('stop_after',4),('instruction','A different policy')]:
            changed={**config,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):validate_run(changed,policy)

    def test_matching_run_data_cannot_replace_the_frozen_plan_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan={'acceptance':dict(EXPECTED_ACCEPTANCE),'dataset':{}}
            for split in ('train','validation','test'):
                raw=(split+'\n').encode();(root/(split+'.jsonl')).write_bytes(raw)
                plan['dataset'][split]={'sha256':hashlib.sha256(raw).hexdigest()}
            probes=root/'probes.jsonl';probes.write_bytes(b'probes\n')
            policy=root/'policy.txt';policy.write_bytes(b'policy\n')
            plan['probe_sha256']=hashlib.sha256(probes.read_bytes()).hexdigest()
            plan['policy_sha256']=hashlib.sha256(policy.read_bytes()).hexdigest()
            validate_inputs(plan,root,probes,policy)
            changed=copy.deepcopy(plan);changed['acceptance']['exact_minimum']=100
            with self.assertRaises(ValueError):validate_inputs(changed,root,probes,policy)
            for file in (root/'test.jsonl',probes,policy):
                original=file.read_bytes();file.write_bytes(original+b'changed')
                with self.subTest(file=file.name),self.assertRaises(ValueError):validate_inputs(plan,root,probes,policy)
                file.write_bytes(original)

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
