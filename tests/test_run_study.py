import unittest
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
from finetune_lab.run_study import build_jobs
from finetune_lab.run_study import main


class RunnerTests(unittest.TestCase):
    def test_base_reused_only_for_same_revision_and_method(self):
        first={'id':'small-1','model':'test/model','revision':'a'*40,'method':'lora','seed':1}
        configs=[first,{**first,'id':'small-2','seed':2},{**first,'id':'changed','revision':'b'*40}]
        jobs=build_jobs(configs,'runs/study','data','policy','probes','hash')
        self.assertEqual(len(jobs),18)
        bases=[j['phase'] for j in jobs if j['phase'].endswith('-base-test')]
        self.assertEqual(bases,['small-1-base-test','changed-base-test'])
        gates=[j for j in jobs if j['gate']]
        self.assertEqual(len(gates),3)

    def test_rejects_path_traversal_or_duplicate_run_names(self):
        first={'id':'ok','model':'test/model','revision':'a'*40,'method':'lora','seed':1}
        for configs in ([first,first],[{**first,'id':'../outside'}]):
            with self.assertRaises(ValueError):build_jobs(configs,'runs','data','policy','probes','hash')

    def exercise_runner(self, exit_code, reconstruction_ok):
        # Fake child processes test orchestration only, never model quality.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); output=root/'runs'; policy=root/'policy'; probes=root/'probes'
            policy.write_bytes(b'policy'); probes.write_bytes(b'probes')
            plan=root/'plan.json'; gate=root/'gate.json'
            gate.write_text(json.dumps({'in_memory_raw_outputs_identical':reconstruction_ok}))
            plan.write_text(json.dumps({'configurations':[{'id':'fixture'}], 'data':'fixture-data',
                'dataset':{},'policy':str(policy),'probes':str(probes),
                'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),
                'probe_sha256':hashlib.sha256(probes.read_bytes()).hexdigest()}))
            jobs=[{'phase':'first','module':'fixture','arguments':[],'gate':str(gate)},
                  {'phase':'second','module':'fixture','arguments':[],'gate':None}]
            with patch('sys.argv',['run_study','--plan',str(plan),'--output',str(output)]), \
                 patch('finetune_lab.run_study.audit_splits',return_value={}), \
                 patch('finetune_lab.run_study.build_jobs',return_value=jobs), \
                 patch('finetune_lab.run_study.subprocess.run',return_value=subprocess.CompletedProcess([],exit_code)) as child:
                if exit_code or not reconstruction_ok:
                    with self.assertRaises(RuntimeError):main()
                    self.assertEqual(child.call_count,1)
                    self.assertFalse((output/'_completed.json').exists())
                else:
                    main(); self.assertEqual(child.call_count,2)
                    self.assertEqual(json.loads((output/'_completed.json').read_text())['phases'],2)
                journal=json.loads((output/'_execution.json').read_text())
                self.assertEqual(journal[0]['exit_code'],exit_code)
                self.assertIn('finished_utc',journal[0])
                calls=child.call_count
                with self.assertRaises(SystemExit):main()
                self.assertEqual(child.call_count,calls)

    def test_failed_child_stops_before_following_phase_and_preserves_journal(self):
        self.exercise_runner(7,True)

    def test_reconstruction_failure_stops_despite_zero_child_exit(self):
        self.exercise_runner(0,False)

    def test_success_completes_and_existing_destination_is_never_replayed(self):
        self.exercise_runner(0,True)


if __name__=='__main__':unittest.main()
