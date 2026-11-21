import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from finetune_lab.data import audit_splits

ROOT=Path(__file__).resolve().parents[1]


class GeneratorTests(unittest.TestCase):
    def test_source_generators_reproduce_distributed_data(self):
        projects={'generate_extraction_evidence_dataset.py':'nvidia-evidence',
            'generate_project2_dataset.py':'project2-nvidia',
            'generate_project3_dataset.py':'project3-nvidia',
            'generate_project4_preferences.py':'project4-dpo',
            'generate_extraction_v2.py':'extraction-v2'}
        for script,project in projects.items():
            with self.subTest(project=project), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); (root/'scripts').mkdir()
                shutil.copyfile(ROOT/'scripts'/script,root/'scripts'/script)
                result=subprocess.run([sys.executable,str(root/'scripts'/script)],cwd=root,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
                actual=root/'experiments'/project/'data'
                audit_splits(actual)
                for name in ('train.jsonl','validation.jsonl','test.jsonl','manifest.json'):
                    self.assertEqual((actual/name).read_bytes(),(ROOT/'experiments'/project/'data'/name).read_bytes(),name)

    def test_new_study_balance_and_unrelated_asset_ids(self):
        root=ROOT/'experiments/extraction-v2/data'
        assets=set()
        for split,count in [('train',192),('validation',96),('test',144)]:
            rows=[json.loads(line) for line in (root/(split+'.jsonl')).read_text().splitlines()]
            self.assertEqual(len(rows),count)
            for category in ('network','hardware','software','access'):
                self.assertEqual(sum(r['target']['category']==category for r in rows),count//4)
            for urgency in ('low','medium','high'):
                self.assertEqual(sum(r['target']['urgency']==urgency for r in rows),count//3)
            for row in rows:
                asset=row['target']['asset']
                self.assertNotIn(asset,assets)
                self.assertIn(asset,row['prompt'])
                assets.add(asset)
