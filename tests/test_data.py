import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from finetune_lab.data import audit_splits,load_rows
from finetune_lab.evaluation import verify_files


class DataTests(unittest.TestCase):
    def build(self,root,leak=False):
        manifest={'splits':{}}
        for split in ('train','validation','test'):
            row={'id':split,'group_id':'shared' if leak else split,'prompt':'synthetic prompt'}
            payload=json.dumps(row)+'\n'
            (root/(split+'.jsonl')).write_text(payload,encoding='utf-8')
            manifest['splits'][split]={'rows':1,'sha256':hashlib.sha256((root/(split+'.jsonl')).read_bytes()).hexdigest()}
        (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')

    def test_hash_and_group_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.build(root)
            self.assertEqual(audit_splits(root)['train']['rows'],1)
            with (root/'test.jsonl').open('a',encoding='utf-8') as out: out.write('\n')
            with self.assertRaises(ValueError): audit_splits(root)
            self.build(root,leak=True)
            with self.assertRaises(ValueError): audit_splits(root)

    def test_duplicate_ids_and_empty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rows.jsonl'
            for payload in ('','{"id":"x"}\n{"id":"x"}\n'):
                path.write_text(payload,encoding='utf-8')
                with self.assertRaises(ValueError): load_rows(path)

    def test_changed_or_added_adapter_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'adapter.json'; path.write_bytes(b'{}')
            hashes={'adapter.json':hashlib.sha256(b'{}').hexdigest()}
            verify_files(root,hashes)
            path.write_bytes(b'{"changed":true}')
            with self.assertRaises(ValueError): verify_files(root,hashes)
            path.write_bytes(b'{}'); (root/'extra.json').write_bytes(b'{}')
            with self.assertRaises(ValueError): verify_files(root,hashes)
