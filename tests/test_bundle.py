import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from finetune_lab.bundle import create,verify


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'run';self.run.mkdir()
        (self.run/'adapter').mkdir()
        weights=self.run/'adapter/adapter_model.safetensors';weights.write_bytes(b'fixture-not-a-model')
        self.write('config.json',{'model':'test/model','revision':'pinned','method':'lora'})
        self.write('adapter-hashes.json',{'adapter_model.safetensors':hashlib.sha256(weights.read_bytes()).hexdigest()})
        for name in ('environment','trainable','training'):self.write(name+'.json',{})
        (self.run/'evaluation').mkdir();self.write('evaluation/summary.json',{'cases':1})
        self.write('evaluation/outputs.json',[{'raw_output':'synthetic'}])
        (self.run/'unrelated-secret.txt').write_text('must not be copied')
        self.output=self.root/'bundle'

    def write(self,name,value):
        (self.run/name).write_text(json.dumps(value),encoding='utf-8')

    def test_allowlisted_copy_and_relocatable_verification(self):
        result=create(self.run,self.output,['evaluation'])
        self.assertNotIn('unrelated-secret.txt',result['files'])
        moved=self.root/'moved';self.output.rename(moved)
        self.assertEqual(verify(moved),result)

    def test_damaged_and_extra_files_are_rejected(self):
        create(self.run,self.output)
        target=self.output/'adapter/adapter_model.safetensors'
        original=target.read_bytes();target.write_bytes(original+b'x')
        with self.assertRaises(ValueError):verify(self.output)
        target.write_bytes(original);(self.output/'extra.txt').write_text('extra')
        with self.assertRaises(ValueError):verify(self.output)

    def test_refuses_overwrite_and_source_destination_overlap(self):
        create(self.run,self.output)
        with self.assertRaises(ValueError):create(self.run,self.output)
        with self.assertRaises(ValueError):create(self.run,self.run/'nested')

    def test_rejects_evidence_traversal_and_corrupt_source(self):
        with self.assertRaises(ValueError):create(self.run,self.output,['../elsewhere'])
        self.assertFalse(self.output.exists())
        (self.run/'adapter/adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):create(self.run,self.output)
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
