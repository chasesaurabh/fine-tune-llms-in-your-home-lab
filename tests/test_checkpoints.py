import unittest
from finetune_lab.checkpoints import IDENTITY_KEYS,check_resume_identity

class ResumeTests(unittest.TestCase):
    def test_schedule_change_is_rejected(self):
        original={key:key for key in IDENTITY_KEYS}
        for key in IDENTITY_KEYS:
            with self.assertRaises(ValueError): check_resume_identity(original,{**original,key:'changed'})

    def test_stop_boundary_does_not_change_plan(self):
        original={key:key for key in IDENTITY_KEYS}
        check_resume_identity({**original,'stop_after_step':3},{**original,'stop_after_step':None})

if __name__=='__main__': unittest.main()
