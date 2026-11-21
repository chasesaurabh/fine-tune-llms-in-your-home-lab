import http.client
import json
import threading
import unittest
from finetune_lab.serve import create_server,validate_request

class FakeRuntime:
    identity={'model':'synthetic-test-double','revision':'fixture','adapter_sha256':'fixture'}
    output='{"category":"network","urgency":"low","asset":"LAB-1234ABCD"}'
    calls=0
    def complete(self,text):self.calls+=1;return self.output

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.runtime=FakeRuntime();self.token='test-only-'+'a'*32
        self.server=create_server(self.runtime,self.token)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
    def request(self,body,token=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        connection.request('POST','/extract',body=body,headers={'Content-Type':'application/json',
            'Authorization':'Bearer '+(self.token if token is None else token)})
        response=connection.getresponse();data=json.loads(response.read());status=response.status
        connection.close();return status,data
    def test_authentication_precedes_model(self):
        status,_=self.request('{}',token='wrong');self.assertEqual(status,401);self.assertEqual(self.runtime.calls,0)
    def test_valid_request_and_asset_mismatch(self):
        body=json.dumps({'text':'LAB-1234ABCD has a network issue.'})
        status,data=self.request(body);self.assertEqual(status,200);self.assertEqual(data['result']['asset'],'LAB-1234ABCD')
        self.runtime.output=self.runtime.output.replace('LAB-1234ABCD','LAB-FFFF0000')
        self.assertEqual(self.request(body)[0],422)
    def test_duplicate_keys_and_large_body_rejected_before_model(self):
        self.assertEqual(self.request('{"text":"LAB-1234ABCD","text":"other"}')[0],400)
        self.assertEqual(self.request('x'*8193)[0],413)
        self.assertEqual(self.runtime.calls,0)
    def test_model_prose_is_not_repaired(self):
        self.runtime.output='Here is the result: '+self.runtime.output
        self.assertEqual(self.request(json.dumps({'text':'LAB-1234ABCD failed.'}))[0],422)
    def test_single_asset_boundary(self):
        for value in ({'text':'No identifier'},{'text':'LAB-1234ABCD and LAB-FFFF0000'},
                      {'text':'LAB-1234ABCD','admin':True},{'text':42}):
            with self.assertRaises(ValueError):validate_request(value)

if __name__=='__main__':unittest.main()
