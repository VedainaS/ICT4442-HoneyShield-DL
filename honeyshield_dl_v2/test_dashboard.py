from pathlib import Path
import json
import unittest
from streamlit.testing.v1 import AppTest

class DashboardTests(unittest.TestCase):
    def test_inference_and_filters(self):
        root=Path(__file__).resolve().parent
        at=AppTest.from_file(str(root/'dashboard.py'),default_timeout=60).run()
        self.assertFalse(at.exception)
        at.button[1].click().run()
        self.assertFalse(at.exception)
        import numpy as np
        from sklearn.metrics import f1_score
        mask=np.load(root/'data/sequences_split.npy')=='test'
        y=np.load(root/'data/sequences_y.npy')[mask]
        for r in json.loads((root/'results/final_run_25ep.json').read_text()):
            scores,threshold=at.session_state['runs'][r['model']]
            self.assertEqual(len(scores),3010)
            self.assertAlmostEqual(f1_score(y,scores>threshold),r['f1_score'],places=10)
        for name in ['mlp','cnn','lstm','attention']:
            at.selectbox(key="model").select(name).run()
            for label in ['False positives','Missed attacks','Flagged only']:
                at.selectbox(key="rows").select(label).run()
                self.assertFalse(at.exception)

if __name__=='__main__':unittest.main()
