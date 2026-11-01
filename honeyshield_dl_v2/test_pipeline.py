"""Regression tests for leakage boundaries, clean windows and model outputs."""
import unittest, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from build_sequences import build_features, build_sequences
from models import MODEL_REGISTRY

class PipelineTests(unittest.TestCase):
    def test_user_disjoint(self):
        m=json.loads(Path('data/split_manifest.json').read_text())
        a,b,c=[set(m[k]['users']) for k in ['train','val','test']]
        self.assertFalse(a&b or a&c or b&c)
        self.assertEqual(len(a|b|c),800)

    def test_saved_data_and_clean_windows(self):
        raw=pd.read_csv('data/raw_logs.csv')
        m=json.loads(Path('data/split_manifest.json').read_text())
        split=np.load('data/sequences_split.npy')
        y=np.load('data/sequences_y.npy')
        X=np.load('data/sequences_X.npy')
        self.assertTrue(np.isfinite(X).all())
        self.assertEqual(X.shape[1:],(10,11))
        fit=raw[raw.user_id.isin(m['train']['users']) & (raw.is_anomaly_ground_truth==0)]
        for name in ['train','val','test']:
            subset,cols=build_features(raw[raw.user_id.isin(m[name]['users'])],fit)
            expected, labels,_=build_sequences(subset,cols)
            mask=[]
            for _,g in subset.groupby('user_id'):
                a=g.sort_values('timestamp').is_anomaly_ground_truth.to_numpy()
                mask.extend(not a[max(0,i-9):i+1].any() for i in range(len(a)))
            mask=np.array(mask) if name!='test' else np.ones(len(labels),dtype=bool)
            np.testing.assert_array_equal(X[split==name],expected[mask])
            np.testing.assert_array_equal(y[split==name],labels[mask])
            if name!='test': self.assertEqual(y[split==name].sum(),0)
        # Test-row modifications cannot change the training-fitted transform.
        probe=raw.iloc[:4].copy(); probe['country']='UNSEEN_CATEGORY'
        transformed,_=build_features(probe,fit)
        self.assertTrue((transformed.country_freq==0).all())
        expected_mean=fit.session_duration.mean()
        metadata=json.loads(Path('data/preprocessing.json').read_text())
        self.assertAlmostEqual(metadata['duration_mean'],expected_mean)

    def test_models_forward_backward(self):
        X=torch.tensor(np.load('data/sequences_X.npy')[:4])
        for name,cls in MODEL_REGISTRY.items():
            with self.subTest(model=name):
                model=cls(10,11);out=model(X)
                self.assertEqual(out.shape,X.shape)
                self.assertTrue(torch.isfinite(out).all())
                out.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

if __name__=='__main__': unittest.main()
