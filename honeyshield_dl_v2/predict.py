"""Score preprocessed (N,10,11) sequences with a saved checkpoint."""
import argparse
import numpy as np
import torch
from models import MODEL_REGISTRY
from train_evaluate import reconstruction_errors

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=list(MODEL_REGISTRY),default='cnn')
    p.add_argument('--input',default='data/sequences_X.npy')
    p.add_argument('--output',default='results/predictions.csv')
    args=p.parse_args()
    saved=torch.load(f'checkpoints/{args.model}.pt',map_location='cpu',weights_only=True)
    model=MODEL_REGISTRY[args.model](saved['seq_len'],saved['num_features'])
    model.load_state_dict(saved['state_dict'])
    X=np.load(args.input)
    if X.ndim!=3 or tuple(X.shape[1:])!=(saved['seq_len'],saved['num_features']):
        raise ValueError('Input must be preprocessed sequences with checkpoint dimensions')
    scores=reconstruction_errors(model,X)
    np.savetxt(args.output,np.column_stack([scores,scores>saved['threshold']]),delimiter=',',header='anomaly_score,predicted_anomaly',comments='')
    print(f'Saved {len(scores)} predictions to {args.output}')
