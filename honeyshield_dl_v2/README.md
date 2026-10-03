# HoneyShield-DL — Code (Interim / Phase 2)

Original implementation (not copied from any external report). All code has
been test-run end-to-end in a sandbox and verified to execute without errors.
**The numbers in this README are 5-epoch sanity-check numbers only — NOT
final results.** Re-run with more epochs/tuning before using any numbers in
the Interim report.

## Setup

```bash
pip install -r requirements.txt
```

(Use Google Colab with a GPU runtime for faster training — just upload these
files, or copy-paste the scripts into cells.)

## Pipeline order

```bash
# 1. Generate the synthetic HoneyShield-style dataset (~15,000 events)
python generate_dataset.py

# 2. Restructure into fixed-length per-user temporal sequences (shared by all 4 models)
python build_sequences.py

# 3. Train + evaluate one model, or all four:
python train_evaluate.py --model mlp --epochs 20
python train_evaluate.py --model cnn --epochs 20
python train_evaluate.py --model lstm --epochs 20
python train_evaluate.py --model attention --epochs 20
python train_evaluate.py --model all --epochs 20     # runs all 4, saves comparison table

# 4. Classical baseline for comparison
python baseline_isolation_forest.py
```

## What's genuinely implemented right now

- [x] Dataset generator (15,000 events, 5 attack categories, 92:8 split) — tested, works
- [x] Sequence/windowing pipeline (N=10 per-user temporal sequences, shared across all models) — tested, works
- [x] All 4 autoencoder architectures (MLP, CNN, LSTM, Attention) — tested, run without errors
- [x] Unsupervised training protocol (train on normal only, threshold from held-out normal validation set, no test leakage)
- [x] Isolation Forest baseline on the same split
- [x] Evaluation: Precision, Recall, F1-score, PR-AUC

## What's NOT done yet (needed before Interim)

- [ ] Proper training (more epochs, hyperparameter tuning — current numbers are from 5 epochs as a correctness check only)
- [ ] Literature review (8-10 papers, summary table)
- [ ] Push to a shared GitHub repo with per-member commits
- [ ] Individual contribution log (signed)
- [ ] Interim report write-up using REAL results from a proper training run

## Known result pattern from the 5-epoch sanity check (not final)

MLP and CNN converged quickly and separated anomalies reasonably well even
in 5 epochs. LSTM and Attention did not converge well in 5 epochs — this is
expected for recurrent/attention models, which typically need more epochs
and a lower/scheduled learning rate. Do not use these numbers in the report;
re-run with `--epochs 20` or more (and consider a learning-rate scheduler)
before recording real results.
