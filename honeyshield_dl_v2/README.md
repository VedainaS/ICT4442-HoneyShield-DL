# HoneyShield-DL

Four autoencoders for synthetic enterprise login anomaly detection: MLP, 1D-CNN, LSTM and self-attention, with an additional Isolation Forest baseline.

Team: Vedaina Srivastava (230911054), Pooja Murali (230911049).

## Run

Tested with Python 3.12 on CPU. Run commands from this directory.

```bash
pip install -r requirements.txt
python build_sequences.py
python train_evaluate.py --model all --epochs 25
python baseline_isolation_forest.py
python analysis_by_attack.py --epochs 25
python -m unittest -v test_pipeline
python predict.py --model cnn
```

The supplied raw dataset is the canonical dataset for the reported run. `python generate_dataset.py` optionally replaces it; regenerate sequences and retrain afterwards. Faker-version differences may change regenerated data. Prediction accepts arrays transformed using the saved training preprocessing, not raw logs.

## Corrected protocol

Split the 800 users first: 544 training, 96 validation, 160 test, seed 42. No user or event is shared across splits. Fit frequency maps and session-duration normalization exclusively on normal events of training users. Unknown categories map to zero. Save fitted parameters in `data/preprocessing.json` and split membership in `data/split_manifest.json`.

Build per-user windows of 10 events with left zero-padding and 11 features. Keep only entirely normal windows for training and validation (including their context). Keep all test windows, labelling each by its last event. This yields 5,680 training, 965 validation and 3,010 test windows, with 250 anomalous test targets.

All four neural models use Adam, learning rate 0.001, batch size 64, MSE loss and 25 epochs. Reset seed 42 before EACH model; use deterministic CPU operations with two threads. The 99th percentile of normal validation scores sets the threshold (approximately 1% validation false positives, not an 8% anomaly rate). Isolation Forest also uses the normal-validation 99th-percentile threshold but sees only the last event.

## Fresh results

| Model | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|
| mlp | 0.1923 | 0.6000 | 0.2913 | 0.1890 |
| cnn | 0.1861 | 0.7520 | 0.2984 | 0.1815 |
| lstm | 0.1863 | 0.6720 | 0.2917 | 0.1754 |
| attention | 0.1868 | 0.7000 | 0.2949 | 0.1867 |
| isolation_forest (baseline) | 0.0588 | 0.0080 | 0.0141 | 0.2140 |

The `pr_auc` JSON key stores sklearn average precision, not trapezoidal PR-curve area. These are one-seed results, not evidence of statistically significant architectural superiority. CNN has the highest F1 in this run, but the scores are close.

## Artifacts

- `results/comparison_results.json`: current four-model metrics.
- `results/final_run_25ep.json`: per-attack recall, false-positive rate and counts.
- `results/training_log.txt`: actual 25-epoch run log.
- `results/test_report.txt`: regression-test output.
- `checkpoints/*.pt`: trained weights, dimensions, threshold, epochs and seed.
- `results/legacy/`: OLD results from the flawed split, retained only for traceability. Do not use them as current results.

## Remaining scientific limitations

The data is synthetic. “Impossible Travel” simulates a changed country, not physical travel speed; device identifiers and geographical distance/time features are absent. Unknown-user testing is harder than the previous random-window split. Windows overlap WITHIN a split, so test observations are correlated. Thresholds estimated on clean normal windows may generate many false positives on test windows whose context contains earlier attacks. Whole-window MSE is scored against a last-event target; this is a deliberate contextual detector and can blame the current event for historical anomalies. Zero padding is reconstructed without masking.

The attention decoder emits the same vector at every time step; this restricts temporal reconstruction. CNN has no compressed temporal bottleneck; model capacities differ. These designs are preserved to avoid silently changing the compared architectures. Inspect per-attack recall alongside false positives rather than treating recall alone as success. No production-performance claim is made.

## Report and submission

The previously supplied DOCX reports describe the OLD experiment. Their split counts, tables, conclusions, error analysis and protocol must be updated using this README and current JSON files before submission. They have not been silently replaced by this ZIP. Contributions and signatures must reflect actual work. A GitHub push and genuine Turnitin report remain separate submission steps.

## AI assistance

The original repository declares Claude assistance for initial code and report text. ChatGPT assisted with the corrected user-disjoint pipeline, clean-window filtering, reproducibility, tests, fresh experiment execution and this documentation. Members should review the changes and accurately declare their own contributions.
