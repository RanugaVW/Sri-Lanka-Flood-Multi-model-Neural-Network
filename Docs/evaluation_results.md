# Model Evaluation Results

## Classification Metrics (Optimal Threshold)

| Target | Threshold | Accuracy | Precision | Recall | F1 Score | PR-AUC |
|---|---|---|---|---|---|---|
| P(flood t+1) | 0.62 | 0.9930 | 0.6473 | 0.3201 | 0.4284 | 0.4358 |
| P(flood t+2) | 0.61 | 0.9885 | 0.4573 | 0.3311 | 0.3841 | 0.3899 |
| P(flood t+3) | 0.61 | 0.9859 | 0.4544 | 0.2912 | 0.3549 | 0.3619 |
| onset | 0.58 | 0.9829 | 0.0393 | 0.2345 | 0.0673 | 0.0276 |

## Regression Metrics

| Target | R2 Score | MAE | RMSE |
|---|---|---|---|
| discharge_t1 | 0.9565 | 2.5494 | 10.7174 |
| zscore_3d_max | 0.0927 | 0.5619 | 2.5601 |
