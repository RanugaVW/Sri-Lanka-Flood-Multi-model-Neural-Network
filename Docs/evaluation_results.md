# Model Evaluation Results

## Classification Metrics (Optimal Threshold)

| Target | Threshold | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| P(flood t+1) | 0.63 | 0.9914 | 0.4674 | 0.3797 | 0.4190 |
| P(flood t+2) | 0.63 | 0.9887 | 0.4702 | 0.3294 | 0.3874 |
| P(flood t+3) | 0.61 | 0.9830 | 0.3664 | 0.3823 | 0.3742 |
| onset | 0.55 | 0.9701 | 0.0216 | 0.2345 | 0.0396 |

## Regression Metrics

| Target | R2 Score | MAE | RMSE |
|---|---|---|---|
| discharge_t1 | 0.9449 | 2.9895 | 12.0661 |
| zscore_3d_max | 0.0729 | 0.6505 | 2.5879 |
