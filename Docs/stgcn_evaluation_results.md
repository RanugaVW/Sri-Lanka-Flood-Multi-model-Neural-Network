# STGCN Model Evaluation Results

This document contains the evaluation metrics for the STGCN model, extracted from the latest training outputs, along with a detailed explanation of what each metric indicates regarding the model's performance.

## STGCN Evaluation Summary

| Stage / Protocol | PR-AUC | ROC-AUC | Brier | ECE | POD | FAR | CSI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **temporal** | 0.6023 | 0.9777 | 0.01793 | 0.0514 | 0.481 | 0.401 | 0.364 |
| **basin** | 0.7669 | 0.9872 | 0.01948 | 0.0917 | 0.451 | 0.112 | 0.427 |
| **temporal** | 0.6633 | 0.9817 | 0.01346 | 0.0070 | 0.474 | 0.301 | 0.394 |

*Note: The second "temporal" row appears to reflect the final best STGCN temporal results based on the optimal threshold of 0.7084.*

---

## Metric Analysis & Feedback

Here is a breakdown of what each result describes in the context of your STGCN model:

### 1. PR-AUC (Precision-Recall Area Under Curve): `0.6633` (Final Temporal)
* **What it means:** It measures the tradeoff between precision (how many predicted events were actual events) and recall (how many actual events were predicted).
* **Feedback:** A score of 0.66 is quite solid for forecasting tasks involving imbalanced data (e.g., floods/extreme events). It indicates the model is reasonably good at finding the positive cases without triggering too many false positives. The `basin` protocol achieved an even better `0.7669`.

### 2. ROC-AUC (Receiver Operating Characteristic Area Under Curve): `0.9817`
* **What it means:** Measures the model's ability to distinguish between classes (e.g., event vs. no-event) at various threshold settings.
* **Feedback:** Your score of ~0.98 is excellent. It means there is a 98% chance the model will correctly distinguish an actual positive occurrence from a negative occurrence. However, because forecasting events are often highly imbalanced, ROC-AUC can sometimes be overly optimistic compared to PR-AUC.

### 3. Brier Score: `0.01346`
* **What it means:** Measures the mean squared difference between the predicted probability and the actual outcome. Lower is better (0 is perfect).
* **Feedback:** At `0.01346`, this is an exceptionally good score. It signifies that the probability estimates your STGCN model outputs are very close to the actual binary outcomes.

### 4. ECE (Expected Calibration Error): `0.0070`
* **What it means:** Measures how well the predicted probabilities correspond to the true likelihood of the event. A perfectly calibrated model has an ECE of 0.
* **Feedback:** An ECE of `0.0070` is extremely low, meaning your model is highly calibrated. When your model predicts a 70% chance of an event, the event actually happens about 70% of the time. The temperature calibration (T=0.367 from your logs) clearly did its job perfectly here.

### 5. POD (Probability of Detection / Recall): `0.474`
* **What it means:** The fraction of actual positive events that were successfully predicted by the model. (True Positives / (True Positives + False Negatives)).
* **Feedback:** A POD of ~47% means the model successfully detects just under half of the actual events. While it might seem low, in extreme event forecasting, it's often a necessary tradeoff to avoid over-predicting (which would drastically increase FAR). 

### 6. FAR (False Alarm Ratio): `0.301`
* **What it means:** The fraction of predicted positive events that did not actually occur. (False Positives / (True Positives + False Positives)).
* **Feedback:** At ~30%, about 3 out of 10 times your model predicts an event, it's a false alarm. In disaster forecasting, some false alarms are tolerated to ensure major events aren't missed, so 30% is generally acceptable. The `basin` evaluation did much better here at `0.112` (only 11% false alarms).

### 7. CSI (Critical Success Index): `0.394`
* **What it means:** Also known as the Threat Score. It evaluates the overall accuracy, explicitly ignoring correct negatives (which dominate in forecasting). Formula: True Positives / (True Positives + False Positives + False Negatives).
* **Feedback:** A score of `0.394` provides a realistic view of your model's accuracy on the events that actually matter. It balances the fact that the model catches 47% of events (POD) but has a 30% false alarm rate (FAR).

### Summary Conclusion
Your STGCN model is **highly calibrated (excellent ECE and Brier)** and shows **phenomenal discriminative ability (ROC-AUC)**. Its strict calibration means it tends to be slightly conservative in its predictions (catching ~47% of events) to keep the false alarm rate reasonable (~30%). This is generally a strong, reliable baseline for forecasting.
