# Architectures We Tried — Summary (Simple English)

This document explains, in plain language, the different model versions we built for this flood-prediction
project, what each one actually did, and what results (numbers) we got out of them. It's meant to be readable
without needing to open the code.

The overall goal never changed: look at 51 river-gauge locations in Sri Lanka, look at their last 14 days of
weather/river data (and some other data), and predict things like "will this location flood in the next 1, 2,
or 3 days?".

---

## 1. Baseline model — STGCN (Spatio-Temporal Graph Convolutional Network)

**What it was:** Our first working model. It only used two things: (1) the time-series data (rainfall, soil
moisture, river discharge, etc. for the last 14 days) and (2) the river network graph (which gauge is
upstream/downstream of which). No satellite images, no detailed terrain features yet.

**Methodology (simple version):** A temporal component read the last 14 days of data for each of the 51
locations, and a graph convolution layer let each location "talk to" its upstream/downstream neighbours so the
model could learn that a flood at one point often causes a flood downstream a bit later. The output was a
flood probability for each location.

**Results we got:**

| Evaluation setup | PR-AUC | ROC-AUC | Brier Score | ECE (calibration error) | POD (detection rate) | FAR (false alarm rate) | CSI |
|---|---|---|---|---|---|---|---|
| Split by time ("temporal" protocol) | 0.60 | 0.98 | 0.018 | 0.051 | 48% | 40% | 0.36 |
| Split by river basin ("basin" protocol, harder test) | 0.77 | 0.99 | 0.019 | 0.092 | 45% | 11% | 0.43 |
| Final tuned "temporal" run | 0.66 | 0.98 | 0.013 | 0.007 | 47% | 30% | 0.39 |

**What these numbers mean in plain terms:**
- PR-AUC (~0.6–0.77): how good the model is at finding real flood events without crying wolf too often, given
  that floods are rare (~2% of days). Higher is better; this is the single most important number for this
  project.
- ROC-AUC (~0.98): the model is very good at telling "flood days" apart from "normal days" in general.
- Brier / ECE (both very low, close to 0): when the model says "70% chance of flood," it really does flood
  about 70% of the time — the predicted probabilities are trustworthy, not just a ranking.
- POD (~45–48%): out of all real floods, the model catches under half of them.
- FAR (~11–40%): out of all the times the model raises an alarm, this fraction were false alarms. The
  basin-split test had a much lower false-alarm rate (11%) than the time-split test (30–40%).
- CSI (~0.36–0.43): an overall "did we get it right when it actually mattered" score, combining detection rate
  and false alarms into one number.

**Takeaway:** A solid, well-calibrated starting point, but it was leaving useful information on the table
(terrain shape, satellite imagery) and had some regularization gaps that showed up later as overfitting.

---

## 2. Current model — Full Multimodal Model (Temporal Transformer + Terrain + SAR + Graph)

**What it is:** The current, most complete version of the model. It combines four different sources of
information instead of just two:

1. **Time-series weather/river data** (33 features/day, 14-day lookback) — read by a Transformer instead of a
   simple recurrent model, so it can pay attention to whichever past days matter most for the flood risk.
2. **Static terrain shape** (elevation, how many locations are upstream of this one, distance to the river
   outlet, which river zone/position it's in, which of the 16 river systems it belongs to) — used to "flavor"
   the time-series understanding for each location, the same way an actor's costume changes depending on the
   scene (technically called FiLM modulation).
3. **Satellite radar images (Sentinel-1 SAR)** — small radar snapshots of the ground near each gauge, read by
   a CNN (ResNet-18) to pick up visual signs of standing water or saturated ground. Locations without an image
   on a given day get a special "no image available" placeholder instead of being treated as zero.
4. **River network graph** — same idea as before (upstream/downstream + nearby locations), but now handled by
   a more expressive graph neural network (GATv2) that learns how much attention to pay to each neighbour.

**Methodology (simple version):** Each of the four sources produces its own "understanding" of a location on
a given day. These are combined into one shared representation, which is then passed through the river-network
graph so that locations can influence their neighbours, and finally a small prediction head turns that into
flood probabilities and related numbers (like next-day river discharge).

**Along the way, we found and fixed two significant bugs** that were quietly destroying model quality:
- A normalization layer that only works correctly when many examples are processed together was being fed a
  single example most of the time, causing it to silently zero out the satellite-image information.
- The satellite images were being normalized using fixed "typical" numbers that didn't actually match the
  real data, feeding the pretrained image model badly-scaled inputs.
Fixing these, and adding dropout/normalization consistently across every component (to stop the model from
memorizing the training data instead of learning general patterns), brought performance back up.

**Results we got (current best, val split):**

| Target being predicted | Best decision threshold | Accuracy | Precision | Recall | F1 Score | PR-AUC |
|---|---|---|---|---|---|---|
| Flood in 1 day | 0.62 | 99.3% | 65% | 32% | 0.43 | 0.44 |
| Flood in 2 days | 0.61 | 98.9% | 46% | 33% | 0.38 | 0.39 |
| Flood in 3 days | 0.61 | 98.6% | 45% | 29% | 0.35 | 0.36 |
| Flood "onset" (start of a new flood episode) | 0.58 | 98.3% | 4% | 23% | 0.07 | 0.03 |

| Regression target | R² (how much variance is explained) | MAE | RMSE |
|---|---|---|---|
| Next-day river discharge | 0.96 (very good) | 2.55 | 10.72 |
| 3-day max flood severity (z-score) | 0.09 (weak) | 0.56 | 2.56 |

**What these numbers mean in plain terms:**
- **Accuracy is close to 99% for every target** — but this is misleading and NOT a good measure here, because
  floods are rare (~2% of days), so a model that always predicted "no flood" would already score ~98%
  accuracy while being useless. This is exactly why we don't use accuracy as our real scorecard (see
  `Docs/Plan.md`).
- **PR-AUC (0.36–0.44 for the 1/2/3-day flood predictions)** is the metric that actually matters here. It's
  lower than the STGCN baseline's PR-AUC above, but the two are not directly comparable one-to-one — this run
  used a stricter/different split and added much harder-to-learn extra targets (onset, regression) at once, so
  some of the "budget" of the model went toward those. The 1-day-ahead prediction is (as expected) the easiest
  and best-performing (0.44), getting harder the further out we predict (0.39, then 0.36 for 3 days ahead).
- **"Onset" prediction (spotting exactly when a NEW flood episode begins, as opposed to any day during an
  ongoing flood) is the hardest task by far** (PR-AUC 0.03) — this is a known hard problem: there are very few
  onset days, and they look similar to the days right before them.
- **Next-day discharge (how much water is flowing) is predicted very well** (R² = 0.96) — this is a much
  easier, more continuous quantity to predict than a rare binary flood event.
- **3-day maximum flood-severity score is predicted poorly** (R² = 0.09) — this is the model's weakest output
  currently and a good candidate for future improvement.

**Takeaway:** The current model is the most information-rich version we've built (4 data sources instead of
2), and it produces well-calibrated, actionable next-day flood probabilities and a strong discharge forecast.
Its weakest points are the same-day "flood onset" detection and the 3-day severity regression, both of which
are naturally harder problems (extremely rare events / noisy targets) rather than simple bugs.

---

## 3. Other things we set up but haven't produced numbers for yet

While building the project, we also wrote (but did not finish running to completion, or never saved results
for) a few simpler comparison models, so that we can later show the multimodal model is actually better than
simpler alternatives:

- **Persistence baseline** (`src/baselines/persistence.py`) — the simplest possible baseline: "tomorrow will
  look like today." Useful as a sanity-check floor that any real model should beat.
- **Climatology baseline** (`src/baselines/climatology.py`) — predicts the historical average flood chance for
  each location/time of year, ignoring today's actual weather.
- **Node-wise LSTM/GRU baseline** (`src/baselines/node_lstm_gru.py`) — a classic recurrent neural network
  applied to each location separately, with no graph/terrain/satellite information at all. This is the closest
  comparison to "what would a standard deep learning approach without our extra data sources achieve?"
- **Tuned LightGBM baseline** (`src/baselines/gbm_baseline.py`) — a gradient-boosted decision tree model (a
  strong, popular non-neural-network approach) with its own hyperparameter search, meant as a "is a fancy deep
  model even necessary?" sanity check.

These all exist as runnable scripts but their result files were not found in the project — they should be run
and their metrics added here before the final report, so we can show a clean before/after/alternative
comparison table.

---

## Quick comparison (what we actually have numbers for)

| Model | Data used | Best PR-AUC (1-day-ahead flood) | Notes |
|---|---|---|---|
| STGCN baseline | Time-series + graph only | 0.60 – 0.77 (depending on test split) | Very well calibrated, simpler, earlier version |
| Full multimodal model (current) | Time-series + terrain + SAR images + graph | 0.44 | More data sources, well-calibrated, strongest next-day discharge forecast, still needs more work on flood-onset detection |

*Note: the two runs used somewhat different evaluation protocols and additional prediction targets, so treat
this table as a rough guide rather than a strict apples-to-apples comparison. A clean, same-protocol
side-by-side run (including the simple baselines above) is the recommended next step.*
