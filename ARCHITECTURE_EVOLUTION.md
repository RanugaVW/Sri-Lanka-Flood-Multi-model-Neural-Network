# Architecture Evolution — Technical Reference

Every architecture generation we built, in order, with the flow, the components, why each exists, and the
numbers. Companion to `ARCHITECTURES_SUMMARY.md` (plain-English version) — this one is the technical one.

**Task (never changed):** 51 river-gauge nodes in Sri Lanka → given the last 14 days, predict flood
probability at t+1/t+2/t+3, flood onset, next-day discharge, and 3-day max discharge z-score.

**Key constraint:** `target_flood_1d` positive rate ≈ **1.9%**. Severe class imbalance drives almost every
design decision below.

---

## 0. Shared preprocessing (identical across all generations)

Applied in `src/data/dataset.py::FloodDataset`.

| # | Step | Detail | Why |
|---|---|---|---|
| 1 | Load panel | `flood_dataset.parquet`, 410,931 rows × 67 cols | One row = one node-day |
| 2 | Filter | keep `valid_sample == True` | Drops rows with insufficient history/targets |
| 3 | Select features | explicit curated list of **33** dynamic columns | Replaced a blind positional `[:33]` cutoff that kept `river_discharge` (exact duplicate of `discharge`, r=1.0) and dropped anomaly columns correlating 0.20–0.29 with the target |
| 4 | Log transform | signed `log1p` on 13 heavy-tailed cols (precip sums/maxes, discharge, API) | Rainfall/discharge are extremely right-skewed; log makes them roughly symmetric so z-scoring is meaningful |
| 5 | Z-score | `StandardScaler` **fit on train rows only** (277,899 rows), reused for val/test | **Leakage control** — fitting on all data would leak test-period statistics |
| 6 | Pivot | long panel → dense `[T, N, F]` array | O(1) sliding-window slicing instead of repeated dataframe filtering |
| 7 | Window | 14-day lookback per sample | Antecedent soil saturation matters for ~2 weeks |
| 8 | Split | precomputed `split_temporal` column: train 2003–2017 · val 2018–2020 · test 2021–2024 | Never random — flood days are autocorrelated, random splits leak. 2025 is dropped by `truncate_after` (ragged: 1,092 rows vs ~18,600/yr). **Note the val prevalence trough — see A7** |
| 9 | Batch | **A7 onward:** `batch_size=16` day snapshots, collated block-diagonally into one 816-node graph (`src/data/batching.py`). Was 1 through A6 | Each sample must keep all 51 nodes together to message-pass, but B samples can be batched as a disconnected graph — batch 1 meant 5,435 very noisy steps/epoch |

**Feature families (33):** precipitation (raw + 2/3/5/7/10/15/30-day sums, 3/7-day maxes, API, wet-days) ·
soil wetness (top/root/profile + anomalies) · discharge (raw, log, anomaly, z-score, percentile, 1/3-day
rise, 3/7-day means) · meteorology (temperature, humidity, radiation).

**Targets (6):** `target_flood_1d/2d/3d` · `target_onset_1d` · `target_next1d_discharge` ·
`target_next3d_max_zscore`.

---

## A1 · STGCN-Base

*Commits `bdc9449` → `6035a90`. The first working model.*

**Flow**

```
[N, 14, 33] ──► GRU (1-layer, h=128) ──► GATv2 ×2 ──► Linear head ──► 6 outputs
                                            ▲
                          flow + spatial edges
```

**Components**

| Component | Model used | Why this component | Why this model |
|---|---|---|---|
| Temporal encoder | 1-layer `nn.GRU`, hidden 128, final hidden state | Compress 14 days × 33 features into one vector | GRU = standard sequence baseline, cheap, fewer params than LSTM |
| Graph | 2× `GATv2Conv`, 4 heads | A flood upstream causes a flood downstream hours/days later — nodes are not independent | GATv2 *learns* how much to weight each neighbour (attention) instead of fixed averaging like GCN |
| Head | single Linear | Map embedding → predictions | Simplest possible |

**Data used:** tabular time series + river graph only. No terrain, no imagery.

**Results** (`Docs/stgcn_evaluation_results.md`)

| Protocol | PR-AUC | ROC-AUC | Brier | ECE | POD | FAR | CSI |
|---|---|---|---|---|---|---|---|
| temporal | 0.6023 | 0.9777 | 0.0179 | 0.0514 | 0.481 | 0.401 | 0.364 |
| basin holdout | 0.7669 | 0.9872 | 0.0195 | 0.0917 | 0.451 | 0.112 | 0.427 |
| temporal (tuned, thr 0.7084) | **0.6633** | 0.9817 | 0.0135 | 0.0070 | 0.474 | 0.301 | 0.394 |

**Limitation:** ignored terrain and imagery entirely; zero regularization in the GNN.

---

## A2 · MMF-Net v1 (first multimodal)

*Commit `4455705`. Added two more modalities + split the output heads.*

**Flow**

```
[N,14,33] ──► GRU ──► FiLM(terrain) ──┐
                                       ├─► Fusion MLP ──► GATv2 ×2 ──► cls head (4)
SAR chips ──► ResNet-18 ──────────────┘                            └─► reg head (2)
```

**What changed vs A1**

| Change | Why |
|---|---|
| **+ FiLM terrain conditioning** | Same rainfall means different flood risk at a steep headwater vs a flat outlet. FiLM lets static terrain *modulate* the temporal features (`γ·h + β`) rather than just being concatenated — conditioning, not extra input |
| **+ SAR CNN branch** (ResNet-18, pretrained) | Radar sees standing water / saturated ground through cloud — the one thing rainfall data can't confirm. ResNet-18 = small, ImageNet-pretrained, residual (trains stably) |
| **+ learned "missing" embedding** | Most node-days have no SAR chip. A learned vector says "no image" explicitly; zero-padding would be read as "image showing zero backscatter" |
| **Split cls/reg heads** | Classification and regression need different output scales; one shared head forces a compromise |

**Results** (val, `Docs/evaluation_results.md` @ this commit)

| Target | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| flood t+1 | 0.63 | 0.4674 | 0.3797 | 0.4190 |
| flood t+2 | 0.63 | 0.4702 | 0.3294 | 0.3874 |
| flood t+3 | 0.61 | 0.3664 | 0.3823 | 0.3742 |
| onset | 0.55 | 0.0216 | 0.2345 | 0.0396 |

Regression: `discharge_t1` R² **0.9449** · `zscore_3d_max` R² 0.0729.

> ⚠️ This generation carried **two silent bugs** (found later, see A3) that were corrupting the SAR branch the
> whole time.

---

## A3 · MMF-Net v2 (Regularized)

*Commits `3345cd9`, `43c598a`, `f0ccc27`. No new modalities — a correctness + regularization pass.*

**Diagnosis that triggered it:** train loss fell monotonically `1.83 → 0.46` while val PR-AUC peaked at epoch
10 (0.3303) then **collapsed to 0.0078** by epoch 14. Textbook overfitting (`Docs/DNN_Improvements_Report.md`).

**Two silent bugs fixed**

| Bug | What happened | Fix |
|---|---|---|
| `BatchNorm1d` on single-sample batches | Only a few of 51 nodes have SAR per snapshot. At V=1: `(x−x)/√(0+ε) = 0` → SAR embedding became **all zeros**, silently. No error, no NaN | → `LayerNorm` (normalizes over features, batch-size independent) |
| Hardcoded dB SAR normalization | Assumed `mean=(−10,−17) dB, std=(5,5)`. If chips aren't in dB, ResNet-18 gets garbage-scaled input | → `InstanceNorm2d(affine=True)` — per-chip, per-channel, **no unit assumption** |

Damage caused by these: val PR-AUC `0.3689 → 0.2242` and **0 detections** on test.

**Regularization added** (Dropout 0.2 + LayerNorm across *every* component; residual added to FiLM)

- **Why dropout 0.2 everywhere:** matches `TemporalEncoder`'s existing rate — consistent regularization strength per stage.
- **Why LayerNorm *before* Dropout:** stops dropout from acting on wildly-scaled activations (dropping a value of 50 ≠ dropping 0.1).
- **Why a residual on FiLM:** `γ·h + β` with large γ/β can completely **erase** the temporal signal. `LayerNorm(γ·h + β + h)` guarantees a path for the temporal features to survive a degenerate terrain branch.

---

## A4 · PLR-Former

*Commit `4a83981`. Replaced the temporal encoder entirely.*

**Why:** a GRU reads scalars raw and cannot cleanly represent **sharp threshold** behaviour — "flood risk jumps
once discharge crosses the 98th percentile". A decision tree gets that free from a split point; a plain
RNN/MLP struggles. This is the well-documented *failure of neural nets on tabular data*.
Fix per Gorishniy et al. (NeurIPS 2022): **a better input layer, not a bigger network**.

**Flow**

```
                 ┌─► Transformer over 14 DAYS ────────┐
[N,14,33] ─PLR──►┤                                     ├─► LayerNorm(sum) ─► FiLM ─► Fusion ─► GATv2 ─► heads
                 └─► Transformer over 33 FEATURES ────┘
```

**Components**

| Component | What it does | Why |
|---|---|---|
| **PLR embedding** | per-feature `x → [sin(2πcx), cos(2πcx)] → Linear → ReLU` | The periodic expansion lets the net represent sharp cut-points in a scalar — exactly what a tree split gives for free |
| **Stream 1: temporal Transformer** (3 layers, 4 heads, CLS token) | self-attention across the 14 days | Answers *when* the critical antecedent conditions built up; attends directly to day −12 without passing through 11 recurrent steps |
| **Stream 2: cross-feature attention** (1 layer, CLS token) | self-attention across the 33 channels | Answers *which features interact* — "heavy rain matters more when soil is already saturated" |
| Merge | `LayerNorm(h_temporal + h_feature)` | Residual-style sum, both streams contribute equally |

Pre-LN (`norm_first=True`) chosen so the stack trains without a warmup-sensitive phase.

**Results** (val, current `Docs/evaluation_results.md`)

| Target | Threshold | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| flood t+1 | 0.62 | 0.6473 | 0.3201 | 0.4284 | **0.4358** |
| flood t+2 | 0.61 | 0.4573 | 0.3311 | 0.3841 | 0.3899 |
| flood t+3 | 0.61 | 0.4544 | 0.2912 | 0.3549 | 0.3619 |
| onset | 0.58 | 0.0393 | 0.2345 | 0.0673 | 0.0276 |

Regression: `discharge_t1` R² **0.9565** · `zscore_3d_max` R² 0.0927.

---

## A5 · PLR-Former + Terra

*Commits `99671f7`, `e9fda84`, `5f505e5`. Fixed the data feeding the model, not the model.*

| Change | Before | After | Why it mattered |
|---|---|---|---|
| **SAR wiring** | 1 site's chip broadcast to all 51 nodes | per-`node_id` index across all **9** site dirs | Nodes were being shown another basin's radar image as if it were their own |
| **Terrain features** | 9 dims, ~half zero-padding | **10 real dims** + learned basin embedding | `upstream_node_count` + `distance_to_outlet_km` derived from the flow-edge chain; `zone`/`position` one-hot; basin as `nn.Embedding(16, 8)` — embedding not one-hot, to avoid bloating input width |
| **Feature selection** | positional `[:33]` cutoff | explicit curated list | Cutoff silently kept a duplicate column and dropped the strongest anomaly predictors |

**Measured** (val split, epoch-8 checkpoint, uncalibrated @ thr 0.5 — run locally via `evaluate_metrics.py`)

| Target | PR-AUC | ROC-AUC | FAR | ECE | POD | CSI | F1 |
|---|---|---|---|---|---|---|---|
| flood t+1 | **0.4387** | 0.9583 | 0.315 | 0.0027 | 0.274 | 0.243 | 0.391 |
| flood t+2 | 0.3800 | 0.9433 | 0.330 | 0.0031 | 0.231 | 0.207 | 0.343 |
| flood t+3 | 0.3569 | 0.9330 | 0.318 | 0.0038 | 0.199 | 0.182 | 0.308 |
| onset | 0.0264 | 0.9039 | 0.000 | 0.0016 | 0.000 | 0.000 | 0.000 |

Regression: `discharge_t1` R² **0.9532** (MAE 2.44) · `zscore_3d_max` R² 0.0406.
Test-split run (earlier session): flood t+1 PR-AUC **0.5154**, ev.det 0.327, mean lead **5.5 days**.

---

## A6 · BiPLR-Former

*Commit `c943fc8`. The 60-epoch Kaggle run completed — see A7 for what it showed.*

**Flow (full current pipeline)**

```
                 ┌─► Transformer over 14 DAYS ──────┐
[N,14,33] ─PLR──►├─► Transformer over 33 FEATURES ──┤─►LN(sum)─► FiLM ─┐
                 └─► BiGRU forward+backward ────────┘    ▲ terrain      │
                                                    basin emb          ├─► Fusion ─► GATv2(+residual) ─► heads
SAR [N,2,512,512] ─► InstanceNorm ─► despeckle ─► ResNet-18 ─► 64d ────┘
```

**Two changes**

| Change | Why |
|---|---|
| **+ Bidirectional GRU stream** (3rd temporal stream) | Self-attention is order-agnostic except through the additive positional embedding — it can see *which* days had rain but represents **trend direction** only weakly. A BiGRU walking oldest→newest and newest→oldest encodes "3 days of *climbing* discharge" distinctly from the same values reversed |
| **+ Residual around GraphGNN** | `GraphGNN` had **no skip path** — unlike FiLM, which has had one since A3. With 2 randomly-initialised GATv2 layers, a not-yet-useful graph signal can *corrupt* the good fused embedding instead of merely contributing nothing. The residual lets training **discount the graph** if message passing isn't earning its place |
| `grad_clip_norm` 1.0 → 2.0 | Gradient logging on a near-identical architecture showed the clip binding on **30–54% of steps** — throttling real learning signal, not just preventing blowups |

⚠️ Adds new params (`dir_gru`, `dir_norm`) → **old checkpoints do not load**; retrain from scratch.

---

## A7 · BiPLR-Former, re-scoped ⭐ current

*The A6 run finished: **test PR-AUC 0.6625** (flood t+1), val-selected at 0.5356, ev.det 0.173,
onset PR-AUC 0.0533, 12.1M params, 8.8 h for 34 epochs. Still ~0.19 AP below LightGBM on this same
panel. A7 is the response, and every change below is tied to a measurement rather than a preference.*

### What the A6 run actually showed

Four things, in descending order of how much they cost:

**1. The validation split cannot carry the selection decision.** Measured directly from the parquet:

| split | node-days | positives | positive rate |
|---|---:|---:|---:|
| train (2003–2017) | 277,899 | 5,635 | 2.03% |
| **val (2018–2020)** | **55,896** | **453** | **0.81%** |
| test (2021–2024) | 75,555 | 1,719 | 2.28% |

The val window is a **2.8× prevalence trough**, and PR-AUC scales with prevalence — so val 0.5356 and
test 0.6625 are not a generalisation gap, they are largely the same model scored at two base rates.
Worse, early stopping, the temperature (T = 1.2704) and the max-F1 threshold (0.3218) were *all* fitted
on those 453 positives. The val trace sat in [0.48, 0.54] for twenty consecutive epochs with no trend:
the checkpoint was chosen off noise. Per-year rates make the cause plain — 2020 alone is 0.27%
(50 positives), the quietest year in the panel.

**2. The SAR branch was 92.4% of the model for a signal with 9/51 node coverage.** 11,209,549 of
12,125,539 parameters, trained end-to-end through a 1.9%-positive target. The sibling project measured
this exact branch as a null **twice** — `M6_cnn` at event PR-AUC 0.2988 vs 0.3007 without imagery, and
`N6_gated` below `N5_bce` at 17× the parameters. Meanwhile our *tabular* encoder is 652k params, and the
sibling's 711k graph-free tabular model scores **0.8355**. The capacity was in the wrong place.

**3. Message passing is measured as a cost on this panel, and the spatial edges carry it.** From the
sibling's paired block bootstrap, graph as the only moving part:

| contrast | Δ PR-AUC | 95% CI |
|---|---:|---|
| `P1 − P0` — add 204 spatial k-NN edges | **−0.0872** | [−0.1074, −0.0680] |
| `P3 − P0_x5` — graph vs graph-free | **−0.0828** | [−0.1003, −0.0668] |
| `P2 − P1` — add 35 flow edges on top | −0.0035 | spans zero |

A6's un-gated residual let training *discount* the graph but never switch it off.

**4. `configs/model.yaml` was decorative.** `FloodModel.__init__` accepted `config` and never read it —
every submodule used its constructor defaults. `sar_cnn.enabled: false` did nothing; `graph.num_layers`
did nothing. Any ablation "configured" through that file would have silently run the same model.

### Changes

| # | Change | Why | Where |
|---|---|---|---|
| 1 | **`FloodModel` now parses `config`** | Every key in `model.yaml` is threaded to its submodule. Ablations become real knobs instead of dead text | `flood_model.py` |
| 2 | **SAR off by default** (`sar_cnn.enabled: false`) → **903,703 params** | Removes 92.4% of the weight budget for a twice-measured null. Lands the model in the 711k–980k regime where comparable models score 0.83+ | `model.yaml`, `flood_model.py` |
| 3 | **Gated SAR fusion** replaces concatenation (`fusion.mode: gated`) | Gate conditioned on state ‖ embedding ‖ presence ‖ **frame age**, zero-init weights + bias −3 → branch starts shut, and `* pres` makes it exactly zero with no frame. Concatenation forced the net to distinguish "no water" from "no picture" on every sample, including the 82% of nodes that can never have one | `fusion.py` |
| 4 | **Graph residual gate `tanh(α)`, α init 0** | At init the GNN block is *exactly* the identity, so the graph is opt-in and must earn α off zero. Also a single readable number: α staying near 0 is reportable evidence, not a suspicion | `graph_gnn.py` |
| 5 | **`graph.mode: flow`** (was flow+spatial) | The spatial edges are where the measured −0.0872 lives; flow edges were neutral. `none` added for the graph-free control | `graph_gnn.py`, `model.yaml` |
| 6 | **Batch 16 snapshots/step** (was 1) | 816 node-days per step instead of 51; 340 steps/epoch instead of 5,435. Block-diagonal edge offsets, with a test asserting batched == single forward so there is no cross-day message leak | `data/batching.py` |
| 7 | **5-epoch linear LR warmup** before cosine | Pre-LN transformers plus two zero-init gates are all sensitive to a large first step at lr 1e-3 | `train.py`, `train.yaml` |
| 8 | **Select on 3-horizon mean PR-AUC** | Triples the positives feeding the selection signal (t+1/t+2/t+3) at zero extra inference cost. Does not fix the prevalence trough, but stops the *noise* half of problem 1 | `train.py` |
| 9 | **`truncate_after: 2024-12-31`** | 2025 has 1,092 valid rows vs ~18,600/complete year — a 3-week sliver folded invisibly into a 4-year average. Matches the sibling's ragged-2025 policy | `dataset.py`, `data.yaml` |
| 10 | **No 107 MB zero-chip when SAR is off** | The dataset allocated and copied `[51,2,512,512]` float32 per snapshot regardless; now a `[51,2,1,1]` placeholder | `dataset.py` |
| 11 | **`ev.det` reporting bug fixed** | The episode summary printed `nan` while the table printed the real 0.173 — `ev_det` was reused as the loop variable and left nan by the last aux head | `evaluate_metrics.py` |
| 12 | **HPO rebuilt on `FloodModel`, maximising PR-AUC** | It hand-rolled a duplicate `TrialFloodModel` (invisible to all of the above) and **minimised val loss** — the one signal this repo documents as wrong at a 2% positive rate | `hpo.py` |

**Flow (A7 headline — SAR off, flow edges only)**

```
                 ┌─► Transformer over 14 DAYS ──────┐
[N,14,33] ─PLR─►├─► Transformer over 33 FEATURES ──┤─►LN(sum)─► FiLM ─► Fusion ─► GATv2 ─► heads
                 └─► BiGRU forward+backward ────────┘    ▲ terrain        MLP      ▲ tanh(α)·h
                                                    basin emb                      residual, α init 0

SAR branch: built only when sar_cnn.enabled, and then entering through the gate:
  [N,2,512,512] ─► InstanceNorm ─► despeckle ─► ResNet-18(frozen) ─► 64d ─► g⊙Wv·pres ─┘
```

### Parameter budget (A7 headline vs A6)

| Component | A6 | A7 headline | A7 share |
|---|---:|---:|---:|
| Temporal encoder | 651,664 | 651,664 | **72.1%** |
| GraphGNN (flow) | 167,168 | 167,169 | 18.5% |
| Fusion | 62,016 | 49,728 | 5.5% |
| FiLM terrain | 18,240 | 18,240 | 2.0% |
| Output heads | 16,902 | 16,902 | 1.9% |
| SAR CNN | 11,209,549 | **0** (opt-in) | — |
| **Total** | **12,125,539** | **903,703** | |

13.4× smaller, and the capacity is now in the encoder that does the work.

⚠️ `FiLMTerrain`/`FusionBlock`/`GraphGNN` all changed shape and `forward` signatures gained
`sar_age_days`. **Old checkpoints do not load — retrain from scratch.**

### What A7 does *not* fix

The prevalence trough itself. Change 8 reduces the selection *variance*, but selection, temperature and
threshold still come from a 0.81%-positive window being used to predict a 2.28% one. The sibling hit the
same wall from the other side: it calibrated on 2020 alone and its Brier *worsened* (0.00718 → 0.00772),
with calibrated ECE nearly doubling. The real fix is a split protocol change — chronological development
folds with enough flood events, chosen without touching the exposed test years — and that is the next
piece of work, not something to settle by trying thresholds until the test number improves.

---

## Parameter budget (current, A6)

| Component | Params | Share |
|---|---|---|
| **SAR CNN (ResNet-18)** | **11,209,549** | **92.4%** |
| Temporal encoder | 651,664 | 5.4% |
| GraphGNN | 167,168 | 1.4% |
| Fusion | 62,016 | 0.5% |
| FiLM terrain | 18,240 | 0.2% |
| Output heads | 16,902 | 0.1% |
| **Total** | **12,125,539** | |

> **The single most important fact on this page:** the SAR branch is **92% of the model** and reaches only
> **9 of 51 nodes**. A sibling project on the same dataset found its SAR branch added *nothing* once the loss
> was fixed — its best model scored **0.8355 PR-AUC with no imagery at 711k params**, beating its own
> 11.9M-param imagery model (0.8310). Our 11.2M SAR params are the obvious first ablation.

---

## Results at a glance

| Gen | Name | Key change | Val PR-AUC (t+1) | discharge R² |
|---|---|---|---|---|
| A1 | STGCN-Base | GRU + GATv2 | **0.6633** † | — |
| A2 | MMF-Net v1 | + FiLM + SAR | — (F1 0.419) | 0.9449 |
| A3 | MMF-Net v2 | bug fixes + regularization | 0.3303 → recovery | — |
| A4 | PLR-Former | GRU → PLR-Transformer | 0.4358 | 0.9565 |
| A5 | +Terra | real terrain, per-node SAR | 0.4387 | 0.9532 |
| A6 | BiPLR-Former | BiGRU + GNN residual | *training* | *training* |

† **Not apples-to-apples.** A1 was evaluated on a different protocol and predicted fewer targets. A2+ carry 6
simultaneous targets (including onset and two regressions), spending capacity on them. Treat the A1 number as
a flag to investigate, **not** proof that the simpler model is better. A clean same-protocol rerun of all
generations is the honest next step.

### External benchmarks (sibling project, same dataset)

| Reference model | PR-AUC |
|---|---|
| Gradient-boosted trees (LightGBM) | 0.8496 |
| MMF-Net `N5_bce` (no graph, no imagery, 711k) | 0.8355 |
| Operational discharge-percentile rule | 0.8164 |
| TF-STGNN (graph-based) | 0.7421 |

**We are well below all four.** The gap is the open problem, not a rounding error.

---

## Training setup (shared, A3 onward)

| Setting | Value | Why |
|---|---|---|
| Loss | **BCE** (not focal) | Ablation: BCE 0.8269 vs focal 0.7592 PR-AUC. Focal is a *defect* here, not a cost |
| Head weights | t+1 = 1.0 · t+2 = 0.3 · t+3 = 0.3 · onset = 0.5 | t+1 is the operational target; others are auxiliary |
| Regression loss | Huber, weight 0.2 | Robust to discharge outliers; low weight so it doesn't dominate classification |
| Optimizer | AdamW, lr 1e-3, wd 1e-4 | Decoupled weight decay |
| Scheduler | **A7:** 5-epoch linear warmup → cosine decay. Plain `CosineAnnealingLR` through A6 | Pre-LN transformers and the two zero-init gates are sensitive to a large first step at lr 1e-3 |
| Early stopping | **A7:** on **mean val PR-AUC over flood t+1/t+2/t+3**, patience 10. Was t+1 alone through A6 | Loss can improve while PR-AUC collapses at 1.9% positive rate — so never loss. Averaging the three horizons triples the positives behind the decision, because val holds only 453 |
| Calibration | temperature scaling, max-F1 threshold on val | Raw logits are overconfident; ECE drops to ~0.002–0.005 |
| Time budget | 10 h hard wall-clock | Kaggle kills sessions at ~12 h and a killed session **saves nothing** |

**Never report plain accuracy as a headline.** At a 1.9% positive rate, always predicting "no flood" scores
~98% accuracy while being useless. Headline metrics are PR-AUC, ev.det, FAR, ECE, POD, CSI, F1.

---

## Open problems

1. **The gap to LightGBM (0.8606) and the sibling's graph-free 711k net (0.8355)** remains the main open
   question. A7 removes four confounds — a null branch holding 92% of params, an un-gated graph with a
   measured cost, batch-size-1 gradient noise, and a config file that wasn't read — but removing
   confounds is not the same as closing the gap. The A7 number is not in yet.
2. **The validation prevalence trough** (0.81% vs 2.28% in test). Selection, temperature and threshold
   all still come from it. Needs a split-protocol fix, not a metric tweak. See A7's closing note.
3. **Onset prediction is broken** (PR-AUC 0.0533, POD 0.000) — and it is broken for *everyone* on this
   panel: the sibling's best onset AP is 0.2225, LightGBM's 0.2061, and its constrained early-warning
   policies warn on 2 of 355 event starts. This is likely a label-definition limit, not a capacity one.
4. **`zscore_3d_max` regression is weak** (R² 0.067) vs discharge (R² 0.961).
5. **Single seed.** `seeds: [42]` has no error bar, and the sibling measured a **0.0483 PR-AUC**
   seed-noise floor (2 sd) — wider than most of the contrasts in the A7 table. Nothing here is a claim
   until it is run at 3–5 seeds.
6. **Baselines never run to completion** — `persistence.py`, `climatology.py`, `node_lstm_gru.py`,
   `gbm_baseline.py` all exist but have no saved results. Required before any paper claim, and the
   cheapest of them (`discharge_pctl`) scores 0.8164 on the sibling's panel, i.e. above our DNN.
7. **SAR may still be worth one honest run** — but as change 3 configures it: frozen pretrained encoder
   through the gate, pretrained on `image_manifest.csv` (3,489 chips, all 51 nodes, **40% positive
   rate**) rather than learned through a 1.9%-positive target. That separates "can a CNN see flooding in
   Sentinel-1?" from "does seeing it help predict tomorrow's discharge?" — two questions A2–A6
   confounded into one null.
