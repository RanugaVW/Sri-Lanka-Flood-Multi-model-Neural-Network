# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A multimodal spatio-temporal GNN that predicts flood risk (1/2/3-day-ahead probability, onset, next-day
discharge, 3-day max z-score) for 51 river-gauge nodes across Sri Lanka, fusing:

1. **Tabular time series** (33 dynamic features/day: precip, soil moisture, discharge, derived antecedent-
   precipitation/anomaly features) via a PLR-tokenizing Transformer.
2. **Static terrain** (elevation + flow-topology-derived drainage proxies + zone/position one-hot + a learned
   basin embedding, from `nodes.csv`/`edges.csv`) via FiLM modulation of the temporal embedding.
3. **Sentinel-1 SAR imagery** (VV/VH chips) via a ResNet-18 CNN with a learned "missing" embedding for
   nodes/days without a chip.
4. **River topology** (35 directed flow edges + 204 distance-weighted spatial edges) via a 2-layer GATv2Conv.

Built for CS3631 (Deep Learning coursework → paper). `Docs/Plan.md` has the full A–Z project/paper plan;
`Docs/ARCHITECTURE_SPEC.md` and `README.md` are the **original planning spec**, not the current implementation
— see "Docs vs. code" below before trusting either of them.

## Commands

There is no test runner config, linter, or build step beyond plain Python/pytest.

```bash
# Install deps
pip install -r requirements.txt

# Train (local paths from configs/data.yaml)
python src/train.py --data_config configs/data.yaml --experiment_dir experiments/<name>

# Train on Kaggle (paths from configs/kaggle_data.yaml, mounted datasets)
python src/train.py --data_config configs/kaggle_data.yaml --experiment_dir /kaggle/working/experiments

# Hyperparameter search (Optuna) — writes best_hparams.yaml
python src/hpo.py --data_config configs/kaggle_data.yaml --experiment_dir /kaggle/working/hpo --n_trials 30 --timeout 3600

# Standalone evaluation of a saved checkpoint (val split, seed 42 by default)
python src/eval/evaluate_metrics.py --experiment_dir experiments/<name> --data_config configs/data.yaml

# Tuned LightGBM baseline (separate script, own Optuna search)
python src/baselines/gbm_baseline.py --data_config configs/kaggle_data.yaml --output_dir experiments/gbm_baseline --n_trials 50

# Tests
pytest tests/
python tests/test_forward_pass.py
```

No dedicated single-test invocation exists beyond standard `pytest tests/test_forward_pass.py::test_forward_and_backward`.

## Architecture

### Data flow (one sample = one full-graph day snapshot, all 51 nodes at once)

`src/data/dataset.py::FloodDataset` loads `data/processed/flood_dataset.parquet` (410,931 rows × 67 cols,
51 nodes × ~8000 days, `target_flood_1d` positive rate ≈1.9%), pivots it into a dense `[T, N, F]` array for
O(1) sliding-window slicing, and returns per `__getitem__`:

- `temporal_features [N, 14, 33]` — 14-day lookback, z-scored (StandardScaler **fit on train split only**,
  reused for val/test — leakage control), heavy-tailed columns (precip/discharge) log1p'd first.
- `terrain_features [N, 10]` + `basin_idx [N]` — static, same every call, built once by
  `src/data/graph_builder.py`. `terrain_features` = `[elevation_m, upstream_node_count, distance_to_outlet_km]`
  (all z-scored across the 51 nodes) + `zone` one-hot (3) + `position` one-hot (4); `upstream_node_count`
  and `distance_to_outlet_km` are derived once from the flow-edge chain in `edges.csv` (depth from the basin
  headwater, and remaining flow-path length to the basin outlet — the flow graph here is a simple per-basin
  chain, no confluences, so these are well-defined). `basin_idx` (0–15, one of 16 river systems) is looked up
  separately as a learned embedding inside `FiLMTerrain` rather than one-hot, to avoid bloating the input width.
- `sar_chips [N, 2, 512, 512]` + `has_sar [N]` — see SAR caveat below.
- `targets [N, 6]`, `valid_mask [N]`, `label_conf [N]`, `event_ids [N]`, plus the static
  `edge_index_flow` / `edge_index_spatial` / `edge_weight_spatial`.

`DataLoader` always runs with `batch_size=1` — each "batch" *is* one graph snapshot (all 51 nodes), so
`train.py::_unpack` immediately does `batch[key][0]` to drop the fake batch dimension. Splits come from the
`split_temporal` column (train=2020–2022, val=2023, test=2024, per `configs/data.yaml`); a `split_basin_holdout`
column exists for a second (unused-by-default) protocol. `valid_sample` filters rows before anything else.

### Model pipeline (`src/models/flood_model.py::FloodModel`)

```
temporal_features ──► TemporalEncoder ──► FiLMTerrain ──┐
                    (temporal_encoder.py) (film_terrain.py) │
                                                             ├─► FusionBlock ──► GraphGNN ──► OutputHeads
sar_chips, has_sar ──────────────► SARCNN ─────────────────┘  (fusion.py)   (graph_gnn.py)  (heads.py)
                                  (sar_cnn.py)
```

- **TemporalEncoder** (`temporal_encoder.py`): *not* the GRU described in `ARCHITECTURE_SPEC.md`/`README.md`.
  It's a PLR (Periodic-Linear-ReLU) numeric-feature tokenizer feeding two parallel Transformer streams — one
  attending across the 14 days, one attending across the 33 feature channels — merged via
  `LayerNorm(h_temporal + h_feature)`. Output is always 128-dim regardless of config.
- **FiLMTerrain**: concatenates the 10-dim terrain vector with an 8-dim learned basin embedding (looked up via
  `basin_idx`), then `Linear(18,64)→ReLU→Dropout→Linear(64,256)` → gamma/beta, applied as
  `LayerNorm(gamma*h + beta + h)` (residual, so a degenerate FiLM branch can't erase the temporal signal).
- **SARCNN**: `InstanceNorm2d` (per-chip, unit-agnostic — replaced a bug-prone hardcoded dB normalization) →
  3×3 avg-pool despeckle → 1×1 conv to 3 channels → pretrained ResNet-18 → `Linear→LayerNorm` to 64-dim.
  Missing chips get a learned `nn.Parameter` embedding, never zero-padding.
- **FusionBlock**: concatenates the 128-dim FiLM output with the 64-dim SAR embedding (192-dim total — the
  spec's 320-dim double-counting-terrain option was deliberately rejected, see comment in `fusion.py`) →
  `Linear→ReLU→Dropout(0.2)→Linear→LayerNorm` (no trailing ReLU — negative dims are needed downstream).
- **GraphGNN**: flow edges (weight=1.0) and spatial edges (weight=`exp(-distance_km/40)`) are concatenated
  into one adjacency and run through 2 stacked `GATv2Conv` layers (4 heads), each followed by
  `LayerNorm`(+`Dropout(0.2)` after layer 1).
- **OutputHeads**: separate `Linear→GELU→Dropout→Linear` MLPs for 4 classification logits
  (`flood_t+1/t+2/t+3`, `onset`) and 2 regression outputs (`discharge_t+1`, `3-day max z-score`). Returns
  **raw logits**, never sigmoid — calibration happens post-hoc in `train.py`, and `BCEWithLogitsLoss` needs
  raw logits for numerical stability.

Every component is dropout+LayerNorm regularized end-to-end (see `Docs/DNN_Improvements_Report.md` for the
exact overfitting diagnosis and fixes — worth reading before touching regularization again, it documents two
previously-shipped silent bugs: `BatchNorm1d` collapsing to zero on single-sample SAR batches, and a hardcoded
dB-scale SAR normalization that assumed the wrong units).

### Loss, training, calibration (`src/losses/multitask_loss.py`, `src/train.py`)

- `MultiTaskLoss`: default is **plain BCE**, not focal — an earlier ablation found focal loss underperforms
  on this dataset (`configs/train.yaml` comment: BCE PR-AUC 0.8269 vs focal 0.7592). Focal/`focal_conf`
  variants exist but are opt-in via `train.yaml: loss:`. Head weights are fixed:
  `flood_t+1=1.0, flood_t+2=0.3, flood_t+3=0.3, onset=0.5`. Regression uses Huber loss, weighted 0.2.
- Early stopping is on **val PR-AUC of flood_t+1**, not loss (loss can improve while PR-AUC collapses at
  ~2% positive rate — this is documented behavior, not a bug to "fix").
- `configs/train.yaml: seeds:` drives a multi-seed deep ensemble — probabilities are averaged across seeds
  *before* calibration.
- Calibration (temperature scaling) and threshold selection (max-F1 on val) both live inline in `train.py`
  (`fit_temperature`, `best_threshold`), **not** in `src/calibration/` — that package only has `__init__.py`;
  the `temperature_scaling.py`/`isotonic.py`/`metrics.py` files described in `ARCHITECTURE_SPEC.md` were
  never built as separate modules.
- Training has a hard wall-clock budget (`max_time_secs = 10.0 * 3600` in `train.py::main`) to guarantee
  evaluation runs before Kaggle's 12h session kill — checkpointing is per-epoch with a resumable
  `last_checkpoint.pth` (optimizer + scheduler state included) alongside the best-PR-AUC `best_model.pth`.

### Evaluation (`src/eval/evaluate_metrics.py`)

Headline metrics follow the "MMF-Net" paper format: PR-AUC, ROC-AUC, `ev.det` (event detection rate — fraction
of the 1,469 flood episodes with ≥1 alarm in the 7-day pre-onset window; **not** inflated by within-episode
autocorrelated days), FAR, ECE (15-bin), POD, CSI, F1. **Never report plain accuracy as a headline metric** —
the ~2% positive rate makes it meaningless (this is called out explicitly in `Docs/Plan.md` Phase N).

### Configs

Four YAMLs are all read independently by each entry point (not composed): `configs/data.yaml` (local paths +
split years) / `configs/kaggle_data.yaml` (Kaggle-mounted dataset paths — placeholders need updating to your
actual mount), `configs/model.yaml` (architecture dims — note many of `TemporalEncoder`'s real hyperparameters,
e.g. transformer depth/heads, are hardcoded in `temporal_encoder.py`'s constructor defaults and not actually
threaded through `model.yaml`), `configs/train.yaml` (optimizer/loss/seeds). `configs/experiments/` is empty —
the per-workpackage ablation configs described in `ARCHITECTURE_SPEC.md` (`wp2_baselines.yaml` etc.) were never
created; workpackage results instead live as loose files under `experiments/wp*/`.

## Docs vs. code — read this before trusting a doc

`README.md` and `Docs/ARCHITECTURE_SPEC.md` describe the *original* Copilot build brief (GRU temporal encoder,
possible 320-dim fusion, `configs/experiments/*.yaml` ablation configs, separate `src/calibration/*.py`
modules). The implementation has since diverged in several places documented above (PLR-Transformer instead of
GRU, 192-dim fusion, calibration inlined in `train.py`). `walkthrough.md` is a chronological build log (also
stale past "Part 7"). `Docs/DNN_Improvements_Report.md` is the most current and accurate architecture doc —
prefer it, then the source, over `README.md`/`ARCHITECTURE_SPEC.md` for anything about current model internals.

## Known gaps worth knowing about before "improving" the model further

- ~~SAR coverage wired for only 1 of 9 sites, broadcast to all 51 nodes~~ — **fixed.** `SARIndex` now builds a
  per-`node_id` sub-index for every site directory under `data/sar_chips/*/frames/` that matches a `node_id`
  in `nodes.csv` (all 9 site IDs — `KEL_HAN`, `KEL_COL`, `KEL_KAD`, `KAL_BUL`, `KAL_KLT`, `KAL_RAT`, `NIL_AKU`,
  `NIL_MAT`, `NIL_PIT` — match exactly), and `FloodDataset.__getitem__` looks up each node's own chip
  independently instead of broadcasting one site's chip to all 51 nodes. `_decode_sar_png` is now
  `lru_cache`'d (same nearest chip is reused across ~`max_age_days` consecutive snapshots per node). The
  `sar_site` constructor arg is gone — site selection is automatic and per-node. Verified locally with a real
  CPU forward pass over the actual dataset (`has_sar` now lights up across 9 distinct node indices instead of
  all-51-or-nothing). See `IMAGE_DATASET.md` and `data/sar_chips/sar_flood_lite/README.txt` for the full
  manifest — `image_dataset.csv` there has a per-image `site_id`/`tabular_key` mapping to `node_id` that could
  drive a future standalone SAR-classifier pretraining step (Plan.md Phase I).
- ~~`nodes.csv` static features are thin (mostly zero-padding)~~ — **fixed.** `graph_builder.py` now derives
  `upstream_node_count` and `distance_to_outlet_km` from the flow-edge chain, one-hots `zone`/`position`, and
  passes `basin` through as a learned embedding inside `FiLMTerrain` (see Architecture above). `TERRAIN_DIM`
  is now 10 (was 9, ~half zero-padded). This is a real architecture change — `FiLMTerrain`/`FloodModel`'s
  `forward()` signatures gained a `basin_idx` argument, so old checkpoints don't load (expected, per the
  "clean best-effort" call — retrain from scratch).
- ~~`tests/test_forward_pass.py` is stale~~ — **fixed**, now matches `FloodModel.forward`'s dict return and
  `MultiTaskLoss.forward(out, targets, mask, conf)` signature, and passes `basin_idx`. Passes via both
  `pytest tests/test_forward_pass.py` and direct execution.
- `src/calibration/` and `configs/experiments/` are effectively empty (see above) — don't assume code lives
  there just because the spec docs say it should.
