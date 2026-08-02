# Copilot Build Brief — Terrain-Aware Multimodal Flood GNN
### Sri Lanka flood early-warning benchmark | Kelani river focus | CS3631

This is a spec document, not a prompt you paste all at once. Give Copilot one section at a time (§4 shows the
recommended order), and paste the relevant piece of this doc into the chat/comment above the file you're generating.
Keep this file itself in your repo at `docs/ARCHITECTURE_SPEC.md` so Copilot (and your teammates) can reference it
inline via `@workspace` or `#file` mentions in every subsequent session.

---

## 0. Hard constraints Copilot must respect (paste this into every session as a system-level instruction)

> Build this using only: PyTorch, PyTorch Geometric, NumPy, Pandas, scikit-learn. Do NOT use any LLM, agentic
> framework, prompt-engineering approach, or pretrained-and-fine-tuned foundation model anywhere in the trainable
> pipeline (CS3631 rules). The CNN branch may use a ResNet-18 **architecture** trained from scratch (ImageNet
> pretrained weights are borderline — ask before using; safest is random init given the constraint against "large
> foundation model fine-tuning"). Every module must be trainable end-to-end on a single Colab/Kaggle T4 GPU.

---

## 1. Target repository folder structure

Give Copilot this tree directly — it's the most reliable way to get consistent file placement across prompts.

```
flood-early-warning/
├── docs/
│   ├── ARCHITECTURE_SPEC.md          # this file
│   ├── README.md                     # update: fix superseded claims per report Appendix B
│   └── IMAGE_DATASET.md              # update: fix superseded claims per report Appendix B
│
├── configs/
│   ├── data.yaml                     # paths, split protocol, valid_sample filters
│   ├── model.yaml                    # architecture hyperparameters (§3 below)
│   ├── train.yaml                    # optimizer, loss weights, epochs, seeds
│   └── experiments/
│       ├── wp2_baselines.yaml
│       ├── wp3_graph_ablation.yaml
│       ├── wp4_calibration.yaml
│       └── wp5_sar_ablation.yaml
│
├── data/
│   ├── raw/                          # cached API pulls (weather, discharge, DEM, SAR) — gitignored
│   ├── processed/
│   │   ├── panel.parquet             # the 410,931-row node-day table
│   │   ├── nodes.csv                 # 51 nodes, static terrain features
│   │   ├── edges.csv                 # 35 flow + 204 spatial edges
│   │   ├── splits/
│   │   │   ├── temporal_train.csv / temporal_val.csv / temporal_test.csv
│   │   │   ├── basin_holdout_train.csv / basin_holdout_gin.csv
│   │   │   └── event_groups.csv      # event_id -> GroupKFold assignment
│   │   └── image_dataset.csv         # SAR chip manifest, once WP1/WP5 complete it
│   └── sar_chips/                    # 2x512x512 dB arrays, one .npy per (node, date)
│
├── src/
│   ├── data/
│   │   ├── dataset.py                # PyG Dataset/InMemoryDataset — see §2.1
│   │   ├── graph_builder.py          # nodes.csv + edges.csv -> torch_geometric.data.Data
│   │   ├── sar_loader.py             # chip loading + presence-mask handling
│   │   └── splits.py                 # split-protocol-aware DataLoader factory
│   │
│   ├── models/
│   │   ├── temporal_encoder.py       # GRU / TCN / temporal-attention (config-switchable) — §2.2
│   │   ├── film_terrain.py           # static MLP -> (gamma, beta), FiLM modulation — §2.3
│   │   ├── sar_cnn.py                # ResNet-18-style branch, presence-mask-aware — §2.4
│   │   ├── fusion.py                 # concat -> Linear[320,192] -> Linear+ReLU[192,128] — §2.5
│   │   ├── graph_gnn.py              # 2-layer GATv2, flow+spatial edges — §2.6
│   │   ├── heads.py                  # per-node MLP -> multi-task output — §2.7
│   │   └── flood_model.py            # top-level nn.Module wiring all of the above
│   │
│   ├── losses/
│   │   └── multitask_loss.py         # focal BCE (label_confidence-weighted) + Huber regression terms
│   │
│   ├── calibration/
│   │   ├── temperature_scaling.py
│   │   ├── isotonic.py
│   │   └── metrics.py                # Brier, ECE(15 bins), reliability diagram, Brier decomposition
│   │
│   ├── eval/
│   │   ├── ranking_metrics.py        # PR-AUC (headline), ROC-AUC
│   │   ├── operational_metrics.py    # POD/recall, FAR, CSI, F1 @ selected thresholds
│   │   ├── event_metrics.py          # per-event lead-time detection (1,469 events)
│   │   └── worst_node_report.py      # Kirschstein-&-Sun-style worst-gauge/node diagnostic
│   │
│   ├── baselines/
│   │   ├── persistence.py
│   │   ├── climatology.py
│   │   ├── discharge_percentile.py
│   │   ├── gbm_baseline.py           # gradient-boosted trees
│   │   └── node_lstm_gru.py          # per-node sequence baseline, no graph
│   │
│   └── train.py                      # single entry point, reads configs/*.yaml
│
├── experiments/                      # one subfolder per WP run, holds logs + checkpoints + metrics.json
│   ├── wp2_baselines/
│   ├── wp3_graph_ablation/
│   ├── wp4_calibration/
│   └── wp5_sar_case_study/
│
├── notebooks/                        # Colab-facing exploratory notebooks only — no production logic here
│
├── tests/
│   ├── test_graph_builder.py
│   ├── test_film_shapes.py
│   ├── test_fusion_shapes.py
│   └── test_no_leakage.py            # asserts valid_sample + split protocols hold
│
├── requirements.txt
└── .gitignore                        # data/raw/, data/sar_chips/, experiments/*/checkpoints/
```

**Migration note if your current repo doesn't look like this:** don't do a single big-bang reorg. Ask Copilot to move
one top-level concern at a time — e.g. "move all model-building code currently in `notebooks/` into
`src/models/`, splitting by module as described in ARCHITECTURE_SPEC.md §1, and update imports" — and run your test
suite after each move.

---

## 2. Architecture spec, module by module (exact shapes from the feasibility report diagram, page 6)

Paste each subsection individually as a Copilot prompt when generating that file.

### 2.1 `data/dataset.py` + `data/graph_builder.py`
- Load `panel.parquet` (410,931 rows × 67 cols), filter to `valid_sample == True` (409,350 rows).
- Build a single `torch_geometric.data.Data` object per timestep (or a `Batch` over a sliding window) with:
  - `x`: static node features (9 terrain features) — used only by the FiLM branch, not concatenated into GATv2 input directly.
  - `edge_index_flow`: directed, 35 edges (upstream→downstream).
  - `edge_index_spatial`: weighted, 204 edges, weight = `exp(-distance_km / 40)`.
  - Keep flow and spatial edges as **separate** `edge_index`/`edge_weight` pairs (don't merge into one adjacency) so
    GATv2 can optionally be run per-edge-type if you adopt the dual-branch ablation from §3 of the earlier review.
- Respect the three split protocols (temporal / basin holdout / event GroupKFold) as **separate DataLoader configs**,
  not baked into the Dataset class — one Dataset, three samplers.

### 2.2 `models/temporal_encoder.py` — Modality 1
- Input: `[batch, 14 days, 33 features]` per node (config: window k ∈ {7,14,30}, default 14 per the diagram).
- Config-switchable: `type: gru | tcn | temporal_attention`.
- GRU variant: single-direction `nn.GRU(input_size=33, hidden_size=128, num_layers=1, batch_first=True)`, take final
  hidden state → `[batch, 128]`.
- Output dimension **must be 128** to match the fusion input in §2.5.

### 2.3 `models/film_terrain.py` — Modality 2
- Input: `[batch, 9]` static terrain features.
- `nn.Sequential(Linear(9, 64), ReLU(), Linear(64, 256))` → split into `gamma [batch,128]`, `beta [batch,128]`.
- Apply to the temporal encoder's output: `modulated = gamma * temporal_out + beta`. **This FiLM output is what
  feeds the fusion block** — don't also concatenate the raw terrain features separately, or you'll double-count
  static information and the ablation in §3 (FiLM vs. plain-concat) won't be clean.

### 2.4 `models/sar_cnn.py` — Modality 3
- Input: `[batch, 2, 512, 512]` (VV+VH dB channels).
- ResNet-18-style CNN (from scratch — see §0 constraint), final pooled embedding `[batch, 64]`.
- **Presence mask**: not every node/date has a SAR chip. Implement a boolean `has_sar` flag per sample; when
  `False`, feed a learned "missing" embedding (a single `nn.Parameter([64])`) instead of running the CNN, rather than
  zero-padding — zero-padding gets confused with a genuine dark/no-water SAR signature.

### 2.5 `models/fusion.py` — Fusion block
- Concatenate `[128 (temporal+FiLM) | 128 (unused if FiLM already modulates — see note) | 64 (SAR or missing-embed)]`.
- **Note on the diagram's "128+128+64=320":** the diagram treats the FiLM-modulated temporal stream and a *separate*
  128-dim terrain branch as two inputs to fusion. Clarify with your team whether FiLM modulation (§2.3) replaces or
  supplements a second terrain embedding — implement whichever your team decided, but make the choice explicit in a
  code comment, since it changes the input dimension to fusion (192 vs 320).
- `Linear(320, 192) → Linear(192, 128) + ReLU` → one 128-dim vector per node.

### 2.6 `models/graph_gnn.py` — Modality 4
- Input: `[51 nodes, 128]` fused per-node vectors + `edge_index_flow` + `edge_index_spatial` (+ weights).
- `torch_geometric.nn.GATv2Conv`, 2 layers, multi-head attention (start with 4 heads, head_dim chosen so output
  stays 128-dim after concat or average — check PyG's `concat=False` option for the final layer to avoid dimension
  blow-up).
- Feed both edge sets into the same GATv2 stack (simplest, matches your original diagram) **or** run two parallel
  GATv2 branches (one per edge type) with a HydroGAT-style learnable gate fusing them — flag this as an explicit
  ablation switch in `model.yaml` (`graph_mode: single | dual_branch_gated`), not two separate code paths.

### 2.7 `models/heads.py` — Output heads
- Per-node MLP: `Linear(128, 64) → ReLU → Linear(64, 6)` for the 6 outputs: `P(flood t+1)`, `P(flood t+2)`,
  `P(flood t+3)`, onset (binary), next-day discharge (regression), 3-day max z-score (regression).
- Apply sigmoid to the three flood-probability outputs and the onset logit; leave discharge/z-score outputs linear.
- **Calibration happens post-hoc** (temperature scaling / isotonic on the validation block only) — don't calibrate
  inside the forward pass; keep `calibration/` as a separate post-training step per the report's WP4.

---

## 3. `configs/model.yaml` — starter values to give Copilot verbatim

```yaml
temporal_encoder:
  type: gru                 # gru | tcn | temporal_attention
  window_days: 14            # try 7, 14, 30 as an ablation
  hidden_dim: 128
  num_layers: 1

film_terrain:
  input_dim: 9
  hidden_dim: 64
  output_dim: 128            # gamma/beta each 128-dim

sar_cnn:
  enabled: true               # false = ablation without SAR
  input_channels: 2
  input_size: 512
  embedding_dim: 64
  backbone: resnet18_scratch
  missing_chip_strategy: learned_embedding   # vs. zero_pad

fusion:
  concat_dim: 320             # confirm 320 vs 192 per §2.5 note before running
  hidden_dim: 192
  output_dim: 128

graph:
  mode: single                # single | dual_branch_gated
  layer: gatv2
  num_layers: 2
  heads: 4
  spatial_edge_decay_km: 40
  learn_edge_weights: true    # ablation: compare against fixed exp(-d/40) per HydroGAT finding

heads:
  hidden_dim: 64
  outputs: [flood_t1, flood_t2, flood_t3, onset, discharge_t1, zscore_3d_max]

loss:
  flood_loss: focal_bce
  focal_gamma: 2.0
  use_label_confidence_weighting: true
  regression_loss: huber
  regression_weight: 0.3
```

---

## 4. Recommended build order (feed to Copilot one at a time, don't ask for everything at once)

1. `graph_builder.py` + `dataset.py` (get data loading and shapes right first — everything downstream depends on it)
2. `temporal_encoder.py` alone, unit-tested in isolation against a dummy `[batch,14,33]` tensor
3. `film_terrain.py` alone, unit-tested against dummy `[batch,9]`
4. `fusion.py`, tested by wiring 2+3 together with a dummy SAR embedding
5. `sar_cnn.py`, including the presence-mask logic
6. `graph_gnn.py` — this is the riskiest module to get right; ask Copilot to generate it with the PyG `GATv2Conv`
   docstring/signature pasted in explicitly, since edge_index/edge_attr conventions are easy to get backwards
7. `heads.py` + `flood_model.py` (wire the whole thing end-to-end, run one forward pass on a single batch before
   writing any training loop)
8. `losses/multitask_loss.py`
9. `train.py` — start with WP2 baselines (persistence/climatology/GBM/per-node LSTM) before touching the full model,
   exactly as your feasibility report's WP ordering specifies
10. `calibration/` + `eval/` modules — last, and only once the model trains end-to-end without errors

---

## 5. Guardrail prompt to reuse before every Copilot session

> Before writing code: check this matches ARCHITECTURE_SPEC.md exactly — same tensor shapes, same file location per
> the §1 folder tree, no LLM/agentic/foundation-model components. If a design choice in the spec is ambiguous (e.g.
> §2.5's 320-vs-192 fusion input), stop and ask rather than guessing.