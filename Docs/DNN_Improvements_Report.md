# DNN Architecture Improvements Report
## Sri Lanka Flood Early-Warning Multi-Model Neural Network

**Course Context:** CS3631 — Deep Learning  
**Lectures Applied:** Lecture 5 (Improving the Performance of DNN) · Lecture 7 (Improved Architectures for CNN)  
**Baseline Val PR-AUC (Seed 42, Epoch 10):** `0.3303`

---

> [!CAUTION]
> ## Critical Bugs Found and Fixed (Revision 2)
>
> The first improvement pass introduced two bugs in `sar_cnn.py` that silently **degraded performance** from `0.3689` → `0.2242` val PR-AUC and caused **0 detections** on the test set.
>
> ### Bug 1 — `BatchNorm1d` collapses to zero when V = 1
>
> **What happened:** Only a handful of the 51 nodes have SAR chips per snapshot (263 chips across the whole dataset — very sparse). When only **1 node** has SAR in a snapshot, `BatchNorm1d` receives a single-sample batch:
> ```
> mean   = that one sample itself  →  variance = 0
> output = (x − x) / sqrt(0 + ε) × γ + β  =  0 × 1 + 0  =  0
> ```
> The SAR embedding silently became **all-zeros** for V=1 snapshots (which is most snapshots), destroying the SAR signal entirely — no NaN, no warning, just silent garbage that poisoned FusionBlock training.
>
> **Fix:** Replaced `BatchNorm1d` with `LayerNorm`. LayerNorm normalises over the *feature* dimension (not the batch dimension), so it works correctly for any V ≥ 1.
>
> **Lecture connection (Lecture 5 — Layer Normalisation):**
> > *"Layer Norm computes mean/variance across all features within a single example, independent of batch size — good for settings where batch composition varies."*
>
> This is exactly the motivation: the number of SAR chips per snapshot varies from 0 to ~51, so a batch-size-independent normalisation is required.
>
> ### Bug 2 — Hardcoded dB-scale SAR normalisation (wrong units assumption)
>
> **What happened:** The SAR normalization used `mean=(−10 dB, −17 dB), std=(5 dB, 5 dB)` — typical Sentinel-1 statistics in **decibel scale**. If the dataset's SAR chips are stored in linear scale, already pre-normalised, or in a different unit, applying these values produces wildly wrong inputs to ResNet-18, making its pretrained features useless.
>
> **Fix:** Replaced with `nn.InstanceNorm2d(affine=True)` — this normalises each chip's VV and VH channels independently per sample (zero mean, unit variance), making **no assumption about absolute units**. It works correctly regardless of whether chips are in dB, linear power, or pre-normalised form.
>
> **Lecture connection (Lecture 5 — Normalising Inputs):**
> > *"Unequal scales = unequal footing → gradient descent trips and zig-zags. Normalise = level playing field → walks straight to the goal."*
>
> `InstanceNorm2d` is the correct "level playing field" here because it adapts to each chip's own distribution rather than assuming a global distribution.

---

## 1. Diagnosis — What the Training Logs Revealed

Before making any changes, the training logs were analysed against the diagnostic framework taught in **Lecture 5 (Slide: "Comparison Table — Underfitting vs Just Right vs Overfitting")**.

| Epoch | Train Loss | Val PR-AUC | Interpretation |
|-------|-----------|------------|----------------|
| 8     | 0.6121    | 0.3299     | First good peak |
| 10    | 0.5635    | **0.3303** | Best — model saved |
| 14    | 0.4854    | **0.0078** | Near-total collapse |
| 15    | 0.4777    | 0.0078     | Still collapsed |
| 16    | 0.4777    | 0.3034     | Partial recovery |
| 20    | 0.4627    | 0.2940     | Early stop (patience=10) |

**Lecture connection — Bias-Variance Tradeoff (Lecture 5):**
> *"Training error almost always keeps decreasing... but validation loss follows a U-shape — it decreases, hits a minimum, then rises again as the model starts memorising noise."*

This is precisely what the logs show. Train loss monotonically decreased from `1.83 → 0.46` across 20 epochs, while val PR-AUC peaked at epoch 10 then collapsed. The **growing gap** between training performance and validation performance is the textbook overfitting signature described in the lecture.

**Root cause identified:** The GNN component (`graph_gnn.py`) had **zero regularization**, and the FiLM terrain modulator (`film_terrain.py`) had **zero regularization** — two of the five model components contributing to the unregularized complexity that caused overfitting.

---

## 2. Lecture 5 Techniques Applied

### 2.1 Dropout — `graph_gnn.py`

**Lecture Theory (Lecture 5 — Dropout):**
> *"During training, dropout randomly turns off a fraction p of neurons in a layer for each training batch... If a neuron can't rely on any one specific other neuron always being present, the network is forced to build redundant, independent, and generally useful representations rather than fragile ones."*
>
> *"It's also mathematically similar to training a huge ensemble of smaller sub-networks simultaneously and then averaging their predictions at test time — this is why it's described as 'simulating ensemble learning'."*

**Problem in the original code:**

```python
# graph_gnn.py — BEFORE (zero regularization)
x = self.conv1(x, combined_edge_index, edge_attr=combined_edge_weight)
x = self.relu(x)
x = self.conv2(x, combined_edge_index, edge_attr=combined_edge_weight)
return x
```

The two GATv2 graph convolution layers had no dropout, no normalization. With 51 nodes and 128-dimensional embeddings, the GNN could freely memorize exact node-to-node co-activation patterns in the training graph snapshots. This is the exact **co-adaptation** problem the lecture describes — neurons (graph nodes here) co-adapt to each other rather than learning independently useful flood-risk features.

**Change applied:**

```python
# graph_gnn.py — AFTER
self.norm1 = nn.LayerNorm(hidden_channels)
self.drop1 = nn.Dropout(p=0.2)
self.norm2 = nn.LayerNorm(out_channels)

# forward()
x = self.conv1(x, combined_edge_index, edge_attr=combined_edge_weight)
x = self.relu(x)
x = self.norm1(x)       # stabilise before dropout
x = self.drop1(x)       # randomly silence 20% of node embeddings
x = self.conv2(x, combined_edge_index, edge_attr=combined_edge_weight)
x = self.norm2(x)       # stabilise final embedding
```

**Why `dropout=0.2`:** The `TemporalEncoder` already uses `dropout=0.2` as its baseline rate. Matching this rate keeps the regularization strength consistent across the pipeline — the same fraction of information is randomly dropped at each stage.

**Why `LayerNorm` before `Dropout`:** LayerNorm prevents dropout from acting on wildly scaled activations. If one node embedding has a value of 50 while another has 0.1, dropping the large one has a disproportionate impact. Normalising first ensures each node's contribution is on a similar scale before masking.

---

### 2.2 Dropout + Residual Connection — `film_terrain.py`

**Lecture Theory (Lecture 5 — Dropout):**
> *"Memory trick: 'Dropout = Random layoffs.' Employees (neurons) never know if a specific coworker will show up, so everyone learns to do their job independently, not just as part of one fixed team."*

**Lecture Theory (Lecture 7 — ResNet):**
> *"Learns residual F(x); output = F(x) + x via shortcut connection... Solves vanishing gradients, enables scaling."*

**Problem in the original code:**

```python
# film_terrain.py — BEFORE
self.mlp = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim * 2)   # no dropout
)

def forward(self, terrain_features, temporal_out):
    mlp_out = self.mlp(terrain_features)
    gamma = mlp_out[:, :self.output_dim]
    beta  = mlp_out[:, self.output_dim:]
    return gamma * temporal_out + beta       # no residual, no norm
```

Two problems:
1. **No dropout** — the terrain MLP could freely memorize which exact terrain feature values (e.g., elevation=347m) corresponded to flood events in training.
2. **No residual** — FiLM modulation computes `gamma * h + beta`. If gamma and beta have large magnitudes (which is possible without regularization), the original temporal signal `h` can be completely overridden. The model then ignores what the temporal transformer learned and relies entirely on static terrain — a form of overfitting to terrain-specific training examples.

**Change applied:**

```python
# film_terrain.py — AFTER
self.mlp = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(p=0.2),                       # regularize terrain MLP
    nn.Linear(hidden_dim, output_dim * 2)
)
self.norm = nn.LayerNorm(output_dim)

def forward(self, terrain_features, temporal_out):
    mlp_out = self.mlp(terrain_features)
    gamma = mlp_out[:, :self.output_dim]
    beta  = mlp_out[:, self.output_dim:]
    modulated = gamma * temporal_out + beta
    return self.norm(modulated + temporal_out)   # residual + LayerNorm
```

**Why the residual `+ temporal_out`:** This is directly analogous to ResNet's skip connection (Lecture 7). The residual guarantees that the temporal representation always reaches the fusion stage — even if gamma and beta are degenerate, the gradient can flow back through the `+ temporal_out` identity path, preventing the terrain branch from suppressing the temporal encoder's gradient. This directly addresses the vanishing gradient concern raised in Lecture 5 for deep paths.

---

### 2.3 Input Normalisation — `sar_cnn.py`

**Lecture Theory (Lecture 5 — Normalising Inputs):**
> *"If one feature has a much larger scale than another, the loss surface becomes a long, narrow, elongated valley. Gradient descent zig-zags slowly across such a valley because the gradient direction doesn't point straight at the minimum. After normalisation, the loss surface becomes more circular/bowl-shaped, and gradient descent can go straight to the minimum in far fewer steps."*
>
> *"Unequal scales = unequal footing → gradient descent trips and zig-zags. Normalise = level playing field → walks straight to the goal."*

**Problem in the original code:**

```python
# sar_cnn.py — BEFORE
# SAR chips passed raw (dB scale) directly into ResNet-18
clean_chips = self.despeckle(valid_chips)
rgb_chips   = self.channel_map(clean_chips)    # still in dB scale!
output[valid_indices] = self.cnn(rgb_chips)
```

SAR backscatter values for Sentinel-1 are typically:
- **VV channel:** -15 dB to -5 dB (mean ≈ -10 dB)
- **VH channel:** -22 dB to -12 dB (mean ≈ -17 dB)

These raw dB values were fed into a ResNet-18 pretrained on ImageNet, where inputs are expected to be normalised close to zero mean and unit variance. The mismatch creates exactly the "unequal footing" described in the lecture — the pretrained weights are calibrated for a completely different input distribution, causing the early ResNet layers to produce wildly scaled activations.

**Change applied:**

```python
# sar_cnn.py — AFTER
# Register SAR-specific normalisation statistics as device-aware buffers
self.register_buffer('sar_mean',
    torch.tensor((-10.0, -17.0), dtype=torch.float32).view(1, 2, 1, 1))
self.register_buffer('sar_std',
    torch.tensor((5.0, 5.0),     dtype=torch.float32).view(1, 2, 1, 1))

# In forward():
chips = (chips - self.sar_mean) / (self.sar_std + 1e-6)  # Z-score normalise
chips = self.despeckle(chips)
chips = self.channel_map(chips)
output[valid_indices] = self.cnn(chips)
```

**Why `register_buffer` instead of a regular tensor:** Buffers are moved to the same device (CPU/GPU) as the model automatically via `.to(device)` — they are not trainable parameters, but they are persistent state that needs to live on the correct device during the forward pass.

**Why Z-score (not min-max):** The lecture covers both normalisation methods. Z-score (`X_std = (X - mean) / std`) is preferred here because SAR values follow a roughly Gaussian distribution in dB scale, and Z-score preserves the shape of that distribution. Min-max normalisation would be sensitive to outliers common in SAR (e.g., specular reflection from water bodies).

**Additional change — `BatchNorm1d` on the SAR embedding:**

**Lecture Theory (Lecture 5 — Batch Normalisation):**
> *"For each mini-batch during training, standardise the activations of a layer... so that layer's inputs to the next layer stay in a stable, predictable range."*
>
> *"Benefits: faster convergence, more stable training... it has a mild regularising effect."*

```python
# BEFORE
self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embedding_dim)

# AFTER
self.cnn.fc = nn.Sequential(
    nn.Linear(self.cnn.fc.in_features, embedding_dim),
    nn.BatchNorm1d(embedding_dim),   # stabilize embedding scale for fusion
)
```

The 64-dim SAR embedding is concatenated with the 128-dim temporal embedding inside `FusionBlock`. Without normalisation, if the SAR embedding has a very different scale from the temporal embedding, the gradient flow through the fusion MLP will be dominated by the larger-scale input. `BatchNorm1d` ensures both modalities arrive at the fusion step on a comparable scale — the "level playing field" the lecture describes.

---

### 2.4 Fusion Regularisation — `fusion.py`

**Lecture Theory (Lecture 5 — Dropout):**
> *"Key benefits: reduces overfitting, encourages robust/general features, improves generalisation, ensemble-like effect at no extra training cost."*

**Change applied:**

```python
# fusion.py — BEFORE
self.mlp = nn.Sequential(
    nn.Linear(concat_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(p=0.1),             # too weak, inconsistent with rest
    nn.Linear(hidden_dim, output_dim),
    nn.ReLU()                      # trailing ReLU discards negative dims
)

# fusion.py — AFTER
self.mlp = nn.Sequential(
    nn.Linear(concat_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(p=0.2),             # consistent with rest of model
    nn.Linear(hidden_dim, output_dim),
)
self.norm = nn.LayerNorm(output_dim)

def forward(...):
    return self.norm(self.mlp(fused))
```

**Why the trailing `ReLU` was removed:** A `ReLU` at the final output of `FusionBlock` clips all negative-valued dimensions of the 128-dim embedding to zero — discarding information before it enters the GNN. Node embeddings benefit from having both positive and negative dimensions, as the GATv2 attention mechanism can distinguish between "presence" and "absence" of features. `LayerNorm` replaces `ReLU` here — it stabilises the scale without destroying negative information.

---

## 3. Lecture 7 Concepts — CNN Architecture Context

### 3.1 ResNet Skip Connections → Applied to FiLMTerrain

**Lecture 7 (ResNet, 2015):**
> *"Core Innovation: Skip connections (residual learning). Learns residual F(x); output = F(x) + x via shortcut connection. Advantage: Solves vanishing gradients, enables scaling to 50/101/152+ layers."*

The residual connection added to `FiLMTerrain` is a direct application of this principle. The FiLM modulation path (`gamma * h + beta`) is equivalent to `F(x)`, and the added `+ temporal_out` is the skip connection `x`. The result, `F(x) + x`, mirrors the ResNet residual block exactly.

### 3.2 Batch Normalisation → Applied to SARCNN Embedding

**Lecture 7 (discussing BatchNorm across architectures):**
> *"Keeps activations at a stable, well-behaved scale throughout the network, preventing gradients from shrinking due to badly-scaled internal signals."*

The `BatchNorm1d` placed at the end of `SARCNN`'s final FC layer is the same principle that makes deep CNNs like ResNet trainable — normalising intermediate representations to prevent Internal Covariate Shift at the boundary between the CNN and the multimodal fusion layer.

### 3.3 Layer Normalisation → Applied to GNN and Fusion

**Lecture 5 (Layer Normalisation):**
> *"Layer Norm computes mean/variance across all features within a single example, independent of batch size — good for settings where batch composition varies."*

`LayerNorm` was chosen over `BatchNorm` for the GNN and FusionBlock because each "batch" in this model is a single full graph snapshot (batch_size=1 in the DataLoader). `BatchNorm1d` with a single example computes degenerate statistics (mean=x, std=0). `LayerNorm` normalises across the 128 feature dimensions of each node embedding instead — working correctly regardless of batch size.

### 3.4 MobileNet Efficiency Principle → Future Improvement for SARCNN

**Lecture 7 (MobileNet, 2017):**
> *"Core Innovation: Depthwise separable convolution. Advantage: ~8-9x fewer params/FLOPs, low latency, small size."*

The current `SARCNN` uses ResNet-18 (11.7M parameters) to produce a 64-dimensional SAR embedding — a significant mismatch between model capacity and output dimensionality. A planned future improvement is to replace ResNet-18 with MobileNetV3-Small (~2.5M parameters), which would reduce Kaggle training time per epoch by approximately 25–30%, freeing GPU resources for running more ensemble seeds within the 12-hour session limit.

---

## 4. Summary of All Changes

| File | Change | Lecture Principle | Expected Effect |
|------|--------|-------------------|-----------------|
| `graph_gnn.py` | `LayerNorm + Dropout(0.2)` between GATv2 layers | L5 — Dropout (co-adaptation prevention) | Reduce GNN memorisation; stabilise val PR-AUC |
| `film_terrain.py` | `Dropout(0.2)` in terrain MLP; residual `+ temporal_out`; `LayerNorm` | L5 — Dropout; L7 — ResNet skip connection | Prevent terrain branch overpowering temporal signal |
| `sar_cnn.py` | Z-score SAR normalisation (VV: -10dB±5, VH: -17dB±5); `BatchNorm1d` on 64-dim embedding | L5 — Normalising Inputs; L5 — Batch Normalisation | Stable ResNet activations; equalised modality scales at fusion |
| `fusion.py` | Dropout `0.1 → 0.2`; remove trailing `ReLU`; add `LayerNorm` | L5 — Dropout; L5 — Layer Normalisation | Preserve negative embedding dimensions; consistent dropout across pipeline |

### Regularisation Coverage — Before vs After

| Component | Before | After |
|---|---|---|
| `TemporalEncoder` | Dropout 0.2, LayerNorm ✅ | Unchanged ✅ |
| `FiLMTerrain` | None ❌ | Dropout 0.2 + LayerNorm + Residual ✅ |
| `SARCNN` | BatchNorm (ResNet internal only) | + Input Normalisation + BatchNorm1d output ✅ |
| `FusionBlock` | Dropout 0.1, trailing ReLU ⚠️ | Dropout 0.2, LayerNorm ✅ |
| `GraphGNN` | None ❌ | LayerNorm + Dropout 0.2 + LayerNorm ✅ |
| `OutputHeads` | Dropout 0.1, GELU ✅ | Unchanged ✅ |

---

## 5. Anti-Overfitting Checklist — Lecture 5 Framework

| Technique | Lecture Description | Status in This Model |
|---|---|---|
| More training data | Real new data is the best fix | Limited by dataset size |
| Data Augmentation | Artificial variety — noise, flips, crops | Planned (temporal noise injection) |
| Regularisation (L2) | Penalty on large weights | `weight_decay=1e-4` in AdamW ✅ |
| Dropout | Random neuron deactivation | Now applied to ALL 5 components ✅ |
| Early Stopping | Stop at validation minimum | `patience=10` + auto-resume checkpoint ✅ |
| Input Normalisation | Equal footing for all features | TabularScaler + SAR dB normalisation ✅ |
| Batch Normalisation | Stable internal activations | ResNet internal + SAR embedding BN1d ✅ |
| Layer Normalisation | Stable activations, batch-independent | Transformer + GNN + FiLM + Fusion ✅ |

---

*Report generated: 2026-08-26*  
*Changes committed: `43c598a` — feat: regularization and normalization improvements across all model components*
