"""Multimodal fusion — how the SAR embedding enters the tabular state.

Three modes, selected by `configs/model.yaml: fusion.mode`:

``none``
    SAR is not used at all. The FiLM-modulated temporal state passes through a
    plain MLP. This is the headline default, and the reason is measured, not
    aesthetic — see the note on coverage below.

``gated`` (default whenever SAR is enabled)
    The image contributes ``g ⊙ W v``, where the gate ``g`` is computed from the
    tabular state, the image embedding, the presence flag and the frame's age in
    days. The gate's output layer is zero-initialised with a negative bias, so
    the branch starts fully shut: at initialisation the gated model is exactly
    the SAR-free model, and any improvement has to be earned by training rather
    than handed over by a wider input layer.

``concat``
    The original behaviour — concatenate [128 temporal | 64 SAR] → 192 and MLP.
    Kept so the contrast against ``gated`` is runnable.

Why `gated` and not `concat`
----------------------------
Sentinel-1 revisits on a ~12-day cycle and only 9 of the 51 gauge nodes have a
chip site at all, so the overwhelming majority of node-days carry no observation.
A fixed slot in a concatenation makes the network spend capacity distinguishing
"no water visible" from "no picture taken" on *every single sample*, including
the ~82% of nodes that can never have a picture. A gate makes that conditional
instead: with `pres = 0` the image term is exactly zero and contributes no
gradient at all.

This mirrors the sibling project's `GatedSarFusion` (see
`Srilanka-Flood-Data-Set-Creation/models/model2/modules.py`), which was written
after that project's concatenated SAR branch returned a null result — its
`M6_cnn` rung scored event PR-AUC 0.2988 against 0.3007 for the identical model
without imagery, at twenty times the parameters.
"""
import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    """Fuse the FiLM-modulated temporal state with the SAR embedding.

    Parameters
    ----------
    mode        : 'none' | 'gated' | 'concat'
    temporal_dim: width of the FiLM output (128)
    sar_dim     : width of the SAR embedding (64)
    hidden_dim  : MLP hidden width (192)
    output_dim  : fusion output width (128)
    dropout     : dropout rate inside the MLP (0.2)
    gate_bias   : initial gate bias in 'gated' mode; -3.0 → sigmoid ≈ 0.047,
                  i.e. the branch starts essentially closed
    max_age_days: divisor that scales frame age into roughly [0, 1]

    Notes
    -----
    In 'none' and 'gated' mode the MLP input is `temporal_dim`, so the SAR
    embedding never widens the fused representation — only `concat` does. That
    keeps the parameter count of the SAR-free and gated models within a few
    thousand of each other, which is what makes an ablation between them a
    statement about the imagery rather than about model size.
    """

    def __init__(
        self,
        mode:         str   = 'gated',
        temporal_dim: int   = 128,
        sar_dim:      int   = 64,
        hidden_dim:   int   = 192,
        output_dim:   int   = 128,
        dropout:      float = 0.2,
        gate_bias:    float = -3.0,
        max_age_days: float = 12.0,
    ):
        super().__init__()
        if mode not in ('none', 'gated', 'concat'):
            raise ValueError(f"fusion.mode must be none|gated|concat, got {mode!r}")
        self.mode = mode
        self.max_age_days = float(max_age_days)

        concat_dim = temporal_dim + sar_dim if mode == 'concat' else temporal_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

        if mode == 'gated':
            self.sar_proj = nn.Linear(sar_dim, temporal_dim)
            # Gate context = tabular state | image embedding | presence | age
            self.gate = nn.Sequential(
                nn.Linear(temporal_dim + sar_dim + 2, temporal_dim),
                nn.GELU(),
                nn.Linear(temporal_dim, temporal_dim),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, gate_bias)
            self.pre_norm = nn.LayerNorm(temporal_dim)

        #: Mean gate opening over node-days that actually carry a frame, from the
        #: last forward pass. A branch whose gate never opens contributes
        #: nothing, and without this the only evidence either way would be a
        #: metric difference too small to attribute to anything.
        self.last_gate_mean: float = float('nan')

    def forward(self, modulated_temporal, sar_embedding=None,
                has_sar=None, sar_age_days=None):
        """
        modulated_temporal : [N, temporal_dim]
        sar_embedding      : [N, sar_dim]  (ignored when mode='none')
        has_sar            : [N] bool     (required when mode='gated')
        sar_age_days       : [N] float    (optional; defaults to max_age_days)
        """
        if self.mode == 'none':
            return self.norm(self.mlp(modulated_temporal))

        if self.mode == 'concat':
            fused = torch.cat([modulated_temporal, sar_embedding], dim=1)
            return self.norm(self.mlp(fused))

        # ── gated ────────────────────────────────────────────────────────────
        h = modulated_temporal
        if has_sar is None:
            raise ValueError("fusion.mode='gated' requires has_sar")
        pres = has_sar.to(h.dtype).unsqueeze(-1)                   # [N, 1]
        if sar_age_days is None:
            age = torch.full_like(pres, self.max_age_days)
        else:
            age = sar_age_days.to(h.dtype).unsqueeze(-1)           # [N, 1]

        ctx = torch.cat([h, sar_embedding, pres, age / self.max_age_days], dim=-1)
        g   = torch.sigmoid(self.gate(ctx))                        # [N, temporal_dim]

        with torch.no_grad():
            denom = pres.sum().clamp_min(1.0) * g.size(-1)
            self.last_gate_mean = float((g * pres).sum() / denom)

        # `* pres` is what makes the branch truly conditional: with no frame the
        # image term is exactly zero, not "the learned missing embedding scaled
        # by whatever the gate happens to emit".
        h = self.pre_norm(h + g * self.sar_proj(sar_embedding) * pres)
        return self.norm(self.mlp(h))
