"""Temporal encoder — PLR Tokenising Transformer + Cross-feature Attention.

Two complementary attention streams that share the same PLR embeddings:

  Stream 1 — Temporal Transformer
    Self-attention across L=14 lookback days (one CLS token per day).
    Answers: *when* did the critical antecedent conditions build up?

  Stream 2 — Cross-feature Attention
    Self-attention across F=33 feature channels (one token per feature),
    averaged over the time axis first.
    Answers: *which features interact* to trigger a flood?
    (e.g. "heavy rain matters more when soil is already saturated")

The two CLS-pooled outputs are merged:  h = LayerNorm(h_temporal + h_feature)

This replaces the previous single-layer GRU, which could not represent
sharp threshold-like decision boundaries on individual scalar features.
Reference: Gorishniy et al. "On Embeddings for Numerical Features in
Tabular Deep Learning", NeurIPS 2022.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────── PLR numerical embedding ──────

class PLREmbedding(nn.Module):
    """Per-feature Periodic-Linear-ReLU embedding.

    Each of the F features gets its own non-linear representation:
        x_f  →  [sin(2π c_f x), cos(2π c_f x)]  →  Linear  →  ReLU

    The periodic expansion lets the network represent sharp thresholds in a
    scalar (e.g. 98th-percentile discharge) — the thing a decision tree gets
    for free from a split point and a plain MLP struggles with.

    Parameters
    ----------
    n_features : number of input scalar features (33)
    d_emb      : embedding dimension per feature (8)
    n_freq     : number of frequency components (8)
    sigma      : frequency initialisation std (small → avoid aliasing)

    Input  : [..., F]
    Output : [..., F, d_emb]
    """
    def __init__(self, n_features: int, d_emb: int = 8,
                 n_freq: int = 8, sigma: float = 0.05):
        super().__init__()
        self.coef   = nn.Parameter(torch.randn(n_features, n_freq) * sigma)
        d_in        = 2 * n_freq
        self.weight = nn.Parameter(torch.empty(n_features, d_in, d_emb))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_emb))
        nn.init.normal_(self.weight, std=d_in ** -0.5)
        self.d_emb = d_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = 2 * math.pi * x.unsqueeze(-1) * self.coef       # [..., F, n_freq]
        z = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # [..., F, 2*n_freq]
        return F.relu(
            torch.einsum('...fi,fio->...fo', z, self.weight) + self.bias
        )                                                     # [..., F, d_emb]


# ─────────────────────────────────────────────── Shared transformer factory ──

def _make_transformer(d_model: int, n_heads: int, ff_mult: int,
                      dropout: float, n_layers: int) -> nn.TransformerEncoder:
    """Pre-LN transformer encoder stack (trains without a warmup-sensitive phase)."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads,
        dim_feedforward=ff_mult * d_model,
        dropout=dropout, activation='gelu',
        batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers,
                                 norm=nn.LayerNorm(d_model))


# ──────────────────────────────────────────────────────── Main encoder ────────

class TemporalEncoder(nn.Module):
    """PLR Tokenising Transformer + Cross-feature Attention.

    Parameters
    ----------
    input_dim   : number of dynamic features (33)
    hidden_dim  : d_model / output dimension (128)
    d_emb       : PLR embedding dim per feature (8)
    n_freq      : number of frequency components per feature (8)
    lookback    : history window in days (14)
    n_layers    : transformer depth for the temporal stream (3)
    n_heads     : attention heads for both streams (4)
    ff_mult     : feedforward multiplier (2 → dim_ff = 256)
    dropout     : dropout rate (0.2)
    feat_layers : transformer depth for the cross-feature stream (1)

    Forward
    -------
    x : [N, L, F]   (N nodes, L lookback days, F dynamic features)
    → [N, hidden_dim]
    """

    def __init__(
        self,
        input_dim:   int   = 33,
        hidden_dim:  int   = 128,
        d_emb:       int   = 8,
        n_freq:      int   = 8,
        lookback:    int   = 14,
        n_layers:    int   = 3,
        n_heads:     int   = 4,
        ff_mult:     int   = 2,
        dropout:     float = 0.2,
        feat_layers: int   = 1,
    ):
        super().__init__()
        self.embed = PLREmbedding(input_dim, d_emb, n_freq)

        # ── Stream 1: Temporal Transformer ───────────────────────────────────
        # One token per day → CLS token pools the whole window
        self.to_temp_tok = nn.Linear(input_dim * d_emb, hidden_dim)
        self.cls_temp    = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_temp    = nn.Parameter(torch.zeros(1, lookback + 1, hidden_dim))
        self.temp_enc    = _make_transformer(hidden_dim, n_heads, ff_mult, dropout, n_layers)
        nn.init.trunc_normal_(self.cls_temp, std=0.02)
        nn.init.trunc_normal_(self.pos_temp, std=0.02)

        # ── Stream 2: Cross-feature Attention ─────────────────────────────────
        # One token per feature → CLS token captures feature interactions
        self.feat_proj = nn.Linear(d_emb, hidden_dim)
        self.feat_tok  = nn.Parameter(torch.zeros(1, input_dim, hidden_dim))
        self.cls_feat  = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.feat_enc  = _make_transformer(hidden_dim, n_heads, ff_mult, dropout, feat_layers)
        nn.init.trunc_normal_(self.feat_tok, std=0.02)
        nn.init.trunc_normal_(self.cls_feat, std=0.02)

        # ── Merge ─────────────────────────────────────────────────────────────
        self.merge = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [N, L, F]
        Returns [N, hidden_dim]
        """
        N, L, F = x.shape

        # Shared PLR embeddings  →  [N, L, F, d_emb]
        z = self.embed(x)

        # ── Stream 1: Temporal ────────────────────────────────────────────────
        tok = self.to_temp_tok(z.flatten(-2))                      # [N, L, hidden]
        cls = self.cls_temp.expand(N, -1, -1)                      # [N, 1, hidden]
        tok = torch.cat([cls, tok], dim=1)                         # [N, L+1, hidden]
        tok = self.drop(tok + self.pos_temp[:, :tok.size(1)])
        h_t = self.temp_enc(tok)[:, 0]                             # [N, hidden]

        # ── Stream 2: Cross-feature ───────────────────────────────────────────
        # Average out the time axis, project each feature token
        feat = self.feat_proj(z.mean(dim=1)) + self.feat_tok       # [N, F, hidden]
        cls  = self.cls_feat.expand(N, -1, -1)                     # [N, 1, hidden]
        feat = torch.cat([cls, feat], dim=1)                       # [N, F+1, hidden]
        h_f  = self.feat_enc(feat)[:, 0]                           # [N, hidden]

        # ── Merge: residual sum then LayerNorm ────────────────────────────────
        return self.merge(h_t + h_f)                                # [N, hidden]
