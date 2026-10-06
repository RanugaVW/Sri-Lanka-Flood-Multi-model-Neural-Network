"""Batching for full-graph day snapshots.

One dataset sample is one day over all N=51 nodes. These helpers collate B of
those into a single B*N-node disconnected graph — the standard PyG batching
trick — so an optimiser step sees `B * 51` node-days instead of 51.

This lives in its own module rather than in `train.py` because `train.py`,
`src/eval/evaluate_metrics.py` and `src/hpo.py` all need it, and
`train.py` already imports from `evaluate_metrics`. Putting it here is what
keeps that from being a circular import, and it means the three entry points
cannot drift onto three different unpacking conventions the way they had.
"""
import numpy as np
import torch


#: Static tensors are identical in every snapshot, so the collate keeps exactly
#: one copy rather than stacking B duplicates of the 51-node terrain matrix and
#: the edge lists.
_STATIC_KEYS = ('terrain_features', 'basin_idx', 'edge_index_flow',
                'edge_index_spatial', 'edge_weight_spatial')


def collate_snapshots(samples):
    """Collate B full-graph day snapshots into one batch.

    Each sample is one day over all N nodes. Stacking B of them gives a
    B*N-node disconnected graph, which is the standard PyG batching trick and
    is what lets the optimiser take a step on 816 node-days (B=16) instead of
    51. Training previously ran at batch_size=1, i.e. 5,435 extremely noisy
    optimiser steps per epoch at lr 1e-3 with no warmup — the val PR-AUC trace
    bounced in [0.48, 0.54] for twenty epochs without a trend, which is the
    signature of gradient noise rather than convergence.
    """
    s0 = samples[0]
    out = {k: s0[k] for k in _STATIC_KEYS}
    for k in ('temporal_features', 'sar_chips', 'has_sar', 'sar_age_days',
              'targets', 'valid_mask', 'label_conf', 'event_ids', 'day_idx'):
        if k in s0:
            out[k] = torch.stack([s[k] for s in samples])
    return out


def _offset_edges(edge_index, n_nodes, B):
    """Block-diagonal replication of one snapshot's edges across B snapshots."""
    if B == 1:
        return edge_index
    off = (torch.arange(B, device=edge_index.device) * n_nodes).view(B, 1, 1)
    return (edge_index.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1)


def _unpack(batch, device):
    """Unpack a collated batch onto device. Returns (inputs, targets, mask, conf).

    Everything node-wise is flattened [B, N, ...] -> [B*N, ...]; the graph is
    the only part that needs the block-diagonal treatment.
    """
    tf = batch['temporal_features'].to(device)            # [B, N, L, F]
    B, N = tf.shape[0], tf.shape[1]

    sar  = batch['sar_chips'].to(device)                  # [B, N, 2, H, W]
    has  = batch['has_sar'].to(device)                    # [B, N]
    age  = batch.get('sar_age_days')
    age  = age.to(device).reshape(B * N) if age is not None else None

    eif = batch['edge_index_flow'].to(device)
    eis = batch['edge_index_spatial'].to(device)
    ews = batch['edge_weight_spatial'].to(device)

    inp = {
        'temporal':  tf.reshape(B * N, *tf.shape[2:]),
        'terrain':   batch['terrain_features'].to(device).repeat(B, 1),
        'basin_idx': batch['basin_idx'].to(device).repeat(B),
        'sar':       sar.reshape(B * N, *sar.shape[2:]),
        'has_sar':   has.reshape(B * N),
        'sar_age':   age,
        'ei_flow':   _offset_edges(eif, N, B),
        'ei_sp':     _offset_edges(eis, N, B),
        'ew_sp':     ews.repeat(B),
        'B':         B,
        'N':         N,
    }
    return (
        inp,
        batch['targets'].to(device).reshape(B * N, -1),
        batch['valid_mask'].to(device).reshape(B * N),
        batch['label_conf'].to(device).reshape(B * N),
    )


def _forward(model, inp):
    return model(
        inp['temporal'], inp['terrain'], inp['basin_idx'],
        inp['sar'],      inp['has_sar'],
        inp['ei_flow'],  inp['ei_sp'], inp['ew_sp'],
        sar_age_days=inp.get('sar_age'),
    )
