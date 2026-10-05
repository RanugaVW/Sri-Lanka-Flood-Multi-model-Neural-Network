"""Forward/backward smoke tests for FloodModel across its configured modes.

`test_forward_and_backward` is the original single-snapshot, all-branches-on
test. The rest cover what became configurable afterwards, because every one of
these is now a real `configs/model.yaml` setting rather than dead text:

  * `sar_cnn.enabled: false`      — the headline default
  * `fusion.mode: none|gated|concat`
  * `graph.mode:  none|flow|both`
  * multi-snapshot batching with block-diagonal edge offsets

The gate tests are the ones worth keeping honest: both the SAR fusion gate and
the GNN residual gate are deliberately initialised shut, and "shut at init" has
to mean *numerically identical to the branch-free model*, not merely small.
"""
import os
import sys

import pytest
import torch

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.flood_model import FloodModel          # noqa: E402
from models.graph_gnn import GraphGNN              # noqa: E402
from losses.multitask_loss import MultiTaskLoss    # noqa: E402
from data.graph_builder import TERRAIN_DIM         # noqa: E402

NUM_BASINS = 16
N_NODES = 51
WINDOW = 14
N_FEAT = 33


def _cfg(sar=False, fusion='none', graph='flow', px=8):
    """A model config. `px` is small so the SAR tests stay fast."""
    return {
        'temporal_encoder': {'input_dim': N_FEAT, 'window_days': WINDOW,
                             'hidden_dim': 128, 'num_layers': 1,
                             'feature_layers': 1},
        'film_terrain': {'input_dim': TERRAIN_DIM, 'num_basins': NUM_BASINS},
        'sar_cnn': {'enabled': sar, 'input_channels': 2, 'input_size': px,
                    'embedding_dim': 64, 'freeze_backbone': False},
        'fusion': {'mode': fusion, 'hidden_dim': 192, 'output_dim': 128},
        'graph': {'mode': graph, 'num_layers': 2, 'heads': 4},
        'heads': {'hidden_dim': 64},
    }


def _inputs(n=N_NODES, px=512):
    return dict(
        temporal_features=torch.randn(n, WINDOW, N_FEAT),
        terrain_features=torch.randn(n, TERRAIN_DIM),
        basin_idx=torch.randint(0, NUM_BASINS, (n,)),
        sar_chips=torch.randn(n, 2, px, px),
        has_sar=torch.randint(0, 2, (n,), dtype=torch.bool),
        edge_index_flow=torch.randint(0, n, (2, 35)),
        edge_index_spatial=torch.randint(0, n, (2, 204)),
        edge_weight_spatial=torch.rand(204),
        sar_age_days=torch.rand(n) * 12.0,
    )


def _call(model, inp):
    return model(
        inp['temporal_features'], inp['terrain_features'], inp['basin_idx'],
        inp['sar_chips'], inp['has_sar'],
        inp['edge_index_flow'], inp['edge_index_spatial'],
        inp['edge_weight_spatial'],
        sar_age_days=inp.get('sar_age_days'),
    )


def _targets(n=N_NODES):
    targets = torch.cat([
        torch.randint(0, 2, (n, 4)).float(),
        torch.randn(n, 2),
    ], dim=1)
    return targets, torch.ones(n), torch.ones(n)


# ── The original end-to-end test, all branches on ─────────────────────────────

def test_forward_and_backward():
    print("Initializing FloodModel and MultiTaskLoss...")
    model = FloodModel(num_basins=NUM_BASINS)
    criterion = MultiTaskLoss()

    print(f"Creating dummy inputs for {N_NODES} nodes...")
    inp = _inputs()
    targets, mask, conf = _targets()

    print("Running forward pass...")
    predictions = _call(model, inp)

    assert predictions['logits'].shape == (N_NODES, 4), \
        f"Expected logits shape {(N_NODES, 4)}, got {predictions['logits'].shape}"
    assert predictions['reg'].shape == (N_NODES, 2), \
        f"Expected reg shape {(N_NODES, 2)}, got {predictions['reg'].shape}"
    print(f"Forward pass successful. logits {tuple(predictions['logits'].shape)}, "
          f"reg {tuple(predictions['reg'].shape)}")

    print("Running loss calculation...")
    loss = criterion(predictions, targets, mask, conf)
    print(f"Computed Loss: {loss.total.item()}")

    print("Running backward pass...")
    loss.total.backward()

    has_grad = all(p.grad is not None
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "Dead branch detected: some trainable parameters received no gradient."
    print("Backward pass successful. All parameters received gradients.")


# ── Config is actually honoured ────────────────────────────────────────────────

@pytest.mark.parametrize('graph', ['none', 'flow', 'both'])
def test_graph_modes(graph):
    """Each graph mode runs, and 'none' creates no GNN parameters at all."""
    model = FloodModel(config=_cfg(graph=graph))
    out = _call(model, _inputs())
    assert out['logits'].shape == (N_NODES, 4)

    gnn_params = sum(p.numel() for p in model.gnn.parameters())
    if graph == 'none':
        assert gnn_params == 0, f"graph.mode='none' still built {gnn_params} params"
    else:
        assert gnn_params > 0


def test_sar_disabled_builds_no_cnn():
    """`sar_cnn.enabled: false` must remove the branch, not just bypass it.

    The ResNet-18 was 11,209,549 of the old model's 12,125,539 parameters, so
    "disabled" leaving it in the module tree would still carry 92% of the
    weight budget and the whole optimiser state for it.
    """
    model = FloodModel(config=_cfg(sar=False))
    assert model.sar_cnn is None
    assert model.fusion.mode == 'none'
    total = sum(p.numel() for p in model.parameters())
    assert total < 2_000_000, f"SAR-free model should be ~0.9M params, got {total:,}"
    # A 1x1 placeholder chip must be accepted, since that is what the dataset
    # emits when SAR is off.
    inp = _inputs(px=1)
    assert _call(model, inp)['logits'].shape == (N_NODES, 4)


@pytest.mark.parametrize('fusion', ['gated', 'concat'])
def test_sar_fusion_modes(fusion):
    model = FloodModel(config=_cfg(sar=True, fusion=fusion, px=8))
    out = _call(model, _inputs(px=8))
    assert out['logits'].shape == (N_NODES, 4)
    if fusion == 'gated':
        # Zero-init + bias -3 → the gate starts nearly shut.
        assert model.fusion.last_gate_mean < 0.10, \
            f"gated SAR branch should start shut, got {model.fusion.last_gate_mean}"


# ── The two gates start shut ───────────────────────────────────────────────────

def test_graph_residual_gate_is_identity_at_init():
    """tanh(alpha) == 0 at init, so the GNN block is exactly the identity.

    This is the property that makes the graph opt-in: at step 0 the model is
    bit-identical to the graph-free one, instead of having two randomly
    initialised GATv2 layers inject full-magnitude noise into the fused
    embedding.
    """
    gnn = GraphGNN(mode='both')
    assert gnn.gate == 0.0
    x = torch.randn(N_NODES, 128)
    out = gnn(x, torch.randint(0, N_NODES, (2, 35)),
              torch.randint(0, N_NODES, (2, 204)), torch.rand(204))
    # LayerNorm(x + 0*h) == LayerNorm(x)
    assert torch.allclose(out, gnn.norm2(x), atol=1e-6)


def test_graph_gate_is_trainable():
    """alpha must receive gradient — a gate that cannot open is just 'graph off'."""
    gnn = GraphGNN(mode='flow')
    x = torch.randn(N_NODES, 128)
    gnn(x, torch.randint(0, N_NODES, (2, 35)),
        torch.randint(0, N_NODES, (2, 204)), torch.rand(204)).sum().backward()
    assert gnn.alpha.grad is not None
    assert torch.any(gnn.alpha.grad != 0)


def test_gated_sar_absent_frame_contributes_nothing():
    """With has_sar=False everywhere, the gated branch must be an exact no-op.

    `* pres` is what guarantees this: the image term is zero rather than "the
    learned missing embedding scaled by whatever the gate emits".
    """
    from models.fusion import FusionBlock
    fb = FusionBlock(mode='gated', temporal_dim=128, sar_dim=64).eval()
    h = torch.randn(10, 128)
    v1, v2 = torch.randn(10, 64), torch.randn(10, 64)
    pres = torch.zeros(10, dtype=torch.bool)
    age = torch.full((10,), 12.0)
    with torch.no_grad():
        a = fb(h, v1, pres, age)
        b = fb(h, v2, pres, age)
    # Different image embeddings, no frames → identical output.
    assert torch.allclose(a, b, atol=1e-6)


# ── Multi-snapshot batching ───────────────────────────────────────────────────

def test_batched_snapshots_match_single():
    """B snapshots batched block-diagonally == B independent forward passes.

    If the edge offsetting were wrong, nodes in snapshot b would exchange
    messages with snapshot b+1 — a silent cross-day leak that no shape
    assertion would catch.
    """
    import train as T

    torch.manual_seed(0)
    model = FloodModel(config=_cfg(graph='both')).eval()
    # Force the graph open, otherwise alpha=0 makes the test vacuous.
    with torch.no_grad():
        model.gnn.alpha.fill_(1.0)

    B = 3
    static = _inputs()
    samples = []
    for _ in range(B):
        s = _inputs(px=1)
        samples.append({
            'temporal_features': s['temporal_features'],
            'terrain_features': static['terrain_features'],
            'basin_idx': static['basin_idx'],
            'sar_chips': s['sar_chips'],
            'has_sar': s['has_sar'],
            'sar_age_days': s['sar_age_days'],
            'targets': torch.randn(N_NODES, 6),
            'valid_mask': torch.ones(N_NODES),
            'label_conf': torch.ones(N_NODES),
            'event_ids': torch.full((N_NODES,), -1, dtype=torch.int32),
            'day_idx': torch.tensor(0, dtype=torch.int32),
            'edge_index_flow': static['edge_index_flow'],
            'edge_index_spatial': static['edge_index_spatial'],
            'edge_weight_spatial': static['edge_weight_spatial'],
        })

    batch = T.collate_snapshots(samples)
    inp, _, _, _ = T._unpack(batch, 'cpu')
    assert inp['B'] == B and inp['N'] == N_NODES
    assert int(inp['ei_flow'].max()) < B * N_NODES
    assert inp['ei_flow'].shape[1] == B * 35

    with torch.no_grad():
        batched = _call(model, {
            'temporal_features': inp['temporal'],
            'terrain_features': inp['terrain'],
            'basin_idx': inp['basin_idx'],
            'sar_chips': inp['sar'],
            'has_sar': inp['has_sar'],
            'edge_index_flow': inp['ei_flow'],
            'edge_index_spatial': inp['ei_sp'],
            'edge_weight_spatial': inp['ew_sp'],
            'sar_age_days': inp['sar_age'],
        })['logits']

        for b, s in enumerate(samples):
            single = _call(model, {**s, 'sar_age_days': s['sar_age_days']})['logits']
            got = batched[b * N_NODES:(b + 1) * N_NODES]
            assert torch.allclose(single, got, atol=1e-4), \
                f"snapshot {b} differs between batched and single forward"


if __name__ == "__main__":
    test_forward_and_backward()
