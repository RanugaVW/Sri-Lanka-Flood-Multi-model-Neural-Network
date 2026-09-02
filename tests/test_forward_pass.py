import torch
import sys
import os

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.flood_model import FloodModel
from losses.multitask_loss import MultiTaskLoss
from data.graph_builder import TERRAIN_DIM

NUM_BASINS = 16


def test_forward_and_backward():
    print("Initializing FloodModel and MultiTaskLoss...")
    model = FloodModel(num_basins=NUM_BASINS)
    criterion = MultiTaskLoss()

    # Dummy Dimensions
    batch_size = 51 # Number of nodes in a single timestep graph
    window_days = 14
    temporal_features_dim = 33

    print(f"Creating dummy inputs for {batch_size} nodes...")
    temporal_features = torch.randn(batch_size, window_days, temporal_features_dim)
    terrain_features = torch.randn(batch_size, TERRAIN_DIM)
    basin_idx = torch.randint(0, NUM_BASINS, (batch_size,))

    # SAR features (only some nodes have it)
    sar_chips = torch.randn(batch_size, 2, 512, 512)
    has_sar = torch.randint(0, 2, (batch_size,), dtype=torch.bool)

    # Edges (dummy fully connected for testing)
    edge_index_flow = torch.randint(0, batch_size, (2, 35))
    edge_index_spatial = torch.randint(0, batch_size, (2, 204))
    edge_weight_spatial = torch.rand(204)

    # Targets [batch_size, 6]
    # First 4 are probs/binary, last 2 are continuous
    targets = torch.cat([
        torch.randint(0, 2, (batch_size, 4)).float(),
        torch.randn(batch_size, 2)
    ], dim=1)
    mask = torch.ones(batch_size)
    conf = torch.ones(batch_size)

    print("Running forward pass...")
    predictions = model(
        temporal_features,
        terrain_features,
        basin_idx,
        sar_chips,
        has_sar,
        edge_index_flow,
        edge_index_spatial,
        edge_weight_spatial
    )

    assert predictions['logits'].shape == (batch_size, 4), \
        f"Expected logits shape {(batch_size, 4)}, got {predictions['logits'].shape}"
    assert predictions['reg'].shape == (batch_size, 2), \
        f"Expected reg shape {(batch_size, 2)}, got {predictions['reg'].shape}"
    print(f"Forward pass successful. logits {tuple(predictions['logits'].shape)}, "
          f"reg {tuple(predictions['reg'].shape)}")

    print("Running loss calculation...")
    loss = criterion(predictions, targets, mask, conf)
    print(f"Computed Loss: {loss.total.item()}")

    print("Running backward pass...")
    loss.total.backward()

    # Check if gradients are computed for all modules
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    if has_grad:
        print("Backward pass successful. All parameters received gradients (no dead branches!).")
    else:
        print("WARNING: Some parameters did not receive gradients. Dead branch detected.")
    assert has_grad, "Dead branch detected: some trainable parameters received no gradient."


if __name__ == "__main__":
    test_forward_and_backward()
