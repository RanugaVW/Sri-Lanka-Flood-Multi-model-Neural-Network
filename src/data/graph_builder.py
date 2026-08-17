import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

# Must match FiLMTerrain(input_dim=9) in flood_model.py
TERRAIN_DIM = 9


def build_static_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Data:
    """
    Builds the static graph topology and node features.
    As per ARCHITECTURE_SPEC.md 2.1:
    - x: static node features — always [num_nodes, TERRAIN_DIM=9], zero-padded
         if nodes.csv has fewer than 9 numeric columns.
    - edge_index_flow: directed, 35 edges (upstream→downstream).
    - edge_index_spatial: weighted, 204 edges, weight = exp(-distance_km / 40).
    """
    # ── Node mapping ──────────────────────────────────────────────────────────
    node_mapping = {nid: i for i, nid in enumerate(nodes_df['node_id'].unique())}

    # ── Terrain features — pad to exactly TERRAIN_DIM columns ─────────────────
    numeric_cols = nodes_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'node_id' in numeric_cols:
        numeric_cols.remove('node_id')

    # Take however many numeric columns exist (up to TERRAIN_DIM)
    terrain_cols = numeric_cols[:TERRAIN_DIM]
    raw = nodes_df[terrain_cols].values.astype(np.float32)   # [N, n_cols_found]

    n_found = raw.shape[1]
    if n_found < TERRAIN_DIM:
        # Zero-pad the missing columns on the right
        pad = np.zeros((raw.shape[0], TERRAIN_DIM - n_found), dtype=np.float32)
        raw = np.concatenate([raw, pad], axis=1)
        print(f"  [graph_builder] nodes.csv has {n_found} terrain cols; "
              f"zero-padded to {TERRAIN_DIM}.")
    elif n_found > TERRAIN_DIM:
        raw = raw[:, :TERRAIN_DIM]

    x = torch.tensor(raw, dtype=torch.float)   # [N, 9]

    # ── Edges ─────────────────────────────────────────────────────────────────
    flow_edges    = edges_df[edges_df['edge_type'] == 'flow']
    spatial_edges = edges_df[edges_df['edge_type'] == 'spatial']

    src_flow = [node_mapping[s] for s in flow_edges['src']]
    dst_flow = [node_mapping[d] for d in flow_edges['dst']]
    edge_index_flow = torch.tensor([src_flow, dst_flow], dtype=torch.long)

    src_sp = [node_mapping[s] for s in spatial_edges['src']]
    dst_sp = [node_mapping[d] for d in spatial_edges['dst']]
    edge_index_spatial = torch.tensor([src_sp, dst_sp], dtype=torch.long)

    distances = spatial_edges['distance_km'].values
    edge_weight_spatial = torch.tensor(np.exp(-distances / 40.0), dtype=torch.float)

    return Data(
        x=x,
        edge_index_flow=edge_index_flow,
        edge_index_spatial=edge_index_spatial,
        edge_weight_spatial=edge_weight_spatial,
        num_nodes=len(node_mapping),
    )
