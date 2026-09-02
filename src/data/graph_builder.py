import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

# Continuous/one-hot static terrain vector fed into FiLMTerrain's MLP:
#   elevation_m, upstream_node_count, distance_to_outlet_km   (3, z-scored)
#   zone one-hot (wet/intermediate/dry)                        (3)
#   position one-hot (upstream/mid/downstream/outlet)          (4)
# Must match FiLMTerrain(input_dim=TERRAIN_DIM) in flood_model.py.
TERRAIN_DIM = 10

ZONES     = ['wet', 'intermediate', 'dry']
POSITIONS = ['upstream', 'mid', 'downstream', 'outlet']


def _flow_topology_features(node_ids: list, flow_edges: pd.DataFrame):
    """Derive drainage-size and river-mouth-proximity proxies from the flow
    edge chain (upstream_of -> downstream_of, one outgoing edge per node —
    this dataset's flow graph is a simple per-basin chain, no confluences).

    Returns
    -------
    upstream_counts   : [N]  — depth from the basin headwater (drainage-size proxy)
    dists_to_outlet_km: [N]  — remaining flow-path length to the basin outlet
    """
    parent, child, edge_dist_km = {}, {}, {}
    for _, row in flow_edges.iterrows():
        src, dst, d = row['src'], row['dst'], float(row['distance_km'])
        parent[dst] = src          # dst's immediate upstream neighbor
        child[src]  = dst          # src's immediate downstream neighbor
        edge_dist_km[src] = d

    upstream_cache: dict = {}

    def upstream_count(nid):
        if nid not in upstream_cache:
            p = parent.get(nid)
            upstream_cache[nid] = 0 if p is None else 1 + upstream_count(p)
        return upstream_cache[nid]

    outlet_cache: dict = {}

    def dist_to_outlet(nid):
        if nid not in outlet_cache:
            c = child.get(nid)
            outlet_cache[nid] = (
                0.0 if c is None else edge_dist_km[nid] + dist_to_outlet(c)
            )
        return outlet_cache[nid]

    upstream_counts    = np.array([upstream_count(n)  for n in node_ids], dtype=np.float32)
    dists_to_outlet_km = np.array([dist_to_outlet(n)  for n in node_ids], dtype=np.float32)
    return upstream_counts, dists_to_outlet_km


def _zscore(v: np.ndarray) -> np.ndarray:
    std = v.std()
    return (v - v.mean()) / std if std > 1e-8 else np.zeros_like(v)


def build_static_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Data:
    """
    Builds the static graph topology and node features.
    - x            : [num_nodes, TERRAIN_DIM] continuous/one-hot terrain vector,
                     see TERRAIN_DIM comment above.
    - basin_idx    : [num_nodes] long — basin identity (16 river systems),
                     consumed as a learned embedding in FiLMTerrain rather
                     than one-hot (keeps input width small; basin identity
                     carries real hydrological-regime information not
                     present elsewhere in the static features).
    - edge_index_flow: directed, 35 edges (upstream→downstream).
    - edge_index_spatial: weighted, 204 edges, weight = exp(-distance_km / 40).
    """
    # ── Node mapping ──────────────────────────────────────────────────────────
    node_ids = nodes_df['node_id'].tolist()
    node_mapping = {nid: i for i, nid in enumerate(node_ids)}

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

    # ── Terrain features ─────────────────────────────────────────────────────
    elevation = _zscore(nodes_df['elevation_m'].values.astype(np.float32))
    upstream_counts, dists_to_outlet = _flow_topology_features(node_ids, flow_edges)
    upstream_counts  = _zscore(upstream_counts)
    dists_to_outlet  = _zscore(dists_to_outlet)

    zone_onehot = pd.get_dummies(
        nodes_df['zone'].astype(str)
    ).reindex(columns=ZONES, fill_value=0).values.astype(np.float32)
    position_onehot = pd.get_dummies(
        nodes_df['position'].astype(str)
    ).reindex(columns=POSITIONS, fill_value=0).values.astype(np.float32)

    x = np.concatenate([
        elevation[:, None], upstream_counts[:, None], dists_to_outlet[:, None],
        zone_onehot, position_onehot,
    ], axis=1)
    assert x.shape[1] == TERRAIN_DIM, f"terrain vector width {x.shape[1]} != TERRAIN_DIM {TERRAIN_DIM}"
    x = torch.tensor(x, dtype=torch.float)   # [N, TERRAIN_DIM]

    # ── Basin identity (learned embedding, not one-hot) ─────────────────────
    basins = sorted(nodes_df['basin'].astype(str).unique())
    basin_to_idx = {b: i for i, b in enumerate(basins)}
    basin_idx = torch.tensor(
        [basin_to_idx[b] for b in nodes_df['basin'].astype(str)], dtype=torch.long
    )

    return Data(
        x=x,
        basin_idx=basin_idx,
        num_basins=len(basins),
        edge_index_flow=edge_index_flow,
        edge_index_spatial=edge_index_spatial,
        edge_weight_spatial=edge_weight_spatial,
        num_nodes=len(node_mapping),
    )
