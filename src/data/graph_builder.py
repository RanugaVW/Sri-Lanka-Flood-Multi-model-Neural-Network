import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

def build_static_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Data:
    """
    Builds the static graph topology and node features.
    As per ARCHITECTURE_SPEC.md 2.1:
    - x: static node features (9 terrain features)
    - edge_index_flow: directed, 35 edges (upstream→downstream).
    - edge_index_spatial: weighted, 204 edges, weight = exp(-distance_km / 40).
    """
    # Create node mapping to 0-indexed integers
    node_mapping = {node_id: idx for idx, node_id in enumerate(nodes_df['node_id'].unique())}
    
    # Extract static features - placeholder for the 9 terrain features
    # Assuming numerical columns in nodes_df are the terrain features
    numeric_cols = nodes_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove 'node_id' if present in numeric
    if 'node_id' in numeric_cols:
        numeric_cols.remove('node_id')
    # Use up to 9 features
    terrain_cols = numeric_cols[:9] 
    
    x = torch.tensor(nodes_df[terrain_cols].values, dtype=torch.float)
    
    # Process edges
    flow_edges = edges_df[edges_df['edge_type'] == 'flow']
    spatial_edges = edges_df[edges_df['edge_type'] == 'spatial']
    
    # Flow edges
    src_flow = [node_mapping[src] for src in flow_edges['src']]
    dst_flow = [node_mapping[dst] for dst in flow_edges['dst']]
    edge_index_flow = torch.tensor([src_flow, dst_flow], dtype=torch.long)
    
    # Spatial edges
    src_spatial = [node_mapping[src] for src in spatial_edges['src']]
    dst_spatial = [node_mapping[dst] for dst in spatial_edges['dst']]
    edge_index_spatial = torch.tensor([src_spatial, dst_spatial], dtype=torch.long)
    
    # Spatial edge weights: exp(-distance_km / 40)
    distances = spatial_edges['distance_km'].values
    edge_weight_spatial = torch.tensor(np.exp(-distances / 40.0), dtype=torch.float)
    
    # Construct Data object
    data = Data(
        x=x,
        edge_index_flow=edge_index_flow,
        edge_index_spatial=edge_index_spatial,
        edge_weight_spatial=edge_weight_spatial,
        num_nodes=len(node_mapping)
    )
    
    return data
