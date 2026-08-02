import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from .graph_builder import build_static_graph

class FloodDataset(Dataset):
    """
    Dataset for the Kelani river flood early-warning benchmark.
    As per ARCHITECTURE_SPEC.md 2.1:
    - Loads panel.parquet, filters to valid_sample == True.
    - Respects split protocols via separate samplers/indices.
    """
    def __init__(self, panel_path, nodes_path, edges_path, window_days=14, split_type='train'):
        print(f"Loading {panel_path}...")
        self.panel_df = pd.read_parquet(panel_path)
        
        # Filter to valid_sample and specific split
        if 'valid_sample' in self.panel_df.columns:
            self.panel_df = self.panel_df[self.panel_df['valid_sample'] == True]
            
        # Example split filtering (assuming split_temporal contains 'train', 'val', 'test')
        if 'split_temporal' in self.panel_df.columns:
            self.panel_df = self.panel_df[self.panel_df['split_temporal'] == split_type]
            
        self.nodes_df = pd.read_csv(nodes_path)
        self.edges_df = pd.read_csv(edges_path)
        
        self.static_graph = build_static_graph(self.nodes_df, self.edges_df)
        self.window_days = window_days
        
        # Define targets as per spec
        self.target_cols = [
            'target_flood_1d', 'target_flood_2d', 'target_flood_3d', 
            'target_onset_1d', 'target_next1d_discharge', 'target_next3d_max_zscore'
        ]
        
        # Select exactly 33 temporal features (as per spec)
        # Exclude targets, identifiers, and metadata
        exclude = set(self.target_cols + ['date', 'node_id', 'basin', 'zone', 'position', 'split_temporal', 'split_basin_holdout', 'valid_sample', 'event_id', 'flood_moderate', 'flood_high', 'flood_severe', 'flood_state', 'label_confidence', 'thr_moderate', 'thr_high', 'thr_severe'])
        self.feature_cols = [c for c in self.panel_df.columns if c not in exclude][:33]
        
        print("Pivot data into 3D array (nodes, dates, features) for fast sliding window access...")
        # Create a complete grid of (nodes x dates) to avoid missing index errors
        self.panel_df['date'] = pd.to_datetime(self.panel_df['date'])
        self.unique_dates = np.sort(self.panel_df['date'].unique())
        self.unique_nodes = self.nodes_df['node_id'].values
        
        # Create a mapping from node_id -> index and date -> index
        self.node2idx = {n: i for i, n in enumerate(self.unique_nodes)}
        self.date2idx = {d: i for i, d in enumerate(self.unique_dates)}
        
        # Initialize 3D tensors: [num_dates, num_nodes, num_features]
        num_d = len(self.unique_dates)
        num_n = len(self.unique_nodes)
        
        self.X_temp = np.zeros((num_d, num_n, len(self.feature_cols)), dtype=np.float32)
        self.Y = np.zeros((num_d, num_n, len(self.target_cols)), dtype=np.float32)
        
        # Fill the arrays
        node_indices = self.panel_df['node_id'].map(self.node2idx).values
        date_indices = self.panel_df['date'].map(self.date2idx).values
        
        self.X_temp[date_indices, node_indices, :] = self.panel_df[self.feature_cols].values
        self.Y[date_indices, node_indices, :] = self.panel_df[self.target_cols].values
        
        # Valid indices are dates where we have at least `window_days` history
        self.valid_idx = np.arange(self.window_days, num_d)
        
    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        # Maps idx to the actual date index ensuring we have `window_days` history
        d_idx = self.valid_idx[idx]
        
        # History: [window_days, num_nodes, num_features] -> transpose to [num_nodes, window_days, num_features]
        x_hist = self.X_temp[d_idx - self.window_days : d_idx, :, :]
        temporal_features = torch.tensor(x_hist).transpose(0, 1) # [51, 14, 33]
        
        # Targets at t
        targets = torch.tensor(self.Y[d_idx, :, :]) # [51, 6]
        
        # Static terrain features [51, 9] (Already computed in graph_builder)
        terrain_features = self.static_graph.x
        
        # Dummy SAR inputs (to be replaced with actual image loading)
        has_sar = torch.zeros(len(self.unique_nodes), dtype=torch.bool)
        sar_chips = torch.zeros((len(self.unique_nodes), 2, 512, 512), dtype=torch.float32)
        
        return {
            'temporal_features': temporal_features,
            'terrain_features': terrain_features,
            'sar_chips': sar_chips,
            'has_sar': has_sar,
            'targets': targets,
            'edge_index_flow': self.static_graph.edge_index_flow,
            'edge_index_spatial': self.static_graph.edge_index_spatial,
            'edge_weight_spatial': self.static_graph.edge_weight_spatial
        }
