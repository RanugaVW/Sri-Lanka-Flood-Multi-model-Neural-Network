import os
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from .graph_builder import build_static_graph

# Sentinel-1 dB decoding from uint8 PNG (per sar_flood_lite README):
#   VV_dB = R/255 * 35 - 30    (range -30..+5 dB)
#   VH_dB = G/255 * 35 - 30    (range -30..+5 dB)
_SAR_SCALE = 35.0 / 255.0
_SAR_SHIFT = -30.0


def _decode_sar_png(path: str) -> np.ndarray:
    """Load a SAR PNG and return a (2, H, W) float32 array in dB [VV, VH]."""
    img = np.array(Image.open(path).convert('RGB'), dtype=np.float32)  # (H, W, 3)
    vv = img[:, :, 0] * _SAR_SCALE + _SAR_SHIFT  # Red → VV dB
    vh = img[:, :, 1] * _SAR_SCALE + _SAR_SHIFT  # Green → VH dB
    return np.stack([vv, vh], axis=0)              # (2, H, W)


class SARIndex:
    """
    Pre-scans all SAR chip directories and builds a
    {site_id → sorted list of (date, filepath)} index for fast lookup.

    Parameters
    ----------
    sar_root : str
        Root directory that contains one subfolder per site
        (e.g. ``sar_flood_lite/frames/``).
    site_ids : list[str]
        Which site sub-directories to index (e.g. ['KEL_HAN']).
        If None, all sub-directories are indexed.
    max_age_days : int
        Maximum allowed gap between the requested date and the
        nearest SAR acquisition.  If the gap is larger,
        ``has_sar`` is set to False.
    """
    def __init__(self, sar_root: str, site_ids=None, max_age_days: int = 12):
        self.sar_root = sar_root
        self.max_age_days = max_age_days
        self.index: dict[str, list[tuple]] = {}  # site_id → [(pd.Timestamp, path)]

        if not os.path.isdir(sar_root):
            print(f"  [SARIndex] SAR root not found: {sar_root} — SAR will be disabled.")
            return

        available = sorted(os.listdir(sar_root))
        sites = site_ids if site_ids else available

        for site in sites:
            site_dir = os.path.join(sar_root, site)
            if not os.path.isdir(site_dir):
                continue
            entries = []
            for fname in sorted(os.listdir(site_dir)):
                if not fname.endswith('.png'):
                    continue
                date_str = fname[:-4]  # strip .png
                try:
                    date = pd.Timestamp(date_str)
                except ValueError:
                    continue
                entries.append((date, os.path.join(site_dir, fname)))
            if entries:
                self.index[site] = entries
        print(f"  [SARIndex] Indexed {sum(len(v) for v in self.index.values())} chips "
              f"across {len(self.index)} sites (max_age={max_age_days}d).")

    def get_chip(self, site_id: str, query_date: pd.Timestamp):
        """
        Returns (chip_array, has_sar):
          - chip_array : np.ndarray (2, H, W) in dB, or None
          - has_sar    : bool
        """
        entries = self.index.get(site_id)
        if not entries:
            return None, False

        # Binary-search for the closest date
        dates = [e[0] for e in entries]
        idx = np.searchsorted(dates, query_date)

        candidates = []
        if idx < len(dates):
            candidates.append((abs((dates[idx] - query_date).days), entries[idx][1]))
        if idx > 0:
            candidates.append((abs((dates[idx - 1] - query_date).days), entries[idx - 1][1]))

        best_days, best_path = min(candidates, key=lambda x: x[0])
        if best_days > self.max_age_days:
            return None, False

        return _decode_sar_png(best_path), True


class FloodDataset(Dataset):
    """
    Dataset for the Kelani river flood early-warning benchmark.
    As per ARCHITECTURE_SPEC.md 2.1:
    - Loads panel.parquet, filters to valid_sample == True.
    - Respects split protocols via separate samplers/indices.
    - Fix 8: Fits a StandardScaler on train split and applies it to all splits.
    - Fix 11: Loads real SAR chips from sar_flood_lite PNG dataset.
    """
    def __init__(
        self,
        panel_path,
        nodes_path,
        edges_path,
        window_days=14,
        split_type='train',
        scaler=None,
        scaler_save_path=None,
        # SAR parameters
        sar_root=None,
        sar_site='KEL_HAN',
        sar_max_age_days=12,
        sar_chip_size=512,
    ):
        print(f"Loading {panel_path} (split={split_type})...")
        self.split_type = split_type
        self.window_days = window_days
        self.sar_chip_size = sar_chip_size
        self.sar_site = sar_site

        # ── SAR index (Fix 11) ────────────────────────────────────────────
        if sar_root and os.path.isdir(sar_root):
            frames_dir = os.path.join(sar_root, 'frames')
            self.sar_index = SARIndex(
                sar_root=frames_dir,
                site_ids=[sar_site],
                max_age_days=sar_max_age_days,
            )
        else:
            self.sar_index = SARIndex(sar_root='__nonexistent__')  # empty index

        # ── Load full panel (unfiltered) to fit scaler on train subset ────
        full_df = pd.read_parquet(panel_path)
        if 'valid_sample' in full_df.columns:
            full_df = full_df[full_df['valid_sample'] == True]

        # ── Define feature columns ────────────────────────────────────────
        self.target_cols = [
            'target_flood_1d', 'target_flood_2d', 'target_flood_3d',
            'target_onset_1d', 'target_next1d_discharge', 'target_next3d_max_zscore'
        ]
        exclude = set(
            self.target_cols + [
                'date', 'node_id', 'basin', 'zone', 'position',
                'split_temporal', 'split_basin_holdout', 'valid_sample',
                'event_id', 'flood_moderate', 'flood_high', 'flood_severe',
                'flood_state', 'label_confidence', 'thr_moderate',
                'thr_high', 'thr_severe'
            ]
        )
        self.feature_cols = [c for c in full_df.columns if c not in exclude][:33]

        # ── Fix 8: Fit scaler ONLY on train split ─────────────────────────
        if scaler is None:
            assert split_type == 'train', (
                "A pre-fitted scaler must be supplied for val/test splits. "
                "Call FloodDataset for 'train' first, then reuse its .scaler."
            )
            train_df = full_df[full_df['split_temporal'] == 'train'] \
                if 'split_temporal' in full_df.columns else full_df
            scaler = StandardScaler()
            scaler.fit(train_df[self.feature_cols].values)
            print(f"  StandardScaler fitted on {len(train_df)} train rows.")
            if scaler_save_path:
                os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
                with open(scaler_save_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"  Scaler saved → {scaler_save_path}")
        self.scaler = scaler

        # ── Filter to the requested split ─────────────────────────────────
        if 'split_temporal' in full_df.columns:
            self.panel_df = full_df[full_df['split_temporal'] == split_type].copy()
        else:
            self.panel_df = full_df.copy()

        self.nodes_df = pd.read_csv(nodes_path)
        self.edges_df = pd.read_csv(edges_path)
        self.static_graph = build_static_graph(self.nodes_df, self.edges_df)

        # ── Pivot into 3-D arrays for fast sliding window access ──────────
        print("Pivoting data into 3D array (nodes × dates × features)...")
        self.panel_df['date'] = pd.to_datetime(self.panel_df['date'])
        self.unique_dates = np.sort(self.panel_df['date'].unique())
        self.unique_nodes = self.nodes_df['node_id'].values

        self.node2idx = {n: i for i, n in enumerate(self.unique_nodes)}
        self.date2idx = {d: i for i, d in enumerate(self.unique_dates)}

        num_d = len(self.unique_dates)
        num_n = len(self.unique_nodes)

        raw_X = np.zeros((num_d, num_n, len(self.feature_cols)), dtype=np.float32)
        self.Y = np.zeros((num_d, num_n, len(self.target_cols)), dtype=np.float32)

        node_indices = self.panel_df['node_id'].map(self.node2idx).values
        date_indices = self.panel_df['date'].map(self.date2idx).values

        raw_X[date_indices, node_indices, :] = self.panel_df[self.feature_cols].values.astype(np.float32)
        self.Y[date_indices, node_indices, :] = self.panel_df[self.target_cols].values.astype(np.float32)

        # ── Apply scaler ──────────────────────────────────────────────────
        original_shape = raw_X.shape
        flat = raw_X.reshape(-1, original_shape[-1])
        self.X_temp = self.scaler.transform(flat).astype(np.float32).reshape(original_shape)

        self.valid_idx = np.arange(self.window_days, num_d)
        print(f"  Ready: {len(self.valid_idx)} valid timesteps.")

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        d_idx = self.valid_idx[idx]

        # Temporal sliding window: [num_nodes, window_days, num_features]
        x_hist = self.X_temp[d_idx - self.window_days: d_idx, :, :]
        temporal_features = torch.tensor(x_hist).transpose(0, 1)  # [51, 14, 33]

        targets = torch.tensor(self.Y[d_idx, :, :])              # [51, 6]
        terrain_features = self.static_graph.x                   # [51, 9]

        # ── Fix 11: Real SAR chip loading ─────────────────────────────────
        # The dataset has one chip per SITE, not per node.  We broadcast the
        # same chip to all nodes that share this site (KEL_HAN by default).
        # Nodes without a matching site keep the zero/missing embedding.
        query_date = pd.Timestamp(self.unique_dates[d_idx])
        chip_array, has_chip = self.sar_index.get_chip(self.sar_site, query_date)

        num_nodes = len(self.unique_nodes)
        sar_chips = torch.zeros((num_nodes, 2, self.sar_chip_size, self.sar_chip_size),
                                dtype=torch.float32)
        has_sar = torch.zeros(num_nodes, dtype=torch.bool)

        if has_chip and chip_array is not None:
            # Resize chip to (2, sar_chip_size, sar_chip_size) if needed
            h, w = chip_array.shape[1], chip_array.shape[2]
            if h != self.sar_chip_size or w != self.sar_chip_size:
                chip_resized = np.zeros((2, self.sar_chip_size, self.sar_chip_size),
                                        dtype=np.float32)
                for c in range(2):
                    ch_img = Image.fromarray(chip_array[c]).resize(
                        (self.sar_chip_size, self.sar_chip_size), Image.BILINEAR
                    )
                    chip_resized[c] = np.array(ch_img, dtype=np.float32)
                chip_array = chip_resized

            chip_tensor = torch.tensor(chip_array)  # (2, H, W)

            # Broadcast to all nodes in this site
            # (In a multi-site setup you would match node_id → site_id here)
            sar_chips[:] = chip_tensor.unsqueeze(0).expand(num_nodes, -1, -1, -1)
            has_sar[:] = True

        return {
            'temporal_features':   temporal_features,
            'terrain_features':    terrain_features,
            'sar_chips':           sar_chips,
            'has_sar':             has_sar,
            'targets':             targets,
            'edge_index_flow':     self.static_graph.edge_index_flow,
            'edge_index_spatial':  self.static_graph.edge_index_spatial,
            'edge_weight_spatial': self.static_graph.edge_weight_spatial,
        }

