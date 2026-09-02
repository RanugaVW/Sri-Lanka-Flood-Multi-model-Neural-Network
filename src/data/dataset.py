"""FloodDataset — graph-snapshot dataset for the Sri Lanka flood model.

Each __getitem__ returns ONE full-graph day snapshot:
  temporal_features : [N, L, F]  — all N nodes, L-day window, F features
  terrain_features  : [N, 10]    — static (same every day, see graph_builder.py)
  basin_idx         : [N]        — basin identity (0-15), static
  sar_chips         : [N, 2, H, W]
  has_sar           : [N] bool
  targets           : [N, 6]     — 4 cls + 2 reg targets
  valid_mask        : [N]        — 1 where node-day is a valid sample
  label_conf        : [N]        — label_confidence (down-weights boundary days)
  event_ids         : [N]        — event code ≥0 inside a flood episode, -1 otherwise
  day_idx           : scalar     — index of day t in the split
  edge_index_flow, edge_index_spatial, edge_weight_spatial  (static graph)

Leakage controls
----------------
* StandardScaler is fitted on the TRAINING split rows only (via split_temporal=='train').
* Callers must pass the train-fitted scaler when constructing val/test datasets.
* Targets are read from pre-computed causal columns in the parquet; no
  future data is accessible through the feature matrix.
"""
import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from .graph_builder import build_static_graph

# ── SAR decoding constants (Sentinel-1 dB from uint8 PNG) ────────────────────
_SAR_SCALE = 35.0 / 255.0
_SAR_SHIFT = -30.0


@lru_cache(maxsize=32)
def _decode_sar_png(path: str) -> np.ndarray:
    """Load a SAR PNG → (2, H, W) float32 array in dB [VV, VH].

    Cached by path: the same nearest chip is reused across ~max_age_days
    consecutive daily snapshots for a given node, so decoding it once per
    distinct file (instead of once per snapshot) avoids redundant PNG
    decode work now that each node looks up its own site independently.
    """
    img = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
    vv  = img[:, :, 0] * _SAR_SCALE + _SAR_SHIFT
    vh  = img[:, :, 1] * _SAR_SCALE + _SAR_SHIFT
    return np.stack([vv, vh], axis=0)


# ── SAR chip index ────────────────────────────────────────────────────────────

class SARIndex:
    """Maps (node_id, query_date) → nearest SAR chip within max_age_days.

    One sub-index is built per site directory under `sar_root/frames/` whose
    name matches a `node_id` — SAR site IDs (e.g. KEL_HAN, KAL_BUL) are the
    same strings as node IDs in nodes.csv, so this covers exactly the nodes
    with real per-node imagery (9 of 51 in the current dataset). Nodes with
    no matching directory simply have no entry and always resolve to
    has_sar=False — no imagery is ever broadcast across nodes.
    """

    def __init__(self, sar_root: str, site_ids=None, max_age_days: int = 12):
        self.sar_root      = sar_root
        self.max_age_days  = max_age_days
        self.index: dict   = {}

        if not sar_root or not os.path.isdir(sar_root):
            return

        frames_dir = os.path.join(sar_root, 'frames')
        if not os.path.isdir(frames_dir):
            frames_dir = sar_root

        available = sorted(os.listdir(frames_dir))
        sites     = site_ids if site_ids else available

        for site in sites:
            site_dir = os.path.join(frames_dir, site)
            if not os.path.isdir(site_dir):
                continue
            entries = []
            for fname in sorted(os.listdir(site_dir)):
                if not fname.endswith('.png'):
                    continue
                try:
                    date = pd.Timestamp(fname[:-4])
                except ValueError:
                    continue
                entries.append((date, os.path.join(site_dir, fname)))
            if entries:
                self.index[site] = entries

        total = sum(len(v) for v in self.index.values())
        print(f"  [SARIndex] {total} chips across {len(self.index)} sites "
              f"(max_age={max_age_days}d)")

    def get_chip(self, node_id: str, query_date: pd.Timestamp):
        entries = self.index.get(node_id)
        if not entries:
            return None, False
        dates = [e[0] for e in entries]
        idx   = np.searchsorted(dates, query_date)
        candidates = []
        if idx < len(dates):
            candidates.append((abs((dates[idx] - query_date).days), entries[idx][1]))
        if idx > 0:
            candidates.append((abs((dates[idx - 1] - query_date).days), entries[idx - 1][1]))
        best_days, best_path = min(candidates, key=lambda c: c[0])
        if best_days > self.max_age_days:
            return None, False
        return _decode_sar_png(best_path), True


# ── Main dataset ──────────────────────────────────────────────────────────────

class FloodDataset(Dataset):
    """One sample = one full-graph day snapshot (all N nodes simultaneously).

    Parameters
    ----------
    panel_path      : path to flood_dataset.parquet
    nodes_path      : path to nodes.csv
    edges_path      : path to edges.csv
    window_days     : lookback window L (default 14)
    split_type      : 'train' | 'val' | 'test'
    scaler          : pre-fitted StandardScaler (required for val/test)
    scaler_save_path: if given, saves the train scaler to this path
    sar_root        : root directory for SAR chips (None → SAR disabled)
    sar_max_age_days: max days between query date and nearest SAR chip, per node
    sar_chip_size   : target chip spatial size in pixels
    """

    TARGET_COLS = [
        'target_flood_1d', 'target_flood_2d', 'target_flood_3d',
        'target_onset_1d',
        'target_next1d_discharge', 'target_next3d_max_zscore',
    ]
    EXCLUDE_COLS = {
        'date', 'node_id', 'basin', 'zone', 'position',
        'split_temporal', 'split_basin_holdout', 'valid_sample',
        'event_id', 'flood_moderate', 'flood_high', 'flood_severe',
        'flood_state', 'label_confidence', 'thr_moderate',
        'thr_high', 'thr_severe',
    }

    def __init__(
        self,
        panel_path:       str,
        nodes_path:       str,
        edges_path:       str,
        window_days:      int   = 14,
        split_type:       str   = 'train',
        scaler=None,
        scaler_save_path: str   = None,
        sar_root:         str   = None,
        sar_max_age_days: int   = 12,
        sar_chip_size:    int   = 512,
    ):
        self.split_type   = split_type
        self.window_days  = window_days
        self.sar_chip_size = sar_chip_size
        self._sar_root    = sar_root
        self._sar_max_age_days = sar_max_age_days

        # ── Load full panel (unfiltered) ──────────────────────────────────────
        print(f"Loading {panel_path}...")
        full_df = pd.read_parquet(panel_path)
        if 'valid_sample' in full_df.columns:
            full_df = full_df[full_df['valid_sample'] == True]

        # ── Feature columns ───────────────────────────────────────────────────
        exclude = self.EXCLUDE_COLS | set(self.TARGET_COLS)
        self.feature_cols = [c for c in full_df.columns if c not in exclude][:33]

        # ── Log1p transform heavy-tailed columns (before z-scoring) ──────────
        LOG1P_COLS = {
            'precipitation_sum', 'precip_sum_2d', 'precip_sum_3d', 'precip_sum_5d',
            'precip_sum_7d', 'precip_sum_15d', 'precip_sum_30d',
            'precip_max_3d', 'precip_max_7d', 'api_k090',
            'discharge', 'discharge_mean_3d', 'discharge_mean_7d',
        }
        for col in LOG1P_COLS:
            if col in full_df.columns:
                full_df[col] = np.sign(full_df[col]) * np.log1p(np.abs(full_df[col]))

        # ── Fit scaler on TRAIN rows only ─────────────────────────────────────
        if scaler is None:
            assert split_type == 'train', (
                "Provide a pre-fitted scaler for val/test. "
                "Construct train dataset first and reuse its .scaler."
            )
            train_rows = (full_df['split_temporal'] == 'train'
                          if 'split_temporal' in full_df.columns
                          else full_df)
            scaler = StandardScaler()
            scaler.fit(full_df.loc[
                full_df['split_temporal'] == 'train', self.feature_cols
            ].values)
            print(f"  StandardScaler fitted on {(full_df['split_temporal']=='train').sum()} train rows.")
            if scaler_save_path:
                os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
                with open(scaler_save_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"  Scaler saved → {scaler_save_path}")
        self.scaler = scaler

        # ── Filter to requested split ─────────────────────────────────────────
        if 'split_temporal' in full_df.columns:
            split_df = full_df[full_df['split_temporal'] == split_type].copy()
        else:
            split_df = full_df.copy()

        # ── Static graph ──────────────────────────────────────────────────────
        self.nodes_df = pd.read_csv(nodes_path)
        self.edges_df = pd.read_csv(edges_path)
        self.static_graph = build_static_graph(self.nodes_df, self.edges_df)

        # ── SAR index — one sub-index per node_id that has its own SAR site ────
        self.sar_index = SARIndex(
            self._sar_root,
            site_ids=self.nodes_df['node_id'].tolist(),
            max_age_days=self._sar_max_age_days,
        )

        # ── Pivot into 3-D dense arrays [T, N, *] ─────────────────────────────
        print("Pivot data into 3D array (nodes, dates, features) for fast sliding window access...")
        split_df['date'] = pd.to_datetime(split_df['date'])
        self.unique_dates = np.sort(split_df['date'].unique())
        self.unique_nodes = self.nodes_df['node_id'].values
        N = len(self.unique_nodes)
        T = len(self.unique_dates)

        node2idx = {n: i for i, n in enumerate(self.unique_nodes)}
        date2idx = {d: i for i, d in enumerate(self.unique_dates)}

        nidx = split_df['node_id'].map(node2idx).values
        tidx = split_df['date'].map(date2idx).values

        # Feature matrix (raw, before z-scoring)
        raw_X = np.zeros((T, N, len(self.feature_cols)), dtype=np.float32)
        raw_X[tidx, nidx, :] = split_df[self.feature_cols].values.astype(np.float32)

        # Apply scaler
        orig_shape = raw_X.shape
        self.X = self.scaler.transform(
            raw_X.reshape(-1, orig_shape[-1])
        ).astype(np.float32).reshape(orig_shape)

        # Targets [T, N, 6]
        self.Y = np.zeros((T, N, len(self.TARGET_COLS)), dtype=np.float32)
        valid_targets = [c for c in self.TARGET_COLS if c in split_df.columns]
        self.Y[tidx, nidx, :len(valid_targets)] = \
            split_df[valid_targets].values.astype(np.float32)

        # Valid mask [T, N]  — 1 where this node-day is a labelled sample
        self.valid_mask = np.zeros((T, N), dtype=np.float32)
        self.valid_mask[tidx, nidx] = 1.0

        # Label confidence [T, N]  — down-weights boundary days
        self.label_conf = np.ones((T, N), dtype=np.float32)
        if 'label_confidence' in split_df.columns:
            lc = split_df['label_confidence'].values.astype(np.float32)
            lc = np.nan_to_num(lc, nan=1.0)
            self.label_conf[tidx, nidx] = lc

        # Event IDs [T, N]  — for episode-level detection metric (ev.det)
        self.event_ids = np.full((T, N), -1, dtype=np.int32)
        if 'event_id' in split_df.columns:
            codes, _ = pd.factorize(split_df['event_id'], use_na_sentinel=True)
            self.event_ids[tidx, nidx] = codes.astype(np.int32)

        # Valid snapshot indices (need full lookback window)
        self.valid_idx = np.arange(window_days, T)
        print(f"  Ready: {len(self.valid_idx)} valid day snapshots "
              f"({N} nodes each).")

        # Node indices that actually have a SAR site — avoids a dict lookup
        # per node per snapshot for the ~80% of nodes that never have SAR.
        self._sar_node_idx = [
            (i, node_id) for i, node_id in enumerate(self.unique_nodes)
            if node_id in self.sar_index.index
        ]

    def __len__(self):
        return len(self.valid_idx)

    def _resize_chip(self, chip_arr: np.ndarray) -> np.ndarray:
        """Resize a decoded (2, H, W) chip to (2, sar_chip_size, sar_chip_size)
        if needed. A no-op for the shipped 512x512 chips at the default size."""
        h, w = chip_arr.shape[1], chip_arr.shape[2]
        if h == self.sar_chip_size and w == self.sar_chip_size:
            return chip_arr
        resized = np.zeros((2, self.sar_chip_size, self.sar_chip_size), dtype=np.float32)
        for c in range(2):
            ch = Image.fromarray(chip_arr[c]).resize(
                (self.sar_chip_size, self.sar_chip_size), Image.BILINEAR)
            resized[c] = np.array(ch, dtype=np.float32)
        return resized

    def __getitem__(self, idx):
        d = self.valid_idx[idx]

        # Temporal window [window_days, N, F] → transpose → [N, window_days, F]
        x_window = self.X[d - self.window_days: d, :, :]    # [L, N, F]
        temporal  = torch.tensor(x_window).permute(1, 0, 2)  # [N, L, F]

        targets    = torch.tensor(self.Y[d],          dtype=torch.float32)  # [N, 6]
        valid_mask = torch.tensor(self.valid_mask[d], dtype=torch.float32)  # [N]
        label_conf = torch.tensor(self.label_conf[d], dtype=torch.float32)  # [N]
        event_ids  = torch.tensor(self.event_ids[d],  dtype=torch.int32)    # [N]

        # SAR chip — looked up independently per node's own site (no broadcast
        # across unrelated basins); only nodes with a SAR site nearby and a
        # chip within max_age_days get a real chip, the rest stay zero/False.
        query_date = pd.Timestamp(self.unique_dates[d])
        N   = len(self.unique_nodes)
        sar = torch.zeros((N, 2, self.sar_chip_size, self.sar_chip_size),
                          dtype=torch.float32)
        has_sar = torch.zeros(N, dtype=torch.bool)

        for i, node_id in self._sar_node_idx:
            chip_arr, has_chip = self.sar_index.get_chip(node_id, query_date)
            if not has_chip or chip_arr is None:
                continue
            chip_arr = self._resize_chip(chip_arr)
            sar[i]     = torch.tensor(chip_arr)
            has_sar[i] = True

        return {
            'temporal_features':    temporal,
            'terrain_features':     self.static_graph.x,
            'basin_idx':            self.static_graph.basin_idx,
            'sar_chips':            sar,
            'has_sar':              has_sar,
            'targets':              targets,
            'valid_mask':           valid_mask,
            'label_conf':           label_conf,
            'event_ids':            event_ids,
            'day_idx':              torch.tensor(d, dtype=torch.int32),
            'edge_index_flow':      self.static_graph.edge_index_flow,
            'edge_index_spatial':   self.static_graph.edge_index_spatial,
            'edge_weight_spatial':  self.static_graph.edge_weight_spatial,
        }
