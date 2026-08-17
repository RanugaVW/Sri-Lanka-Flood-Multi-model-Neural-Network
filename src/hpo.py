"""
Hyperparameter optimisation for the Sri Lanka Flood DNN using Optuna.

Usage (from repo root, e.g. in a Kaggle notebook cell):
    !python src/hpo.py \
        --data_config configs/kaggle_data.yaml \
        --experiment_dir /kaggle/working/hpo \
        --n_trials 30 \
        --timeout 3600

The best hyperparameters are printed at the end and written to
    <experiment_dir>/best_hparams.yaml
so you can copy them back into configs/train.yaml and configs/model.yaml.
"""

import os
import sys
import yaml
import random
import pickle
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Make src imports work
sys.path.insert(0, os.path.dirname(__file__))
from data.dataset import FloodDataset
from models.flood_model import FloodModel
from losses.multitask_loss import MultiTaskLoss

import optuna
from optuna.samplers import TPESampler


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ── Single trial objective ─────────────────────────────────────────────────────
def objective(trial, data_cfg, device, experiment_dir, n_epochs_hpo=15, patience_hpo=5):
    """
    We use a shortened n_epochs_hpo run (default 15) for speed.
    Objective: minimise validation loss.
    """
    set_seed(42)

    # ── Suggest hyperparameters ────────────────────────────────────────────
    lr             = trial.suggest_float('learning_rate',    1e-5, 5e-4, log=True)
    weight_decay   = trial.suggest_float('weight_decay',     1e-6, 1e-3, log=True)
    focal_gamma    = trial.suggest_float('focal_gamma',      1.0,  5.0)
    focal_alpha    = trial.suggest_float('focal_alpha',      0.25, 0.95)
    reg_weight     = trial.suggest_float('regression_weight',0.1,  0.5)
    dropout        = trial.suggest_float('dropout',          0.05, 0.3)
    gru_hidden_dim = trial.suggest_categorical('gru_hidden_dim', [64, 128, 256])
    gru_num_layers = trial.suggest_int('gru_num_layers',     1,    3)
    window_days    = trial.suggest_categorical('window_days', [7, 14, 30])
    grad_clip_norm = trial.suggest_float('grad_clip_norm',   0.5,  5.0)

    # Derived: fusion/output dim from GRU hidden dim (SAR always 64)
    fusion_concat_dim = gru_hidden_dim + 64

    # ── Build datasets ────────────────────────────────────────────────────
    scaler_path = os.path.join(experiment_dir, 'hpo_scaler.pkl')
    train_dataset = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='train',
        window_days=window_days,
        scaler=None,
        scaler_save_path=scaler_path,
    )
    val_dataset = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='val',
        window_days=window_days,
        scaler=train_dataset.scaler,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

    # ── Build model with suggested params ─────────────────────────────────
    # Override defaults via direct constructor arguments
    from models.temporal_encoder import TemporalEncoder
    from models.film_terrain import FiLMTerrain
    from models.sar_cnn import SARCNN
    from models.fusion import FusionBlock
    from models.graph_gnn import GraphGNN
    from models.heads import OutputHeads

    class TrialFloodModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_encoder = TemporalEncoder(
                input_dim=33,
                hidden_dim=gru_hidden_dim,
                num_layers=gru_num_layers,
                dropout=dropout,
            )
            self.film_terrain = FiLMTerrain(
                input_dim=9,
                hidden_dim=64,
                output_dim=gru_hidden_dim,
            )
            self.sar_cnn = SARCNN(embedding_dim=64)
            self.fusion  = FusionBlock(
                concat_dim=fusion_concat_dim,
                hidden_dim=fusion_concat_dim,
                output_dim=gru_hidden_dim,
            )
            self.gnn   = GraphGNN(
                in_channels=gru_hidden_dim,
                hidden_channels=gru_hidden_dim,
                out_channels=gru_hidden_dim,
                heads=4,
            )
            self.heads = OutputHeads(
                in_features=gru_hidden_dim,
                dropout=dropout,
            )

        def forward(self, temporal_features, terrain_features, sar_chips,
                    has_sar, edge_index_flow, edge_index_spatial, edge_weight_spatial):
            temporal_out      = self.temporal_encoder(temporal_features)
            modulated_temporal = self.film_terrain(terrain_features, temporal_out)
            sar_embedding     = self.sar_cnn(sar_chips, has_sar)
            fused             = self.fusion(modulated_temporal, sar_embedding)
            gnn_out           = self.gnn(fused, edge_index_flow,
                                         edge_index_spatial, edge_weight_spatial)
            return self.heads(gnn_out)

    model = TrialFloodModel().to(device)

    criterion = MultiTaskLoss(
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        regression_weight=reg_weight,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs_hpo, eta_min=1e-7
    )

    best_val_loss  = float('inf')
    patience_count = 0

    for epoch in range(n_epochs_hpo):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            temporal_features  = batch['temporal_features'].to(device)
            terrain_features   = batch['terrain_features'][0].to(device)
            sar_chips          = batch['sar_chips'][0].to(device)
            has_sar            = batch['has_sar'][0].to(device)
            targets            = batch['targets'][0].to(device)
            edge_index_flow    = batch['edge_index_flow'][0].to(device)
            edge_index_spatial = batch['edge_index_spatial'][0].to(device)
            edge_weight_spatial = batch['edge_weight_spatial'][0].to(device)

            preds = model(temporal_features[0], terrain_features,
                          sar_chips, has_sar,
                          edge_index_flow, edge_index_spatial, edge_weight_spatial)
            lc = criterion(preds, targets)
            lc.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                temporal_features  = batch['temporal_features'].to(device)
                terrain_features   = batch['terrain_features'][0].to(device)
                sar_chips          = batch['sar_chips'][0].to(device)
                has_sar            = batch['has_sar'][0].to(device)
                targets            = batch['targets'][0].to(device)
                edge_index_flow    = batch['edge_index_flow'][0].to(device)
                edge_index_spatial = batch['edge_index_spatial'][0].to(device)
                edge_weight_spatial = batch['edge_weight_spatial'][0].to(device)
                preds = model(temporal_features[0], terrain_features,
                              sar_chips, has_sar,
                              edge_index_flow, edge_index_spatial, edge_weight_spatial)
                val_loss += criterion(preds, targets).total.item()
        val_loss /= max(len(val_loader), 1)

        # Pruning: let Optuna kill unpromising trials early
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience_hpo:
                break

    return best_val_loss


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for Flood DNN")
    parser.add_argument('--data_config',    type=str, default='configs/kaggle_data.yaml')
    parser.add_argument('--experiment_dir', type=str, default='experiments/hpo')
    parser.add_argument('--n_trials',       type=int, default=30,
                        help='Number of Optuna trials')
    parser.add_argument('--timeout',        type=int, default=None,
                        help='Wall-clock timeout in seconds (optional)')
    parser.add_argument('--n_epochs_hpo',   type=int, default=15,
                        help='Epochs per trial (keep short for speed)')
    args = parser.parse_args()

    os.makedirs(args.experiment_dir, exist_ok=True)
    data_cfg = load_config(args.data_config)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"HPO device: {device}")

    # Persist study to SQLite so you can resume if the notebook times out
    storage = f"sqlite:///{os.path.join(args.experiment_dir, 'optuna_study.db')}"
    study   = optuna.create_study(
        study_name='flood_hpo',
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial, data_cfg, device, args.experiment_dir, args.n_epochs_hpo
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    # ── Report results ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  HPO Complete")
    print("="*60)
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best val loss: {study.best_value:.4f}")
    print("  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Save best params to YAML
    out_path = os.path.join(args.experiment_dir, 'best_hparams.yaml')
    with open(out_path, 'w') as f:
        yaml.dump({'best_val_loss': study.best_value, **study.best_params}, f)
    print(f"\n  Best params saved → {out_path}")
    print("  Copy these values into configs/train.yaml and configs/model.yaml")
    print("="*60)


if __name__ == '__main__':
    main()
