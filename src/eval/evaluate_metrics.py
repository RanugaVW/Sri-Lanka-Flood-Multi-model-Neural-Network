import os
import pickle
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              average_precision_score, roc_auc_score,
                              brier_score_loss, confusion_matrix)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataset import FloodDataset
from models.flood_model import FloodModel

import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, val_loader, device, output_file=None):
    all_class_preds = []
    all_class_targets = []
    all_reg_preds = []
    all_reg_targets = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in val_loader:
            temporal_features = batch['temporal_features'].to(device)
            terrain_features = batch['terrain_features'][0].to(device)
            sar_chips = batch['sar_chips'][0].to(device)
            has_sar = batch['has_sar'][0].to(device)
            targets = batch['targets'][0].to(device)
            
            edge_index_flow = batch['edge_index_flow'][0].to(device)
            edge_index_spatial = batch['edge_index_spatial'][0].to(device)
            edge_weight_spatial = batch['edge_weight_spatial'][0].to(device)
            
            predictions = model(
                temporal_features[0],
                terrain_features, 
                sar_chips, 
                has_sar, 
                edge_index_flow, 
                edge_index_spatial, 
                edge_weight_spatial
            )
            
            # Classification indices: 0 to 3
            # NOTE: OutputHeads already applies sigmoid internally — do NOT apply again.
            class_probs = predictions[:, :4].cpu().numpy()
            class_targets = targets[:, :4].int().cpu().numpy()
            
            # Regression indices: 4 to 5
            reg_preds = predictions[:, 4:].cpu().numpy()
            reg_targets = targets[:, 4:].cpu().numpy()
            
            all_class_preds.append(class_probs)
            all_class_targets.append(class_targets)
            all_reg_preds.append(reg_preds)
            all_reg_targets.append(reg_targets)
            
    # Concatenate all batches
    all_class_preds = np.concatenate(all_class_preds, axis=0)
    all_class_targets = np.concatenate(all_class_targets, axis=0)
    all_reg_preds = np.concatenate(all_reg_preds, axis=0)
    all_reg_targets = np.concatenate(all_reg_targets, axis=0)

    SEP  = "=" * 80
    DASH = "-" * 80

    # ── Classification table (terminal-style, matching screenshot) ─────────
    cls_names = ['flood_t+1', 'flood_t+2', 'flood_t+3', 'onset']
    reg_names = ['discharge_t1', 'zscore_3d_max']

    lines = []
    lines.append(f"\n{SEP}")
    lines.append(f"{'FLOOD DNN  EVALUATION SUMMARY':^80}")
    lines.append(SEP)
    hdr = f"{'Stage / Protocol':<20} {'PR-AUC':>7} {'ROC-AUC':>8} {'Brier':>8} {'ECE':>7} {'POD':>6} {'FAR':>6} {'CSI':>6}"
    lines.append(hdr)
    lines.append(DASH)

    for i, name in enumerate(cls_names):
        y_true  = all_class_targets[:, i]
        y_probs = all_class_preds[:, i]

        try:
            pr_auc  = average_precision_score(y_true, y_probs)
            roc_auc = roc_auc_score(y_true, y_probs)
        except ValueError:
            pr_auc = roc_auc = float('nan')

        brier = brier_score_loss(y_true, y_probs)

        # ECE
        n_bins = 10
        boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for j in range(n_bins):
            lo, hi  = boundaries[j], boundaries[j + 1]
            mask = (y_probs >= lo) & (y_probs <= hi)
            if mask.sum() > 0:
                ece += np.abs(y_probs[mask].mean() - y_true[mask].mean()) * mask.mean()

        # Threshold sweep for best CSI
        best_csi, best_pod, best_far = -1, 0, 0
        for thr in np.arange(0.01, 1.0, 0.01):
            y_pred = (y_probs >= thr).astype(int)
            if len(np.unique(y_true)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            else:
                tp = int(np.sum((y_true == 1) & (y_pred == 1)))
                fp = int(np.sum((y_true == 0) & (y_pred == 1)))
                fn = int(np.sum((y_true == 1) & (y_pred == 0)))
                tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            pod = tp / (tp + fn) if (tp + fn) > 0 else 0
            far = fp / (tp + fp) if (tp + fp) > 0 else 0
            csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
            if csi > best_csi:
                best_csi, best_pod, best_far = csi, pod, far

        row = (f"{name:<20} {pr_auc:>7.4f} {roc_auc:>8.4f} {brier:>8.5f} "
               f"{ece:>7.4f} {best_pod:>6.3f} {best_far:>6.3f} {best_csi:>6.3f}")
        lines.append(row)

    lines.append(SEP)

    # ── Regression sub-table ───────────────────────────────────────────────
    lines.append(f"\n{'Regression Targets':<20} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    lines.append("-" * 46)
    for i, name in enumerate(reg_names):
        y_true = all_reg_targets[:, i]
        y_pred = all_reg_preds[:, i]
        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        lines.append(f"{name:<20} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f}")
    lines.append(SEP)

    results = "\n".join(lines)

    # Print to terminal
    print(results)

    # Optionally write to file
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(results)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flood Early-Warning Model")
    parser.add_argument('--experiment_dir', type=str, default='experiments/wp2_baselines',
                        help='Root experiment directory (same as used during training)')
    parser.add_argument('--data_config', type=str, default='configs/data.yaml',
                        help='Path to data config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Which seed checkpoint to evaluate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Configs
    data_cfg  = load_config(args.data_config)
    model_cfg = load_config('configs/model.yaml')

    # ── Load the scaler that was fit during training (no data leakage) ────
    scaler_path = os.path.join(args.experiment_dir, 'scaler.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")
    else:
        print(f"[WARNING] Scaler not found at {scaler_path}. "
              f"Val features will NOT be normalized — results may be inaccurate.")

    # ── Load val dataset with the train scaler ────────────────────────────
    val_dataset = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='val',
        scaler=scaler,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ── Load model checkpoint ─────────────────────────────────────────────
    model = FloodModel(config=model_cfg).to(device)
    checkpoint_path = os.path.join(
        args.experiment_dir, f'seed_{args.seed}', 'checkpoints', 'best_model.pth'
    )
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    os.makedirs('Docs', exist_ok=True)
    evaluate_model(model, val_loader, device, output_file='Docs/evaluation_results.md')
    print("Evaluation complete. Results written to Docs/evaluation_results.md")


if __name__ == '__main__':
    main()
