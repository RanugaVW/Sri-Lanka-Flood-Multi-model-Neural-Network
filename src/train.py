"""
train.py — Main training entry point for the Sri Lanka Flood Early-Warning Model.

Epoch log format (matches STGCN-style output):
    epoch X/Y | loss 0.XXXX (cls 0.XXXX reg 0.XXXX) | val pr_auc 0.XXXX (best 0.XXXX)

Final summary table format:
    Stage / Protocol   PR-AUC  ROC-AUC  Brier   ECE    POD    FAR    CSI
    ─────────────────────────────────────────────────────────────────────────
    temporal           0.XXXX  0.XXXX   0.XXXXX 0.XXXX 0.XXX  0.XXX  0.XXX

Complete model checkpoint (saved per seed):
    best_model.pth    — full checkpoint dict (weights + config + metadata)
    best_model_full.pth — entire model object via torch.save(model)
    scaler.pkl        — fitted StandardScaler (for inference on new data)
"""

import os
import sys
import json
import time
import pickle
import random
import argparse

import numpy as np
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix
)

sys.path.insert(0, os.path.dirname(__file__))
from data.dataset import FloodDataset
from models.flood_model import FloodModel
from losses.multitask_loss import MultiTaskLoss
from eval.evaluate_metrics import evaluate_model


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_batch(model, batch, device):
    """Unpack one batch dict and run a forward pass. Returns (predictions, targets)."""
    temporal_features   = batch['temporal_features'].to(device)
    terrain_features    = batch['terrain_features'][0].to(device)
    sar_chips           = batch['sar_chips'][0].to(device)
    has_sar             = batch['has_sar'][0].to(device)
    targets             = batch['targets'][0].to(device)
    edge_index_flow     = batch['edge_index_flow'][0].to(device)
    edge_index_spatial  = batch['edge_index_spatial'][0].to(device)
    edge_weight_spatial = batch['edge_weight_spatial'][0].to(device)

    preds = model(
        temporal_features[0],   # one timestep → all 51 nodes
        terrain_features,
        sar_chips,
        has_sar,
        edge_index_flow,
        edge_index_spatial,
        edge_weight_spatial,
    )
    return preds, targets


# ── Per-epoch train ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_norm=1.0):
    model.train()
    tot, tot_cls, tot_reg = 0.0, 0.0, 0.0
    for batch in loader:
        optimizer.zero_grad()
        preds, targets = _run_batch(model, batch, device)
        lc = criterion(preds, targets)
        lc.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        tot     += lc.total.item()
        tot_cls += lc.cls.item()
        tot_reg += lc.reg.item()
    n = max(len(loader), 1)
    return tot / n, tot_cls / n, tot_reg / n


# ── Per-epoch validation (loss + quick PR-AUC on flood_t1) ────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    tot = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            preds, targets = _run_batch(model, batch, device)
            lc = criterion(preds, targets)
            tot += lc.total.item()
            # Collect flood_t+1 probabilities for quick PR-AUC (index 0)
            all_probs.append(preds[:, 0].cpu().numpy())
            all_labels.append(targets[:, 0].int().cpu().numpy())

    n = max(len(loader), 1)
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = float('nan')

    return tot / n, pr_auc


# ── Full metrics at a single optimal threshold ─────────────────────────────────
def compute_threshold_metrics(y_true: np.ndarray, y_probs: np.ndarray):
    """Returns (pr_auc, roc_auc, brier, ece, pod, far, csi, opt_thresh)."""
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
        lo, hi = boundaries[j], boundaries[j + 1]
        mask = (y_probs >= lo) & (y_probs <= hi)
        if mask.sum() > 0:
            ece += np.abs(y_probs[mask].mean() - y_true[mask].mean()) * mask.mean()

    # Sweep threshold for best CSI
    best_csi, best_pod, best_far, best_thresh = -1, 0, 0, 0.5
    for thr in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_probs >= thr).astype(int)
        if len(np.unique(y_true)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        else:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
        if csi > best_csi:
            best_csi, best_pod, best_far, best_thresh = csi, pod, far, thr

    return pr_auc, roc_auc, brier, ece, best_pod, best_far, best_csi, best_thresh


def collect_val_predictions(model, loader, device):
    """Run model over val set and return (class_probs [N,4], class_targets [N,4])."""
    all_probs, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            preds, targets = _run_batch(model, batch, device)
            all_probs.append(preds[:, :4].cpu().numpy())
            all_targets.append(targets[:, :4].int().cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


# ── Pretty print helpers ───────────────────────────────────────────────────────
SEP  = "=" * 80
DASH = "-" * 80
HDR  = f"{'Stage / Protocol':<22} {'PR-AUC':>7} {'ROC-AUC':>8} {'Brier':>8} {'ECE':>7} {'POD':>6} {'FAR':>6} {'CSI':>6}"
ROW_FMT = "{stage:<22} {pr:>7.4f} {roc:>8.4f} {brier:>8.5f} {ece:>7.4f} {pod:>6.3f} {far:>6.3f} {csi:>6.3f}"


def print_summary_table(rows: list[dict]):
    print(f"\n{SEP}")
    print(f"{'FLOOD DNN — EVALUATION SUMMARY':^80}")
    print(SEP)
    print(HDR)
    print(DASH)
    for r in rows:
        print(ROW_FMT.format(**r))
    print(SEP)


# ── Complete model save ────────────────────────────────────────────────────────
def save_complete_checkpoint(
    model, optimizer, scheduler, epoch, best_val_loss, val_metrics,
    model_cfg, train_cfg, scaler, checkpoint_dir, seed
):
    """
    Saves everything needed to resume training or run inference:
      best_model.pth      — full checkpoint dict (weights + configs + metrics)
      best_model_full.pth — entire torch model object (for quick loading)
      scaler.pkl          — fitted StandardScaler
      training_summary.json — human-readable training metadata
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Full checkpoint dict
    ckpt = {
        'epoch':          epoch,
        'seed':           seed,
        'model_state':    model.state_dict(),
        'optimizer_state':optimizer.state_dict(),
        'scheduler_state':scheduler.state_dict(),
        'best_val_loss':  best_val_loss,
        'val_metrics':    val_metrics,
        'model_cfg':      model_cfg,
        'train_cfg':      train_cfg,
    }
    torch.save(ckpt, os.path.join(checkpoint_dir, 'best_model.pth'))

    # 2. Full model object (easy inference — just torch.load and call model(x))
    torch.save(model, os.path.join(checkpoint_dir, 'best_model_full.pth'))

    # 3. Scaler
    with open(os.path.join(checkpoint_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # 4. Human-readable JSON summary
    summary = {
        'seed':         seed,
        'best_epoch':   epoch + 1,
        'best_val_loss':float(best_val_loss),
        'val_metrics':  {k: float(v) for k, v in (val_metrics or {}).items()},
        'model_cfg':    model_cfg,
        'train_cfg':    {k: str(v) for k, v in train_cfg.items()},
    }
    with open(os.path.join(checkpoint_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [Checkpoint] Saved to {checkpoint_dir}/")


# ── Single-seed training run ───────────────────────────────────────────────────
def run_single_seed(seed, data_cfg, train_cfg, model_cfg, device, experiment_dir):
    print(f"\n{SEP}")
    print(f"{'  Training  seed=' + str(seed):^80}")
    print(SEP)
    set_seed(seed)
    t_start = time.time()

    # ── Datasets ───────────────────────────────────────────────────────────
    scaler_path = os.path.join(experiment_dir, 'scaler.pkl')
    train_dataset = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='train',
        scaler=None,
        scaler_save_path=scaler_path,
        sar_root=data_cfg['data_paths'].get('sar_chips'),
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_dataset = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='val',
        scaler=train_dataset.scaler,
        sar_root=data_cfg['data_paths'].get('sar_chips'),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"  Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # ── Model, Loss, Optimizer, Scheduler ─────────────────────────────────
    model = FloodModel(config=model_cfg).to(device)
    criterion = MultiTaskLoss(
        focal_gamma=model_cfg['loss']['focal_gamma'],
        focal_alpha=model_cfg['loss'].get('focal_alpha', 0.25),
        regression_weight=model_cfg['loss']['regression_weight'],
    )
    lr = float(train_cfg['learning_rate'])
    wd = float(train_cfg['weight_decay'])
    optimizer = (optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                 if train_cfg['optimizer'] == 'adamw'
                 else optim.Adam(model.parameters(), lr=lr))
    epochs    = train_cfg['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    patience       = train_cfg.get('patience', 10)
    grad_clip_norm = float(train_cfg.get('grad_clip_norm', 1.0))

    seed_dir       = os.path.join(experiment_dir, f'seed_{seed}')
    checkpoint_dir = os.path.join(seed_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss  = float('inf')
    best_pr_auc    = 0.0
    patience_count = 0
    best_epoch     = 0

    print(f"\n  {'epoch':>7}  {'loss':>8}  {'cls':>8}  {'reg':>8}  {'val pr_auc':>10}  {'best':>8}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")

    for epoch in range(epochs):
        tr_loss, tr_cls, tr_reg = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip_norm
        )
        val_loss, val_pr_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Match screenshot format exactly:
        # epoch 13/60 | loss 0.0164 (cls 0.0056 reg 0.0537) | val pr_auc 0.5648 (best 0.6002)
        is_best = val_loss < best_val_loss
        if is_best:
            best_pr_auc = val_pr_auc
        print(
            f"  epoch {epoch+1:>2}/{epochs} | "
            f"loss {tr_loss:.4f} (cls {tr_cls:.4f} reg {tr_reg:.4f}) | "
            f"val pr_auc {val_pr_auc:.4f} (best {best_pr_auc:.4f})"
        )

        if is_best:
            best_val_loss  = val_loss
            best_pr_auc    = val_pr_auc
            best_epoch     = epoch
            patience_count = 0
            # Save complete checkpoint every time we improve
            save_complete_checkpoint(
                model, optimizer, scheduler, epoch,
                best_val_loss, {'pr_auc': val_pr_auc},
                model_cfg, train_cfg, train_dataset.scaler,
                checkpoint_dir, seed
            )
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  early stopping at epoch {epoch+1} (patience {patience})")
                break

    elapsed = time.time() - t_start

    # ── Final evaluation on best checkpoint ───────────────────────────────
    print(f"\n  Loading best checkpoint (epoch {best_epoch+1}) for final evaluation...")
    ckpt = torch.load(
        os.path.join(checkpoint_dir, 'best_model.pth'),
        map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    class_probs, class_targets = collect_val_predictions(model, val_loader, device)

    # Compute full metrics for flood_t+1 (primary target)
    y_true  = class_targets[:, 0]
    y_probs = class_probs[:, 0]
    pr, roc, brier, ece, pod, far, csi, opt_thr = compute_threshold_metrics(y_true, y_probs)

    print(f"\n  Elapsed: {elapsed:.1f}s | Optimal Threshold: {opt_thr:.4f}")
    print(f"  {'Model':<12} {'PR-AUC':>7} {'ROC-AUC':>8} {'Brier':>8} {'ECE':>7} {'POD':>6} {'FAR':>6} {'CSI':>6}")
    print(f"  {'FloodDNN':<12} {pr:>7.4f} {roc:>8.4f} {brier:>8.5f} {ece:>7.4f} {pod:>6.3f} {far:>6.3f} {csi:>6.3f}")

    # Update the checkpoint with full final metrics
    val_metrics = dict(pr_auc=pr, roc_auc=roc, brier=brier, ece=ece, pod=pod, far=far, csi=csi)
    save_complete_checkpoint(
        model, optimizer, scheduler, best_epoch,
        best_val_loss, val_metrics,
        model_cfg, train_cfg, train_dataset.scaler,
        checkpoint_dir, seed
    )

    return best_val_loss, val_metrics


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Flood Early-Warning Model")
    parser.add_argument('--experiment_dir', type=str,
                        default='experiments/wp2_baselines')
    parser.add_argument('--data_config', type=str,
                        default='configs/data.yaml')
    args = parser.parse_args()

    data_cfg  = load_config(args.data_config)
    train_cfg = load_config('configs/train.yaml')
    model_cfg = load_config('configs/model.yaml')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    seeds = train_cfg.get('seeds', [42])
    seed_results = {}   # seed → (best_val_loss, val_metrics)

    for seed in seeds:
        best_val_loss, val_metrics = run_single_seed(
            seed=seed,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            device=device,
            experiment_dir=args.experiment_dir,
        )
        seed_results[seed] = (best_val_loss, val_metrics)

    # ── Summary table (one row per seed, matching screenshot format) ───────
    rows = []
    for seed, (loss, metrics) in seed_results.items():
        rows.append(dict(
            stage=f'seed_{seed}',
            pr=metrics.get('pr_auc', 0),
            roc=metrics.get('roc_auc', 0),
            brier=metrics.get('brier', 0),
            ece=metrics.get('ece', 0),
            pod=metrics.get('pod', 0),
            far=metrics.get('far', 0),
            csi=metrics.get('csi', 0),
        ))

    # Aggregate row (mean across seeds)
    if len(rows) > 1:
        rows.append(dict(
            stage='mean (all seeds)',
            pr=np.mean([r['pr']    for r in rows]),
            roc=np.mean([r['roc']  for r in rows]),
            brier=np.mean([r['brier'] for r in rows]),
            ece=np.mean([r['ece']  for r in rows]),
            pod=np.mean([r['pod']  for r in rows]),
            far=np.mean([r['far']  for r in rows]),
            csi=np.mean([r['csi']  for r in rows]),
        ))

    print_summary_table(rows)

    # Point to the best seed checkpoint
    best_seed = min(seed_results, key=lambda s: seed_results[s][0])
    best_ckpt = os.path.join(
        args.experiment_dir, f'seed_{best_seed}', 'checkpoints', 'best_model.pth'
    )
    print(f"\n  Best seed: {best_seed}")
    print(f"  Checkpoint : {best_ckpt}")
    print(f"  Full model : {best_ckpt.replace('best_model.pth','best_model_full.pth')}")
    print(f"  Scaler     : {best_ckpt.replace('best_model.pth','scaler.pkl')}")


if __name__ == '__main__':
    main()
