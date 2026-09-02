"""evaluate_metrics.py — Paper-format evaluation for the Sri Lanka Flood Model.

Matches the metric set and table format from the companion MMF-Net paper:
  PR-AUC | ev.det | FAR | ECE | Params

ev.det (Event Detection Rate)
------------------------------
An episode counts as "detected" if the model's calibrated probability exceeded
the threshold on at least one day in the 7-day window before the flood onset.
This is the operationally meaningful metric — a model that only fires during
an ongoing flood gives no actionable warning.

ECE (Expected Calibration Error)
---------------------------------
Computed with 15 equal-width bins.  After temperature scaling, the model's
stated probabilities should match empirical frequencies.

Usage (standalone)
------------------
    python src/eval/evaluate_metrics.py \\
        --experiment_dir experiments/overhaul \\
        --data_config    configs/data.yaml

Or call compute_paper_metrics() directly from train.py.
"""

import os
import json
import pickle
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, r2_score,
    mean_absolute_error, mean_squared_error,
)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataset    import FloodDataset
from models.flood_model import FloodModel


# ─────────────────────────────────────────────── metric primitives ─────────────

def _ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx   = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece   = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(ece)


def _contingency(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tp   = int(((yhat == 1) & (y == 1)).sum())
    fp   = int(((yhat == 1) & (y == 0)).sum())
    fn   = int(((yhat == 0) & (y == 1)).sum())
    tn   = int(((yhat == 0) & (y == 0)).sum())
    pod  = tp / max(tp + fn, 1)
    far  = fp / max(tp + fp, 1)
    csi  = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1   = 2 * prec * pod / max(prec + pod, 1e-12)
    return dict(pod=pod, far=far, csi=csi, precision=prec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


def _event_detection(p: np.ndarray, event: np.ndarray, day: np.ndarray,
                     node: np.ndarray, threshold: float,
                     max_lead: int = 7) -> dict:
    """Episode-level detection rate and mean lead time.

    Parameters
    ----------
    p         : calibrated probability per valid node-day
    event     : event code ≥0 for flood episode days, -1 otherwise
    day       : day index per valid node-day
    node      : node index per valid node-day
    threshold : decision threshold (from validation)
    max_lead  : how many days before onset to look for an alarm

    Returns
    -------
    dict with 'event_detection_rate', 'mean_lead_days', 'n_events'
    """
    # Build per-node alarm day sets
    alarms: dict = {}
    for nd in np.unique(node):
        sel = node == nd
        alarms[int(nd)] = np.sort(day[sel][p[sel] >= threshold])

    detected, leads = 0, []
    ev_ids = np.unique(event[event >= 0])

    for e in ev_ids:
        sel   = event == e
        onset = int(day[sel].min())
        nd    = int(node[sel][0])
        a     = alarms.get(nd, np.empty(0, dtype=int))
        w     = a[(a >= onset - max_lead) & (a <= onset - 1)]
        if w.size:
            detected += 1
            leads.append(onset - int(w.min()))

    n = len(ev_ids)
    return {
        'n_events':             n,
        'event_detection_rate': detected / max(n, 1),
        'mean_lead_days':       float(np.mean(leads)) if leads else float('nan'),
    }


# ─────────────────────────────────────────────── paper-format table ────────────

CLS_NAMES = ['flood_t+1', 'flood_t+2', 'flood_t+3', 'onset']
REG_NAMES = ['discharge_t1', 'zscore_3d_max']

SEP  = '=' * 88
DASH = '-' * 88


def compute_paper_metrics(
    cls_probs:    np.ndarray,   # [M, 4]  calibrated probabilities
    cls_targets:  np.ndarray,   # [M, 4]
    reg_preds:    np.ndarray,   # [M, 2]
    reg_targets:  np.ndarray,   # [M, 2]
    event:        np.ndarray,   # [M]
    day:          np.ndarray,   # [M]
    node:         np.ndarray,   # [M]
    threshold:    float,
    n_params:     int   = 0,
    n_seeds:      int   = 1,
    output_file:  str   = None,
) -> str:
    """Compute and format the paper metrics table.

    Primary classification target: flood_t+1 (index 0).
    ev.det is computed only for the primary target.
    """
    lines = []
    lines.append(f"\n{SEP}")
    lines.append(f"{'FLOOD DNN  EVALUATION SUMMARY':^88}")
    lines.append(SEP)
    lines.append(
        f"  Seeds: {n_seeds}  |  Params: {n_params:,}  |  Threshold (val): {threshold:.4f}")
    lines.append(DASH)

    # ── Classification table ────────────────────────────────────────────────
    hdr = (f"{'Target':<18} {'PR-AUC':>7} {'ROC-AUC':>8} {'ev.det':>7} "
           f"{'FAR':>6} {'ECE':>7} {'POD':>6} {'CSI':>6} {'F1':>6}")
    lines.append(hdr)
    lines.append(DASH)

    for i, name in enumerate(CLS_NAMES):
        y = cls_targets[:, i].astype(float)
        p = cls_probs[:, i]

        try:
            pr_auc  = average_precision_score(y, p)
            roc_auc = roc_auc_score(y, p)
        except ValueError:
            pr_auc = roc_auc = float('nan')

        ece  = _ece(y, p)
        cont = _contingency(y, p, threshold)

        # ev.det only for primary target (all others are aux)
        ev_det = float('nan')
        if i == 0:
            ev_info = _event_detection(p, event, day, node, threshold)
            ev_det  = ev_info['event_detection_rate']
            ml_days = ev_info['mean_lead_days']
            n_ev    = ev_info['n_events']

        lines.append(
            f"{name:<18} {pr_auc:>7.4f} {roc_auc:>8.4f} {ev_det:>7.3f} "
            f"{cont['far']:>6.3f} {ece:>7.4f} {cont['pod']:>6.3f} "
            f"{cont['csi']:>6.3f} {cont['f1']:>6.3f}"
        )

    lines.append(SEP)

    # ── Episode summary (primary target) ─────────────────────────────────────
    lines.append(f"\nEpisode detection (flood_t+1, 7-day lead window):")
    lines.append(f"  Episodes in test set : {n_ev}")
    lines.append(f"  Detected (ev.det)    : {ev_det:.3f}")
    lines.append(f"  Mean lead days       : {ml_days:.1f}")
    lines.append(DASH)

    # ── Regression sub-table ─────────────────────────────────────────────────
    lines.append(f"\n{'Regression Target':<20} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    lines.append('-' * 46)
    for i, name in enumerate(REG_NAMES):
        yt = reg_targets[:, i]
        yp = reg_preds[:, i]
        r2   = r2_score(yt, yp)
        mae  = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        lines.append(f"{name:<20} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f}")
    lines.append(SEP)

    # ── Metric legend (matches friend's paper table caption) ─────────────────
    lines.append("")
    lines.append("Metric definitions:")
    lines.append("  PR-AUC  — Precision-Recall Area Under Curve (headline metric, imbalanced baseline).")
    lines.append("  ev.det  — Event detection rate: fraction of flood episodes with ≥1 alarm in the")
    lines.append("            7-day pre-onset window.  Operationally meaningful; not inflated by")
    lines.append("            within-episode days (discharge autocorrelation makes those nearly free).")
    lines.append("  FAR     — False Alarm Ratio: FP/(TP+FP) at the validation-chosen threshold.")
    lines.append("  ECE     — Expected Calibration Error (15 bins); lower = better-calibrated probs.")
    lines.append(SEP)

    result = "\n".join(lines)
    print(result)

    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(result)

    return result


# ─────────────────────────────────────────────── standalone evaluate ────────────

def evaluate_model(model, val_loader, device, output_file=None):
    """Legacy entry point used by scripts that call evaluate_metrics directly."""
    all_cls_logits, all_cls_targets = [], []
    all_reg_preds,  all_reg_targets  = [], []
    all_events, all_days, all_nodes  = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            temporal  = batch['temporal_features'][0].to(device)
            terrain   = batch['terrain_features'][0].to(device)
            basin_idx = batch['basin_idx'][0].to(device)
            sar       = batch['sar_chips'][0].to(device)
            has_sar   = batch['has_sar'][0].to(device)
            targets   = batch['targets'][0].to(device)
            mask      = batch['valid_mask'][0].cpu().numpy() > 0
            ei_flow   = batch['edge_index_flow'][0].to(device)
            ei_sp     = batch['edge_index_spatial'][0].to(device)
            ew_sp     = batch['edge_weight_spatial'][0].to(device)

            out = model(temporal, terrain, basin_idx, sar, has_sar, ei_flow, ei_sp, ew_sp)

            lg  = out['logits'].cpu().numpy()
            reg = out['reg'].cpu().numpy()
            tgt = targets.cpu().numpy()
            ev  = batch['event_ids'][0].numpy()
            day = int(batch['day_idx'][0].numpy())

            all_cls_logits.append(lg[mask])
            all_cls_targets.append(tgt[mask, :4].astype(int))
            all_reg_preds.append(reg[mask])
            all_reg_targets.append(tgt[mask, 4:])
            all_events.append(ev[mask])
            all_days.append(np.full(mask.sum(), day))
            all_nodes.append(np.where(mask)[0])

    cls_logits  = np.concatenate(all_cls_logits)
    cls_targets = np.concatenate(all_cls_targets)
    reg_preds   = np.concatenate(all_reg_preds)
    reg_targets = np.concatenate(all_reg_targets)

    # Default: no calibration, threshold=0.5
    cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))

    return compute_paper_metrics(
        cls_probs=cls_probs,
        cls_targets=cls_targets,
        reg_preds=reg_preds,
        reg_targets=reg_targets,
        event=np.concatenate(all_events),
        day=np.concatenate(all_days),
        node=np.concatenate(all_nodes),
        threshold=0.5,
        output_file=output_file,
    )


# ─────────────────────────────────────────────── CLI ──────────────────────────

def load_config(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Flood Model — paper metrics')
    parser.add_argument('--experiment_dir', default='experiments/overhaul')
    parser.add_argument('--data_config',    default='configs/data.yaml')
    parser.add_argument('--seed',           type=int, default=42)
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_cfg = load_config(args.data_config)
    model_cfg = load_config('configs/model.yaml')

    # Load scaler
    scaler_path = os.path.join(args.experiment_dir, 'scaler.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    # Load temperature
    T = 1.0
    temp_path = os.path.join(args.experiment_dir, 'temperature.json')
    if os.path.exists(temp_path):
        with open(temp_path) as f:
            T = json.load(f)['temperature']

    # Load threshold
    thr = 0.5
    thr_path = os.path.join(args.experiment_dir, 'threshold.json')
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            thr = json.load(f)['threshold']

    val_ds = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='val',
        scaler=scaler,
        sar_root=data_cfg['data_paths'].get('sar_chips'),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Load model
    model = FloodModel(config=model_cfg).to(device)
    ckpt_path = os.path.join(args.experiment_dir, f'seed_{args.seed}',
                             'checkpoints', 'best_model.pth')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    out_path = os.path.join(args.experiment_dir, 'evaluation_results.md')
    evaluate_model(model, val_loader, device, output_file=out_path)
    print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
