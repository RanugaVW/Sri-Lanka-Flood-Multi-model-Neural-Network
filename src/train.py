"""
train.py — Main training entry point for the Sri Lanka Flood Early-Warning Model.

Key design decisions (aligned with the companion MMF-Net ablation):
  * Early stopping on val PR-AUC (flood_t+1) — not on total loss.
    At a 2% positive rate, total loss can decrease while PR-AUC degrades.
  * Temperature scaling fitted on validation logits, applied to test.
  * Threshold chosen on validation, applied to test (no test-set leakage).
  * Multi-seed deep ensemble: probabilities averaged before calibration.
  * All intermediate results written to Drive-safe experiment_dir.

Output files per seed
---------------------
  <experiment_dir>/seed_<k>/checkpoints/best_model.pth   — weights only
  <experiment_dir>/scaler.pkl
After all seeds:
  <experiment_dir>/temperature.json   — calibrated temperature T
  <experiment_dir>/threshold.json     — optimal val threshold
  <experiment_dir>/evaluation_results.md   — paper-format table
"""

import os, sys, json, time, pickle, random, argparse

import numpy as np
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(__file__))
from data.dataset          import FloodDataset
from models.flood_model    import FloodModel
from losses.multitask_loss import MultiTaskLoss
from eval.evaluate_metrics import evaluate_model, compute_paper_metrics

# ── Config helpers ─────────────────────────────────────────────────────────────

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
    torch.backends.cudnn.benchmark     = False


# ── Batch unpacking ────────────────────────────────────────────────────────────

def _unpack(batch, device):
    """Unpack one DataLoader batch onto device.  Returns (inputs_dict, targets, mask, conf)."""
    return (
        {
            'temporal':  batch['temporal_features'][0].to(device),
            'terrain':   batch['terrain_features'][0].to(device),
            'sar':       batch['sar_chips'][0].to(device),
            'has_sar':   batch['has_sar'][0].to(device),
            'ei_flow':   batch['edge_index_flow'][0].to(device),
            'ei_sp':     batch['edge_index_spatial'][0].to(device),
            'ew_sp':     batch['edge_weight_spatial'][0].to(device),
        },
        batch['targets'][0].to(device),
        batch['valid_mask'][0].to(device),
        batch['label_conf'][0].to(device),
    )


def _forward(model, inp):
    return model(
        inp['temporal'], inp['terrain'],
        inp['sar'],      inp['has_sar'],
        inp['ei_flow'],  inp['ei_sp'], inp['ew_sp'],
    )


# ── One epoch ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    tot = tot_cls = tot_reg = 0.0
    for batch in loader:
        inp, targets, mask, conf = _unpack(batch, device)
        optimizer.zero_grad()
        out = _forward(model, inp)
        lc  = criterion(out, targets, mask, conf)
        lc.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        tot     += lc.total.item()
        tot_cls += lc.cls.item()
        tot_reg += lc.reg.item()
    n = max(len(loader), 1)
    return tot / n, tot_cls / n, tot_reg / n


# ── Validation — collect logits + PR-AUC ──────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels, all_events, all_days, all_nodes = [], [], [], [], []
    tot = 0.0
    node_count = None

    for batch in loader:
        inp, targets, mask, conf = _unpack(batch, device)
        out = _forward(model, inp)
        lc  = criterion(out, targets, mask, conf)
        tot += lc.total.item()

        # Primary head (flood_t+1 = index 0)
        m   = mask.cpu().numpy() > 0
        lg  = out['logits'][:, 0].cpu().numpy()
        y   = targets[:, 0].int().cpu().numpy()
        ev  = batch['event_ids'][0].numpy()
        day = int(batch['day_idx'][0].numpy())
        N   = len(m)
        node_count = N

        all_logits.append(lg[m])
        all_labels.append(y[m])
        all_events.append(ev[m])
        all_days.append(np.full(m.sum(), day))
        all_nodes.append(np.where(m)[0])

    n = max(len(loader), 1)
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs  = 1.0 / (1.0 + np.exp(-logits))

    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = float('nan')

    collected = {
        'logit':  logits,
        'y':      labels,
        'event':  np.concatenate(all_events),
        'day':    np.concatenate(all_days),
        'node':   np.concatenate(all_nodes),
    }
    return tot / n, pr_auc, collected


# ── Temperature calibration ────────────────────────────────────────────────────

def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> float:
    """Scalar temperature T minimising val NLL of sigmoid(logit / T)."""
    lg  = torch.as_tensor(logits.astype(np.float32))
    tgt = torch.as_tensor(y.astype(np.float32))
    log_t = torch.zeros(1, requires_grad=True)
    opt   = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            lg / log_t.exp(), tgt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits / max(T, 1e-3)))


# ── Threshold search ───────────────────────────────────────────────────────────

def best_threshold(y: np.ndarray, probs: np.ndarray, n: int = 200) -> float:
    """Choose threshold maximising F1 on the validation set only."""
    grid = np.quantile(probs, np.linspace(0.5, 0.9999, n))
    best_thr, best_f1 = 0.5, -1.0
    for t in np.unique(grid):
        yhat = (probs >= t).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr


# ── Test-set logit collection ──────────────────────────────────────────────────

@torch.no_grad()
def collect_all_heads(model, loader, device):
    """Collect logits and targets for all 4 cls heads + 2 reg heads on test/val."""
    model.eval()
    cls_logits, cls_targets = [], []
    reg_preds,  reg_targets = [], []
    events, days, nodes     = [], [], []
    masks                   = []

    for batch in loader:
        inp, targets, mask, _ = _unpack(batch, device)
        out = _forward(model, inp)
        m   = mask.cpu().numpy() > 0

        cls_logits.append(out['logits'].cpu().numpy()[m])
        cls_targets.append(targets[:, :4].int().cpu().numpy()[m])
        reg_preds.append(out['reg'].cpu().numpy()[m])
        reg_targets.append(targets[:, 4:].cpu().numpy()[m])

        ev  = batch['event_ids'][0].numpy()
        day = int(batch['day_idx'][0].numpy())
        events.append(ev[m])
        days.append(np.full(m.sum(), day))
        nodes.append(np.where(m)[0])

    return {
        'cls_logits':  np.concatenate(cls_logits),
        'cls_targets': np.concatenate(cls_targets),
        'reg_preds':   np.concatenate(reg_preds),
        'reg_targets': np.concatenate(reg_targets),
        'event':       np.concatenate(events),
        'day':         np.concatenate(days),
        'node':        np.concatenate(nodes),
    }


# ── Checkpoint save ────────────────────────────────────────────────────────────

def save_checkpoint(model, checkpoint_dir, seed):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(checkpoint_dir, 'best_model.pth'))


# ── Single-seed training ───────────────────────────────────────────────────────

def run_single_seed(seed, data_cfg, train_cfg, model_cfg, device, experiment_dir, global_start_time=None, max_time_secs=None):
    SEP  = '=' * 80
    print(f"\n{SEP}")
    print(f"{'  Training  seed=' + str(seed):^80}")
    print(SEP)
    set_seed(seed)

    scaler_path = os.path.join(experiment_dir, 'scaler.pkl')

    # ── Datasets ────────────────────────────────────────────────────────────
    train_ds = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='train',
        scaler=None,
        scaler_save_path=scaler_path,
        sar_root=data_cfg['data_paths'].get('sar_chips'),
    )
    val_ds = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='val',
        scaler=train_ds.scaler,
        sar_root=data_cfg['data_paths'].get('sar_chips'),
    )
    test_ds = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='test',
        scaler=train_ds.scaler,
        sar_root=data_cfg['data_paths'].get('sar_chips'),
    )

    # DataLoader batch_size=1: each "batch" is one full-graph snapshot
    kw = dict(batch_size=1, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"  Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)} snapshots")

    # ── Model & optimiser ───────────────────────────────────────────────────
    model = FloodModel(config=model_cfg).to(device)
    print(f"  Parameters: {model.n_params():,}")

    loss_cfg  = model_cfg.get('loss', {})
    criterion = MultiTaskLoss(
        loss=train_cfg.get('loss', 'bce'),
        focal_alpha=loss_cfg.get('focal_alpha', 0.75),
        focal_gamma=loss_cfg.get('focal_gamma', 2.0),
        regression_weight=loss_cfg.get('regression_weight', 0.2),
    )

    lr       = float(train_cfg['learning_rate'])
    wd       = float(train_cfg.get('weight_decay', 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs   = train_cfg['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    patience   = train_cfg.get('patience', 10)
    grad_clip  = float(train_cfg.get('grad_clip_norm', 1.0))

    seed_dir   = os.path.join(experiment_dir, f'seed_{seed}')
    ckpt_dir   = os.path.join(seed_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training loop ───────────────────────────────────────────────────────
    best_pr_auc    = -1.0
    patience_count = 0
    best_epoch     = 0
    t_start        = time.time()
    start_epoch    = 0

    checkpoint_path = os.path.join(ckpt_dir, 'last_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"  [resuming] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        best_pr_auc    = ckpt['best_pr_auc']
        patience_count = ckpt['patience_count']
        best_epoch     = ckpt['best_epoch']
        start_epoch    = ckpt['epoch'] + 1

    print(f"\n  {'epoch':>7}  {'loss':>8}  {'cls':>8}  {'reg':>8}  {'val_pr_auc':>10}  {'best':>8}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")

    for epoch in range(start_epoch, epochs):
        tr_loss, tr_cls, tr_reg = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip)
        val_loss, val_pr_auc, _ = validate(
            model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_pr_auc > best_pr_auc
        print(
            f"  epoch {epoch+1:>2}/{epochs} | "
            f"loss {tr_loss:.4f} (cls {tr_cls:.4f} reg {tr_reg:.4f}) | "
            f"val pr_auc {val_pr_auc:.4f} (best {max(best_pr_auc, val_pr_auc):.4f})"
        )

        if is_best:
            best_pr_auc    = val_pr_auc
            best_epoch     = epoch
            patience_count = 0
            save_checkpoint(model, ckpt_dir, seed)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stop at epoch {epoch+1} (best PR-AUC {best_pr_auc:.4f})")
                break

        # Save state for resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_pr_auc': best_pr_auc,
            'patience_count': patience_count,
            'best_epoch': best_epoch,
        }, checkpoint_path)
        
        # Check time limit
        if global_start_time is not None and max_time_secs is not None:
            if time.time() - global_start_time > max_time_secs:
                print(f"  [time limit] Max time {max_time_secs}s reached. Stopping seed {seed} training early.")
                break

    elapsed = time.time() - t_start
    print(f"\n  Elapsed: {elapsed:.1f}s  |  Best epoch: {best_epoch+1}")

    # ── Load best checkpoint → collect val + test logits ────────────────────
    print(f"  [eval] Loading best model (epoch {best_epoch+1}, PR-AUC {best_pr_auc:.4f}) for inference...")
    model.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'best_model.pth'),
        map_location=device, weights_only=True))
    model.eval()

    print(f"  [eval] Running val inference...")
    _, _, val_collected  = validate(model, val_loader,  criterion, device)
    print(f"  [eval] Running test inference...")
    test_all = collect_all_heads(model, test_loader, device)
    print(f"  [eval] Inference complete.")

    return val_collected, test_all, best_pr_auc, model.n_params()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train Flood Early-Warning Model')
    parser.add_argument('--experiment_dir', default='experiments/overhaul')
    parser.add_argument('--data_config',    default='configs/data.yaml')
    args = parser.parse_args()

    data_cfg  = load_config(args.data_config)
    train_cfg = load_config('configs/train.yaml')
    model_cfg = load_config('configs/model.yaml')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.experiment_dir, exist_ok=True)
    seeds = train_cfg.get('seeds', [42])

    global_start_time = time.time()
    # 10.0h training budget → leaves ~2h for val/test inference + evaluation
    # before Kaggle's 12h hard kill. Each epoch ≈ 2000s, so this fits ~18 epochs
    # before the budget fires. The best checkpoint is always saved per epoch.
    max_time_secs = 10.0 * 3600

    # ── Train all seeds ──────────────────────────────────────────────────────
    all_val_logits  = []
    all_test_data   = []
    val_ref = test_ref = None
    n_params = 0

    for seed in seeds:
        val_col, test_all, _, np_ = run_single_seed(
            seed, data_cfg, train_cfg, model_cfg, device, args.experiment_dir, 
            global_start_time, max_time_secs)
        all_val_logits.append(val_col['logit'])
        all_test_data.append(test_all)
        val_ref  = val_col
        test_ref = test_all
        n_params = np_

        if time.time() - global_start_time > max_time_secs:
            print(f"  [time limit] Stopping seeds loop. Proceeding to evaluation...")
            break

    # ── Ensemble: average probabilities across seeds ─────────────────────────
    def ens_prob(logit_list):
        return np.mean([1.0 / (1.0 + np.exp(-l)) for l in logit_list], axis=0)

    val_prob_raw = ens_prob(all_val_logits)

    # For test: stack per-head logits and average
    n_cls = all_test_data[0]['cls_logits'].shape[1]
    test_cls_prob = np.mean(
        [1.0 / (1.0 + np.exp(-d['cls_logits'])) for d in all_test_data], axis=0)
    test_reg_pred = np.mean([d['reg_preds'] for d in all_test_data], axis=0)

    # ── Temperature calibration on validation ────────────────────────────────
    print("\n[calibration] Fitting temperature scaling on validation set...")
    val_logits_ens = np.log(val_prob_raw.clip(1e-7, 1 - 1e-7) /
                            (1 - val_prob_raw.clip(1e-7, 1 - 1e-7)))
    T = fit_temperature(val_logits_ens, val_ref['y'])
    print(f"  Temperature T = {T:.4f}")

    # Calibrated probabilities
    val_prob_cal  = apply_temperature(val_logits_ens, T)
    test_cls_logits_ens = np.log(test_cls_prob.clip(1e-7, 1-1e-7) /
                                  (1-test_cls_prob.clip(1e-7, 1-1e-7)))
    test_cls_prob_cal = 1.0 / (1.0 + np.exp(-test_cls_logits_ens / max(T, 1e-3)))

    # ── Threshold from validation ────────────────────────────────────────────
    thr = best_threshold(val_ref['y'], val_prob_cal)
    print(f"[threshold]   Optimal F1 threshold (val) = {thr:.4f}")

    # ── Save calibration artefacts ───────────────────────────────────────────
    with open(os.path.join(args.experiment_dir, 'temperature.json'), 'w') as f:
        json.dump({'temperature': T}, f, indent=2)
    with open(os.path.join(args.experiment_dir, 'threshold.json'), 'w') as f:
        json.dump({'threshold': thr, 'n_seeds': len(seeds)}, f, indent=2)

    # ── Final evaluation on test set ─────────────────────────────────────────
    out_path = os.path.join(args.experiment_dir, 'evaluation_results.md')
    compute_paper_metrics(
        cls_probs=test_cls_prob_cal,
        cls_targets=test_ref['cls_targets'],
        reg_preds=test_reg_pred,
        reg_targets=test_ref['reg_targets'],
        event=test_ref['event'],
        day=test_ref['day'],
        node=test_ref['node'],
        threshold=thr,
        n_params=n_params,
        n_seeds=len(seeds),
        output_file=out_path,
    )
    print(f"\n[done] Results written to {out_path}")


if __name__ == '__main__':
    main()
