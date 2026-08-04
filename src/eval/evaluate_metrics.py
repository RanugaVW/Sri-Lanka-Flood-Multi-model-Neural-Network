import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error, average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix

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
            # Apply sigmoid to get probabilities
            class_probs = torch.sigmoid(predictions[:, :4]).cpu().numpy()
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
    
    # Calculate Metrics
    class_names = ['P(flood t+1)', 'P(flood t+2)', 'P(flood t+3)', 'onset']
    reg_names = ['discharge_t1', 'zscore_3d_max']
    
    results = "# Model Evaluation Results\n\n"
    
    results += "## Classification Metrics (Optimal Threshold)\n\n"
    results += "| Target | PR-AUC | ROC-AUC | Brier | ECE | POD | FAR | CSI |\n"
    results += "|---|---|---|---|---|---|---|---|\n"
    
    for i, name in enumerate(class_names):
        y_true = all_class_targets[:, i]
        y_probs = all_class_preds[:, i]
        
        try:
            pr_auc = average_precision_score(y_true, y_probs)
            roc_auc = roc_auc_score(y_true, y_probs)
        except ValueError:
            pr_auc = float('nan')
            roc_auc = float('nan')
            
        brier = brier_score_loss(y_true, y_probs)
        
        # ECE calculation
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for j in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[j], bin_boundaries[j+1]
            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper) if j < n_bins - 1 else (y_probs >= bin_lower) & (y_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                acc_in_bin = y_true[in_bin].mean()
                conf_in_bin = y_probs[in_bin].mean()
                ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin

        best_csi = -1
        best_pod = 0
        best_far = 0
        
        for thresh in np.arange(0.01, 1.0, 0.01):
            y_pred = (y_probs >= thresh).astype(int)
            # handle cases with only one class gracefully
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
                best_csi = csi
                best_pod = pod
                best_far = far
                
        results += f"| {name} | {pr_auc:.4f} | {roc_auc:.4f} | {brier:.5f} | {ece:.4f} | {best_pod:.3f} | {best_far:.3f} | {best_csi:.3f} |\n"
        
    results += "\n## Regression Metrics\n\n"
    results += "| Target | R2 Score | MAE | RMSE |\n"
    results += "|---|---|---|---|\n"
    
    for i, name in enumerate(reg_names):
        y_true = all_reg_targets[:, i]
        y_pred = all_reg_preds[:, i]
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        results += f"| {name} | {r2:.4f} | {mae:.4f} | {rmse:.4f} |\n"
        
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(results)
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Flood Early-Warning Model")
    parser.add_argument('--experiment_dir', type=str, default='experiments/wp2_baselines', help='Experiment directory')
    parser.add_argument('--data_config', type=str, default='configs/data.yaml', help='Path to data config file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Configs
    data_cfg = load_config(args.data_config)
    model_cfg = load_config('configs/model.yaml')
    
    # Load val dataset
    val_dataset = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = FloodModel(config=model_cfg).to(device)
    checkpoint_path = os.path.join(args.experiment_dir, 'checkpoints', 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    # weights_only=True isn't supported in all older torch versions, so let's omit if not needed
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    evaluate_model(model, val_loader, device, output_file='Docs/evaluation_results.md')
    print("Evaluation complete. Results written to Docs/evaluation_results.md")

if __name__ == '__main__':
    main()
