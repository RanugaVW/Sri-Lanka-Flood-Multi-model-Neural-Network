import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import FloodDataset
from models.flood_model import FloodModel
from losses.multitask_loss import MultiTaskLoss
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Unpack the dictionary
        temporal_features = batch['temporal_features'].to(device)
        terrain_features = batch['terrain_features'][0].to(device) # Batched incorrectly by default, just take the first
        sar_chips = batch['sar_chips'][0].to(device)
        has_sar = batch['has_sar'][0].to(device)
        targets = batch['targets'][0].to(device)
        
        edge_index_flow = batch['edge_index_flow'][0].to(device)
        edge_index_spatial = batch['edge_index_spatial'][0].to(device)
        edge_weight_spatial = batch['edge_weight_spatial'][0].to(device)
        
        # Forward pass
        predictions = model(
            temporal_features[0], # removing the batch dimension since one item is one timestep for all nodes
            terrain_features, 
            sar_chips, 
            has_sar, 
            edge_index_flow, 
            edge_index_spatial, 
            edge_weight_spatial
        )
        
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="Train Flood Early-Warning Model")
    parser.add_argument('--experiment_dir', type=str, default='experiments/wp2_baselines', help='Experiment directory')
    parser.add_argument('--data_config', type=str, default='configs/data.yaml', help='Path to data config file')
    args = parser.parse_args()

    # Load Configs
    data_cfg = load_config(args.data_config)
    train_cfg = load_config('configs/train.yaml')
    model_cfg = load_config('configs/model.yaml')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset & DataLoaders
    print("Loading datasets...")
    train_dataset = FloodDataset(
        panel_path=data_cfg['data_paths']['panel'],
        nodes_path=data_cfg['data_paths']['nodes'],
        edges_path=data_cfg['data_paths']['edges_flow'],
        split_type='train'
    )
    
    # We use batch_size 1 because each 'item' is an entire graph for all 51 nodes at a single timestep
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    print(f"Train Dataset size: {len(train_dataset)}")

    # 2. Model setup
    model = FloodModel(config=model_cfg).to(device)
    print("Model initialized.")

    # 3. Loss & Optimizer
    criterion = MultiTaskLoss(
        focal_gamma=model_cfg['loss']['focal_gamma'],
        regression_weight=model_cfg['loss']['regression_weight']
    )
    
    if train_cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=float(train_cfg['learning_rate']), weight_decay=float(train_cfg['weight_decay']))
    else:
        optimizer = optim.Adam(model.parameters(), lr=float(train_cfg['learning_rate']))

    # 4. Training Loop
    # We will do a small test run of 1 epoch for now
    epochs = 1 # override config for quick test
    checkpoint_dir = os.path.join(args.experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
if __name__ == '__main__':
    main()
