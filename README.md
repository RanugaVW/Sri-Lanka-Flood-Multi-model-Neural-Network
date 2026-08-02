# 🌊 Terrain-Aware Multimodal Flood GNN

> A state-of-the-art Multi-modal Graph Neural Network (GNN) for Early Flood Warning in the Kelani River Basin, Sri Lanka.

This repository contains the architecture and training scaffolding for a spatio-temporal early warning system. It fuses historical tabular weather/discharge data, static terrain features, and SAR (Synthetic Aperture Radar) imagery into a unified graph representation to predict imminent flooding and water discharge up to 3 days in advance.

## 🏗️ Architecture Overview

The model is built with PyTorch and PyTorch Geometric and consists of four main modalities that are fused into a single node representation before message passing:

1. **Temporal Encoding**: A GRU processes 14-day historical windows of weather, soil, and river metrics.
2. **Terrain Modulation**: Static node features (elevation, flow accumulation) pass through an MLP to generate FiLM (Feature-wise Linear Modulation) parameters, which dynamically modulate the temporal outputs.
3. **SAR Imagery**: Sentinel-1 SAR chips (VV/VH) are processed via a custom ResNet-18 CNN. Missing imagery is handled elegantly via a learned presence-mask embedding.
4. **Spatial & Flow GNN**: Modalities are fused and passed through a 2-layer `GATv2Conv` Multi-Head Attention GNN across 51 localized nodes using both physical water-flow edges and spatial-proximity edges.

## 📂 Project Structure

```text
├── configs/                # YAML configuration files
│   ├── model.yaml          # Hyperparameters and layer dims
│   ├── train.yaml          # Learning rate, epochs, optimizer
│   └── kaggle_data.yaml    # Pre-configured paths for Kaggle GPU training
├── data/                   # Dataset directory (Ignored by Git)
│   ├── raw/
│   ├── processed/          # Contains flood_dataset.parquet, nodes.csv, edges.csv
│   └── sar_chips/          
├── src/
│   ├── baselines/          # Persistence, Climatology, GBM, and Node-LSTM ablations
│   ├── data/               # Sliding-window Dataset and Graph Builder
│   ├── losses/             # Focal BCE + Huber Multi-task loss
│   ├── models/             # PyTorch Neural Network Modules
│   └── train.py            # Main training loop
├── tests/                  # Standalone architecture validation
└── walkthrough.md          # Chronological log of development progress
```

## 🚀 Training on Kaggle (Recommended)

Due to the size of the dataset and the complexity of the GNN, training is best done on a GPU. This repository is pre-configured to run seamlessly on Kaggle.

1. Create a new Kaggle Notebook and set the Accelerator to **GPU T4x2**.
2. Mount your Tabular and Image Kaggle datasets via the Kaggle UI (Add Input).
3. Run the following in your first notebook cell to start training:

```bash
# Clone the repository
!git clone https://github.com/RanugaVW/Sri-Lanka-Flood-Multi-model-Neural-Network.git
%cd Sri-Lanka-Flood-Multi-model-Neural-Network

# Install requirements
!pip install -r requirements.txt

# Run the training loop using the Kaggle config
!python src/train.py \
    --data_config configs/kaggle_data.yaml \
    --experiment_dir /kaggle/working/experiments
```

*(Note: Don't forget to update the placeholder dataset names in `configs/kaggle_data.yaml` to match your actual Kaggle dataset mount paths).*

## 🔬 Baselines

The repository also implements standard forecasting baselines in `src/baselines/` to prove the efficacy of the GNN:
- **Persistence Model**: Predicts tomorrow's state will equal today's state.
- **Climatology Model**: Predicts the historical average probability of flooding per node.
- **GBM**: Gradient-Boosted Trees (LightGBM).
- **Node-LSTM**: A purely temporal ablation that uses the deep neural network components but skips the spatial GNN.

---
*Built for the CS3631 Benchmark.*
