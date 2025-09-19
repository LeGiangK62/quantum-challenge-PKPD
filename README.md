# PK/PD Modeling System

A comprehensive machine learning system for Pharmacokinetic/Pharmacodynamic (PK/PD) modeling with advanced neural network architectures and training strategies.

## 🚀 **Features**

### **Multiple Training Modes**
- **Separate**: Independent PK and PD model training
- **Joint**: Simultaneous PK/PD training with shared loss
- **Shared**: Shared encoder with separate heads
- **Dual Stage**: Two-stage training approach
- **Integrated**: Fully integrated PK/PD modeling
- **Two Stage Shared**: Two-stage with shared components

### **Advanced Encoders**
- **MLP**: Multi-layer perceptron
- **ResMLP**: Residual MLP with skip connections
- **MoE**: Mixture of Experts
- **ResMLP+MoE**: Combined residual and mixture of experts
- **Adaptive ResMLP+MoE**: Adaptive mixture of experts

### **Advanced Techniques**
- **Mixup Augmentation**: Data augmentation for improved generalization
- **Contrastive Learning**: Self-supervised learning approach
- **Feature Engineering**: Advanced data preprocessing and feature creation

## 📁 **Project Structure**

```
TeamPNU_exp/
├── main.py                    # Main training script
├── config.py                  # Configuration management
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── DIRECTORY_STRUCTURE.md     # Directory structure documentation
├── data/                      # Data processing modules
│   ├── loaders.py            # Data loading utilities
│   └── splits.py             # Data splitting strategies
├── models/                    # Model architectures
│   ├── encoders.py           # Encoder implementations
│   ├── heads.py              # Output head implementations
│   └── unified_model.py      # Unified model architecture
├── training/                  # Training modules
│   └── unified_trainer.py    # Unified training system
├── utils/                     # Utility functions
│   ├── helpers.py            # Helper functions
│   ├── logging.py            # Logging utilities
│   └── factory.py            # Model/trainer factory
└── results/                   # Output directory
    ├── runs/                 # Hierarchical experiment results
    ├── logs/                 # Training logs
    └── models/               # Model files (symlinks)
```

## 🛠️ **Installation**

1. **Clone the repository**
```bash
git clone <repository-url>
cd TeamPNU_exp
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 🚀 **Quick Start**

### **Basic Training**
```bash
# Simple training with default settings
python main.py

# Custom configuration
python main.py --mode separate --encoder resmlp_moe --epochs 100 --batch_size 32
```

### **Advanced Training**
```bash
# With feature engineering and mixup
python main.py --mode shared --encoder resmlp_moe --use_fe --use_mixup --epochs 150

# Different PK/PD encoders
python main.py --mode joint --encoder_pk mlp --encoder_pd resmlp --epochs 200

# With contrastive learning
python main.py --mode separate --encoder resmlp_moe --lambda_contrast 0.1 --epochs 100
```

### **Custom Run Name**
```bash
# Specify custom run name
python main.py --run_name my_experiment --mode separate --encoder mlp
```

## ⚙️ **Configuration**

### **Training Modes**
- `separate`: Train PK and PD models independently
- `joint`: Train PK/PD models jointly with shared loss
- `shared`: Use shared encoder with separate heads
- `dual_stage`: Two-stage training approach
- `integrated`: Fully integrated PK/PD modeling
- `two_stage_shared`: Two-stage with shared components

### **Encoders**
- `mlp`: Standard multi-layer perceptron
- `resmlp`: Residual MLP with skip connections
- `moe`: Mixture of Experts
- `resmlp_moe`: Combined ResMLP and MoE
- `adaptive_resmlp_moe`: Adaptive mixture of experts

### **Key Parameters**
```bash
--mode {separate,joint,shared,dual_stage,integrated,two_stage_shared}
--encoder {mlp,resmlp,moe,resmlp_moe,adaptive_resmlp_moe}
--encoder_pk ENCODER_PK    # PK-specific encoder
--encoder_pd ENCODER_PD    # PD-specific encoder
--epochs EPOCHS            # Number of training epochs
--batch_size BATCH_SIZE    # Batch size
--lr LEARNING_RATE         # Learning rate
--hidden HIDDEN            # Hidden dimension
--depth DEPTH              # Network depth
--dropout DROPOUT          # Dropout rate
```

### **Advanced Features**
```bash
--use_fe                   # Enable feature engineering
--use_mixup                # Enable mixup augmentation
--lambda_contrast LAMBDA   # Contrastive learning weight
--temperature TEMP         # Temperature for contrastive learning
```

## 📊 **Output Structure**

The system creates a hierarchical directory structure for organized experiment management:

```
results/
├── runs/
│   └── {run_name}/
│       └── {mode}/
│           └── {encoder} or {encoder_pk-encoder_pd}/
│               └── s{seed}/
│                   ├── model.pth          # Trained model
│                   ├── config.json        # Configuration
│                   ├── scalers.pkl        # Data scalers
│                   └── results.json       # Training results
├── logs/
│   └── {run_name}/
│       └── {mode}/
│           └── {encoder} or {encoder_pk-encoder_pd}/
│               └── s{seed}/
│                   └── {run_name}_{timestamp}.log
└── models/                    # Backward compatibility symlinks
    └── {mode}/
        └── {encoder} or {encoder_pk-encoder_pd}/
            └── s{seed}/
                └── {run_name}.pth -> ../../runs/{run_name}/{mode}/{encoder}/s{seed}/model.pth
```

## 📈 **Example Results**

### **Training Output**
```
=== PK/PD Modeling ===
Run name: separate_resmlp_moe_s42_250919_0920_fe_mixup
Mode: separate | Encoder: resmlp_moe | Epochs: 100
Batch size: 32 | Learning rate: 0.001
Device: cuda:0

=== Training Results ===
Best validation loss: 0.897320
Best PK RMSE: 0.748186
Best PD RMSE: 1.075058
Training time: 46.90 seconds

=== Output Files ===
Model saved: results/runs/separate_resmlp_moe_s42_250919_0920_fe_mixup/separate/resmlp_moe/s42/model.pth
Configuration saved: results/runs/separate_resmlp_moe_s42_250919_0920_fe_mixup/separate/resmlp_moe/s42/config.json
Results saved: results/runs/separate_resmlp_moe_s42_250919_0920_fe_mixup/separate/resmlp_moe/s42/results.json
```
