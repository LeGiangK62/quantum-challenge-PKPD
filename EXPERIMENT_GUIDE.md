# Experiment Guide

This guide explains how to run comprehensive experiments across different modes and encoders.

## 🚀 **Quick Start**

### **Quick Test (5 minutes)**
```bash
# Run quick tests to verify everything works
./quick_test.sh
```

### **Basic Experiments (30 minutes)**
```bash
# Run basic mode-encoder combinations
./run_experiments.sh basic
```

### **All Experiments (2-3 hours)**
```bash
# Run all experiment types
./run_experiments.sh all
```

## 📋 **Available Experiment Types**

### **1. Basic Experiments (`basic`)**
- **Purpose**: Test all mode-encoder combinations
- **Duration**: ~30 minutes
- **Experiments**: 36 combinations
  - 6 modes × 6 encoders
  - Modes: separate, joint, shared, dual_stage, integrated, two_stage_shared
  - Encoders: mlp, resmlp, moe, resmlp_moe, adaptive_resmlp_moe, cnn

```bash
./run_experiments.sh basic
```

### **2. Advanced Experiments (`advanced`)**
- **Purpose**: Test different PK/PD encoder combinations
- **Duration**: ~20 minutes
- **Experiments**: 18 combinations
  - 3 modes × 3 PK encoders × 3 PD encoders (excluding same combinations)
  - Modes: joint, dual_stage, integrated
  - Encoders: mlp, resmlp, cnn

```bash
./run_experiments.sh advanced
```

### **3. Ablation Experiments (`ablation`)**
- **Purpose**: Test feature engineering and augmentation effects
- **Duration**: ~15 minutes
- **Experiments**: 6 combinations
  - Feature engineering: with/without
  - Mixup: with/without
  - Contrastive learning: with/without

```bash
./run_experiments.sh ablation
```

### **4. Hyperparameter Experiments (`hyperparameter`)**
- **Purpose**: Test different hyperparameter settings
- **Duration**: ~25 minutes
- **Experiments**: 9 combinations
  - Learning rates: 0.0001, 0.001, 0.01
  - Batch sizes: 16, 32, 64
  - Hidden dimensions: 32, 64, 128

```bash
./run_experiments.sh hyperparameter
```

### **5. CNN-Specific Experiments (`cnn`)**
- **Purpose**: Test CNN encoder parameters
- **Duration**: ~15 minutes
- **Experiments**: 6 combinations
  - Kernel sizes: 3, 5, 7
  - Number of filters: 32, 64, 128

```bash
./run_experiments.sh cnn
```

### **6. All Experiments (`all`)**
- **Purpose**: Run all experiment types
- **Duration**: ~2-3 hours
- **Experiments**: 90+ combinations

```bash
./run_experiments.sh all
```

## ⚙️ **Configuration**

### **Base Configuration**
```bash
BASE_EPOCHS=50          # Number of training epochs
BASE_BATCH_SIZE=32      # Batch size
BASE_LR=0.001          # Learning rate
BASE_PATIENCE=20       # Early stopping patience
```

### **Customizing Experiments**
Edit `run_experiments.sh` to modify:
- Number of epochs
- Batch size
- Learning rate
- Experiment combinations
- Additional arguments

## 📊 **Output Structure**

### **Results Directory**
```
results/
├── experiments/                    # Experiment results
│   ├── basic_summary_TIMESTAMP.txt
│   ├── advanced_summary_TIMESTAMP.txt
│   └── ...
├── experiment_logs/               # Experiment logs
│   ├── experiment_basic_TIMESTAMP.log
│   ├── experiment_advanced_TIMESTAMP.log
│   └── ...
└── runs/                         # Individual experiment results
    ├── basic_separate_mlp_TIMESTAMP/
    ├── basic_joint_resmlp_TIMESTAMP/
    └── ...
```

### **Individual Experiment Results**
Each experiment creates:
```
results/runs/{run_name}/{mode}/{encoder}/s{seed}/
├── model.pth          # Trained model
├── config.json        # Configuration
├── scalers.pkl        # Data scalers
└── results.json       # Training results
```

## 🔍 **Monitoring Experiments**

### **Real-time Monitoring**
```bash
# Watch experiment progress
tail -f results/experiment_logs/experiment_basic_TIMESTAMP.log

# Check specific experiment
ls -la results/runs/basic_separate_mlp_TIMESTAMP/
```

### **Experiment Summary**
After completion, check the summary file:
```bash
cat results/experiments/basic_summary_TIMESTAMP.txt
```

## 🧪 **Example Commands**

### **Run Specific Experiment**
```bash
# Single experiment
python main.py --run_name my_test --mode separate --encoder resmlp_moe --epochs 10

# With feature engineering
python main.py --run_name my_test --mode separate --encoder resmlp_moe --epochs 10 --use_fe

# Different PK/PD encoders
python main.py --run_name my_test --mode joint --encoder_pk mlp --encoder_pd cnn --epochs 10
```

### **Custom Experiment Script**
```bash
#!/bin/bash
# custom_experiment.sh

python main.py --run_name custom_1 --mode separate --encoder resmlp_moe --epochs 100 --use_fe --use_mixup
python main.py --run_name custom_2 --mode joint --encoder_pk resmlp --encoder_pd cnn --epochs 100 --use_fe
python main.py --run_name custom_3 --mode shared --encoder adaptive_resmlp_moe --epochs 100 --use_fe --lambda_contrast 0.1
```

## 📈 **Analyzing Results**

### **Compare Experiments**
```bash
# Compare different modes
grep "Best validation loss" results/runs/*/separate/*/results.json
grep "Best validation loss" results/runs/*/joint/*/results.json

# Compare different encoders
grep "Best PK RMSE" results/runs/*/separate/mlp/*/results.json
grep "Best PK RMSE" results/runs/*/separate/resmlp_moe/*/results.json
```

### **Extract Key Metrics**
```python
import json
import glob

# Load all results
results = []
for file in glob.glob("results/runs/*/*/*/results.json"):
    with open(file) as f:
        data = json.load(f)
        results.append({
            'file': file,
            'mode': data.get('mode'),
            'encoder': data.get('encoder'),
            'best_pk_rmse': data.get('best_pk_rmse'),
            'best_pd_rmse': data.get('best_pd_rmse'),
            'best_val_loss': data.get('best_val_loss')
        })

# Sort by performance
results.sort(key=lambda x: x['best_val_loss'])
for r in results[:5]:  # Top 5
    print(f"{r['mode']}-{r['encoder']}: {r['best_val_loss']:.4f}")
```

## 🚨 **Troubleshooting**

### **Common Issues**

**1. Out of Memory**
```bash
# Reduce batch size
BASE_BATCH_SIZE=16

# Reduce hidden dimension
--hidden 32
```

**2. Long Training Time**
```bash
# Reduce epochs for testing
BASE_EPOCHS=10

# Use smaller models
--hidden 32 --depth 2
```

**3. Experiment Fails**
```bash
# Check logs
tail -f results/experiment_logs/experiment_basic_TIMESTAMP.log

# Run single experiment to debug
python main.py --run_name debug --mode separate --encoder mlp --epochs 1
```

### **Resume Failed Experiments**
```bash
# Check which experiments failed
grep "failed" results/experiment_logs/experiment_basic_TIMESTAMP.log

# Re-run specific experiment
python main.py --run_name failed_experiment_name --mode separate --encoder mlp --epochs 50
```

## 📚 **Best Practices**

1. **Start with Quick Test**: Always run `./quick_test.sh` first
2. **Use Appropriate Epochs**: 50 for full experiments, 5-10 for testing
3. **Monitor Resources**: Check GPU memory and disk space
4. **Save Results**: Results are automatically saved to `results/runs/`
5. **Document Changes**: Keep track of configuration changes
6. **Analyze Systematically**: Use the provided analysis scripts

## 🎯 **Expected Results**

### **Typical Performance Ranges**
- **PK RMSE**: 0.5 - 1.5
- **PD RMSE**: 0.8 - 2.0
- **Validation Loss**: 0.5 - 2.0

### **Best Performing Combinations**
Based on typical results:
1. **ResMLP+MoE** with feature engineering
2. **CNN** for sequence-like data
3. **Joint mode** with different PK/PD encoders
4. **Mixup augmentation** for better generalization

---

**Happy Experimenting! 🧪🚀**
