# PK/PD Modeling System

## 📁 **Project Structure**

### **🔬 Core System Files**
- `main.py` - Main training script
- `config.py` - Configuration management
- `competition_solver.py` - Competition tasks solver
- `dose_optimizer.py` - Dose optimization
- `population_simulator.py` - Population simulation
- `ensemble_optimizer.py` - Ensemble optimization
- `hyperparameter_optimizer.py` - Hyperparameter optimization
- `model_integration.py` - Model integration
- `Baseline_ODE.py` - Baseline ODE model

### **🧹 Unified Files**
- `unified_augmentation.py` - **Unified Data Augmentation System**
  - Mixup Augmentation
  - Contrastive Learning Augmentation
  - Extended Dose Data Generation
  - Massive Patient Data Generation
- `unified_task_runner.py` - **Unified Task Execution System**
  - Single model task execution
  - Multi-model comparison execution
- `unified_competition.py` - **Unified Competition System**
  - All competition tasks execution
  - Result analysis and comparison

### **📊 Data and Models**
- `data/` - Data files
- `models/` - Model definitions
- `training/` - Training modes
- `utils/` - Utility functions
- `results/` - Result files

### **🔧 Scripts and Configuration**
- `requirements.txt` - Dependencies
- `run_mode_tests.sh` - Mode tests
- `quick_mode_encoder_test.sh` - Quick tests
- `run_comprehensive_experiments.sh` - Comprehensive experiments
- `analyze_experiment_results.py` - Result analysis

## 🚀 **Usage**

### **1. Data Augmentation**
```bash
python unified_augmentation.py
```

### **2. Model Training**
```bash
python main.py --mode shared --encoder resmlp_moe --epochs 150 --use_mixup --lambda_contrast 0.3
```

### **3. Competition Tasks Execution**
```bash
python unified_task_runner.py
```

### **4. Unified Competition System**
```bash
python unified_competition.py
```

### **5. Basic Usage**
```bash
# Default execution
python main.py

# Specify options
python main.py --mode separate --encoder mlp --epochs 100
```

### **6. Advanced Encoder Configuration**
```bash
# Use different encoders for PK and PD
python main.py --mode joint --encoder_pk mlp --encoder_pd resmlp_moe

# PK: simple MLP, PD: complex MoE
python main.py --mode separate --encoder_pk mlp --encoder_pd moe

# Dual stage with different front/back encoders
python main.py --mode dual_stage --encoder_pk resmlp --encoder_pd moe

# Shared mode with specific encoder
python main.py --mode shared --encoder resmlp_moe
```

## 📈 **Key Achievements**

- ✅ **File Organization**: 30+ files → 15 core files
- ✅ **Duplicate Removal**: 6 augmentation files → 1 unified file
- ✅ **Task Execution**: 5 task execution files → 1 unified file
- ✅ **Competition**: 3 competition files → 1 unified file
- ✅ **Code Reusability**: Unified system for improved consistency

## ✨ **Key Features**

- **Multiple Training Modes**: separate, joint, shared, dual_stage, integrated
- **Diverse Encoders**: MLP, ResMLP, MoE, ResMLP+MoE, DualStage
- **Advanced Techniques**: Contrastive Learning, Mixup Augmentation
- **Extensible Architecture**: Easy to add new modes or encoders
- **Clean Logging**: Simplified log output
- **Uncertainty Quantification**: Monte Carlo Dropout, Ensemble methods
- **Competition Task Solver**: Automated solution for all competition questions
- **Dose Optimization**: Advanced dose finding algorithms with population simulation
- **Scenario Analysis**: Comprehensive analysis of different dosing scenarios

## 🔧 **Installation**

```bash
# Install required packages
pip install -r requirements.txt

# All required packages are in requirements.txt
```

## 🎯 **Next Steps**

1. **Model Training**: Data augmentation with `unified_augmentation.py` then training
2. **Task Execution**: Performance evaluation with `unified_task_runner.py`
3. **Result Analysis**: Comprehensive analysis with `unified_competition.py`
4. **Optimization**: Hyperparameter tuning and model improvement

---

**Cleanup Complete! Now it's a much cleaner and easier to manage structure.** 🎉