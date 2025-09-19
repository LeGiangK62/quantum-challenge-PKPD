# 🎯 **Experiment Type-Mode-Model Combination Performance Analysis**

## **🏆 Top 10 Best Performing Combinations:**

| Rank | Experiment Type | Mode | Model | PD Average | Seed Count |
|------|-----------------|------|-------|------------|------------|
| 1 | **seed_exp_fe** | **joint** | **mlp-mlp** | **0.2490** | 5 |
| 2 | **seed_exp_fe** | **shared** | **resmlp-resmlp** | **0.2511** | 5 |
| 3 | **seed_exp_fe** | **shared** | **moe-moe** | **0.2537** | 5 |
| 4 | **seed_exp_fe_mc** | **shared** | **mlp-mlp** | **0.2548** | 5 |
| 5 | **seed_exp_fe** | **joint** | **resmlp-resmlp** | **0.2551** | 5 |
| 6 | **seed_exp_fe** | **shared** | **mlp-mlp** | **0.2563** | 5 |
| 7 | **seed_exp_fe** | **joint** | **resmlp_moe** | **0.2567** | 4 |
| 8 | **seed_exp_fe** | **dual_stage** | **mlp-mlp** | **0.2573** | 5 |
| 9 | **seed_exp_fe_mc** | **joint** | **resmlp-resmlp** | **0.2593** | 5 |
| 10 | **seed_exp_fe_mc** | **dual_stage** | **mlp-mlp** | **0.2612** | 5 |

## **📊 Best Performance by Experiment Type:**

| Experiment Type | Best Combination | PD Average |
|-----------------|------------------|------------|
| **seed_exp_fe** | joint + mlp-mlp | 0.2490 |
| **seed_exp_fe_mc** | shared + mlp-mlp | 0.2548 |
| **seed_exp_fe_mixup** | separate + mlp-mlp | 0.3684 |
| **seed_exp_fe_mixup_ct** | separate + mlp-mlp | 0.3709 |
| **seed_exp** | separate + adaptive_resmlp_moe | 0.7101 |

## **🎯 Best Performance by Mode:**

| Mode | Best Combination | PD Average |
|------|------------------|------------|
| **joint** | seed_exp_fe + mlp-mlp | 0.2490 |
| **shared** | seed_exp_fe + resmlp-resmlp | 0.2511 |
| **dual_stage** | seed_exp_fe + mlp-mlp | 0.2573 |
| **two_stage_shared** | seed_exp_fe + mlp-mlp | 0.2692 |
| **separate** | seed_exp_fe + resmlp_moe | 0.3333 |

## **🏗️ Best Performance by Model:**

| Model | Best Combination | PD Average |
|-------|------------------|------------|
| **mlp-mlp** | seed_exp_fe + joint | 0.2490 |
| **resmlp-resmlp** | seed_exp_fe + shared | 0.2511 |
| **moe-moe** | seed_exp_fe + shared | 0.2537 |
| **resmlp_moe** | seed_exp_fe + joint | 0.2567 |
| **adaptive_resmlp_moe** | seed_exp_fe + separate | 0.3426 |

## **💡 Key Insights:**

1. **Best Performance**: `seed_exp_fe + joint + mlp-mlp` (0.2490)
2. **Feature Engineering is Key**: seed_exp_fe dominates the top rankings
3. **MLP-MLP is Most Stable**: Shows good performance across multiple combinations
4. **Joint Mode is Superior**: Appears frequently in top rankings
5. **Separate Mode Shows Lower Performance**: Not present in top rankings

## **📈 Performance Summary:**

- **Total Combinations Analyzed**: 35
- **Best PD RMSE**: 0.2490
- **Worst PD RMSE**: 0.7254
- **Performance Range**: 0.4764 (difference between best and worst)

## **🔍 Detailed Analysis:**

### **Experiment Type Impact:**
- **seed_exp_fe**: Consistently performs best across all modes
- **seed_exp_fe_mc**: Second best, showing good stability
- **seed_exp_fe_mixup**: Moderate performance with separate mode
- **seed_exp_fe_mixup_ct**: Similar to mixup, slightly better
- **seed_exp**: Baseline performance, significantly lower than feature engineering variants

### **Mode Effectiveness:**
- **joint**: Most effective for feature engineering experiments
- **shared**: Good balance of performance and stability
- **dual_stage**: Competitive performance with feature engineering
- **two_stage_shared**: Moderate performance
- **separate**: Lowest performance across all experiment types

### **Model Performance:**
- **mlp-mlp**: Most versatile and stable across different modes
- **resmlp-resmlp**: Strong performance with shared mode
- **moe-moe**: Good performance with shared mode
- **resmlp_moe**: Competitive with joint mode
- **adaptive_resmlp_moe**: Limited data but shows potential
