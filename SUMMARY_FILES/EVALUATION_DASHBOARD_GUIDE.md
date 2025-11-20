# Probe Evaluation Summary and Dashboard Guide

**Generated:** November 19, 2025  
**Project:** Minimal Embedding Dimension of Phonetic Features

---

## 📊 Overview

This document summarizes the probe evaluation results and provides guidance for navigating the interactive dashboard.

### Evaluation Statistics

- **Total Probes Trained:** 684
- **Total Plots Generated:** 72 (36 Accuracy + 36 F1 Score)
- **Models Evaluated:** 6 (HUBERT_BASE, HUBERT_LARGE, WAV2VEC2_BASE, WAV2VEC2_LARGE, WAVLM_BASE, WAVLM_LARGE)
- **Probe Architectures:** 2 (Linear, MLP 1×200)
- **Phonetic Features:** 3 (Voiced, Fricative, Nasal)

### Performance Summary

#### Linear Probe Architecture
- **Average Accuracy:** Varies by model (0.84-0.97 across layers)
- **Average F1 Score:** Varies by model and feature (0.38-0.91)
- **Total Parameters per Probe:** 769 (base models), 1025 (large models)

#### MLP 1×200 Probe Architecture
- **Architecture:** Input → 200 Hidden Units (ReLU) → 1 Output
- **Average Accuracy:** Similar to linear, slight improvements in some cases
- **Average F1 Score:** Competitive with linear probes
- **Total Parameters per Probe:** 154,001 (base models), 205,201 (large models)

---

## 📁 File Organization

### Source Code (src/)
- `plot_evaluation_results.py` - Script to generate all evaluation plots from JSON summaries
- `create_dashboard.py` - Script to create interactive HTML dashboard
- `build_probe.py` - Original probe training and evaluation pipeline
- `build_dataset_for_probe.py` - Dataset preparation utilities
- `compute.py` - Computation utilities
- `utils.py` - Helper functions

### Configuration (configs/)
- `probe_models.py` - Probe class definitions (LinearProbe, MLPProbe_1x200)
- `probe_architectures.py` - JSON-based architecture loader (legacy)
- `probe_architectures.json` - Architecture specifications (legacy)
- `phoneme_features.py` - Phonetic feature definitions
- `model_config.py` - Model configuration

### Outputs (02_OUTPUTS/TIMIT_Outputs/)

#### Probe Files
```
probes/
├── linear/
│   ├── linear_evaluation_summary.json     [342 records]
│   └── probes/                            [Individual .pt files]
└── mlp_1x200/
    ├── mlp_1x200_evaluation_summary.json  [342 records]
    └── probes/                            [Individual .pt files]
```

#### Plot Files
```
plots/01_Evaluation_plots/
├── 01_Accuracy/                           [36 plots]
│   ├── linear/                            [15 plots]
│   ├── mlp_1x200/                         [15 plots]
│   └── {6 MODEL dirs}/                    [3 plots each]
└── 02_F1_Score/                           [36 plots - same structure]
```

#### Dashboard
```
plots/evaluation_dashboard.html            [Interactive visualization]
```

---

## 🎯 Interactive Dashboard Guide

### Accessing the Dashboard

**Local Path:**
```
file:///home/narthana/MinimalEMbeddingDimension_of_PhoneticFeatures_01/02_OUTPUTS/TIMIT_Outputs/plots/evaluation_dashboard.html
```

Open this file in any modern web browser (Chrome, Firefox, Safari, Edge).

### Dashboard Features

#### 1. Statistics Overview
- Top section displays key metrics
- Architecture-specific average accuracy and F1 scores
- Total probe counts and model information

#### 2. Tab Navigation

**📊 Features per Model Tab (Variant 1)**
- View: All 3 phonetic features on the same graph
- Organization: Separate plots for each model × architecture combination
- Count: 12 plots per metric (24 total)
- Use case: Compare how different features behave within a single model

**📈 Models per Feature Tab (Variant 2)**
- View: All 6 models on the same graph
- Organization: Separate plots for each feature × architecture combination
- Count: 6 plots per metric (12 total)
- Use case: Compare model performance on a specific phonetic feature

**🔄 Architecture Comparison Tab (Variant 3)**
- View: Both architectures (linear vs MLP) on the same graph
- Organization: Separate plots for each model × feature combination
- Count: 18 plots per metric (36 total)
- Use case: Compare probe architecture effectiveness for specific tasks

**🗂️ Complete Overview Tab**
- View: All 72 plots with dynamic filtering
- Filters available:
  - Metric (Accuracy / F1 Score)
  - Architecture (Linear / MLP 1×200)
  - Model (6 options)
  - Feature (Voiced / Fricative / Nasal)
- Use case: Flexible exploration and cross-comparison

#### 3. Interactive Features

- **Click to Enlarge:** Click any plot to view full-size in modal overlay
- **Hover Effects:** Plots highlight on hover for better visibility
- **Responsive Design:** Works on desktop, tablet, and mobile devices
- **Real-time Filtering:** Overview tab filters update instantly

---

## 📈 Plot Interpretation Guide

### X-Axis: Layer Number
- Base models: 0-12 (13 layers)
- Large models: 0-24 (25 layers)

### Y-Axis: Performance Metric
- Accuracy: 0.0 to 1.0 (higher is better)
- F1 Score: 0.0 to 1.0 (harmonic mean of precision and recall)

### Common Patterns

#### Voiced Feature
- Generally high performance across all models
- Often shows layer-wise improvement in middle layers
- Linear and MLP probes perform similarly

#### Fricative Feature
- More challenging than voiced
- Performance varies more across layers
- Some models show distinct "sweet spot" layers

#### Nasal Feature
- Most challenging feature to classify
- Lower F1 scores, especially in early layers
- Benefits more from non-linear (MLP) probes in some cases

---

## 🔄 Regenerating Plots and Dashboard

### Generate Plots
```bash
cd /home/narthana/MinimalEMbeddingDimension_of_PhoneticFeatures_01
./venv/bin/python src/plot_evaluation_results.py
```

### Create Dashboard
```bash
cd /home/narthana/MinimalEMbeddingDimension_of_PhoneticFeatures_01
./venv/bin/python src/create_dashboard.py
```

---

## 📊 Data Sources

All plots are generated from evaluation summary JSON files:
- `02_OUTPUTS/TIMIT_Outputs/probes/linear/linear_evaluation_summary.json`
- `02_OUTPUTS/TIMIT_Outputs/probes/mlp_1x200/mlp_1x200_evaluation_summary.json`

Each JSON contains 342 records with the following structure:
```json
{
  "model": "HUBERT_BASE",
  "layer": 0,
  "feature": "voiced",
  "input_dim": 768,
  "train_samples": 3679,
  "test_samples": 648,
  "architecture": "linear",
  "num_hidden_layers": 0,
  "hidden_units": [],
  "total_params": 769,
  "hyperparameters": {
    "epochs": 15,
    "batch_size": 512,
    "learning_rate": 0.001
  },
  "metrics": {
    "accuracy": 0.889,
    "precision": 0.901,
    "recall": 0.911,
    "f1": 0.906
  }
}
```

---

## 🎓 Key Insights

### Model Comparison
- Large models (25 layers) generally outperform base models (13 layers)
- Middle-to-late layers often contain the most predictive representations
- WAVLM models show strong performance on all features

### Architecture Comparison
- Linear probes are surprisingly effective for most tasks
- MLP probes (1×200) offer marginal improvements, especially for nasal feature
- Parameter efficiency: Linear probes have ~200× fewer parameters

### Feature Analysis
- Voiced: Easiest to classify (F1 > 0.85 in most cases)
- Fricative: Moderate difficulty (F1 ~0.70-0.85)
- Nasal: Most challenging (F1 ~0.40-0.70)

---

## 📞 Support

For questions or issues with the dashboard or plots:
1. Check that all plot files exist in `02_OUTPUTS/TIMIT_Outputs/plots/01_Evaluation_plots/`
2. Verify JSON summary files are present and valid
3. Regenerate plots if any are missing
4. Recreate dashboard if layout issues occur

---

**Document Version:** 1.0  
**Last Updated:** November 19, 2025
