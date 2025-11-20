# Output Path Configuration Guide

## Overview

The `configs/output_paths.py` module provides centralized path management for the entire project. This prevents confusion about where files are saved and makes it easy to work with new datasets.

## Why Use This?

**Before:** Paths were scattered across different scripts, leading to:
- ❌ Files saved in random locations
- ❌ Hard to find outputs
- ❌ Difficult to add new datasets
- ❌ Path inconsistencies

**After:** Centralized configuration provides:
- ✅ Single source of truth for all paths
- ✅ Consistent directory structure
- ✅ Easy dataset switching
- ✅ Automatic directory creation
- ✅ Type-safe path handling

---

## Quick Start

### Import the module

```python
from configs.output_paths import (
    get_dataset_dir,
    get_probe_file_path,
    get_dashboard_path,
    ensure_directory_exists
)
```

### Basic Usage Examples

```python
# Get dataset directory for TIMIT
dataset_dir = get_dataset_dir("TIMIT")
# Returns: 02_OUTPUTS/TIMIT_Outputs/datasets

# Get specific model's embedding directory
model_dir = get_model_dataset_dir("HUBERT_BASE", "TIMIT")
# Returns: 02_OUTPUTS/TIMIT_Outputs/datasets/HUBERT_BASE/

# Get path for a specific layer file
layer_path = get_layer_file_path("HUBERT_BASE", layer_num=5, split="train", dataset_name="TIMIT")
# Returns: 02_OUTPUTS/TIMIT_Outputs/datasets/HUBERT_BASE/layer_05_train.pkl

# Get probe file path
probe_path = get_probe_file_path("linear", "HUBERT_BASE", 5, "voiced", "TIMIT")
# Returns: 02_OUTPUTS/TIMIT_Outputs/probes/linear/probes/HUBERT_BASE/layer_05/voiced.pt

# Create directory if it doesn't exist
output_dir = ensure_directory_exists(probe_path.parent)
```

---

## Complete Function Reference

### Dataset Embeddings

#### `get_dataset_dir(dataset_name="TIMIT")`
Get the base directory for dataset embeddings.

**Args:**
- `dataset_name`: "TIMIT", "BUCKEYE", or custom name

**Returns:** `Path` to dataset base directory

**Example:**
```python
timit_dir = get_dataset_dir("TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/datasets

buckeye_dir = get_dataset_dir("BUCKEYE")
# 02_OUTPUTS/Buckeye_Outputs/datasets
```

---

#### `get_model_dataset_dir(model_name, dataset_name="TIMIT")`
Get directory for a specific model's embeddings.

**Args:**
- `model_name`: e.g., "HUBERT_BASE", "WAV2VEC2_LARGE"
- `dataset_name`: Dataset name

**Returns:** `Path` to model directory

**Example:**
```python
model_dir = get_model_dataset_dir("WAVLM_LARGE", "TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/datasets/WAVLM_LARGE/
```

---

#### `get_layer_file_path(model_name, layer_num, split="train", dataset_name="TIMIT")`
Get path for a specific layer's embedding file.

**Args:**
- `model_name`: Model name
- `layer_num`: Layer number (0-indexed)
- `split`: "train" or "test"
- `dataset_name`: Dataset name

**Returns:** `Path` to pickle file

**Example:**
```python
train_path = get_layer_file_path("HUBERT_BASE", 12, "train", "TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/datasets/HUBERT_BASE/layer_12_train.pkl

test_path = get_layer_file_path("HUBERT_BASE", 12, "test", "TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/datasets/HUBERT_BASE/layer_12_test.pkl
```

---

### Probe Outputs

#### `get_probe_base_dir(dataset_name="TIMIT")`
Get base directory for all probes.

**Example:**
```python
probe_base = get_probe_base_dir("TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/probes
```

---

#### `get_probe_architecture_dir(architecture_name, dataset_name="TIMIT")`
Get directory for a specific probe architecture.

**Example:**
```python
linear_dir = get_probe_architecture_dir("linear", "TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/probes/linear/

mlp_dir = get_probe_architecture_dir("mlp_1x200", "TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/probes/mlp_1x200/
```

---

#### `get_probe_file_path(architecture_name, model_name, layer_num, feature_name, dataset_name="TIMIT")`
Get path for a trained probe file.

**Example:**
```python
probe_path = get_probe_file_path("linear", "HUBERT_BASE", 5, "voiced", "TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/probes/linear/probes/HUBERT_BASE/layer_05/voiced.pt
```

---

#### `get_probe_evaluation_summary_path(architecture_name, dataset_name="TIMIT")`
Get path for evaluation summary JSON.

**Example:**
```python
summary_path = get_probe_evaluation_summary_path("linear", "TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/probes/linear/linear_evaluation_summary.json
```

---

### Visualization Outputs

#### `get_plots_base_dir(dataset_name="TIMIT")`
Get base directory for all plots.

**Example:**
```python
plots_dir = get_plots_base_dir("TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/plots
```

---

#### `get_evaluation_plots_dir(dataset_name="TIMIT")`
Get directory for evaluation plots.

**Example:**
```python
eval_plots_dir = get_evaluation_plots_dir("TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/plots/01_Evaluation_plots/
```

---

#### `get_dashboard_path(dataset_name="TIMIT")`
Get path for interactive HTML dashboard.

**Example:**
```python
dashboard = get_dashboard_path("TIMIT")
# 02_OUTPUTS/TIMIT_Outputs/plots/evaluation_dashboard.html
```

---

### Utility Functions

#### `ensure_directory_exists(path)`
Create directory if it doesn't exist (including parent directories).

**Example:**
```python
output_dir = ensure_directory_exists(Path("02_OUTPUTS/NewDataset/probes"))
# Creates all necessary parent directories
```

---

#### `get_all_model_names(dataset_name="TIMIT")`
Get list of all models that have embeddings.

**Example:**
```python
models = get_all_model_names("TIMIT")
# ['HUBERT_BASE', 'HUBERT_LARGE', 'WAV2VEC2_BASE', ...]
```

---

#### `get_all_architecture_names(dataset_name="TIMIT")`
Get list of all trained probe architectures.

**Example:**
```python
architectures = get_all_architecture_names("TIMIT")
# ['linear', 'mlp_1x200']
```

---

#### `print_path_structure(dataset_name="TIMIT")`
Print complete path structure for debugging.

**Example:**
```python
print_path_structure("TIMIT")
# Displays full directory structure with examples
```

---

## Working with New Datasets

### Example: Adding BUCKEYE Dataset

```python
from configs.output_paths import *

# 1. Get dataset directory (automatically handles new datasets)
buckeye_dataset_dir = get_dataset_dir("BUCKEYE")
ensure_directory_exists(buckeye_dataset_dir)

# 2. Save embeddings
for model in ["HUBERT_BASE", "WAV2VEC2_BASE"]:
    for layer in range(13):
        train_path = get_layer_file_path(model, layer, "train", "BUCKEYE")
        test_path = get_layer_file_path(model, layer, "test", "BUCKEYE")
        
        ensure_directory_exists(train_path.parent)
        # Save your embeddings to train_path and test_path

# 3. Train probes
probe_dir = get_probe_architecture_dir("linear", "BUCKEYE")
ensure_directory_exists(probe_dir)

# 4. Save evaluation summary
summary_path = get_probe_evaluation_summary_path("linear", "BUCKEYE")
# Save your evaluation results to summary_path

# 5. Generate dashboard
dashboard_path = get_dashboard_path("BUCKEYE")
# Create dashboard and save to dashboard_path
```

---

## Migration Guide

### Updating Existing Scripts

**Old approach:**
```python
# Hardcoded paths - BAD
output_path = "02_OUTPUTS/TIMIT_Outputs/probes/linear/HUBERT_BASE/layer_05/voiced.pt"
```

**New approach:**
```python
# Using centralized config - GOOD
from configs.output_paths import get_probe_file_path, ensure_directory_exists

output_path = get_probe_file_path("linear", "HUBERT_BASE", 5, "voiced", "TIMIT")
ensure_directory_exists(output_path.parent)
```

---

### Common Patterns

#### Saving Embeddings

```python
from configs.output_paths import get_layer_file_path, ensure_directory_exists
import pickle

def save_embeddings(embeddings, model_name, layer_num, split, dataset_name="TIMIT"):
    output_path = get_layer_file_path(model_name, layer_num, split, dataset_name)
    ensure_directory_exists(output_path.parent)
    
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Saved to: {output_path}")
```

#### Loading Embeddings

```python
from configs.output_paths import get_layer_file_path
import pickle

def load_embeddings(model_name, layer_num, split, dataset_name="TIMIT"):
    input_path = get_layer_file_path(model_name, layer_num, split, dataset_name)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {input_path}")
    
    with open(input_path, 'rb') as f:
        return pickle.load(f)
```

#### Saving Probes

```python
from configs.output_paths import get_probe_file_path, ensure_directory_exists
import torch

def save_probe(probe_model, architecture, model_name, layer, feature, dataset="TIMIT"):
    output_path = get_probe_file_path(architecture, model_name, layer, feature, dataset)
    ensure_directory_exists(output_path.parent)
    
    torch.save(probe_model.state_dict(), output_path)
    print(f"Probe saved to: {output_path}")
```

---

## Directory Structure

```
02_OUTPUTS/
├── TIMIT_Outputs/
│   ├── datasets/           # Model embeddings
│   │   ├── HUBERT_BASE/
│   │   │   ├── layer_00_train.pkl
│   │   │   ├── layer_00_test.pkl
│   │   │   └── ...
│   │   └── ...
│   ├── probes/             # Trained probes
│   │   ├── linear/
│   │   │   ├── linear_evaluation_summary.json
│   │   │   └── probes/
│   │   │       └── HUBERT_BASE/
│   │   │           └── layer_00/
│   │   │               ├── voiced.pt
│   │   │               ├── fricative.pt
│   │   │               └── nasal.pt
│   │   └── ...
│   ├── plots/              # Visualizations
│   │   ├── evaluation_dashboard.html
│   │   └── 01_Evaluation_plots/
│   │       ├── 01_Accuracy/
│   │       └── 02_F1_Score/
│   └── evaluations/        # Legacy evaluation files
│
├── Buckeye_Outputs/        # Same structure for BUCKEYE
│   ├── datasets/
│   ├── probes/
│   └── plots/
│
└── {YourDataset}_Outputs/  # Automatically created for new datasets
    ├── datasets/
    ├── probes/
    └── plots/
```

---

## Benefits

1. **No More Confusion** - Single source of truth for all paths
2. **Easy Dataset Switching** - Just change the dataset_name parameter
3. **Automatic Organization** - Consistent structure for all datasets
4. **Type Safety** - Returns Path objects, not strings
5. **Auto-create Directories** - Use `ensure_directory_exists()`
6. **Discovery** - Use `get_all_model_names()` to find existing data
7. **Debugging** - Use `print_path_structure()` to see everything

---

## Testing

Run the module directly to see the complete path structure:

```bash
cd /home/narthana/MinimalEMbeddingDimension_of_PhoneticFeatures_01
./venv/bin/python configs/output_paths.py
```

---

## Next Steps

1. Update existing scripts to use this configuration
2. Test with TIMIT dataset (already working)
3. Add BUCKEYE dataset using these paths
4. Create new datasets with consistent structure
