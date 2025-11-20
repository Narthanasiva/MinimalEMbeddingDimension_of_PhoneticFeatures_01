"""
Centralized output path configuration for the entire project.

All output directories and file paths are defined here to ensure consistency
and prevent confusion about where files are saved.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# MAIN OUTPUT DIRECTORIES
# ============================================================================

# Base output directory
OUTPUT_BASE = PROJECT_ROOT / "02_OUTPUTS"

# Dataset-specific output directories
TIMIT_OUTPUT_BASE = OUTPUT_BASE / "TIMIT_Outputs"
BUCKEYE_OUTPUT_BASE = OUTPUT_BASE / "Buckeye_Outputs"  # For future use


# ============================================================================
# DATASET EMBEDDINGS (Model Layer Outputs)
# ============================================================================

def get_dataset_dir(dataset_name="TIMIT"):
    """Get the dataset directory for a specific dataset."""
    if dataset_name.upper() == "TIMIT":
        return TIMIT_OUTPUT_BASE / "datasets"
    elif dataset_name.upper() == "BUCKEYE":
        return BUCKEYE_OUTPUT_BASE / "datasets"
    else:
        return OUTPUT_BASE / f"{dataset_name}_Outputs" / "datasets"


def get_model_dataset_dir(model_name, dataset_name="TIMIT"):
    """
    Get the directory for a specific model's embeddings.
    
    Args:
        model_name: e.g., "HUBERT_BASE", "WAV2VEC2_LARGE"
        dataset_name: e.g., "TIMIT", "BUCKEYE"
    
    Returns:
        Path: Directory where layer embeddings are saved
        Example: 02_OUTPUTS/TIMIT_Outputs/datasets/HUBERT_BASE/
    """
    return get_dataset_dir(dataset_name) / model_name


def get_layer_file_path(model_name, layer_num, split="train", dataset_name="TIMIT"):
    """
    Get the path for a specific layer's embedding file.
    
    Args:
        model_name: e.g., "HUBERT_BASE"
        layer_num: Layer number (0-indexed)
        split: "train" or "test"
        dataset_name: Dataset name
    
    Returns:
        Path: Full path to the pickle file
        Example: 02_OUTPUTS/TIMIT_Outputs/datasets/HUBERT_BASE/layer_00_train.pkl
    """
    model_dir = get_model_dataset_dir(model_name, dataset_name)
    return model_dir / f"layer_{layer_num:02d}_{split}.pkl"


# ============================================================================
# PROBE OUTPUTS
# ============================================================================

def get_probe_base_dir(dataset_name="TIMIT"):
    """Get the base directory for all probes."""
    if dataset_name.upper() == "TIMIT":
        return TIMIT_OUTPUT_BASE / "probes"
    elif dataset_name.upper() == "BUCKEYE":
        return BUCKEYE_OUTPUT_BASE / "probes"
    else:
        return OUTPUT_BASE / f"{dataset_name}_Outputs" / "probes"


def get_probe_architecture_dir(architecture_name, dataset_name="TIMIT"):
    """
    Get the directory for a specific probe architecture.
    
    Args:
        architecture_name: e.g., "linear", "mlp_1x200"
        dataset_name: Dataset name
    
    Returns:
        Path: Directory for this architecture's probes
        Example: 02_OUTPUTS/TIMIT_Outputs/probes/linear/
    """
    return get_probe_base_dir(dataset_name) / architecture_name


def get_probe_file_path(architecture_name, model_name, layer_num, feature_name, dataset_name="TIMIT"):
    """
    Get the path for a specific trained probe file.
    
    Args:
        architecture_name: e.g., "linear", "mlp_1x200"
        model_name: e.g., "HUBERT_BASE"
        layer_num: Layer number (0-indexed)
        feature_name: e.g., "voiced", "fricative", "nasal"
        dataset_name: Dataset name
    
    Returns:
        Path: Full path to the probe file
        Example: 02_OUTPUTS/TIMIT_Outputs/probes/linear/probes/HUBERT_BASE/layer_00/voiced.pt
    """
    arch_dir = get_probe_architecture_dir(architecture_name, dataset_name)
    probe_dir = arch_dir / "probes" / model_name / f"layer_{layer_num:02d}"
    return probe_dir / f"{feature_name}.pt"


def get_probe_evaluation_summary_path(architecture_name, dataset_name="TIMIT"):
    """
    Get the path for the evaluation summary JSON file.
    
    Args:
        architecture_name: e.g., "linear", "mlp_1x200"
        dataset_name: Dataset name
    
    Returns:
        Path: Path to evaluation summary JSON
        Example: 02_OUTPUTS/TIMIT_Outputs/probes/linear/linear_evaluation_summary.json
    """
    arch_dir = get_probe_architecture_dir(architecture_name, dataset_name)
    return arch_dir / f"{architecture_name}_evaluation_summary.json"


# ============================================================================
# EVALUATION OUTPUTS
# ============================================================================

def get_evaluation_base_dir(dataset_name="TIMIT"):
    """Get the base directory for evaluation outputs."""
    if dataset_name.upper() == "TIMIT":
        return TIMIT_OUTPUT_BASE / "evaluations"
    elif dataset_name.upper() == "BUCKEYE":
        return BUCKEYE_OUTPUT_BASE / "evaluations"
    else:
        return OUTPUT_BASE / f"{dataset_name}_Outputs" / "evaluations"


def get_evaluation_metrics_dir(architecture_name, dataset_name="TIMIT"):
    """
    Get the directory for evaluation metrics (legacy system).
    
    Example: 02_OUTPUTS/TIMIT_Outputs/evaluations/Evaluation_score/linear/
    """
    eval_base = get_evaluation_base_dir(dataset_name)
    return eval_base / "Evaluation_score" / architecture_name


# ============================================================================
# VISUALIZATION OUTPUTS
# ============================================================================

def get_plots_base_dir(dataset_name="TIMIT"):
    """Get the base directory for all plots."""
    if dataset_name.upper() == "TIMIT":
        return TIMIT_OUTPUT_BASE / "plots"
    elif dataset_name.upper() == "BUCKEYE":
        return BUCKEYE_OUTPUT_BASE / "plots"
    else:
        return OUTPUT_BASE / f"{dataset_name}_Outputs" / "plots"


def get_evaluation_plots_dir(dataset_name="TIMIT"):
    """
    Get the directory for evaluation plots.
    
    Example: 02_OUTPUTS/TIMIT_Outputs/plots/01_Evaluation_plots/
    """
    return get_plots_base_dir(dataset_name) / "01_Evaluation_plots"


def get_metric_plots_dir(metric_name, dataset_name="TIMIT"):
    """
    Get the directory for a specific metric's plots.
    
    Args:
        metric_name: "01_Accuracy" or "02_F1_Score"
        dataset_name: Dataset name
    
    Returns:
        Path: Directory for this metric's plots
        Example: 02_OUTPUTS/TIMIT_Outputs/plots/01_Evaluation_plots/01_Accuracy/
    """
    return get_evaluation_plots_dir(dataset_name) / metric_name


def get_architecture_plots_dir(metric_name, architecture_name, dataset_name="TIMIT"):
    """
    Get the directory for architecture-specific plots.
    
    Example: 02_OUTPUTS/TIMIT_Outputs/plots/01_Evaluation_plots/01_Accuracy/linear/
    """
    return get_metric_plots_dir(metric_name, dataset_name) / architecture_name


def get_model_plots_dir(metric_name, model_name, dataset_name="TIMIT"):
    """
    Get the directory for model-specific comparison plots.
    
    Example: 02_OUTPUTS/TIMIT_Outputs/plots/01_Evaluation_plots/01_Accuracy/HUBERT_BASE/
    """
    return get_metric_plots_dir(metric_name, dataset_name) / model_name


def get_dashboard_path(dataset_name="TIMIT"):
    """
    Get the path for the interactive HTML dashboard.
    
    Returns:
        Path: Path to dashboard HTML file
        Example: 02_OUTPUTS/TIMIT_Outputs/plots/evaluation_dashboard.html
    """
    return get_plots_base_dir(dataset_name) / "evaluation_dashboard.html"


# ============================================================================
# LOG OUTPUTS
# ============================================================================

LOGS_DIR = PROJECT_ROOT / "logs"


def get_log_file_path(log_name, dataset_name="TIMIT"):
    """
    Get the path for a log file.
    
    Args:
        log_name: Name of the log file (without extension)
        dataset_name: Dataset name
    
    Returns:
        Path: Full path to log file
        Example: logs/TIMIT_probe_training.log
    """
    return LOGS_DIR / f"{dataset_name}_{log_name}.log"


# ============================================================================
# SUMMARY FILES
# ============================================================================

SUMMARY_DIR = PROJECT_ROOT / "SUMMARY_FILES"


def get_summary_file_path(summary_name):
    """
    Get the path for a summary file.
    
    Args:
        summary_name: Name of the summary file
    
    Returns:
        Path: Full path to summary file
        Example: SUMMARY_FILES/plot_summary.txt
    """
    return SUMMARY_DIR / summary_name


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directory_exists(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Path object or string
    
    Returns:
        Path: The created/existing directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_all_model_names(dataset_name="TIMIT"):
    """
    Get list of all model names that have embeddings.
    
    Args:
        dataset_name: Dataset name
    
    Returns:
        list: List of model names found in the dataset directory
    """
    dataset_dir = get_dataset_dir(dataset_name)
    if not dataset_dir.exists():
        return []
    
    return [d.name for d in dataset_dir.iterdir() if d.is_dir()]


def get_all_architecture_names(dataset_name="TIMIT"):
    """
    Get list of all probe architectures that have been trained.
    
    Args:
        dataset_name: Dataset name
    
    Returns:
        list: List of architecture names found in the probes directory
    """
    probe_dir = get_probe_base_dir(dataset_name)
    if not probe_dir.exists():
        return []
    
    return [d.name for d in probe_dir.iterdir() if d.is_dir()]


def print_path_structure(dataset_name="TIMIT"):
    """
    Print the complete path structure for a dataset.
    Useful for debugging and documentation.
    """
    print(f"\n{'='*70}")
    print(f"OUTPUT PATH STRUCTURE FOR {dataset_name.upper()}")
    print(f"{'='*70}\n")
    
    print("📁 DATASET EMBEDDINGS:")
    print(f"   Base: {get_dataset_dir(dataset_name)}")
    print(f"   Model example: {get_model_dataset_dir('HUBERT_BASE', dataset_name)}")
    print(f"   Layer file example: {get_layer_file_path('HUBERT_BASE', 0, 'train', dataset_name)}")
    
    print("\n📊 PROBE OUTPUTS:")
    print(f"   Base: {get_probe_base_dir(dataset_name)}")
    print(f"   Architecture example: {get_probe_architecture_dir('linear', dataset_name)}")
    print(f"   Probe file example: {get_probe_file_path('linear', 'HUBERT_BASE', 0, 'voiced', dataset_name)}")
    print(f"   Evaluation summary example: {get_probe_evaluation_summary_path('linear', dataset_name)}")
    
    print("\n📈 VISUALIZATION OUTPUTS:")
    print(f"   Base: {get_plots_base_dir(dataset_name)}")
    print(f"   Evaluation plots: {get_evaluation_plots_dir(dataset_name)}")
    print(f"   Dashboard: {get_dashboard_path(dataset_name)}")
    
    print("\n📝 LOGS:")
    print(f"   Base: {LOGS_DIR}")
    print(f"   Example: {get_log_file_path('probe_training', dataset_name)}")
    
    print("\n📋 SUMMARIES:")
    print(f"   Base: {SUMMARY_DIR}")
    print(f"   Example: {get_summary_file_path('QUICK_START.txt')}")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Print the complete path structure
    print_path_structure("TIMIT")
    
    # Example: Get paths for a new dataset
    print("\nExample for BUCKEYE dataset:")
    print(f"Dataset dir: {get_dataset_dir('BUCKEYE')}")
    print(f"Probe dir: {get_probe_base_dir('BUCKEYE')}")
    print(f"Dashboard: {get_dashboard_path('BUCKEYE')}")
