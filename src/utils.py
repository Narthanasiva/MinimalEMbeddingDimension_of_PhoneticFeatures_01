# src/utils.py
import os

def make_dataset_output_folders(dataset_name: str):
    """
    Create output folders for a given dataset under 02_OUTPUTS/.
    """
    base_path = os.path.join("02_OUTPUTS", f"{dataset_name}_Outputs")
    subfolders = [
        "datasets",
        "probes",
        "evaluations",
        "attributions",
        "topk_neurons",
        "plots"
    ]
    for subfolder in subfolders:
        path = os.path.join(base_path, subfolder)
        os.makedirs(path, exist_ok=True)
