#!/usr/bin/env python3
"""
Generate evaluation plots from probe evaluation summary JSON files.

Creates three variants of line charts for accuracy and F1 scores:
1. All features per model per architecture (12 plots per metric)
2. All models per feature per architecture (6 plots per metric)
3. Both architectures per model per feature (18 plots per metric)

Total: 72 plots (36 accuracy + 36 F1)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Configuration
SUMMARY_DIR = Path("02_OUTPUTS/TIMIT_Outputs/probes")
OUTPUT_BASE = Path("02_OUTPUTS/TIMIT_Outputs/plots/01_Evaluation_plots")
ARCHITECTURES = ["linear", "mlp_1x200"]
FEATURES = ["voiced", "fricative", "nasal"]
MODELS = ["HUBERT_BASE", "HUBERT_LARGE", "WAV2VEC2_BASE", "WAV2VEC2_LARGE", "WAVLM_BASE", "WAVLM_LARGE"]

# Color schemes
FEATURE_COLORS = {"voiced": "#1f77b4", "fricative": "#ff7f0e", "nasal": "#2ca02c"}
MODEL_COLORS = {
    "HUBERT_BASE": "#1f77b4",
    "HUBERT_LARGE": "#ff7f0e", 
    "WAV2VEC2_BASE": "#2ca02c",
    "WAV2VEC2_LARGE": "#d62728",
    "WAVLM_BASE": "#9467bd",
    "WAVLM_LARGE": "#8c564b"
}
ARCH_COLORS = {"linear": "#1f77b4", "mlp_1x200": "#ff7f0e"}

# Line styles
MODEL_STYLES = {
    "HUBERT_BASE": "-",
    "HUBERT_LARGE": "--",
    "WAV2VEC2_BASE": "-",
    "WAV2VEC2_LARGE": "--",
    "WAVLM_BASE": "-",
    "WAVLM_LARGE": "--"
}


def load_evaluation_summary(architecture: str) -> List[Dict]:
    """Load evaluation summary JSON for a given architecture."""
    summary_path = SUMMARY_DIR / architecture / f"{architecture}_evaluation_summary.json"
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records from {summary_path}")
    return data


def organize_data_by_architecture(data: List[Dict]) -> Dict:
    """
    Organize evaluation data by model, layer, and feature.
    
    Returns nested dict: {model: {layer: {feature: {accuracy, f1_score}}}}
    """
    organized = {}
    
    for record in data:
        model = record["model"]
        layer = record["layer"]
        feature = record["feature"]
        
        if model not in organized:
            organized[model] = {}
        if layer not in organized[model]:
            organized[model][layer] = {}
        
        organized[model][layer][feature] = {
            "accuracy": record["metrics"]["accuracy"],
            "f1_score": record["metrics"]["f1"]
        }
    
    return organized


def plot_variant1_features_per_model_per_arch(all_data: Dict[str, Dict], output_dir: Path, metric: str):
    """
    Variant 1: Plot all features (3 lines) for each model+architecture combination.
    
    Creates: 2 architectures × 6 models = 12 plots per metric
    Saves to: {metric_name}/{architecture}/{model_name}.png
    """
    metric_key = "accuracy" if metric == "01_Accuracy" else "f1_score"
    metric_label = "Accuracy" if metric == "01_Accuracy" else "F1 Score"
    
    for architecture in ARCHITECTURES:
        arch_dir = output_dir / metric / architecture
        arch_dir.mkdir(parents=True, exist_ok=True)
        
        data = all_data[architecture]
        
        for model in MODELS:
            if model not in data:
                print(f"Warning: {model} not found in {architecture} data")
                continue
            
            plt.figure(figsize=(10, 6))
            
            for feature in FEATURES:
                layers = sorted(data[model].keys())
                scores = []
                
                for layer in layers:
                    if feature in data[model][layer]:
                        scores.append(data[model][layer][feature][metric_key])
                    else:
                        scores.append(None)
                
                # Filter out None values
                valid_layers = [l for l, s in zip(layers, scores) if s is not None]
                valid_scores = [s for s in scores if s is not None]
                
                if valid_scores:
                    plt.plot(valid_layers, valid_scores, 
                            marker='o', label=feature.capitalize(),
                            color=FEATURE_COLORS[feature], linewidth=2, markersize=5)
            
            plt.xlabel("Layer", fontsize=12)
            plt.ylabel(metric_label, fontsize=12)
            plt.title(f"{metric_label} vs Layer - {model} ({architecture})", fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = arch_dir / f"{model}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Created: {output_path}")


def plot_variant2_models_per_feature_per_arch(all_data: Dict[str, Dict], output_dir: Path, metric: str):
    """
    Variant 2: Plot all models (6 lines) for each feature+architecture combination.
    
    Creates: 2 architectures × 3 features = 6 plots per metric
    Saves to: {metric_name}/{architecture}/{feature}.png
    """
    metric_key = "accuracy" if metric == "01_Accuracy" else "f1_score"
    metric_label = "Accuracy" if metric == "01_Accuracy" else "F1 Score"
    
    for architecture in ARCHITECTURES:
        arch_dir = output_dir / metric / architecture
        arch_dir.mkdir(parents=True, exist_ok=True)
        
        data = all_data[architecture]
        
        for feature in FEATURES:
            plt.figure(figsize=(10, 6))
            
            for model in MODELS:
                if model not in data:
                    continue
                
                layers = sorted(data[model].keys())
                scores = []
                
                for layer in layers:
                    if feature in data[model][layer]:
                        scores.append(data[model][layer][feature][metric_key])
                    else:
                        scores.append(None)
                
                valid_layers = [l for l, s in zip(layers, scores) if s is not None]
                valid_scores = [s for s in scores if s is not None]
                
                if valid_scores:
                    plt.plot(valid_layers, valid_scores,
                            marker='o', label=model,
                            color=MODEL_COLORS[model], linestyle=MODEL_STYLES[model],
                            linewidth=2, markersize=5)
            
            plt.xlabel("Layer", fontsize=12)
            plt.ylabel(metric_label, fontsize=12)
            plt.title(f"{metric_label} vs Layer - {feature.capitalize()} ({architecture})", 
                     fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=9, ncol=2)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = arch_dir / f"{feature}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Created: {output_path}")


def plot_variant3_archs_per_model_per_feature(all_data: Dict[str, Dict], output_dir: Path, metric: str):
    """
    Variant 3: Plot both architectures (2 lines) for each model+feature combination.
    
    Creates: 6 models × 3 features = 18 plots per metric
    Saves to: {metric_name}/{model_name}/{feature}.png
    """
    metric_key = "accuracy" if metric == "01_Accuracy" else "f1_score"
    metric_label = "Accuracy" if metric == "01_Accuracy" else "F1 Score"
    
    for model in MODELS:
        model_dir = output_dir / metric / model
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for feature in FEATURES:
            plt.figure(figsize=(10, 6))
            
            for architecture in ARCHITECTURES:
                data = all_data[architecture]
                
                if model not in data:
                    print(f"Warning: {model} not found in {architecture} data")
                    continue
                
                layers = sorted(data[model].keys())
                scores = []
                
                for layer in layers:
                    if feature in data[model][layer]:
                        scores.append(data[model][layer][feature][metric_key])
                    else:
                        scores.append(None)
                
                valid_layers = [l for l, s in zip(layers, scores) if s is not None]
                valid_scores = [s for s in scores if s is not None]
                
                if valid_scores:
                    plt.plot(valid_layers, valid_scores,
                            marker='o', label=architecture,
                            color=ARCH_COLORS[architecture],
                            linewidth=2, markersize=5)
            
            plt.xlabel("Layer", fontsize=12)
            plt.ylabel(metric_label, fontsize=12)
            plt.title(f"{metric_label} vs Layer - {model} ({feature.capitalize()})",
                     fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = model_dir / f"{feature}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Created: {output_path}")


def main():
    """Generate all evaluation plots."""
    print("=" * 70)
    print("EVALUATION PLOTTING SCRIPT")
    print("=" * 70)
    
    # Load data for all architectures
    print("\n[1/4] Loading evaluation summaries...")
    all_data = {}
    for architecture in ARCHITECTURES:
        try:
            raw_data = load_evaluation_summary(architecture)
            all_data[architecture] = organize_data_by_architecture(raw_data)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    # Generate plots for both metrics
    metrics = ["01_Accuracy", "02_F1_Score"]
    
    for metric in metrics:
        metric_name = "Accuracy" if metric == "01_Accuracy" else "F1 Score"
        print(f"\n{'=' * 70}")
        print(f"Generating {metric_name} Plots")
        print(f"{'=' * 70}")
        
        # Variant 1: Features per model per architecture
        print(f"\n[2/4] Variant 1: All features per model per architecture (12 plots)...")
        plot_variant1_features_per_model_per_arch(all_data, OUTPUT_BASE, metric)
        
        # Variant 2: Models per feature per architecture
        print(f"\n[3/4] Variant 2: All models per feature per architecture (6 plots)...")
        plot_variant2_models_per_feature_per_arch(all_data, OUTPUT_BASE, metric)
        
        # Variant 3: Architectures per model per feature
        print(f"\n[4/4] Variant 3: Both architectures per model per feature (18 plots)...")
        plot_variant3_archs_per_model_per_feature(all_data, OUTPUT_BASE, metric)
    
    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"\nTotal plots generated: 72 (36 Accuracy + 36 F1 Score)")
    print(f"Output directory: {OUTPUT_BASE}")
    print("\nDirectory structure:")
    print("  01_Accuracy/")
    print("    linear/          - 12 model plots + 3 feature plots")
    print("    mlp_1x200/       - 12 model plots + 3 feature plots")
    print("    {MODEL}/         - 3 feature plots (18 total)")
    print("  02_F1_Score/")
    print("    [same structure as Accuracy]")


if __name__ == "__main__":
    main()
