#!/usr/bin/env python3
"""
Training and evaluation script for class-based probes.
Trains linear and mlp_1x200 probes for each model/layer/feature combination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from configs.phoneme_features import FEATURE_NAMES
from configs.probe_models import build_probe, list_available_probes


def find_dataset_dirs(root: Path) -> Dict[str, Path]:
    """Find all model dataset directories."""
    mapping = {}
    for candidate in sorted(root.glob("*")):
        if not candidate.is_dir():
            continue
        if candidate.name.startswith("probes"):
            continue
        if candidate.name.lower() == "evaluations":
            continue
        if not any(candidate.glob("layer_*_train.pkl")):
            continue
        mapping[candidate.name] = candidate
    return mapping


def load_layer_data(
    dataset_dir: Path, layer: int, feature: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Load train and test data for a specific layer and feature."""
    train_path = dataset_dir / f"layer_{layer}_train.pkl"
    test_path = dataset_dir / f"layer_{layer}_test.pkl"
    
    if not train_path.exists() or not test_path.exists():
        return None
    
    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)
    
    if feature not in train_df.columns or feature not in test_df.columns:
        return None
    
    if len(train_df) == 0 or len(test_df) == 0:
        return None
    
    x_train = torch.stack([
        item.float() if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.float32)
        for item in train_df["embedding"]
    ])
    x_test = torch.stack([
        item.float() if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.float32)
        for item in test_df["embedding"]
    ])
    
    y_train = torch.tensor(train_df[feature].astype(np.float32).to_numpy(), dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_df[feature].astype(np.float32).to_numpy(), dtype=torch.float32).unsqueeze(1)
    
    return x_train, y_train, x_test, y_test


def train_probe(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Tuple[nn.Module, List[float]]:
    """Train a probe model."""
    model = model.to(device)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = []
    model.train()
    
    for _ in range(epochs):
        running_loss = 0.0
        sample_count = 0
        
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            running_loss += float(loss.item()) * xb.size(0)
            sample_count += xb.size(0)
        
        history.append(running_loss / max(sample_count, 1))
    
    return model, history


def evaluate_probe(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    """Evaluate a probe model."""
    model = model.to(device)
    model.eval()
    
    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            logits_list.append(logits.cpu())
            labels_list.append(yb.cpu())
    
    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    probs = torch.sigmoid(logits_all)
    preds = (probs >= 0.5).float()
    
    y_true = labels_all.numpy().ravel()
    y_pred = preds.numpy().ravel()
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def get_probe_metadata(probe_name: str, input_dim: int, hidden_dim: int = 200) -> Dict[str, Any]:
    """Get architecture metadata for a probe."""
    if probe_name == "linear":
        return {
            "architecture": "linear",
            "num_hidden_layers": 0,
            "hidden_units": [],
            "total_params": input_dim + 1,  # weights + bias
            "activation": "none",
        }
    elif probe_name == "mlp_1x200":
        total_params = (input_dim * hidden_dim + hidden_dim) + (hidden_dim * 1 + 1)
        return {
            "architecture": "mlp_1x200",
            "num_hidden_layers": 1,
            "hidden_units": [hidden_dim],
            "total_params": total_params,
            "activation": "relu",
        }
    else:
        return {
            "architecture": probe_name,
            "num_hidden_layers": "unknown",
            "hidden_units": [],
            "total_params": "unknown",
            "activation": "unknown",
        }


def main():
    parser = argparse.ArgumentParser(description="Train class-based probes")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("02_OUTPUTS/TIMIT_Outputs"),
        help="Root directory containing model datasets",
    )
    parser.add_argument(
        "--probes",
        type=str,
        nargs="+",
        default=["linear", "mlp_1x200"],
        help="Probe architectures to train",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: dataset-root/class_probes)",
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    output_dir = args.output_dir or (args.dataset_root / "class_probes")
    
    dataset_dirs = find_dataset_dirs(args.dataset_root)
    if not dataset_dirs:
        raise RuntimeError(f"No datasets found in {args.dataset_root}")
    
    print(f"Found {len(dataset_dirs)} model datasets")
    print(f"Training probes: {', '.join(args.probes)}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}\n")
    
    for probe_name in args.probes:
        if probe_name not in list_available_probes():
            raise ValueError(f"Unknown probe: {probe_name}")
        
        probe_output = output_dir / probe_name
        probe_output.mkdir(parents=True, exist_ok=True)
        
        all_records = []
        
        print(f"\n{'='*60}")
        print(f"Training: {probe_name}")
        print(f"{'='*60}\n")
        
        for model_name, dataset_dir in dataset_dirs.items():
            print(f">>> Model: {model_name}")
            
            layer_files = sorted(dataset_dir.glob("layer_*_train.pkl"))
            layers = sorted({int(p.stem.split("_")[1]) for p in layer_files})
            
            if not layers:
                print(f"    [skip] No layers found")
                continue
            
            model_probe_dir = probe_output / "probes" / model_name
            model_probe_dir.mkdir(parents=True, exist_ok=True)
            
            for layer in layers:
                print(f"  Layer {layer}")
                
                for feature in FEATURE_NAMES:
                    data = load_layer_data(dataset_dir, layer, feature)
                    if data is None:
                        print(f"    - {feature}: no data, skipped")
                        continue
                    
                    x_train, y_train, x_test, y_test = data
                    input_dim = x_train.shape[1]
                    
                    # Build probe
                    probe = build_probe(probe_name, input_dim=input_dim)
                    
                    # Train
                    probe, loss_history = train_probe(
                        probe, x_train, y_train, device, args.epochs, args.batch_size, args.lr
                    )
                    
                    # Evaluate
                    metrics = evaluate_probe(probe, x_test, y_test, device, args.batch_size)
                    
                    # Save probe
                    layer_dir = model_probe_dir / f"layer_{layer:02d}"
                    layer_dir.mkdir(parents=True, exist_ok=True)
                    probe_path = layer_dir / f"{feature}.pt"
                    torch.save(probe.cpu().state_dict(), probe_path)
                    
                    # Get architecture metadata
                    arch_meta = get_probe_metadata(probe_name, input_dim)
                    
                    # Record
                    record = {
                        "model": model_name,
                        "layer": layer,
                        "feature": feature,
                        "input_dim": input_dim,
                        "train_samples": int(x_train.shape[0]),
                        "test_samples": int(x_test.shape[0]),
                        **arch_meta,
                        "hyperparameters": {
                            "epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "learning_rate": args.lr,
                        },
                        "loss_history": [float(v) for v in loss_history],
                        "metrics": metrics,
                        "probe_path": str(probe_path),
                    }
                    all_records.append(record)
                    
                    print(f"    - {feature}: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}")
        
        # Save evaluation summary
        summary_path = probe_output / f"{probe_name}_evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2)
        
        print(f"\n✅ {probe_name}: {len(all_records)} probes trained")
        print(f"   Summary saved: {summary_path}")
    
    print(f"\n{'='*60}")
    print("All probes trained successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
