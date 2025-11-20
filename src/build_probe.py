"""
Training, evaluation, and visualization pipeline for phonetic feature probes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.phoneme_features import FEATURE_NAMES
from configs.probe_architectures import (
	ProbeArchitectureSpec,
	create_spec_from_dict,
	get_probe_architecture,
	list_probe_architectures,
)


def _set_seed(seed: Optional[int]) -> None:
	if seed is None:
		return
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
	if name == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(name)


def _canonical_model_name(name: str) -> str:
	return name.replace("-", "_").replace(" ", "_").lower()


def _find_dataset_dirs(root: Path) -> Dict[str, Path]:
	mapping: Dict[str, Path] = {}
	for candidate in sorted(root.glob("*")):
		if not candidate.is_dir():
			continue
		if candidate.name.startswith("probes_"):
			continue
		if candidate.name.lower() == "evaluations":
			continue
		if not any(candidate.glob("layer_*_train.pkl")):
			continue
		mapping[_canonical_model_name(candidate.name)] = candidate
	return mapping


def _spec_from_dict(config: Dict[str, Any], fallback_name: Optional[str] = None) -> ProbeArchitectureSpec:
	return create_spec_from_dict(config, fallback_name=fallback_name)


def _spec_from_file(path: Path, fallback_name: Optional[str] = None) -> ProbeArchitectureSpec:
	with open(path, "r", encoding="utf-8") as handle:
		config = json.load(handle)
	return _spec_from_dict(config, fallback_name=fallback_name or path.stem)


def _load_architectures_from_file(path: Path) -> List[ProbeArchitectureSpec]:
	with open(path, "r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if isinstance(payload, dict):
		entries = payload.get("architectures") or payload.get("probes") or payload.get("items")
		if entries is None:
			raise ValueError(
				f"Architecture file '{path}' must contain an 'architectures' list or be a list itself."
			)
	elif isinstance(payload, list):
		entries = payload
	else:
		raise ValueError(f"Unsupported architecture file format for '{path}'.")

	specs: List[ProbeArchitectureSpec] = []
	for idx, entry in enumerate(entries):
		if isinstance(entry, str):
			specs.append(get_probe_architecture(entry))
		elif isinstance(entry, dict):
			fallback = f"{path.stem}_{idx}" if "name" not in entry else None
			specs.append(_spec_from_dict(entry, fallback_name=fallback))
		else:
			raise ValueError(
				f"Invalid architecture entry at index {idx} in '{path}'. Expected string or dict, got {type(entry).__name__}."
			)

	return specs


def _resolve_architecture_specs(
	architecture_names: Optional[Sequence[str]],
	custom_arch_paths: Optional[Sequence[Path]],
	architecture_file: Optional[Path],
) -> List[ProbeArchitectureSpec]:
	resolved: Dict[str, ProbeArchitectureSpec] = {}
	selection_order: List[str] = []

	if architecture_file is not None:
		file_specs = _load_architectures_from_file(architecture_file)
		for spec in file_specs:
			resolved[spec.name] = spec
		if not architecture_names:
			selection_order.extend([spec.name for spec in file_specs])

	if custom_arch_paths:
		for path in custom_arch_paths:
			spec = _spec_from_file(path)
			resolved[spec.name] = spec
			selection_order.append(spec.name)

	names = list(architecture_names) if architecture_names else []
	include_all = any(name.lower() == "all" for name in names)
	if include_all:
		names = sorted(list_probe_architectures())
	else:
		names = list(dict.fromkeys(names))

	if names:
		for name in names:
			if name.lower() == "all":
				continue
			spec = get_probe_architecture(name)
			resolved[spec.name] = spec
			selection_order.append(spec.name)
	elif not selection_order:
		default_spec = get_probe_architecture("mlp_1x200")
		resolved[default_spec.name] = default_spec
		selection_order.append(default_spec.name)

	unique_order = []
	seen = set()
	for name in selection_order:
		if name not in resolved:
			continue
		if name not in seen:
			unique_order.append(name)
			seen.add(name)

	if include_all:
		for name in sorted(list_probe_architectures()):
			if name not in seen:
				unique_order.append(name)
				seen.add(name)

	if not unique_order:
		raise ValueError("No probe architectures resolved. Provide --architecture, --architecture-file, or --custom-architecture.")

	missing = [name for name in unique_order if name not in resolved]
	if missing:
		raise ValueError(f"Unknown architecture(s): {', '.join(missing)}")

	return [resolved[name] for name in unique_order]


def _stack_tensor(series: pd.Series) -> torch.Tensor:
	return torch.stack([item.float() if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.float32) for item in series])


def _load_layer_split(dataset_dir: Path, layer: int, feature: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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

	x_train = _stack_tensor(train_df["embedding"])
	x_test = _stack_tensor(test_df["embedding"])
	y_train = torch.tensor(train_df[feature].astype(np.float32).to_numpy(), dtype=torch.float32).unsqueeze(1)
	y_test = torch.tensor(test_df[feature].astype(np.float32).to_numpy(), dtype=torch.float32).unsqueeze(1)
	return x_train, y_train, x_test, y_test


def _build_dataloader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, use_cuda: bool, num_workers: int) -> DataLoader:
	dataset = TensorDataset(x, y)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		drop_last=False,
		num_workers=num_workers,
		pin_memory=use_cuda,
	)


def _train_probe(
	spec: ProbeArchitectureSpec,
	x_train: torch.Tensor,
	y_train: torch.Tensor,
	device: torch.device,
	epochs: int,
	batch_size: int,
	lr: float,
	num_workers: int,
) -> Tuple[Any, List[float]]:
	if spec.kind == "pytorch_mlp":
		model = spec.build(input_dim=x_train.shape[1], output_dim=1).to(device)
		loader = _build_dataloader(x_train, y_train, batch_size, True, device.type == "cuda", num_workers)
		criterion = nn.BCEWithLogitsLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
		history: List[float] = []

		for _ in range(epochs):
			running_loss = 0.0
			sample_count = 0
			model.train()
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

	if spec.kind == "sklearn_linear_svc":
		params = dict(spec.extra)
		model = LinearSVC(**params)
		x_np = x_train.cpu().numpy()
		y_np = y_train.cpu().numpy().ravel().astype(int)
		model.fit(x_np, y_np)
		return model, []

	if spec.kind == "sklearn_svc":
		params = dict(spec.extra)
		model = SVC(**params)
		x_np = x_train.cpu().numpy()
		y_np = y_train.cpu().numpy().ravel().astype(int)
		model.fit(x_np, y_np)
		return model, []

	raise ValueError(f"Unsupported probe kind: {spec.kind}")


def _evaluate_probe(
	spec: ProbeArchitectureSpec,
	model: Any,
	x_test: torch.Tensor,
	y_test: torch.Tensor,
	device: torch.device,
	batch_size: int,
	num_workers: int,
) -> Dict[str, float]:
	if spec.kind == "pytorch_mlp":
		loader = _build_dataloader(x_test, y_test, batch_size, False, device.type == "cuda", num_workers)
		model.eval()
		logits_list: List[torch.Tensor] = []
		labels_list: List[torch.Tensor] = []
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

	else:
		x_np = x_test.cpu().numpy()
		y_true = y_test.cpu().numpy().ravel().astype(int)
		pred_array = model.predict(x_np)
		y_pred = np.asarray(pred_array, dtype=float)

	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred, zero_division=0)
	recall = recall_score(y_true, y_pred, zero_division=0)
	f1 = f1_score(y_true, y_pred, zero_division=0)

	return {
		"accuracy": float(accuracy),
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
	}


def _detect_available_features(dataset_dir: Path) -> List[str]:
	train_files = sorted(dataset_dir.glob("layer_*_train.pkl"))
	if not train_files:
		return list(FEATURE_NAMES)
	df = pd.read_pickle(train_files[0])
	return [col for col in df.columns if col not in {"embedding", "phone"}]


def _ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def _plot_scores(df: pd.DataFrame, title: str, ylabel: str, output_path: Path) -> None:
	if df.empty:
		return
	plt.figure(figsize=(10, 6))
	layers = np.array([int(col) for col in df.columns], dtype=int)
	plotted = False
	for label, row in df.iterrows():
		values = row.to_numpy(dtype=float)
		mask = np.isfinite(values)
		if not mask.any():
			continue
		plt.plot(layers[mask], values[mask], marker="o", label=str(label))
		plotted = True
	if not plotted:
		plt.close()
		return
	plt.title(title)
	plt.xlabel("Layer")
	plt.ylabel(ylabel)
	plt.xticks(layers)
	plt.ylim(0.0, 1.0)
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()


def run_probe_pipeline(
	dataset_root: Path,
	architecture_spec: ProbeArchitectureSpec,
	model_names: Optional[Sequence[str]],
	features: Optional[Sequence[str]],
	epochs: int,
	batch_size: int,
	lr: float,
	device_name: str,
	num_workers: int,
	probes_output_root: Path,
	eval_output_root: Path,
	quiet: bool,
	seed: Optional[int],
) -> List[Dict[str, object]]:
	_set_seed(seed)
	device = _resolve_device(device_name)

	dataset_root = dataset_root.resolve()
	probes_output_root = probes_output_root.resolve()
	eval_output_root = eval_output_root.resolve()

	available_dirs = _find_dataset_dirs(dataset_root)
	if not available_dirs:
		raise RuntimeError(f"No probe datasets found in {dataset_root}")

	if model_names is None or "all" in [name.lower() for name in model_names]:
		selected = available_dirs
	else:
		selected = {}
		for name in model_names:
			key = _canonical_model_name(name)
			if key not in available_dirs:
				raise ValueError(f"Dataset for model '{name}' not found under {dataset_root}")
			selected[key] = available_dirs[key]

	records: List[Dict[str, object]] = []
	skipped: List[Tuple[str, int, str]] = []

	probes_base_dir = _ensure_dir(probes_output_root / f"probes_{architecture_spec.name}")

	for key, dataset_dir in selected.items():
		model_label = dataset_dir.name
		if not quiet:
			print(f"\n>>> Training probes for dataset: {model_label} ({architecture_spec.name})")

		layer_files = sorted(dataset_dir.glob("layer_*_train.pkl"))
		layers = sorted({int(path.stem.split("_")[1]) for path in layer_files})
		if not layers:
			if not quiet:
				print(f"[skip] {model_label}: no layer files detected")
			continue

		available_features = _detect_available_features(dataset_dir)
		if features is None:
			feature_list = [f for f in FEATURE_NAMES if f in available_features]
		else:
			invalid = [f for f in features if f not in FEATURE_NAMES]
			if invalid:
				raise ValueError(f"Unsupported feature(s) requested: {', '.join(invalid)}")
			feature_list = [f for f in features if f in available_features]
		if not feature_list:
			if not quiet:
				print(f"[skip] {model_label}: no matching features found")
			continue

		model_probe_dir = _ensure_dir(probes_base_dir / model_label)

		for layer in layers:
			if not quiet:
				print(f"  Layer {layer}")
			for feature in feature_list:
				split = _load_layer_split(dataset_dir, layer, feature)
				if split is None:
					skipped.append((model_label, layer, feature))
					if not quiet:
						print(f"    - {feature}: no data, skipped")
					continue

				x_train, y_train, x_test, y_test = split
				model, loss_history = _train_probe(
					architecture_spec,
					x_train,
					y_train,
					device,
					epochs,
					batch_size,
					lr,
					num_workers,
				)

				metrics = _evaluate_probe(
					architecture_spec,
					model,
					x_test,
					y_test,
					device,
					batch_size,
					num_workers,
				)

				layer_dir = _ensure_dir(model_probe_dir / f"layer_{layer:02d}")
				if architecture_spec.kind == "pytorch_mlp":
					model_path = layer_dir / f"{feature}.pt"
					model_cpu = model.cpu()
					torch.save(model_cpu.state_dict(), model_path)
					del model
					if device.type == "cuda":
						torch.cuda.empty_cache()
				else:
					model_path = layer_dir / f"{feature}.joblib"
					joblib.dump(model, model_path)

				metadata = {
					"architecture": architecture_spec.name,
					"architecture_kind": architecture_spec.kind,
					"architecture_params": architecture_spec.extra,
					"model_dataset": model_label,
					"layer": layer,
					"feature": feature,
					"input_dim": int(x_train.shape[1]),
					"train_samples": int(x_train.shape[0]),
					"test_samples": int(x_test.shape[0]),
					"hyperparameters": {
						"epochs": epochs,
						"batch_size": batch_size,
						"lr": lr,
					},
					"loss_history": [float(v) for v in loss_history],
					"metrics": metrics,
				}

				with open(layer_dir / f"{feature}_metadata.json", "w", encoding="utf-8") as handle:
					json.dump(metadata, handle, indent=2)

				record = {
					"architecture": architecture_spec.name,
					"architecture_kind": architecture_spec.kind,
					"model_key": key,
					"model_label": model_label,
					"layer": layer,
					"feature": feature,
					"model_path": str(model_path),
					**metrics,
				}
				records.append(record)

				if not quiet:
					print(
						f"    - {feature}: acc={metrics['accuracy']:.3f} "
						f"f1={metrics['f1']:.3f}"
					)

	scores_root = _ensure_dir(eval_output_root / "evaluations")
	score_dir = _ensure_dir(scores_root / "Evaluation_score" / architecture_spec.name)
	viz_dir = _ensure_dir(scores_root / "Visualization" / architecture_spec.name)

	if records:
		metrics_df = pd.DataFrame(records)
		metrics_df.sort_values(["model_label", "layer", "feature"], inplace=True)

		raw_metrics_path = score_dir / "raw_metrics.json"
		with open(raw_metrics_path, "w", encoding="utf-8") as handle:
			json.dump(records, handle, indent=2)

		model_layer_accuracy = (
			metrics_df.groupby(["model_label", "layer"])["accuracy"].mean().unstack(fill_value=np.nan)
		)
		model_layer_accuracy.sort_index(axis=0, inplace=True)
		model_layer_accuracy.sort_index(axis=1, inplace=True)
		model_layer_accuracy.to_csv(score_dir / "model_vs_layer_accuracy.csv", float_format="%.6f")

		model_layer_f1 = (
			metrics_df.groupby(["model_label", "layer"])["f1"].mean().unstack(fill_value=np.nan)
		)
		model_layer_f1.sort_index(axis=0, inplace=True)
		model_layer_f1.sort_index(axis=1, inplace=True)
		model_layer_f1.to_csv(score_dir / "model_vs_layer_f1.csv", float_format="%.6f")

		feature_layer_accuracy = (
			metrics_df.groupby(["feature", "layer"])["accuracy"].mean().unstack(fill_value=np.nan)
		)
		feature_layer_accuracy = feature_layer_accuracy.reindex(sorted(feature_layer_accuracy.index))
		feature_layer_accuracy.sort_index(axis=1, inplace=True)
		feature_layer_accuracy.to_csv(score_dir / "feature_vs_layer_accuracy.csv", float_format="%.6f")

		feature_layer_f1 = (
			metrics_df.groupby(["feature", "layer"])["f1"].mean().unstack(fill_value=np.nan)
		)
		feature_layer_f1 = feature_layer_f1.reindex(sorted(feature_layer_f1.index))
		feature_layer_f1.sort_index(axis=1, inplace=True)
		feature_layer_f1.to_csv(score_dir / "feature_vs_layer_f1.csv", float_format="%.6f")

		_plot_scores(
			model_layer_accuracy,
			"Accuracy vs Layer (per model)",
			"Accuracy",
			viz_dir / "model_vs_layer_accuracy.png",
		)
		_plot_scores(
			model_layer_f1,
			"F1 vs Layer (per model)",
			"F1 Score",
			viz_dir / "model_vs_layer_f1.png",
		)
		_plot_scores(
			feature_layer_accuracy,
			"Accuracy vs Layer (per feature)",
			"Accuracy",
			viz_dir / "feature_vs_layer_accuracy.png",
		)
		_plot_scores(
			feature_layer_f1,
			"F1 vs Layer (per feature)",
			"F1 Score",
			viz_dir / "feature_vs_layer_f1.png",
		)

	if skipped and not quiet:
		print("\nSkipped combinations due to missing data:")
		for model_label, layer, feature in skipped:
			print(f"  - {model_label} | layer {layer} | feature {feature}")

	if not quiet:
		print("\nProbe training complete.")

	return records


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train, evaluate, and visualise phonetic feature probes",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		required=True,
		help="Root directory containing per-model probe datasets",
	)
	parser.add_argument(
		"--models",
		type=str,
		nargs="*",
		default=None,
		help="Model dataset names to include (omit to use all)",
	)
	parser.add_argument(
		"--architecture",
		type=str,
		nargs="+",
		default=None,
		help=(
			"Probe architecture name(s) to run. Use 'all' to iterate through every registered architecture. "
			"Omit to use the default (mlp_1x200). "
			f"Available: {', '.join(list_probe_architectures())}"
		),
	)
	parser.add_argument(
		"--architectures-file",
		type=Path,
		default=None,
		help="JSON file listing architectures to load/run (names or inline definitions with a 'type' field).",
	)
	parser.add_argument(
		"--custom-architecture",
		type=Path,
		nargs="*",
		default=None,
		help="Additional JSON files defining custom probe architectures",
	)
	parser.add_argument(
		"--features",
		type=str,
		nargs="*",
		default=None,
		help="Subset of features to train (default: all available)",
	)
	parser.add_argument("--epochs", type=int, default=15, help="Training epochs per probe")
	parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training/eval")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		choices=["auto", "cpu", "cuda"],
		help="Compute device",
	)
	parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
	parser.add_argument(
		"--probes-output-root",
		type=Path,
		default=None,
		help="Base directory for saving trained probe weights",
	)
	parser.add_argument(
		"--eval-output-root",
		type=Path,
		default=None,
		help="Base directory for evaluation tables and plots",
	)
	parser.add_argument("--seed", type=int, default=None, help="Random seed")
	parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
	return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
	args = _parse_args(argv)

	custom_paths = args.custom_architecture or []
	architecture_specs = _resolve_architecture_specs(
		architecture_names=args.architecture,
		custom_arch_paths=custom_paths,
		architecture_file=args.architectures_file,
	)

	probes_root = args.probes_output_root or args.dataset_root
	evaluation_root = args.eval_output_root or args.dataset_root

	total_records = 0
	for spec in architecture_specs:
		if not args.quiet and len(architecture_specs) > 1:
			print(f"\n=== Architecture: {spec.name} ===")
		records = run_probe_pipeline(
			dataset_root=args.dataset_root,
			architecture_spec=spec,
			model_names=args.models,
			features=args.features,
			epochs=args.epochs,
			batch_size=args.batch_size,
			lr=args.lr,
			device_name=args.device,
			num_workers=args.num_workers,
			probes_output_root=probes_root,
			eval_output_root=evaluation_root,
			quiet=args.quiet,
			seed=args.seed,
		)
		total_records += len(records)

	if not args.quiet:
		print(
			f"\nSaved trained probes for {total_records} feature-layer combinations "
			f"across {len(architecture_specs)} architecture(s)."
		)


if __name__ == "__main__":
	main()

