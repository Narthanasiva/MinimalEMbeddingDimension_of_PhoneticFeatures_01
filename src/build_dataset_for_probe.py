"""
Main dataset creation pipeline for phonetic feature probing
Supports multiple models and TIMIT-structured datasets
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.model_config import get_model_config
from configs.phoneme_features import PHONEME_FEATURES, FEATURE_NAMES
from src.utils import read_phn, frame_bounds, phone_avg, phone_frames


def create_probe_dataset(
    dataset_path: Union[str, Path],
    model_name: str,
    output_path: Union[str, Path],
    train_split: float = 0.95,
    frame_shift: float = 0.020,
    frame_len: float = 0.025,
    device: Optional[str] = None,
    features: Optional[List[str]] = None,
    verbose: bool = True,
):
    """
    Create probe dataset for phonetic feature analysis
    
    Args:
        dataset_path: Path to dataset folder containing .wav and .phn files
        model_name: Model identifier (e.g., 'wavlm-base', 'hubert-large')
                   Options: wavlm-base, wavlm-large, wav2vec2-base, wav2vec2-large,
                           hubert-base, hubert-large
        output_path: Path to save output .pkl files
        train_split: Fraction of data for training (default: 0.95)
        frame_shift: Frame shift in seconds (default: 0.020)
        frame_len: Frame length in seconds (default: 0.025)
        device: Device to run model on ('cuda' or 'cpu', auto-detect if None)
        features: List of features to extract (default: all available)
        verbose: Print progress information
    
    Returns:
        Dict with statistics about the created dataset
    """
    # Setup paths
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get model configuration
    model_config = get_model_config(model_name)
    sample_rate = model_config["sample_rate"]
    num_layers = model_config["num_layers"]
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Setup features
    if features is None:
        features = FEATURE_NAMES
    else:
        # Validate features
        invalid = set(features) - set(FEATURE_NAMES)
        if invalid:
            raise ValueError(f"Invalid features: {invalid}. Available: {FEATURE_NAMES}")
    
    if verbose:
        print(f"{'='*60}")
        print(f"Creating Probe Dataset")
        print(f"{'='*60}")
        print(f"Model: {model_name} ({model_config['model_name']})")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_path}")
        print(f"Layers: {num_layers} (0 to {num_layers-1})")
        print(f"Hidden size: {model_config['hidden_size']}")
        print(f"Device: {device}")
        print(f"Features: {', '.join(features)}")
        print(f"Train/Test split: {train_split:.1%} / {1-train_split:.1%}")
        print(f"{'='*60}\n")
    
    # Load model
    if verbose:
        print(f"Loading {model_name}...")
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_config["model_name"])
    model = AutoModel.from_pretrained(
        model_config["model_name"],
        output_hidden_states=True
    ).to(device).eval()
    
    if verbose:
        print(f"✓ Model loaded successfully\n")
    
    # Find all audio files
    wav_files = sorted(list(dataset_path.glob("*.wav")) + list(dataset_path.glob("*.WAV")))
    
    if len(wav_files) == 0:
        raise ValueError(f"No .wav files found in {dataset_path}")
    
    if verbose:
        print(f"Found {len(wav_files)} audio files\n")
    
    # Initialize data storage
    train_rows = defaultdict(list)
    test_rows = defaultdict(list)
    
    # Process each audio file
    for wav in tqdm(wav_files, desc="Extracting embeddings", disable=not verbose):
        stem = wav.stem
        phn = dataset_path / f"{stem}.phn"
        
        if not phn.exists():
            warnings.warn(f"{stem}: no .phn file, skipping")
            continue
        
        # Load audio
        speech, sr = torchaudio.load(wav)
        if sr != sample_rate:
            speech = torchaudio.functional.resample(speech, sr, sample_rate)
        speech = speech.squeeze()
        
        # Extract embeddings
        inp = feature_extractor(
            speech,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in inp.items()})
            hstates = [h.squeeze(0).cpu() for h in out.hidden_states]
        
        # Get frame times and phoneme intervals
        times = frame_bounds(len(hstates[0]), frame_shift, frame_len)
        phone_intervals = read_phn(phn, sample_rate)
        
        # Random train/test split
        is_train = np.random.rand() < train_split
        
        # Extract features for each layer
        for layer_idx in range(num_layers):
            target_rows = train_rows if is_train else test_rows
            
            if is_train:
                # Training: average embeddings per phoneme
                for ph, vec in phone_avg(hstates[layer_idx], times, phone_intervals):
                    if ph in PHONEME_FEATURES:
                        row = {
                            "embedding": vec,
                            "phone": ph,
                            **{f: PHONEME_FEATURES[ph][f] for f in features}
                        }
                        target_rows[layer_idx].append(row)
            else:
                # Testing: frame-level embeddings
                for ph, vec in phone_frames(hstates[layer_idx], times, phone_intervals):
                    if ph in PHONEME_FEATURES:
                        row = {
                            "embedding": vec,
                            "phone": ph,
                            **{f: PHONEME_FEATURES[ph][f] for f in features}
                        }
                        target_rows[layer_idx].append(row)
    
    # Save datasets
    if verbose:
        print(f"\nSaving datasets...")
    
    stats = {
        "model": model_name,
        "dataset": str(dataset_path),
        "num_layers": num_layers,
        "train_samples_per_layer": {},
        "test_samples_per_layer": {},
    }
    
    for layer_idx in range(num_layers):
        # Convert to DataFrames
        train_df = pd.DataFrame(train_rows[layer_idx])
        test_df = pd.DataFrame(test_rows[layer_idx])
        
        # Save
        train_df.to_pickle(output_path / f"layer_{layer_idx}_train.pkl")
        test_df.to_pickle(output_path / f"layer_{layer_idx}_test.pkl")
        
        # Statistics
        stats["train_samples_per_layer"][layer_idx] = len(train_df)
        stats["test_samples_per_layer"][layer_idx] = len(test_df)
    
    if verbose:
        print(f"✓ All datasets saved to {output_path}\n")
        print(f"{'='*60}")
        print(f"Dataset Statistics")
        print(f"{'='*60}")
        total_train = sum(stats["train_samples_per_layer"].values())
        total_test = sum(stats["test_samples_per_layer"].values())
        print(f"Total training samples: {total_train:,}")
        print(f"Total test samples: {total_test:,}")
        print(f"Samples per layer (train): {total_train // num_layers:,}")
        print(f"Samples per layer (test): {total_test // num_layers:,}")
        print(f"{'='*60}\n")
    
    return stats


def create_probe_dataset_batch(
    dataset_path: Union[str, Path],
    model_names: List[str],
    output_base_path: Union[str, Path],
    **kwargs
):
    """
    Create probe datasets for multiple models
    
    Args:
        dataset_path: Path to dataset folder
        model_names: List of model identifiers
        output_base_path: Base path for outputs (subdirectories created per model)
        **kwargs: Additional arguments passed to create_probe_dataset
    
    Returns:
        Dict mapping model_name -> statistics
    """
    output_base_path = Path(output_base_path)
    results = {}
    
    for model_name in model_names:
        print(f"\n{'#'*60}")
        print(f"Processing: {model_name}")
        print(f"{'#'*60}\n")
        
        output_path = output_base_path / model_name.replace("-", "_").upper()
        
        try:
            stats = create_probe_dataset(
                dataset_path=dataset_path,
                model_name=model_name,
                output_path=output_path,
                **kwargs
            )
            results[model_name] = stats
        except Exception as e:
            print(f"✗ Error processing {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage
    create_probe_dataset(
        dataset_path="01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole",
        model_name="wavlm-base",
        output_path="02_OUTPUTS/TIMIT_Outputs/WAVLM_BASE",
    )
