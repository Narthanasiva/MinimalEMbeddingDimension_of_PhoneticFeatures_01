"""
utils.py
--------
Small helper functions reused across modules.
Keep these simple and generic.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os
import csv
import json
import numpy as np

try:
    from sphfile import SPHFile  # optional: for .sph reading
except ImportError:
    SPHFile = None

# --------------------------
# Output folders
# --------------------------

def make_dataset_output_folders(dataset_name: str, base_root: str = "02_OUTPUTS") -> None:
    """
    Create 02_OUTPUTS/<DATASET>_Outputs with standard subfolders, if missing.
    """
    base_path = os.path.join(base_root, f"{dataset_name}_Outputs")
    subfolders = ["datasets", "probes", "evaluations", "attributions", "topk_neurons", "plots"]
    for sub in subfolders:
        os.makedirs(os.path.join(base_path, sub), exist_ok=True)


# --------------------------
# CSV writing
# --------------------------

def write_to_csv(data: List[List[Any]], path: str) -> None:
    """
    Write a 2D list into a CSV file.
    We DO NOT write headers because columns depend on embedding dim.
    Rows are assumed to be flat lists: [dim_0,...,dim_767, phoneme, voiced, fric, nasal]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)


# --------------------------
# Audio file discovery (simple)
# --------------------------

def find_audio_files(dataset_path: str) -> List[str]:
    """
    Recursively find audio files under dataset_path.
    Adjust extensions to match your dataset (e.g., .wav).
    """
    audio_exts = {".wav", ".flac", ".sph"}
    found = []
    for root, _, files in os.walk(dataset_path):
        for name in files:
            lower = name.lower()
            if any(lower.endswith(ext) for ext in audio_exts):
                found.append(os.path.join(root, name))
    found.sort()
    return found


def load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load audio from .wav/.flac using librosa, or .sph using sphfile.

    Returns mono float32 waveform in [-1,1] and sampling rate (resampled to target_sr).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".wav", ".flac"}:
        import librosa
        wav, sr = librosa.load(path, sr=target_sr, mono=True)
        return wav.astype(np.float32), sr
    if ext == ".sph":
        if SPHFile is None:
            raise RuntimeError("sphfile not installed. Run: pip install sphfile")
        sph = SPHFile(path)
        data = sph.content
        sr = int(sph.sample_rate)
        if data.dtype == np.int16:
            wav = (data.astype(np.float32) / 32768.0)
        else:
            wav = data.astype(np.float32)
        if sr != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return wav, sr
    raise ValueError(f"Unsupported audio format: {path}")


# --------------------------
# Config loader
# --------------------------

def load_model_layers_config(config_path: str = "configs/model_layers.json") -> Dict[str, int]:
    """
    Load a simple JSON mapping model_name -> num_layers
    """
    if not os.path.exists(config_path):
        print(f"[WARN] Missing {config_path}. Returning empty dict.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------
# Phoneme -> feature mapping
# --------------------------

# Your sets from the prompt:
_VOICED = {
    "b","d","g","dx","jh","z","zh","v","dh",
    "m","n","ng","em","en","eng","nx",
    "l","r","w","y","hh","hv","el",
    "iy","ih","eh","ey","ae","aa","aw","ay","ah","ao",
    "oy","ow","uh","uw","ux","er","ax","ix","axr","ax-h",
}
_FRIC = {"s","sh","z","zh","f","th","v","dh","hh","hv"}
_NASAL = {"m","n","ng","em","en","eng","nx"}

# All phones set (from prompt — you can expand later if needed)
_ALL_PHONES = {
    "b","d","g","p","t","k","bcl","dcl","gcl","pcl","tcl","kcl",
    "dx","q","jh","ch","s","sh","z","zh","f","th","v","dh",
    "m","n","ng","em","en","eng","nx",
    "l","r","w","y","hh","hv","el",
    "iy","ih","eh","ey","ae","aa","aw","ay","ah","ao","oy","ow",
    "uh","uw","ux","er","ax","ix","axr","ax-h",
    "pau","epi","h#"
}

def build_feature_dict() -> Dict[str, Dict[str, int]]:
    """
    Build phoneme -> {voiced, fricative, nasal} mapping.
    Unknown phones default to 0,0,0 (e.g., pauses).
    """
    feat = {}
    for ph in _ALL_PHONES:
        feat[ph] = {
            "voiced": 1 if ph in _VOICED else 0,
            "fricative": 1 if ph in _FRIC else 0,
            "nasal": 1 if ph in _NASAL else 0,
        }
    return feat



def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert seconds to integer samples (rounded)."""
    return int(round(seconds * sample_rate))
