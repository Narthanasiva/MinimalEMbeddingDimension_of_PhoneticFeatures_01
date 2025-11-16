"""
build_dataset_for_probe.py
--------------------------
Builds probing-ready datasets (train/test CSVs per model & layer) from a speech corpus.

This version implements:
  - Model loading for: WavLM (base/large), Wav2Vec2 (base/large), HuBERT (base/large)
  - Audio framing: 25 ms window, 20 ms overlap (=> hop = 5 ms)
  - (Placeholder) Phoneme alignment: returns empty labels for now -> you'll fill for TIMIT
  - Frame-wise embedding extraction for ALL transformer layers
  - TRAIN: average contiguous same-phoneme segments (your rule)
  - TEST : keep each frame as one row (no averaging)

IMPORTANT
---------
* Whisper encoder is different (expects log-mel input over long windows, ~30s). We include a
  clearly marked TODO for Whisper so today’s flow works for WavLM/W2V2/HuBERT without breaking.

* Phoneme alignment is dataset-specific (e.g., TIMIT .phn files). You will fill the
  `get_phoneme_labels()` function later to return one phoneme per frame (use frame START time).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import os
import math
import csv

import numpy as np
import librosa
import torch
from transformers import (
    AutoModel,
    AutoFeatureExtractor,   # works for WavLM/Wav2Vec2/HuBERT feature extraction
)

from .utils import (
    make_dataset_output_folders,
    write_to_csv,
    find_audio_files,
    load_model_layers_config,
    build_feature_dict,
    seconds_to_samples,
    load_audio,
)


# --------------------------
# Top-level driver
# --------------------------

def run_build_dataset_for_probe(
    dataset_name: str,
    dataset_path: str,
    model_list: List[str],
    output_root: str = "02_OUTPUTS",
    train_ratio: float = 0.95,
    sample_rate: int = 16000,
    win_ms: float = 25.0,
    overlap_ms: float = 20.0,
) -> None:
    """
    Orchestrates dataset building for one dataset (e.g., TIMIT).

    * Splits utterances into TRAIN/TEST by file (utterance-level split).
    * For each model:
        - Loads model & feature extractor
        - Builds TRAIN rows (averaged contiguous same-phoneme regions)
        - Builds TEST rows  (frame-wise)
        - Writes per-layer CSVs

    Args:
        dataset_name:  e.g., "TIMIT"
        dataset_path:  root folder of your dataset's audio
        model_list:    e.g., ["wavlm_base","wav2vec2_base","hubert_base",...]
        output_root:   base outputs path (we create <DATASET>_Outputs/* inside)
        train_ratio:   fraction of utterances to place in TRAIN split
        sample_rate:   audio sampling rate (TIMIT uses 16kHz)
        win_ms:        frame window size (25 ms)
        overlap_ms:    frame overlap (20 ms) -> hop = 5 ms
    """
    # 1) Prepare output folders for this dataset
    make_dataset_output_folders(dataset_name, base_root=output_root)

    # 2) Collect audio files
    all_files = find_audio_files(dataset_path)
    if not all_files:
        print(f"[WARN] No audio files found under: {dataset_path}")
        return

    # 3) Utterance-level split (simple 95/5; replace later with official split if you have it)
    cutoff = int(math.ceil(len(all_files) * train_ratio))
    train_files = all_files[:cutoff]
    test_files  = all_files[cutoff:]

    print(f"[INFO] {dataset_name}: Total={len(all_files)} | Train={len(train_files)} | Test={len(test_files)}")

    # 4) Model -> num_layers
    model_layers = load_model_layers_config()

    # 5) Build phoneme->binary dict once
    phone_feat_dict = build_feature_dict()

    # 6) Process each model
    for model_name in model_list:
        model_config = model_layers.get(model_name)
        if model_config is None:
            print(f"[WARN] Unknown model in configs: {model_name}. Skipping.")
            continue
        
        # Handle both old format (int) and new format (dict)
        if isinstance(model_config, dict):
            num_layers = model_config.get("num_layers")
            hidden_size = model_config.get("hidden_size")
        else:
            num_layers = model_config
            hidden_size = None  # Will be inferred from model output
        
        if num_layers is None:
            print(f"[WARN] Invalid config for model: {model_name}. Skipping.")
            continue

        print(f"[INFO] Model: {model_name} | layers={num_layers}" + 
              (f" | hidden_size={hidden_size}" if hidden_size else ""))

        # 6a) Load model + feature extractor
        model, feature_extractor = load_model(model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 6b) Containers for per-layer rows
        train_data: Dict[int, List[List[Any]]] = {i: [] for i in range(num_layers)}
        test_data:  Dict[int, List[List[Any]]] = {i: [] for i in range(num_layers)}

        # 6c) TRAIN: averaged segments
        for idx, audio_path in enumerate(train_files, 1):
            if idx % 50 == 0:
                print(f"[INFO] TRAIN {idx}/{len(train_files)}")

            layer_to_rows = process_audio(
                audio_path=audio_path,
                model=model,
                feature_extractor=feature_extractor,
                model_name=model_name,
                mode="train",
                sample_rate=sample_rate,
                win_ms=win_ms,
                overlap_ms=overlap_ms,
                phone_feat_dict=phone_feat_dict,
                device=device,
            )
            for layer_id, rows in layer_to_rows.items():
                train_data[layer_id].extend(rows)

        # 6d) TEST: frame-wise
        for idx, audio_path in enumerate(test_files, 1):
            if idx % 50 == 0:
                print(f"[INFO] TEST {idx}/{len(test_files)}")

            layer_to_rows = process_audio(
                audio_path=audio_path,
                model=model,
                feature_extractor=feature_extractor,
                model_name=model_name,
                mode="test",
                sample_rate=sample_rate,
                win_ms=win_ms,
                overlap_ms=overlap_ms,
                phone_feat_dict=phone_feat_dict,
                device=device,
            )
            for layer_id, rows in layer_to_rows.items():
                test_data[layer_id].extend(rows)

        # 6e) Write per-layer CSVs
        out_dir = os.path.join(output_root, f"{dataset_name}_Outputs", "datasets", model_name)
        os.makedirs(out_dir, exist_ok=True)

        for layer_id in range(num_layers):
            train_csv = os.path.join(out_dir, f"layer_{layer_id}_train.csv")
            test_csv  = os.path.join(out_dir, f"layer_{layer_id}_test.csv")
            write_to_csv(train_data[layer_id], train_csv)
            write_to_csv(test_data[layer_id],  test_csv)

        print(f"[INFO] Finished: {model_name}")


# --------------------------
# Model loading
# --------------------------

def load_model(model_name: str):
    """
    Load a pretrained SSL model + its feature extractor for raw audio.
    Returns:
      model: HF model with hidden_states
      feature_extractor: HF feature extractor to preprocess raw audio
    """
    # Map your friendly names -> Hugging Face checkpoints
    name_map = {
        "wavlm_base":      "microsoft/wavlm-base",
        "wavlm_large":     "microsoft/wavlm-large",
        "wav2vec2_base":   "facebook/wav2vec2-base",
        "wav2vec2_large":  "facebook/wav2vec2-large-robust",
        "hubert_base":     "facebook/hubert-base-ls960",
        "hubert_large":    "facebook/hubert-large-ll60k",
        # "whisper_encoder": "openai/whisper-base"  # SPECIAL (see note below)
    }

    if model_name == "whisper_encoder":
        # Whisper expects log-mel features over long context (~30s). It is not
        # designed for 25 ms raw frames. We keep it out of today's flow to avoid
        # wrong embeddings. You can add a separate whisper path later:
        #   from transformers import WhisperModel, WhisperProcessor
        #   processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        #   model = WhisperModel.from_pretrained("openai/whisper-base")
        #   -> Prepare log-mels for longer windows, then get encoder hidden_states
        raise NotImplementedError(
            "Whisper encoder needs a separate log-mel pipeline; add it later."
        )

    ckpt = name_map.get(model_name)
    if ckpt is None:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Load model with hidden states ON
    model = AutoModel.from_pretrained(ckpt, output_hidden_states=True)

    # Feature extractor for raw audio -> normalized float tensors
    feature_extractor = AutoFeatureExtractor.from_pretrained(ckpt)
    return model, feature_extractor


# --------------------------
# Audio framing + alignment
# --------------------------

def split_audio_to_frames(
    audio_waveform: np.ndarray,
    sample_rate: int,
    win_ms: float,
    overlap_ms: float
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """
    Slice audio into overlapping frames.

    Returns:
      frames:      list of 1-D numpy arrays (each frame)
      frame_times: list of (start_sec, end_sec) per frame

    Example:
      win_ms = 25.0
      overlap_ms = 20.0  -> hop_ms = 5.0
    """
    # Convert ms -> samples
    win = seconds_to_samples(win_ms / 1000.0, sample_rate)       # 25ms -> 400 samples @16k
    hop = seconds_to_samples((win_ms - overlap_ms) / 1000.0, sample_rate)  # 5ms -> 80 samples @16k

    if hop <= 0 or win <= 0:
        raise ValueError("Window/hop must be positive.")

    n = len(audio_waveform)
    frames: List[np.ndarray] = []
    frame_times: List[Tuple[float, float]] = []

    # Slide start indices: 0, hop, 2*hop, ...
    for start in range(0, max(1, n - win + 1), hop):
        end = start + win
        if end > n:
            break
        frame = audio_waveform[start:end]  # shape (win,)
        frames.append(frame)
        frame_times.append((start / sample_rate, end / sample_rate))

    return frames, frame_times


def get_phoneme_labels(
    audio_path: str,
    frame_times: List[Tuple[float, float]],
    sample_rate: int,
) -> List[str]:
    """Return phoneme label per frame-start using a sibling .phn file.

    .phn format lines: <start_sample> <end_sample> <phoneme>
    If no .phn found, fallback to 'pau' labels.
    """
    base, _ = os.path.splitext(audio_path)
    phn_path = base + ".phn"
    if not os.path.exists(phn_path):
        return ["pau"] * len(frame_times)

    intervals: List[Tuple[int, int, str]] = []
    with open(phn_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            try:
                s = int(parts[0]); e = int(parts[1]); ph = parts[2]
                intervals.append((s, e, ph))
            except ValueError:
                continue

    labels: List[str] = []
    for (start_sec, _end_sec) in frame_times:
        sample = int(round(start_sec * sample_rate))
        lab = "pau"
        for s, e, ph in intervals:
            if s <= sample < e:
                lab = ph
                break
        labels.append(lab)
    return labels


# --------------------------
# Embedding extraction
# --------------------------

def extract_embeddings(
    frames: List[np.ndarray],
    model: torch.nn.Module,
    feature_extractor,
    device: torch.device,
    sample_rate: int = 16000,
) -> List[List[List[float]]]:
    """
    For each frame:
      1) Convert raw samples -> model input using feature_extractor
      2) Run model forward with output_hidden_states=True
      3) For each layer: pool the time dimension to get ONE vector per frame (mean over time)
      4) Return nested Python lists: embeddings[frame_idx][layer_idx] = 1D list (dim ~768)

    NOTE:
      - This per-frame loop is simple but not the fastest. Later you can batch frames
        to speed up inference (keep it simple for now).
    """
    out: List[List[List[float]]] = []

    for frame in frames:
        # Feature extractor expects float32 array in range [-1, 1]
        inputs = feature_extractor(
            frame,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        # Move tensors to device (cpu/gpu)
        input_values = inputs.get("input_values")
        if input_values is None:
            # Some extractors may use "input_features"
            input_values = inputs.get("input_features")
        if input_values is None:
            raise RuntimeError("Feature extractor did not return input_values or input_features.")

        input_values = input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)
            # hidden_states is a tuple: (embeddings_layer0, hidden1, hidden2, ..., hiddenN)
            # Each element has shape (batch=1, time_steps, hidden_dim)
            hidden_states = outputs.hidden_states

        # Convert each layer to a single vector by mean pooling over time
        per_frame_per_layer: List[List[float]] = []
        for h in hidden_states:
            # h: (1, T, D)
            h_np = h.squeeze(0).detach().cpu().numpy()  # (T, D)
            if h_np.ndim == 1:
                # If somehow collapsed, ensure 2D
                h_np = h_np[None, :]
            vec = h_np.mean(axis=0)  # average over time dimension -> (D,)
            per_frame_per_layer.append(vec.tolist())

        out.append(per_frame_per_layer)

    return out


# --------------------------
# Phoneme -> binary mapping
# --------------------------

def map_phoneme_to_binary(
    phoneme: str,
    phone_feat_dict: Dict[str, Dict[str, int]]
) -> Tuple[int, int, int]:
    """
    Convert phoneme -> (voiced, fricative, nasal) using your prepared dict.
    Unknown labels default to (0, 0, 0).
    """
    info = phone_feat_dict.get(phoneme, {"voiced": 0, "fricative": 0, "nasal": 0})
    return info["voiced"], info["fricative"], info["nasal"]


# --------------------------
# Per-utterance processing
# --------------------------

def process_audio(
    audio_path: str,
    model: torch.nn.Module,
    feature_extractor,
    model_name: str,
    mode: str,
    sample_rate: int,
    win_ms: float,
    overlap_ms: float,
    phone_feat_dict: Dict[str, Dict[str, int]],
    device: torch.device,
) -> Dict[int, List[List[Any]]]:
    """
    Process ONE utterance and produce rows per layer.

    mode:
      - "train": average over contiguous same-phoneme segments
      - "test" : keep one row per frame

    Returns:
      {layer_id: [[dim_0..dim_767, phoneme, voiced, fricative, nasal], ...], ...}
    """
    # 1) Load audio as mono 16k (librosa is simple for beginners)
    #    (sr=sample_rate will resample if needed)
    wav, sr = load_audio(audio_path, target_sr=sample_rate)
    if wav.size == 0:
        return {}

    # 2) Slice into frames and compute frame times
    frames, frame_times = split_audio_to_frames(wav, sample_rate, win_ms, overlap_ms)
    if not frames:
        return {}

    # 3) One phoneme per frame (based on frame START) — fill this for TIMIT later
    phonemes = get_phoneme_labels(audio_path, frame_times, sample_rate=sample_rate)

    # 4) Extract embeddings for ALL layers, per frame
    embeddings = extract_embeddings(
        frames=frames,
        model=model,
        feature_extractor=feature_extractor,
        device=device,
        sample_rate=sample_rate,
    )
    if not embeddings:
        return {}

    n_frames = len(frames)
    if len(phonemes) != n_frames or len(embeddings) != n_frames:
        print(f"[WARN] Mismatch: frames={n_frames}, phonemes={len(phonemes)}, embeds={len(embeddings)}")
        return {}

    # Each embeddings[i] has one vector per layer
    num_layers = len(embeddings[0]) if n_frames > 0 else 0
    layer_to_frame_rows: Dict[int, List[List[Any]]] = {i: [] for i in range(num_layers)}

    # 5) Build frame-wise rows (common pre-step for train/test)
    for i in range(n_frames):
        ph = phonemes[i]
        v, f, n = map_phoneme_to_binary(ph, phone_feat_dict)

        for layer_id in range(num_layers):
            vec = embeddings[i][layer_id]  # Python list of dims
            row = list(vec) + [ph, v, f, n]
            layer_to_frame_rows[layer_id].append(row)

    # 6) Train: average contiguous same-phoneme segments per layer
    if mode.lower() == "train":
        layer_to_rows: Dict[int, List[List[Any]]] = {}
        for layer_id, rows in layer_to_frame_rows.items():
            averaged = average_consecutive_embeddings(rows)
            layer_to_rows[layer_id] = averaged
        return layer_to_rows

    # 7) Test: return frame-wise rows
    return layer_to_frame_rows


# --------------------------
# Averaging contiguous segments
# --------------------------

def average_consecutive_embeddings(rows: List[List[Any]]) -> List[List[Any]]:
    """
    Average over contiguous runs of the SAME phoneme.

    Input row format (per frame):
      [dim_0, ..., dim_D-1, phoneme, voiced, fricative, nasal]

    We walk rows in order (time order). When phoneme changes, we finalize the block:
      - mean of D-dim vectors in the block
      - append one output row: [mean_vec..., phoneme, voiced, fricative, nasal]
    """
    if not rows:
        return []

    out: List[List[Any]] = []

    # Initialize first block from row 0
    cur_ph = rows[0][-4]
    cur_v  = rows[0][-3]
    cur_f  = rows[0][-2]
    cur_n  = rows[0][-1]
    vecs: List[List[float]] = []

    def finalize(block_vecs: List[List[float]], ph: str, v: int, f: int, n: int):
        if not block_vecs:
            return
        # Mean over 2D list -> 1D vector
        arr = np.asarray(block_vecs, dtype=np.float32)  # shape (T_block, D)
        mean_vec = arr.mean(axis=0).tolist()
        out.append(mean_vec + [ph, v, f, n])

    for row in rows:
        dims = row[:-4]
        ph   = row[-4]
        v, f, n = row[-3], row[-2], row[-1]

        if ph == cur_ph:
            vecs.append(dims)
        else:
            # close previous block
            finalize(vecs, cur_ph, cur_v, cur_f, cur_n)
            # start new
            cur_ph, cur_v, cur_f, cur_n = ph, v, f, n
            vecs = [dims]

    # close last block
    finalize(vecs, cur_ph, cur_v, cur_f, cur_n)

    return out


# --------------------------
# CSV wrapper (kept for readability)
# --------------------------

def write_to_csv(data: List[List[Any]], path: str) -> None:
    from .utils import write_to_csv as _w
    _w(data, path)
