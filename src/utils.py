"""
Utility functions for dataset creation
"""

from pathlib import Path
from typing import List, Tuple
import torch


def read_phn(pfile: Path, sample_rate: int = 16000) -> List[Tuple[float, float, str]]:
    """
    Read a TIMIT .PHN file and return phoneme time intervals
    
    Args:
        pfile: Path to .phn file
        sample_rate: Audio sample rate (default: 16000)
    
    Returns:
        List of (start_time, end_time, phoneme) tuples in seconds
    """
    rows: List[Tuple[float, float, str]] = []
    
    with open(pfile) as fh:
        for line in fh:
            s, e, ph = line.strip().split()
            rows.append((int(s) / sample_rate, int(e) / sample_rate, ph))
    
    return rows


def frame_bounds(n: int, frame_shift: float = 0.020, frame_len: float = 0.025) -> List[Tuple[float, float]]:
    """
    Calculate time bounds for each frame
    
    Args:
        n: Number of frames
        frame_shift: Time shift between frames in seconds
        frame_len: Frame length in seconds
    
    Returns:
        List of (start_time, end_time) tuples for each frame
    """
    return [(i * frame_shift, i * frame_shift + frame_len) for i in range(n)]


def phone_avg(layer_embs: torch.Tensor, times: List[Tuple[float, float]], 
              phone_intervals: List[Tuple[float, float, str]]):
    """
    Average embeddings for each phoneme segment
    
    Args:
        layer_embs: Tensor of shape (T, D) with frame embeddings
        times: List of (start, end) tuples for each frame
        phone_intervals: List of (start, end, phoneme) tuples
    
    Yields:
        (phoneme, averaged_embedding) pairs
    """
    for ps, pe, ph in phone_intervals:
        mask = torch.tensor([(t0 >= ps) and (t1 <= pe) for t0, t1 in times])
        if mask.any():
            yield ph, layer_embs[mask].mean(dim=0)


def phone_frames(layer_embs: torch.Tensor, times: List[Tuple[float, float]], 
                 phone_intervals: List[Tuple[float, float, str]]):
    """
    Assign each frame to its corresponding phoneme
    
    Args:
        layer_embs: Tensor of shape (T, D) with frame embeddings
        times: List of (start, end) tuples for each frame
        phone_intervals: List of (start, end, phoneme) tuples
    
    Yields:
        (phoneme, frame_embedding) pairs
    """
    phone_idx = 0
    for idx, (t0, t1) in enumerate(times):
        while phone_idx < len(phone_intervals) - 1 and t0 >= phone_intervals[phone_idx][1]:
            phone_idx += 1
        yield phone_intervals[phone_idx][2], layer_embs[idx]


def label_counts(rows: List[dict], features: List[str]):
    """
    Count label distribution for each feature
    
    Args:
        rows: List of data rows with feature labels
        features: List of feature names to count
    
    Returns:
        Dict mapping feature -> {label: count}
    """
    import collections
    c = {f: collections.Counter() for f in features}
    for r in rows:
        for f in features:
            c[f][r[f]] += 1
    return {f: dict(c[f]) for f in features}
