"""
Phoneme to Phonetic Feature Mapping
Maps TIMIT phonemes to binary phonetic features
"""

from typing import Dict

# Set of all phonemes that are voiced
_voiced = {
    "b", "d", "g", "dx", "jh", "z", "zh", "v", "dh",
    "m", "n", "ng", "em", "en", "eng", "nx",
    "l", "r", "w", "y", "hh", "hv", "el",
    "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao",
    "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h",
}

# Set of fricative phonemes
_fric = {"s", "sh", "z", "zh", "f", "th", "v", "dh", "hh", "hv"}

# Set of nasal phonemes
_nasal = {"m", "n", "ng", "em", "en", "eng", "nx"}


def build_feature_dict() -> Dict[str, Dict[str, int]]:
    """
    Build a dictionary mapping each TIMIT phoneme to its phonetic features
    
    Returns:
        Dict mapping phoneme -> {feature_name: binary_value}
        Example: {"s": {"voiced": 0, "fricative": 1, "nasal": 0}}
    """
    phones = {
        "b", "d", "g", "p", "t", "k", "bcl", "dcl", "gcl", "pcl", "tcl", "kcl",
        "dx", "q", "jh", "ch", "s", "sh", "z", "zh", "f", "th", "v", "dh",
        "m", "n", "ng", "em", "en", "eng", "nx",
        "l", "r", "w", "y", "hh", "hv", "el",
        "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow",
        "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h",
        "pau", "epi", "h#"
    }

    table = {}
    for ph in phones:
        table[ph] = {
            "voiced": int(ph in _voiced),
            "fricative": int(ph in _fric),
            "nasal": int(ph in _nasal),
        }
    
    return table


# Pre-built feature dictionary
PHONEME_FEATURES = build_feature_dict()

# List of feature names
FEATURE_NAMES = ["voiced", "fricative", "nasal"]
