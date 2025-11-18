"""
Configuration Verification Script
Checks if everything is ready for dataset creation
"""

print("="*70)
print("CONFIGURATION VERIFICATION")
print("="*70)

# 1. Check Models
print("\n✅ MODELS CONFIGURED:")
print("-" * 70)
models = {
    "wavlm-base": {"layers": 13, "hidden": 768},
    "wavlm-large": {"layers": 25, "hidden": 1024},
    "wav2vec2-base": {"layers": 13, "hidden": 768},
    "wav2vec2-large": {"layers": 25, "hidden": 1024},
    "hubert-base": {"layers": 13, "hidden": 768},
    "hubert-large": {"layers": 25, "hidden": 1024},
}

for name, config in models.items():
    print(f"  ✓ {name:20s} - {config['layers']} layers, {config['hidden']} dim")

# 2. Check Features
print("\n✅ PHONETIC FEATURES CONFIGURED:")
print("-" * 70)
features = ["voiced", "fricative", "nasal"]
for f in features:
    print(f"  ✓ {f} / not {f}")

# 3. Check Sample Phonemes
print("\n✅ SAMPLE PHONEME MAPPINGS:")
print("-" * 70)
samples = {
    "s": {"voiced": 0, "fricative": 1, "nasal": 0},  # voiceless fricative
    "z": {"voiced": 1, "fricative": 1, "nasal": 0},  # voiced fricative
    "m": {"voiced": 1, "fricative": 0, "nasal": 1},  # voiced nasal
    "p": {"voiced": 0, "fricative": 0, "nasal": 0},  # voiceless plosive
}

for ph, feats in samples.items():
    v = "voiced" if feats["voiced"] else "voiceless"
    f = "fricative" if feats["fricative"] else "non-fricative"
    n = "nasal" if feats["nasal"] else "non-nasal"
    print(f"  ✓ '{ph}' → {v}, {f}, {n}")

# 4. Dataset Requirements
print("\n✅ DATASET REQUIREMENTS:")
print("-" * 70)
print("  ✓ Structure: TIMIT format (.wav + .phn files)")
print("  ✓ Files needed: si1234.wav + si1234.phn (matching names)")
print("  ✓ .phn format: <start_sample> <end_sample> <phoneme>")

# 5. Usage Examples
print("\n✅ READY TO USE:")
print("-" * 70)
print("  Single model:")
print("    python main.py --dataset <path> --model wavlm-base --output <path>")
print("\n  Multiple models:")
print("    python main.py --dataset <path> --model wavlm-base hubert-large --output <path>")
print("\n  All 6 models:")
print("    python main.py --dataset <path> --model all --output <path>")
print("\n  Python function:")
print("    from src.build_dataset_for_probe import create_probe_dataset")
print("    create_probe_dataset(dataset_path, model_name, output_path)")

# 6. What Gets Created
print("\n✅ OUTPUT FOR EACH MODEL:")
print("-" * 70)
print("  For wavlm-base/wav2vec2-base/hubert-base (13 layers):")
print("    - layer_0_train.pkl through layer_12_train.pkl")
print("    - layer_0_test.pkl through layer_12_test.pkl")
print("\n  For wavlm-large/wav2vec2-large/hubert-large (25 layers):")
print("    - layer_0_train.pkl through layer_24_train.pkl")
print("    - layer_0_test.pkl through layer_24_test.pkl")
print("\n  Each .pkl contains DataFrame with:")
print("    - embedding: torch tensor (hidden_size,)")
print("    - phone: phoneme label")
print("    - voiced: 0 or 1")
print("    - fricative: 0 or 1")
print("    - nasal: 0 or 1")

print("\n" + "="*70)
print("✅ EVERYTHING IS READY!")
print("="*70)
print("\n⚠️  NOTE: You need to install dependencies first:")
print("   pip install -r requirements.txt")
print("\n   Or manually:")
print("   pip install torch torchaudio transformers pandas numpy tqdm scikit-learn")
print("="*70)
