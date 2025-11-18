#!/usr/bin/env python3
"""
Example usage of the probe dataset creation pipeline
"""

from pathlib import Path
from src.build_dataset_for_probe import create_probe_dataset, create_probe_dataset_batch


# =============================================================================
# Example 1: Create dataset for a single model
# =============================================================================
def example_single_model():
    """Create probe dataset for WavLM Base model"""
    
    stats = create_probe_dataset(
        dataset_path="01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole",
        model_name="wavlm-base",
        output_path="02_OUTPUTS/TIMIT_Outputs/WAVLM_BASE",
        train_split=0.95,
        verbose=True
    )
    
    print("Dataset creation completed!")
    print(f"Total train samples: {sum(stats['train_samples_per_layer'].values())}")
    print(f"Total test samples: {sum(stats['test_samples_per_layer'].values())}")


# =============================================================================
# Example 2: Create datasets for multiple models
# =============================================================================
def example_multiple_models():
    """Create probe datasets for multiple models"""
    
    results = create_probe_dataset_batch(
        dataset_path="01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole",
        model_names=["wavlm-base", "wavlm-large", "hubert-base"],
        output_base_path="02_OUTPUTS/TIMIT_Outputs",
        train_split=0.95,
        verbose=True
    )
    
    print("\nAll models processed!")
    for model_name, stats in results.items():
        if "error" not in stats:
            total_samples = sum(stats['train_samples_per_layer'].values())
            print(f"{model_name}: {total_samples:,} training samples")


# =============================================================================
# Example 3: Create datasets for all available models
# =============================================================================
def example_all_models():
    """Create probe datasets for all supported models"""
    
    from configs.model_config import list_available_models
    
    all_models = list_available_models()
    print(f"Processing {len(all_models)} models: {', '.join(all_models)}")
    
    results = create_probe_dataset_batch(
        dataset_path="01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole",
        model_names=all_models,
        output_base_path="02_OUTPUTS/TIMIT_Outputs",
        train_split=0.95,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for model_name, stats in results.items():
        if "error" in stats:
            print(f"✗ {model_name}: FAILED - {stats['error']}")
        else:
            total = sum(stats['train_samples_per_layer'].values())
            print(f"✓ {model_name}: {total:,} samples")


# =============================================================================
# Example 4: Use with custom dataset (same TIMIT structure)
# =============================================================================
def example_custom_dataset():
    """Use a custom dataset with the same structure as TIMIT"""
    
    # Your custom dataset should have:
    # - .wav files (audio)
    # - .phn files (phoneme alignments)
    # Same naming convention: si1234.wav and si1234.phn
    
    stats = create_probe_dataset(
        dataset_path="path/to/your/custom/dataset",
        model_name="wav2vec2-base",
        output_path="02_OUTPUTS/CustomDataset_Outputs/WAV2VEC2_BASE",
        train_split=0.95,
        verbose=True
    )


# =============================================================================
# Example 5: Programmatic usage with specific features
# =============================================================================
def example_specific_features():
    """Extract only specific phonetic features"""
    
    stats = create_probe_dataset(
        dataset_path="01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole",
        model_name="hubert-large",
        output_path="02_OUTPUTS/TIMIT_Outputs/HUBERT_LARGE",
        train_split=0.95,
        features=["voiced", "nasal"],  # Only these features
        verbose=True
    )


# =============================================================================
# Example 6: Using CPU explicitly (for systems without GPU)
# =============================================================================
def example_cpu_only():
    """Run on CPU (slower but works without GPU)"""
    
    stats = create_probe_dataset(
        dataset_path="01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_sample",
        model_name="wavlm-base",
        output_path="02_OUTPUTS/TIMIT_Outputs/WAVLM_BASE_CPU",
        device="cpu",
        verbose=True
    )


if __name__ == "__main__":
    # Uncomment the example you want to run
    
    # example_single_model()
    # example_multiple_models()
    # example_all_models()
    # example_custom_dataset()
    # example_specific_features()
    # example_cpu_only()
    
    print("Please uncomment one of the examples to run!")
