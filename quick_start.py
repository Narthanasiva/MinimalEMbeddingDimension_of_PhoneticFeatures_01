"""
Quick start script - demonstrates basic usage
"""

from src.build_dataset_for_probe import create_probe_dataset

# Simple example: Create dataset for WavLM Base
print("Creating probe dataset for WavLM Base...\n")

stats = create_probe_dataset(
    dataset_path="01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole",
    model_name="wavlm-base",
    output_path="02_OUTPUTS/TIMIT_Outputs/WAVLM_BASE",
    train_split=0.95,
    verbose=True
)

print("\n✅ Dataset created successfully!")
print(f"\nTotal training samples: {sum(stats['train_samples_per_layer'].values()):,}")
print(f"Total test samples: {sum(stats['test_samples_per_layer'].values()):,}")
