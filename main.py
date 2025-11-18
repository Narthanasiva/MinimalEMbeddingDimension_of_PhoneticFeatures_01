#!/usr/bin/env python3
"""
CLI for creating probe datasets for phonetic feature analysis
"""

import argparse
from pathlib import Path
from src.build_dataset_for_probe import create_probe_dataset, create_probe_dataset_batch
from configs.model_config import list_available_models


def main():
    parser = argparse.ArgumentParser(
        description="Create probe datasets for phonetic feature analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model
  python main.py --dataset ./01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole \\
                 --model wavlm-base \\
                 --output ./02_OUTPUTS/TIMIT_Outputs

  # Multiple models
  python main.py --dataset ./01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole \\
                 --model wavlm-base wavlm-large hubert-base \\
                 --output ./02_OUTPUTS/TIMIT_Outputs

  # All models
  python main.py --dataset ./01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole \\
                 --model all \\
                 --output ./02_OUTPUTS/TIMIT_Outputs
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset folder containing .wav and .phn files"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=True,
        help=f"Model name(s) or 'all'. Available: {', '.join(list_available_models())}"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Base output path (subdirectories created per model)"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Fraction of data for training (default: 0.95)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto)"
    )
    
    parser.add_argument(
        "--frame-shift",
        type=float,
        default=0.020,
        help="Frame shift in seconds (default: 0.020)"
    )
    
    parser.add_argument(
        "--frame-len",
        type=float,
        default=0.025,
        help="Frame length in seconds (default: 0.025)"
    )
    
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help="Phonetic features to extract (default: all)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Handle model selection
    if "all" in args.model:
        model_names = list_available_models()
    else:
        model_names = args.model
        # Validate model names
        available = set(list_available_models())
        invalid = set(model_names) - available
        if invalid:
            parser.error(f"Invalid model(s): {', '.join(invalid)}. "
                        f"Available: {', '.join(available)}")
    
    # Handle device
    device = None if args.device == "auto" else args.device
    
    # Common kwargs
    common_kwargs = {
        "train_split": args.train_split,
        "frame_shift": args.frame_shift,
        "frame_len": args.frame_len,
        "device": device,
        "features": args.features,
        "verbose": not args.quiet,
    }
    
    # Process
    if len(model_names) == 1:
        # Single model
        output_path = Path(args.output) / model_names[0].replace("-", "_").upper()
        create_probe_dataset(
            dataset_path=args.dataset,
            model_name=model_names[0],
            output_path=output_path,
            **common_kwargs
        )
    else:
        # Multiple models
        create_probe_dataset_batch(
            dataset_path=args.dataset,
            model_names=model_names,
            output_base_path=args.output,
            **common_kwargs
        )
    
    print("\n✅ All datasets created successfully!")


if __name__ == "__main__":
    main()
