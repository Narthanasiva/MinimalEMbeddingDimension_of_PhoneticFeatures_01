"""
main.py
-------
Simple CLI runner to call different stages of your pipeline.
Right now, we only implement --stage build_dataset.
"""

import argparse
from src.build_dataset_for_probe import run_build_dataset_for_probe

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, required=True, choices=["build_dataset"])
    ap.add_argument("--dataset", type=str, required=True, help="e.g., TIMIT")
    ap.add_argument("--dataset_path", type=str, required=True, help="root path of dataset audio")
    ap.add_argument("--models", nargs="+", default=[
        "wavlm_base", "wavlm_large",
        "wav2vec2_base", "wav2vec2_large",
        "hubert_base", "hubert_large",
    ])
    ap.add_argument("--train_ratio", type=float, default=0.95)
    return ap.parse_args()

def main():
    args = parse_args()
    if args.stage == "build_dataset":
        run_build_dataset_for_probe(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            model_list=args.models,
            train_ratio=args.train_ratio
        )

if __name__ == "__main__":
    main()
