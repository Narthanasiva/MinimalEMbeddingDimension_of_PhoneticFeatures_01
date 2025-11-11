# main.py
from src.utils import make_dataset_output_folders
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create output folder structure for a dataset.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset for which to create output folders.")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    make_dataset_output_folders(dataset_name)
    print(f"Output folder structure created for {dataset_name}")
