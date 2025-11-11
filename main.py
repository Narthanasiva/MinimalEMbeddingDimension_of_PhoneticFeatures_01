# main.py
from src.utils import make_dataset_output_folders

if __name__ == "__main__":
    dataset_name = "TIMIT"
    make_dataset_output_folders(dataset_name)
    print(f"Output folder structure created for {dataset_name}")
