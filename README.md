| File                         | Purpose                                                                     |
| ---------------------------- | --------------------------------------------------------------------------- |
| `build_dataset_for_probe.py` | Extracts frame-level embeddings + labels from audio; saves as `.csv`        |
| `build_probe.py`             | Trains probe classifiers (e.g., 768 → 200 → 1 MLP); saves models + scores   |
| `compute.py`                 | Computes interpretability scores (e.g., Integrated Gradients)               |
| `is_subset_existence.py`     | Analyzes whether a minimal dimension subset exists per feature/layer        |
| `utils.py`                   | Common helpers (path builders, saving/loading, logging, etc.)               |
| `__init__.py`                | 📦 Declares this folder as a Python package; helps you **import functions** |
