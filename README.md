# Minimal Embedding Dimension of Phonetic Features

Automated pipeline for creating probe datasets for phonetic feature analysis using self-supervised speech models.

## 🎯 Features

- **Multi-Model Support**: WavLM, Wav2Vec2, HuBERT (Base & Large variants)
- **Dataset Agnostic**: Works with any TIMIT-structured dataset
- **Single Function Call**: Create probe datasets with one command
- **Layer-wise Extraction**: Extracts embeddings from all transformer layers
- **Phonetic Features**: Analyzes voiced, fricative, and nasal features

## 📦 Supported Models

| Model | Identifier | Layers | Hidden Size |
|-------|-----------|--------|-------------|
| WavLM Base | `wavlm-base` | 13 | 768 |
| WavLM Large | `wavlm-large` | 25 | 1024 |
| Wav2Vec2 Base | `wav2vec2-base` | 13 | 768 |
| Wav2Vec2 Large | `wav2vec2-large` | 25 | 1024 |
| HuBERT Base | `hubert-base` | 13 | 768 |
| HuBERT Large | `hubert-large` | 25 | 1024 |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Command Line Usage

**Single Model:**
```bash
python main.py \
  --dataset ./01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole \
  --model wavlm-base \
  --output ./02_OUTPUTS/TIMIT_Outputs
```

**Multiple Models:**
```bash
python main.py \
  --dataset ./01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole \
  --model wavlm-base wavlm-large hubert-base \
  --output ./02_OUTPUTS/TIMIT_Outputs
```

**All Models:**
```bash
python main.py \
  --dataset ./01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole \
  --model all \
  --output ./02_OUTPUTS/TIMIT_Outputs
```

### Programmatic Usage

```python
from src.build_dataset_for_probe import create_probe_dataset

# Single model
stats = create_probe_dataset(
    dataset_path="01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole",
    model_name="wavlm-base",
    output_path="02_OUTPUTS/TIMIT_Outputs/WAVLM_BASE",
    train_split=0.95,
    verbose=True
)

# Multiple models
from src.build_dataset_for_probe import create_probe_dataset_batch

results = create_probe_dataset_batch(
    dataset_path="01_Raw_Phonetic_Annotated_Datasets/01_TIMIT_raw_dataset_whole",
    model_names=["wavlm-base", "wavlm-large", "hubert-base"],
    output_base_path="02_OUTPUTS/TIMIT_Outputs",
)
```

## 📁 Dataset Structure

Your dataset should follow the TIMIT structure:

```
dataset_folder/
├── si1234.wav    # Audio file
├── si1234.phn    # Phoneme alignments
├── si5678.wav
├── si5678.phn
└── ...
```

### .phn File Format
```
0 3050 h#
3050 4559 sh
4559 5723 ix
...
```
Each line: `start_sample end_sample phoneme`

## 🔧 Advanced Options

```bash
python main.py \
  --dataset <path> \
  --model <model-name> \
  --output <path> \
  --train-split 0.95 \        # Train/test split ratio
  --device cuda \              # cuda, cpu, or auto
  --frame-shift 0.020 \        # Frame shift in seconds
  --frame-len 0.025 \          # Frame length in seconds
  --features voiced nasal \    # Specific features to extract
  --quiet                      # Suppress progress output
```

## 📊 Output Format

For each model and layer, two pickle files are created:

```
02_OUTPUTS/TIMIT_Outputs/WAVLM_BASE/
├── layer_0_train.pkl
├── layer_0_test.pkl
├── layer_1_train.pkl
├── layer_1_test.pkl
└── ...
```

Each file contains a pandas DataFrame with:
- `embedding`: Torch tensor of shape (hidden_size,)
- `phone`: Phoneme label (e.g., "s", "m", "iy")
- `voiced`: Binary label (0/1)
- `fricative`: Binary label (0/1)
- `nasal`: Binary label (0/1)

## 💡 Use Cases

1. **Phonetic Feature Probing**: Train linear probes to analyze what phonetic features are encoded in each layer
2. **Layer Comparison**: Compare how different layers encode phonetic information
3. **Model Comparison**: Compare phonetic encoding across different model architectures
4. **Custom Datasets**: Use your own phonetically annotated speech data

## 📚 Examples

See `examples/example_usage.py` for detailed usage examples:
- Single model processing
- Batch processing multiple models
- Custom datasets
- Specific feature extraction
- CPU-only execution

## 🏗️ Project Structure

```
.
├── main.py                          # CLI entry point
├── requirements.txt                 # Dependencies
├── configs/
│   └── model_config.py             # Model registry
├── src/
│   ├── build_dataset_for_probe.py  # Main pipeline - Extracts frame-level embeddings + labels
│   ├── build_probe.py              # Trains probe classifiers (e.g., 768 → 200 → 1 MLP)
│   ├── compute.py                  # Computes interpretability scores (e.g., Integrated Gradients)
│   ├── is_subset_existence.py      # Analyzes minimal dimension subset per feature/layer
│   ├── phoneme_features.py         # Phoneme-feature mapping
│   └── utils.py                    # Common helpers (path builders, saving/loading, etc.)
└── examples/
    └── example_usage.py            # Usage examples
```

## ⚙️ Configuration

### Adding a New Model

Edit `configs/model_config.py`:

```python
MODEL_REGISTRY = {
    "your-model": {
        "model_name": "org/model-name",
        "num_layers": 13,
        "hidden_size": 768,
        "sample_rate": 16000,
    },
}
```

### Adding New Phonetic Features

Edit `src/phoneme_features.py` to add new feature sets.

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add support for new models
- Add new phonetic features
- Improve documentation
- Report issues

## 📄 License

[Your License Here]

## 📧 Contact

[Your Contact Information]
