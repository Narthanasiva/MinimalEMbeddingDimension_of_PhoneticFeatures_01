"""
Model Configuration Registry
Maps model names to their HuggingFace identifiers and specifications
"""

MODEL_REGISTRY = {
    "wavlm-base": {
        "model_name": "microsoft/wavlm-base",
        "num_layers": 13,  # 0-12 (input + 12 transformer layers)
        "hidden_size": 768,
        "sample_rate": 16000,
    },
    "wavlm-large": {
        "model_name": "microsoft/wavlm-large",
        "num_layers": 25,  # 0-24 (input + 24 transformer layers)
        "hidden_size": 1024,
        "sample_rate": 16000,
    },
    "wav2vec2-base": {
        "model_name": "facebook/wav2vec2-base",
        "num_layers": 13,  # 0-12
        "hidden_size": 768,
        "sample_rate": 16000,
    },
    "wav2vec2-large": {
        "model_name": "facebook/wav2vec2-large",
        "num_layers": 25,  # 0-24
        "hidden_size": 1024,
        "sample_rate": 16000,
    },
    "hubert-base": {
        "model_name": "facebook/hubert-base-ls960",
        "num_layers": 13,  # 0-12
        "hidden_size": 768,
        "sample_rate": 16000,
    },
    "hubert-large": {
        "model_name": "facebook/hubert-large-ll60k",
        "num_layers": 25,  # 0-24
        "hidden_size": 1024,
        "sample_rate": 16000,
    },
}


def get_model_config(model_name: str) -> dict:
    """
    Get configuration for a specific model
    
    Args:
        model_name: One of 'wavlm-base', 'wavlm-large', 'wav2vec2-base', 
                    'wav2vec2-large', 'hubert-base', 'hubert-large'
    
    Returns:
        Dictionary with model configuration
    
    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Available models: {available}"
        )
    return MODEL_REGISTRY[model_name]


def list_available_models():
    """List all available models"""
    return list(MODEL_REGISTRY.keys())
