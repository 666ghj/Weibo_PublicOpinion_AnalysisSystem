# -*- coding: utf-8 -*-
"""
Qwen3模型配置文件
定义不同规模的模型参数和配置
"""

# Qwen3模型配置
QWEN3_MODELS = {
    "0.6B": {
        "base_model": "Qwen/Qwen3-0.6B",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dim": 1024,
        "max_length": 32768,
        "recommended_batch_size": 32,
        "recommended_lr": 1e-3,
        "lora_r": 16,
        "lora_alpha": 32
    },
    "4B": {
        "base_model": "Qwen/Qwen3-4B",
        "embedding_model": "Qwen/Qwen3-Embedding-4B", 
        "embedding_dim": 2560,
        "max_length": 32768,
        "recommended_batch_size": 16,
        "recommended_lr": 5e-4,
        "lora_r": 32,
        "lora_alpha": 64
    },
    "8B": {
        "base_model": "Qwen/Qwen3-8B",
        "embedding_model": "Qwen/Qwen3-Embedding-8B",
        "embedding_dim": 4096,
        "max_length": 32768,
        "recommended_batch_size": 8,
        "recommended_lr": 2e-4,
        "lora_r": 64,
        "lora_alpha": 128
    }
}

# 模型文件路径配置
MODEL_PATHS = {
    "embedding": {
        "0.6B": "./models/qwen3_embedding_0.6b_sentiment.pth",
        "4B": "./models/qwen3_embedding_4b_sentiment.pth", 
        "8B": "./models/qwen3_embedding_8b_sentiment.pth"
    },
    "lora": {
        "0.6B": "./models/qwen3_lora_0.6b_final",
        "4B": "./models/qwen3_lora_4b_final",
        "8B": "./models/qwen3_lora_8b_final"
    }
}