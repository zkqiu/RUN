#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT训练配置文件
"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    trust_remote_code: bool = True
    torch_dtype: str = "float16"  # float16, float32, bfloat16
    device_map: str = "auto"
    use_cache: bool = False

@dataclass
class DataConfig:
    """数据配置"""
    train_data_path: str = "./data/chatml_data.json"
    eval_data_path: Optional[str] = None
    max_length: int = 2048
    ignore_index: int = -100
    data_loader_num_workers: int = 4
    dataloader_pin_memory: bool = True

@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "./outputs/sft_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # 日志和保存
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    save_strategy: str = "steps"
    evaluation_strategy: str = "no"
    
    # 优化器
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    
    # 精度
    fp16: bool = True
    bf16: bool = False
    
    # 其他
    remove_unused_columns: bool = False
    report_to: Optional[str] = None
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

@dataclass
class InferenceConfig:
    """推理配置"""
    model_path: str = "./outputs/sft_model"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

# 预定义配置
CONFIGS = {
    "qwen2.5-0.5b": {
        "model": ModelConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            torch_dtype="float16"
        ),
        "data": DataConfig(
            max_length=2048
        ),
        "training": TrainingConfig(
            per_device_train_batch_size=8,
            learning_rate=5e-5,
            num_train_epochs=3
        )
    },
    
    "qwen2.5-1.5b": {
        "model": ModelConfig(
            model_name="Qwen/Qwen2.5-1.5B",
            torch_dtype="float16"
        ),
        "data": DataConfig(
            max_length=2048
        ),
        "training": TrainingConfig(
            per_device_train_batch_size=4,
            learning_rate=3e-5,
            num_train_epochs=3
        )
    },
    
    "qwen2.5-7b": {
        "model": ModelConfig(
            model_name="Qwen/Qwen2.5-7B",
            torch_dtype="bfloat16"
        ),
        "data": DataConfig(
            max_length=2048
        ),
        "training": TrainingConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            num_train_epochs=2
        )
    },
    
    "llama3-8b": {
        "model": ModelConfig(
            model_name="meta-llama/Llama-3-8B",
            torch_dtype="bfloat16"
        ),
        "data": DataConfig(
            max_length=2048
        ),
        "training": TrainingConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            num_train_epochs=2
        )
    }
}

def get_config(config_name: str = "qwen2.5-0.5b"):
    """获取预定义配置"""
    if config_name not in CONFIGS:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name]

def create_custom_config(
    model_name: str,
    output_dir: str = "./outputs/sft_model",
    data_path: str = "./data/chatml_data.json",
    **kwargs
):
    """创建自定义配置"""
    config = {
        "model": ModelConfig(model_name=model_name),
        "data": DataConfig(train_data_path=data_path),
        "training": TrainingConfig(output_dir=output_dir)
    }
    
    # 更新自定义参数
    for key, value in kwargs.items():
        if key in ["num_epochs", "batch_size", "learning_rate", "max_length"]:
            if key == "num_epochs":
                config["training"].num_train_epochs = value
            elif key == "batch_size":
                config["training"].per_device_train_batch_size = value
            elif key == "learning_rate":
                config["training"].learning_rate = value
            elif key == "max_length":
                config["data"].max_length = value
        else:
            # 其他参数直接设置到training配置中
            if hasattr(config["training"], key):
                setattr(config["training"], key, value)
    
    return config
