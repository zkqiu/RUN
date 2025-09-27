#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT (Supervised Fine-Tuning) LLM训练代码
支持ChatML格式数据
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import logging
from tqdm import tqdm
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-0.5B",
        metadata={"help": "预训练模型路径或名称"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )

@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        default="./data/chatml_data.json",
        metadata={"help": "训练数据路径"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"}
    )
    ignore_index: int = field(
        default=-100,
        metadata={"help": "忽略的标签索引"}
    )

@dataclass
class TrainingArguments:
    """训练相关参数"""
    output_dir: str = field(
        default="./outputs/sft_model",
        metadata={"help": "输出目录"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "每设备训练批次大小"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "每设备评估批次大小"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "学习率"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "权重衰减"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "预热比例"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "日志记录步数"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "保存步数"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "评估步数"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "保存模型数量限制"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "是否使用fp16"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "数据加载器工作进程数"}
    )

class ChatMLDataset(Dataset):
    """ChatML格式数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"加载了 {len(self.data)} 条训练数据")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 解析ChatML格式
        messages = item.get('messages', [])
        if not messages:
            raise ValueError(f"数据项 {idx} 缺少messages字段")
        
        # 构建对话文本
        conversation = self._format_conversation(messages)
        
        # 编码文本
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        # 创建标签（只对assistant的回复计算损失）
        labels = self._create_labels(messages, input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        """格式化对话为训练文本"""
        conversation = ""
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                conversation += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                conversation += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        return conversation
    
    def _create_labels(self, messages: List[Dict], input_ids: torch.Tensor) -> torch.Tensor:
        """创建训练标签，只对assistant回复计算损失"""
        labels = input_ids.clone()
        
        # 找到assistant回复的位置
        text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        assistant_start = text.find("<|im_start|>assistant")
        
        if assistant_start != -1:
            # 找到assistant开始token的位置
            assistant_tokens = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
            assistant_start_idx = self._find_token_position(input_ids, assistant_tokens)
            
            if assistant_start_idx != -1:
                # 在assistant回复之前的所有token都设为ignore_index
                labels[:assistant_start_idx] = self.ignore_index
                
                # 找到assistant结束的位置
                assistant_end = text.find("<|im_end|>", assistant_start)
                if assistant_end != -1:
                    end_tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
                    assistant_end_idx = self._find_token_position(input_ids, end_tokens, assistant_start_idx)
                    
                    if assistant_end_idx != -1:
                        # assistant结束后的token也设为ignore_index
                        labels[assistant_end_idx:] = self.ignore_index
        else:
            # 如果没有找到assistant回复，所有标签都设为ignore_index
            labels.fill_(self.ignore_index)
        
        return labels
    
    def _find_token_position(self, input_ids: torch.Tensor, target_tokens: List[int], start_idx: int = 0) -> int:
        """在input_ids中查找目标token序列的位置"""
        for i in range(start_idx, len(input_ids) - len(target_tokens) + 1):
            if torch.equal(input_ids[i:i+len(target_tokens)], torch.tensor(target_tokens)):
                return i
        return -1

class SFTTrainer:
    """SFT训练器"""
    
    def __init__(self, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
            device_map="auto"
        )
        
        # 调整模型配置
        self.model.config.use_cache = False
        
    def prepare_dataset(self):
        """准备训练和验证数据集"""
        # 加载训练数据
        train_dataset = ChatMLDataset(
            data_path=self.data_args.data_path,
            tokenizer=self.tokenizer,
            max_length=self.data_args.max_length,
            ignore_index=self.data_args.ignore_index
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        return train_dataset, data_collator
    
    def train(self):
        """开始训练"""
        # 准备数据
        train_dataset, data_collator = self.prepare_dataset()
        
        # 创建训练参数
        training_args = TrainingArguments(
            output_dir=self.training_args.output_dir,
            num_train_epochs=self.training_args.num_train_epochs,
            per_device_train_batch_size=self.training_args.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
            learning_rate=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay,
            warmup_ratio=self.training_args.warmup_ratio,
            logging_steps=self.training_args.logging_steps,
            save_steps=self.training_args.save_steps,
            eval_steps=self.training_args.eval_steps,
            save_total_limit=self.training_args.save_total_limit,
            fp16=self.training_args.fp16,
            dataloader_num_workers=self.training_args.dataloader_num_workers,
            remove_unused_columns=False,
            report_to=None,  # 禁用wandb等
            save_strategy="steps",
            evaluation_strategy="no",  # 暂时不进行评估
            load_best_model_at_end=False,
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        logger.info("开始SFT训练...")
        trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        
        logger.info(f"训练完成，模型已保存到 {self.training_args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="SFT LLM训练")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="模型名称")
    parser.add_argument("--data_path", type=str, default="./data/chatml_data.json", help="数据路径")
    parser.add_argument("--output_dir", type=str, default="./outputs/sft_model", help="输出目录")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    
    args = parser.parse_args()
    
    # 创建参数对象
    model_args = ModelArguments(model_name_or_path=args.model_name)
    data_args = DataArguments(data_path=args.data_path, max_length=args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # 创建训练器并开始训练
    trainer = SFTTrainer(model_args, data_args, training_args)
    trainer.train()

if __name__ == "__main__":
    main()
