#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT微调脚本
使用Hugging Face Trainer进行BERT模型微调，支持自定义数据集
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDataset:
    """自定义数据集类，支持多种数据格式"""
    
    def __init__(self, data_path: str, text_column: str = "text", label_column: str = "label"):
        """
        初始化自定义数据集
        
        Args:
            data_path: 数据文件路径 (支持.json, .jsonl, .csv, .txt)
            text_column: 文本列名
            label_column: 标签列名
        """
        self.data_path = data_path
        self.text_column = text_column
        self.label_column = label_column
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """加载数据文件"""
        data_path = Path(self.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
            
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.suffix == '.jsonl':
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif data_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        elif data_path.suffix == '.txt':
            # 假设是简单的文本文件，每行一个样本
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data.append({
                        self.text_column: line.strip(),
                        self.label_column: 0  # 默认标签
                    })
        else:
            raise ValueError(f"不支持的文件格式: {data_path.suffix}")
            
        logger.info(f"成功加载 {len(data)} 个样本")
        return data
    
    def get_dataset(self, tokenizer, max_length: int = 512) -> Dataset:
        """将数据转换为Hugging Face Dataset格式"""
        
        def tokenize_function(examples):
            return tokenizer(
                examples[self.text_column],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # 创建Dataset
        dataset = Dataset.from_list(self.data)
        
        # 获取标签映射
        if self.data:
            unique_labels = sorted(list(set([item[self.label_column] for item in self.data])))
            self.label2id = {label: i for i, label in enumerate(unique_labels)}
            self.id2label = {i: label for i, label in enumerate(unique_labels)}
            logger.info(f"标签映射: {self.label2id}")
        else:
            self.label2id = {0: 0}
            self.id2label = {0: 0}
        
        # 转换标签为数字
        def map_labels(examples):
            examples[self.label_column] = [self.label2id[label] for label in examples[self.label_column]]
            return examples
        
        dataset = dataset.map(map_labels, batched=True)
        
        # 分词
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    parser = argparse.ArgumentParser(description='BERT微调脚本')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--text_column', type=str, default='text', help='文本列名')
    parser.add_argument('--label_column', type=str, default='label', help='标签列名')
    parser.add_argument('--test_data_path', type=str, help='测试数据路径（可选）')
    
    # 模型相关参数
    parser.add_argument('--model_name', type=str, default='bert-base-chinese', 
                       help='预训练模型名称')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    
    # 训练相关参数
    parser.add_argument('--output_dir', type=str, default='./bert_finetuned', help='输出目录')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16, help='训练批次大小')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16, help='评估批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=500, help='预热步数')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--logging_steps', type=int, default=100, help='日志记录步数')
    parser.add_argument('--eval_steps', type=int, default=500, help='评估步数')
    parser.add_argument('--save_steps', type=int, default=500, help='保存步数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb记录')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='早停耐心值')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 初始化wandb（如果启用）
    if args.use_wandb:
        wandb.init(project="bert-finetune", config=vars(args))
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载分词器
    logger.info(f"加载分词器: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 加载自定义数据集
    logger.info("加载训练数据...")
    train_dataset = CustomDataset(
        data_path=args.data_path,
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    train_data = train_dataset.get_dataset(tokenizer, args.max_length)
    
    # 加载测试数据（如果提供）
    eval_data = None
    if args.test_data_path:
        logger.info("加载测试数据...")
        test_dataset = CustomDataset(
            data_path=args.test_data_path,
            text_column=args.text_column,
            label_column=args.label_column
        )
        # 使用相同的标签映射
        test_dataset.label2id = train_dataset.label2id
        test_dataset.id2label = train_dataset.id2label
        eval_data = test_dataset.get_dataset(tokenizer, args.max_length)
    
    # 如果没有测试数据，从训练数据中分割一部分作为验证集
    if eval_data is None:
        train_data = train_data.train_test_split(test_size=0.1, seed=args.seed)
        eval_data = train_data['test']
        train_data = train_data['train']
    
    # 加载模型
    logger.info(f"加载模型: {args.model_name}")
    num_labels = len(train_dataset.label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=train_dataset.id2label,
        label2id=train_dataset.label2id
    )
    
    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb" if args.use_wandb else None,
        seed=args.seed,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存最终模型
    logger.info("保存最终模型...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存标签映射
    with open(os.path.join(args.output_dir, "label_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "label2id": train_dataset.label2id,
            "id2label": train_dataset.id2label
        }, f, ensure_ascii=False, indent=2)
    
    # 最终评估
    logger.info("进行最终评估...")
    eval_results = trainer.evaluate()
    logger.info(f"最终评估结果: {eval_results}")
    
    # 保存评估结果
    with open(os.path.join(args.output_dir, "eval_results.json"), 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"训练完成！模型保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
