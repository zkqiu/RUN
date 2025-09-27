#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT训练启动脚本
"""

import os
import sys
import argparse
from data_utils import create_sample_data
from sft_train import main as train_main

def setup_data():
    """设置示例数据"""
    data_dir = "./data"
    data_path = os.path.join(data_dir, "chatml_data.json")
    
    if not os.path.exists(data_path):
        print("创建示例数据...")
        os.makedirs(data_dir, exist_ok=True)
        create_sample_data()
        print(f"示例数据已创建: {data_path}")
    else:
        print(f"数据文件已存在: {data_path}")

def main():
    parser = argparse.ArgumentParser(description="SFT训练启动脚本")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="模型名称")
    parser.add_argument("--data_path", type=str, default="./data/chatml_data.json", help="数据路径")
    parser.add_argument("--output_dir", type=str, default="./outputs/sft_model", help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--setup_data", action="store_true", help="是否创建示例数据")
    
    args = parser.parse_args()
    
    # 设置数据
    if args.setup_data:
        setup_data()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"数据文件不存在: {args.data_path}")
        print("请使用 --setup_data 创建示例数据，或提供正确的数据路径")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建训练命令
    train_args = [
        "--model_name", args.model_name,
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--num_epochs", str(args.num_epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--max_length", str(args.max_length)
    ]
    
    # 设置sys.argv以传递给训练脚本
    sys.argv = ["sft_train.py"] + train_args
    
    print("开始SFT训练...")
    print(f"模型: {args.model_name}")
    print(f"数据: {args.data_path}")
    print(f"输出: {args.output_dir}")
    print(f"轮数: {args.num_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print("-" * 50)
    
    # 启动训练
    train_main()

if __name__ == "__main__":
    main()
