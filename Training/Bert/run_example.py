#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT微调示例运行脚本
演示如何使用finetune_bert.py进行模型微调
"""

import subprocess
import sys
import os

def run_training():
    """运行BERT微调训练"""
    
    # 检查数据文件是否存在
    if not os.path.exists("sample_data.json"):
        print("错误：找不到示例数据文件 sample_data.json")
        return
    
    # 构建训练命令
    cmd = [
        sys.executable, "finetune_bert.py",
        "--data_path", "sample_data.json",
        "--text_column", "text", 
        "--label_column", "label",
        "--model_name", "bert-base-chinese",
        "--output_dir", "./bert_finetuned",
        "--num_train_epochs", "2",
        "--per_device_train_batch_size", "4",
        "--learning_rate", "2e-5",
        "--max_length", "128"
    ]
    
    print("开始BERT微调训练...")
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # 运行训练命令
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("训练完成！")
        print("输出:", result.stdout)
        
    except subprocess.CalledProcessError as e:
        print("训练失败！")
        print("错误信息:", e.stderr)
        print("返回码:", e.returncode)
        
    except FileNotFoundError:
        print("错误：找不到Python解释器或finetune_bert.py文件")
        print("请确保已安装所需依赖：pip install -r requirements.txt")

if __name__ == "__main__":
    run_training()
