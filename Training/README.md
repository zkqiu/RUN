# BERT微调脚本

这是一个使用Hugging Face Trainer进行BERT模型微调的完整脚本，支持自定义数据集。

## 功能特点

- 支持多种数据格式：JSON、JSONL、CSV、TXT
- 自动标签映射和编码
- 支持早停机制
- 集成wandb日志记录
- 自动评估和指标计算
- 支持自定义超参数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python finetune_bert.py --data_path sample_data.json --text_column text --label_column label
```

### 完整参数示例

```bash
python finetune_bert.py \
    --data_path sample_data.json \
    --text_column text \
    --label_column label \
    --test_data_path test_data.json \
    --model_name bert-base-chinese \
    --output_dir ./my_bert_model \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --use_wandb
```

## 数据格式

### JSON格式
```json
[
    {"text": "这是一个正面的评论", "label": "positive"},
    {"text": "这是一个负面的评论", "label": "negative"}
]
```

### JSONL格式
```jsonl
{"text": "这是一个正面的评论", "label": "positive"}
{"text": "这是一个负面的评论", "label": "negative"}
```

### CSV格式
```csv
text,label
"这是一个正面的评论",positive
"这是一个负面的评论",negative
```

## 参数说明

### 数据相关
- `--data_path`: 训练数据文件路径
- `--text_column`: 文本列名（默认：text）
- `--label_column`: 标签列名（默认：label）
- `--test_data_path`: 测试数据文件路径（可选）

### 模型相关
- `--model_name`: 预训练模型名称（默认：bert-base-chinese）
- `--max_length`: 最大序列长度（默认：512）

### 训练相关
- `--output_dir`: 输出目录（默认：./bert_finetuned）
- `--num_train_epochs`: 训练轮数（默认：3）
- `--per_device_train_batch_size`: 训练批次大小（默认：16）
- `--per_device_eval_batch_size`: 评估批次大小（默认：16）
- `--learning_rate`: 学习率（默认：2e-5）
- `--warmup_steps`: 预热步数（默认：500）
- `--weight_decay`: 权重衰减（默认：0.01）

### 其他
- `--use_wandb`: 启用wandb日志记录
- `--early_stopping_patience`: 早停耐心值（默认：3）
- `--seed`: 随机种子（默认：42）

## 输出文件

训练完成后，输出目录将包含：
- `pytorch_model.bin`: 模型权重
- `config.json`: 模型配置
- `tokenizer.json`: 分词器文件
- `label_mapping.json`: 标签映射
- `eval_results.json`: 评估结果
- `logs/`: 训练日志

## 使用训练好的模型

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# 加载模型和分词器
model_path = "./my_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 加载标签映射
with open(f"{model_path}/label_mapping.json", 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)

# 预测
text = "这是一条测试文本"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=-1)
predicted_label_id = torch.argmax(predictions, dim=-1).item()
predicted_label = label_mapping["id2label"][str(predicted_label_id)]

print(f"预测标签: {predicted_label}")
```
