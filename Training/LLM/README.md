# SFT LLM 训练代码

这是一个用于大语言模型监督微调（Supervised Fine-Tuning, SFT）的完整训练框架，支持ChatML格式数据。

## 功能特性

- 🚀 支持多种预训练模型（Qwen2.5、Llama3等）
- 📝 完整的ChatML格式数据处理
- ⚙️ 灵活的配置系统
- 🔧 支持多种训练优化技术
- 💬 交互式推理和批量推理
- 📊 完整的训练监控和日志

## 文件结构

```text
Training/LLM/
├── sft_train.py          # 主训练脚本
├── data_utils.py         # 数据处理工具
├── config.py             # 配置文件
├── inference.py          # 推理脚本
├── requirements.txt      # 依赖包
├── README.md            # 说明文档
└── data/                # 数据目录
    └── chatml_data.json # 示例数据
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

使用ChatML格式准备训练数据：

```python
from data_utils import ChatMLDataProcessor

processor = ChatMLDataProcessor()

# 创建示例数据
sample_data = [
    processor.create_chatml_sample(
        system="你是一个有用的AI助手。",
        user="什么是机器学习？",
        assistant="机器学习是人工智能的一个分支..."
    ),
    # 更多数据...
]

# 保存数据
processor.save_chatml_data(sample_data, "./data/chatml_data.json")
```

或者直接运行数据生成脚本：

```bash
python data_utils.py
```

### 3. 开始训练

#### 使用默认配置训练

```bash
python sft_train.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_path "./data/chatml_data.json" \
    --output_dir "./outputs/sft_model" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5
```

#### 使用预定义配置

```python
from config import get_config
from sft_train import SFTTrainer

# 获取预定义配置
config = get_config("qwen2.5-0.5b")

# 创建训练器
trainer = SFTTrainer(
    model_args=config["model"],
    data_args=config["data"], 
    training_args=config["training"]
)

# 开始训练
trainer.train()
```

### 4. 模型推理

#### 交互式聊天

```bash
python inference.py \
    --model_path "./outputs/sft_model" \
    --mode chat \
    --system_prompt "你是一个有用的AI助手。"
```

#### 批量推理

```bash
python inference.py \
    --model_path "./outputs/sft_model" \
    --mode batch \
    --data_path "./data/test_data.json" \
    --output_path "./outputs/predictions.json"
```

## 配置说明

### 模型配置

- `model_name`: 预训练模型名称或路径
- `trust_remote_code`: 是否信任远程代码
- `torch_dtype`: 数据类型（float16/float32/bfloat16）
- `device_map`: 设备映射策略

### 数据配置

- `train_data_path`: 训练数据路径
- `max_length`: 最大序列长度
- `ignore_index`: 忽略的标签索引
- `data_loader_num_workers`: 数据加载器工作进程数

### 训练配置

- `num_train_epochs`: 训练轮数
- `per_device_train_batch_size`: 每设备批次大小
- `gradient_accumulation_steps`: 梯度累积步数
- `learning_rate`: 学习率
- `weight_decay`: 权重衰减
- `warmup_ratio`: 预热比例
- `fp16`: 是否使用半精度训练

## ChatML数据格式

训练数据应使用ChatML格式，示例：

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "你是一个有用的AI助手。"
      },
      {
        "role": "user", 
        "content": "什么是机器学习？"
      },
      {
        "role": "assistant",
        "content": "机器学习是人工智能的一个分支..."
      }
    ]
  }
]
```

## 支持的模型

- Qwen2.5系列（0.5B, 1.5B, 7B, 14B, 32B, 72B）
- Llama3系列（8B, 70B）
- 其他兼容的Hugging Face模型

## 训练优化建议

### 内存优化

1. **使用梯度检查点**：减少显存占用
2. **调整批次大小**：根据GPU显存调整
3. **使用DeepSpeed**：支持ZeRO优化
4. **量化训练**：使用8bit或4bit量化

### 训练策略

1. **学习率调度**：使用warmup + linear decay
2. **梯度裁剪**：防止梯度爆炸
3. **权重衰减**：防止过拟合
4. **早停机制**：避免过训练

### 数据质量

1. **数据清洗**：移除低质量样本
2. **数据平衡**：确保任务类型平衡
3. **数据增强**：增加数据多样性
4. **格式统一**：确保ChatML格式正确

## 常见问题

### Q: 训练时显存不足怎么办？

A: 可以尝试以下方法：

- 减小批次大小
- 增加梯度累积步数
- 使用DeepSpeed ZeRO
- 启用梯度检查点
- 使用量化训练

### Q: 如何评估模型效果？

A: 可以：

- 使用验证集计算困惑度
- 进行人工评估
- 使用自动评估指标（BLEU、ROUGE等）
- 进行A/B测试

### Q: 训练速度太慢怎么办？

A: 可以：

- 使用更大的批次大小
- 启用混合精度训练（fp16）
- 使用多GPU训练
- 优化数据加载
- 使用更快的存储设备

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

- v1.0.0: 初始版本，支持基本的SFT训练
- 支持ChatML格式数据处理
- 支持多种预训练模型
- 提供完整的训练和推理流程
