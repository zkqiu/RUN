#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT LLM使用示例
"""

import os
import json
from data_utils import ChatMLDataProcessor
from sft_train import SFTTrainer, ModelArguments, DataArguments, TrainingArguments
from inference import SFTInference

def create_example_data():
    """创建示例训练数据"""
    processor = ChatMLDataProcessor()
    
    # 创建多种类型的对话数据
    sample_data = [
        # 编程问答
        processor.create_chatml_sample(
            system="你是一个专业的编程助手，请用中文回答问题。",
            user="如何在Python中实现快速排序？",
            assistant="以下是Python中快速排序的实现：\n\n```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    \n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    \n    return quick_sort(left) + middle + quick_sort(right)\n\n# 测试\ntest_array = [64, 34, 25, 12, 22, 11, 90]\nprint(quick_sort(test_array))\n```\n\n时间复杂度：平均O(n log n)，最坏O(n²)\n空间复杂度：O(log n)"
        ),
        
        # 数学问题
        processor.create_chatml_sample(
            system="你是一个数学老师，请详细解释数学概念。",
            user="请解释什么是导数，并给出几何意义。",
            assistant="导数是微积分中的核心概念，表示函数在某点的瞬时变化率。\n\n**定义：**\n如果函数f(x)在点x处可导，则导数f'(x)为：\n\nf'(x) = lim[h→0] [f(x+h) - f(x)] / h\n\n**几何意义：**\n导数表示函数图像在该点处切线的斜率。\n\n**例子：**\n对于f(x) = x²：\n- f'(x) = 2x\n- 在x=1处，f'(1) = 2，表示切线斜率为2\n- 在x=0处，f'(0) = 0，表示切线水平\n\n**物理意义：**\n在物理学中，导数表示瞬时速度、加速度等变化率。"
        ),
        
        # 创意写作
        processor.create_chatml_sample(
            system="你是一个创意写作助手，请帮助用户进行创意写作。",
            user="请写一个关于时间旅行的科幻故事开头。",
            assistant="**时间裂缝**\n\n当李博士按下那个红色的按钮时，他从未想过自己会真的成功。实验室里的一切都在颤抖，空气中弥漫着臭氧的味道。突然，一道刺眼的白光闪过，李博士感到自己被一股无形的力量拉扯着。\n\n当他再次睁开眼睛时，发现自己站在一个完全陌生的街道上。汽车是悬浮的，建筑物高耸入云，天空中还有飞行器在穿梭。他看了看手腕上的手表——显示的是2157年。\n\n\"我成功了...\"李博士喃喃自语，但随即意识到，成功可能意味着他再也回不去了。"
        ),
        
        # 多轮对话
        processor.create_conversation_sample(
            system="你是一个有用的AI助手。",
            conversation=[
                {"role": "user", "content": "什么是深度学习？"},
                {"role": "assistant", "content": "深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别、自然语言处理、语音识别等领域取得了突破性进展。"},
                {"role": "user", "content": "深度学习有哪些应用？"},
                {"role": "assistant", "content": "深度学习的应用非常广泛：\n\n1. **计算机视觉**：图像分类、目标检测、人脸识别\n2. **自然语言处理**：机器翻译、文本生成、情感分析\n3. **语音技术**：语音识别、语音合成\n4. **推荐系统**：个性化推荐、内容过滤\n5. **自动驾驶**：环境感知、路径规划\n6. **医疗诊断**：医学影像分析、疾病预测\n7. **游戏AI**：AlphaGo、游戏NPC智能"}
            ]
        ),
        
        # 翻译任务
        processor.create_chatml_sample(
            system="你是一个专业的翻译助手。",
            user="请将以下英文翻译成中文：'Artificial intelligence is transforming the way we work and live.'",
            assistant="人工智能正在改变我们的工作和生活方式。"
        ),
        
        # 代码调试
        processor.create_chatml_sample(
            system="你是一个代码调试专家，请帮助用户解决编程问题。",
            user="我的Python代码出现了错误，请帮我看看：\n```python\nfor i in range(5):\n    print(i)\n    if i == 3:\n        break\n    else:\n        print('继续')\n```",
            assistant="你的代码逻辑有问题。在Python中，`for`循环的`else`子句只有在循环正常结束时才会执行，如果使用了`break`，`else`就不会执行。\n\n**问题分析：**\n- 当`i == 3`时，执行`break`跳出循环\n- 由于使用了`break`，`else`子句不会执行\n- 所以永远不会打印'继续'\n\n**修正后的代码：**\n```python\nfor i in range(5):\n    print(i)\n    if i == 3:\n        break\n    print('继续')  # 移到这里\n```\n\n或者如果你想要不同的逻辑：\n```python\nfor i in range(5):\n    print(i)\n    if i == 3:\n        break\nelse:\n    print('循环正常结束')  # 只有没有break时才执行\n```"
        )
    ]
    
    # 保存数据
    os.makedirs("./data", exist_ok=True)
    processor.save_chatml_data(sample_data, "./data/chatml_data.json")
    print("示例数据已创建！")

def train_example():
    """训练示例"""
    print("开始训练示例...")
    
    # 创建参数
    model_args = ModelArguments(
        model_name_or_path="Qwen/Qwen2.5-0.5B",
        trust_remote_code=True
    )
    
    data_args = DataArguments(
        data_path="./data/chatml_data.json",
        max_length=1024,  # 使用较小的长度进行快速训练
        ignore_index=-100
    )
    
    training_args = TrainingArguments(
        output_dir="./outputs/sft_model",
        num_train_epochs=1,  # 只训练1个epoch作为示例
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=10,
        fp16=True
    )
    
    # 创建训练器
    trainer = SFTTrainer(model_args, data_args, training_args)
    
    # 开始训练
    trainer.train()
    print("训练完成！")

def inference_example():
    """推理示例"""
    print("开始推理示例...")
    
    # 创建推理器
    inference = SFTInference("./outputs/sft_model")
    
    # 测试对话
    test_messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "请解释什么是机器学习？"}
    ]
    
    response = inference.generate_response(test_messages)
    print(f"用户: {test_messages[-1]['content']}")
    print(f"助手: {response}")
    
    # 交互式聊天
    print("\n开始交互式聊天（输入'quit'退出）:")
    inference.chat(system_prompt="你是一个有用的AI助手。")

def main():
    """主函数"""
    print("SFT LLM 使用示例")
    print("=" * 50)
    
    # 1. 创建示例数据
    print("1. 创建示例数据...")
    create_example_data()
    
    # 2. 训练模型
    print("\n2. 训练模型...")
    train_example()
    
    # 3. 推理测试
    print("\n3. 推理测试...")
    inference_example()

if __name__ == "__main__":
    main()
