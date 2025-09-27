#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatML格式数据处理工具
"""

import json
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ChatMLDataProcessor:
    """ChatML格式数据处理器"""
    
    def __init__(self):
        self.system_prompt = "你是一个有用的AI助手。"
    
    def create_chatml_sample(self, 
                            system: str = None, 
                            user: str = "", 
                            assistant: str = "") -> Dict[str, Any]:
        """
        创建单个ChatML格式的对话样本
        
        Args:
            system: 系统提示词
            user: 用户输入
            assistant: 助手回复
            
        Returns:
            ChatML格式的对话字典
        """
        messages = []
        
        if system:
            messages.append({
                "role": "system",
                "content": system
            })
        
        if user:
            messages.append({
                "role": "user", 
                "content": user
            })
        
        if assistant:
            messages.append({
                "role": "assistant",
                "content": assistant
            })
        
        return {
            "messages": messages
        }
    
    def create_conversation_sample(self, 
                                 system: str = None,
                                 conversation: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        创建多轮对话样本
        
        Args:
            system: 系统提示词
            conversation: 对话列表，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            
        Returns:
            ChatML格式的对话字典
        """
        messages = []
        
        if system:
            messages.append({
                "role": "system",
                "content": system
            })
        
        if conversation:
            messages.extend(conversation)
        
        return {
            "messages": messages
        }
    
    def save_chatml_data(self, data: List[Dict[str, Any]], output_path: str):
        """
        保存ChatML格式数据到JSON文件
        
        Args:
            data: 数据列表
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存 {len(data)} 条数据到 {output_path}")
    
    def load_chatml_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        加载ChatML格式数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            数据列表
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"已加载 {len(data)} 条数据从 {data_path}")
        return data
    
    def validate_chatml_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        验证ChatML格式数据的有效性
        
        Args:
            data: 数据列表
            
        Returns:
            是否有效
        """
        for i, item in enumerate(data):
            if 'messages' not in item:
                logger.error(f"数据项 {i} 缺少 'messages' 字段")
                return False
            
            messages = item['messages']
            if not isinstance(messages, list) or len(messages) == 0:
                logger.error(f"数据项 {i} 的 'messages' 字段为空或不是列表")
                return False
            
            for j, message in enumerate(messages):
                if not isinstance(message, dict):
                    logger.error(f"数据项 {i} 的消息 {j} 不是字典")
                    return False
                
                if 'role' not in message or 'content' not in message:
                    logger.error(f"数据项 {i} 的消息 {j} 缺少 'role' 或 'content' 字段")
                    return False
                
                role = message['role']
                if role not in ['system', 'user', 'assistant']:
                    logger.error(f"数据项 {i} 的消息 {j} 的 'role' 字段值无效: {role}")
                    return False
        
        logger.info("数据验证通过")
        return True

def create_sample_data():
    """创建示例训练数据"""
    processor = ChatMLDataProcessor()
    
    # 创建示例数据
    sample_data = [
        # 单轮对话
        processor.create_chatml_sample(
            system="你是一个有用的AI助手，请用中文回答问题。",
            user="什么是机器学习？",
            assistant="机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。通过分析大量数据，机器学习算法可以识别模式并做出预测或决策。"
        ),
        
        # 多轮对话
        processor.create_conversation_sample(
            system="你是一个编程助手，专门帮助解决编程问题。",
            conversation=[
                {"role": "user", "content": "如何在Python中读取CSV文件？"},
                {"role": "assistant", "content": "在Python中读取CSV文件可以使用pandas库。以下是基本用法：\n\n```python\nimport pandas as pd\n\n# 读取CSV文件\ndf = pd.read_csv('file.csv')\nprint(df.head())\n```"},
                {"role": "user", "content": "如果CSV文件很大怎么办？"},
                {"role": "assistant", "content": "对于大型CSV文件，你可以使用以下方法：\n\n1. 分块读取：\n```python\ndf_chunks = pd.read_csv('large_file.csv', chunksize=10000)\nfor chunk in df_chunks:\n    process(chunk)\n```\n\n2. 只读取需要的列：\n```python\ndf = pd.read_csv('large_file.csv', usecols=['col1', 'col2'])\n```"}
            ]
        ),
        
        # 代码生成任务
        processor.create_chatml_sample(
            system="你是一个代码生成助手，请根据用户需求生成相应的代码。",
            user="请写一个Python函数来计算斐波那契数列的第n项。",
            assistant="以下是计算斐波那契数列第n项的Python函数：\n\n```python\ndef fibonacci(n):\n    \"\"\"\n    计算斐波那契数列的第n项\n    \n    Args:\n        n (int): 要计算的项数\n    \n    Returns:\n        int: 第n项的值\n    \"\"\"\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range(2, n + 1):\n            a, b = b, a + b\n        return b\n\n# 测试函数\nfor i in range(10):\n    print(f\"F({i}) = {fibonacci(i)}\")\n```\n\n这个函数使用迭代方法，时间复杂度为O(n)，空间复杂度为O(1)。"
        ),
        
        # 数学问题
        processor.create_chatml_sample(
            system="你是一个数学助手，请帮助解决数学问题。",
            user="请解释什么是导数，并给出一个例子。",
            assistant="导数是微积分中的一个重要概念，它表示函数在某一点处的瞬时变化率。\n\n**定义：**\n如果函数f(x)在点x处可导，那么导数f'(x)定义为：\n\nf'(x) = lim[h→0] [f(x+h) - f(x)] / h\n\n**几何意义：**\n导数表示函数图像在该点处的切线斜率。\n\n**例子：**\n考虑函数f(x) = x²\n\nf'(x) = lim[h→0] [(x+h)² - x²] / h\n     = lim[h→0] [x² + 2xh + h² - x²] / h\n     = lim[h→0] [2xh + h²] / h\n     = lim[h→0] [2x + h]\n     = 2x\n\n所以f(x) = x²的导数是f'(x) = 2x。"
        ),
        
        # 翻译任务
        processor.create_chatml_sample(
            system="你是一个翻译助手，请将中文翻译成英文。",
            user="人工智能正在改变我们的生活方式。",
            assistant="Artificial intelligence is changing our way of life."
        ),
        
        # 创意写作
        processor.create_chatml_sample(
            system="你是一个创意写作助手，请帮助用户进行创意写作。",
            user="请写一个关于时间旅行的短故事开头。",
            assistant="**时间裂缝**\n\n当李博士按下那个红色的按钮时，他从未想过自己会真的成功。实验室里的一切都在颤抖，空气中弥漫着臭氧的味道。突然，一道刺眼的白光闪过，李博士感到自己被一股无形的力量拉扯着。\n\n当他再次睁开眼睛时，发现自己站在一个完全陌生的街道上。汽车是悬浮的，建筑物高耸入云，天空中还有飞行器在穿梭。他看了看手腕上的手表——显示的是2157年。\n\n\"我成功了...\"李博士喃喃自语，但随即意识到，成功可能意味着他再也回不去了。"
        )
    ]
    
    # 验证数据
    if processor.validate_chatml_data(sample_data):
        # 保存数据
        processor.save_chatml_data(sample_data, "./data/chatml_data.json")
        print("示例数据创建成功！")
    else:
        print("数据验证失败！")

if __name__ == "__main__":
    create_sample_data()
