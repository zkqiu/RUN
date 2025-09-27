#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT模型推理脚本
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SFTInference:
    """SFT模型推理类"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        self.model.eval()
        logger.info(f"模型已加载: {model_path}")
    
    def format_chatml(self, messages: List[Dict[str, str]]) -> str:
        """格式化消息为ChatML格式"""
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
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """
        生成回复
        
        Args:
            messages: 对话消息列表
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus采样参数
            top_k: top-k采样参数
            repetition_penalty: 重复惩罚
            do_sample: 是否采样
            
        Returns:
            生成的回复
        """
        # 格式化输入
        input_text = self.format_chatml(messages)
        
        # 添加assistant开始标记
        input_text += "<|im_start|>assistant\n"
        
        # 编码输入
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # 清理输出（移除可能的结束标记）
        if generated_text.endswith("<|im_end|>"):
            generated_text = generated_text[:-10]
        
        return generated_text.strip()
    
    def chat(self, system_prompt: str = None, max_turns: int = 10):
        """
        交互式聊天
        
        Args:
            system_prompt: 系统提示词
            max_turns: 最大对话轮数
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        print("开始聊天！输入 'quit' 退出，输入 'clear' 清空对话历史。")
        print("-" * 50)
        
        for turn in range(max_turns):
            # 获取用户输入
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() == 'quit':
                print("再见！")
                break
            elif user_input.lower() == 'clear':
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                print("对话历史已清空。")
                continue
            
            if not user_input:
                continue
            
            # 添加用户消息
            messages.append({"role": "user", "content": user_input})
            
            # 生成回复
            print("助手: ", end="", flush=True)
            try:
                response = self.generate_response(messages)
                print(response)
                
                # 添加助手回复
                messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                print(f"生成回复时出错: {e}")
    
    def batch_inference(self, 
                       data_path: str, 
                       output_path: str,
                       max_new_tokens: int = 512,
                       **generation_kwargs):
        """
        批量推理
        
        Args:
            data_path: 输入数据路径
            output_path: 输出数据路径
            max_new_tokens: 最大生成token数
            **generation_kwargs: 其他生成参数
        """
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        
        for i, item in enumerate(data):
            messages = item.get('messages', [])
            
            # 移除最后的assistant消息（如果有的话）
            if messages and messages[-1]['role'] == 'assistant':
                messages = messages[:-1]
            
            try:
                response = self.generate_response(
                    messages, 
                    max_new_tokens=max_new_tokens,
                    **generation_kwargs
                )
                
                # 添加生成的回复
                result_item = item.copy()
                result_item['messages'].append({
                    "role": "assistant",
                    "content": response
                })
                result_item['generated_response'] = response
                
                results.append(result_item)
                
                print(f"处理完成 {i+1}/{len(data)}")
                
            except Exception as e:
                print(f"处理第 {i+1} 条数据时出错: {e}")
                results.append(item)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"批量推理完成，结果已保存到 {output_path}")

def main():
    parser = argparse.ArgumentParser(description="SFT模型推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--mode", type=str, choices=["chat", "batch"], default="chat", help="推理模式")
    parser.add_argument("--data_path", type=str, help="批量推理数据路径")
    parser.add_argument("--output_path", type=str, help="批量推理输出路径")
    parser.add_argument("--system_prompt", type=str, help="系统提示词")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus采样参数")
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = SFTInference(args.model_path)
    
    if args.mode == "chat":
        # 交互式聊天
        inference.chat(system_prompt=args.system_prompt)
    
    elif args.mode == "batch":
        # 批量推理
        if not args.data_path or not args.output_path:
            print("批量推理模式需要指定 --data_path 和 --output_path")
            return
        
        inference.batch_inference(
            data_path=args.data_path,
            output_path=args.output_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

if __name__ == "__main__":
    main()
