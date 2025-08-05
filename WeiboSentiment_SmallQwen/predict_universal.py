#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3微博情感分析统一预测接口
支持0.6B、4B、8B三种规格的Embedding和LoRA模型
"""

import os
import sys
import argparse
import torch
from typing import List, Dict, Tuple, Any

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_config import QWEN3_MODELS, MODEL_PATHS
from qwen3_embedding_universal import Qwen3EmbeddingUniversal
from qwen3_lora_universal import Qwen3LoRAUniversal


class Qwen3UniversalPredictor:
    """Qwen3统一预测器"""
    
    def __init__(self):
        self.models = {}  # 存储已加载的模型 {model_key: {model: obj, display_name: str}}
        
    def _get_model_key(self, model_type: str, model_size: str) -> str:
        """生成模型键值"""
        return f"{model_type}_{model_size}"
    
    def load_model(self, model_type: str, model_size: str) -> None:
        """加载指定的模型"""
        if model_type not in ['embedding', 'lora']:
            raise ValueError(f"不支持的模型类型: {model_type}")
        if model_size not in ['0.6B', '4B', '8B']:
            raise ValueError(f"不支持的模型大小: {model_size}")
            
        model_path = MODEL_PATHS[model_type][model_size]
        model_key = self._get_model_key(model_type, model_size)
        
        # 检查训练好的模型文件是否存在
        if not os.path.exists(model_path):
            print(f"训练好的模型文件不存在: {model_path}")
            print(f"请先训练 {model_type.upper()}-{model_size} 模型，或检查模型路径配置")
            return
        
        print(f"加载 {model_type.upper()}-{model_size} 模型...")
        
        try:
            if model_type == 'embedding':
                model = Qwen3EmbeddingUniversal(model_size)
                model.load_model(model_path)
            else:  # lora
                model = Qwen3LoRAUniversal(model_size)
                model.load_model(model_path)
            
            self.models[model_key] = {
                'model': model,
                'display_name': f"Qwen3-{model_type.title()}-{model_size}"
            }
            print(f"{model_type.upper()}-{model_size} 模型加载成功")
            
        except Exception as e:
            print(f"加载 {model_type.upper()}-{model_size} 模型失败: {e}")
            print(f"这可能是因为基础模型下载失败或训练好的模型文件损坏")
    
    def load_all_models(self, model_dir: str = './models') -> None:
        """加载所有可用的模型"""
        print("开始加载所有可用的Qwen3模型...")
        
        loaded_count = 0
        for model_type in ['embedding', 'lora']:
            for model_size in ['0.6B', '4B', '8B']:
                try:
                    self.load_model(model_type, model_size)
                    loaded_count += 1
                except Exception as e:
                    print(f"跳过 {model_type}-{model_size}: {e}")
        
        print(f"\n已加载 {loaded_count} 个模型")
        self._print_loaded_models()
    
    def load_specific_models(self, model_configs: List[Tuple[str, str]]) -> None:
        """加载指定的模型配置
        Args:
            model_configs: [(model_type, model_size), ...] 的列表
        """
        print("加载指定的Qwen3模型...")
        
        for model_type, model_size in model_configs:
            try:
                self.load_model(model_type, model_size)
            except Exception as e:
                print(f"跳过 {model_type}-{model_size}: {e}")
        
        print(f"\n已加载 {len(self.models)} 个模型")
        self._print_loaded_models()
    
    def _print_loaded_models(self):
        """打印已加载的模型列表"""
        if self.models:
            print("已加载模型:")
            for model_info in self.models.values():
                print(f"  - {model_info['display_name']}")
        else:
            print("没有成功加载任何模型")
    
    def predict_single(self, text: str, model_key: str = None) -> Dict[str, Tuple[int, float]]:
        """单文本预测
        Args:
            text: 要预测的文本
            model_key: 指定模型键值，None表示使用所有模型
        Returns:
            {model_name: (prediction, confidence), ...}
        """
        results = {}
        
        if model_key and model_key in self.models:
            # 使用指定模型
            model_info = self.models[model_key]
            try:
                prediction, confidence = model_info['model'].predict_single(text)
                results[model_info['display_name']] = (prediction, confidence)
            except Exception as e:
                print(f"模型 {model_info['display_name']} 预测失败: {e}")
                results[model_info['display_name']] = (0, 0.0)
        else:
            # 使用所有模型
            for model_info in self.models.values():
                try:
                    prediction, confidence = model_info['model'].predict_single(text)
                    results[model_info['display_name']] = (prediction, confidence)
                except Exception as e:
                    print(f"模型 {model_info['display_name']} 预测失败: {e}")
                    results[model_info['display_name']] = (0, 0.0)
        
        return results
    
    def predict_batch(self, texts: List[str]) -> Dict[str, List[int]]:
        """批量预测"""
        results = {}
        
        for model_info in self.models.values():
            try:
                predictions = model_info['model'].predict(texts)
                results[model_info['display_name']] = predictions
            except Exception as e:
                print(f"模型 {model_info['display_name']} 预测失败: {e}")
                results[model_info['display_name']] = [0] * len(texts)
        
        return results
    
    def ensemble_predict(self, text: str) -> Tuple[int, float]:
        """集成预测"""
        if len(self.models) < 2:
            raise ValueError("集成预测需要至少2个模型")
        
        results = self.predict_single(text)
        
        # 加权平均（这里使用简单平均，可以根据模型性能调整权重）
        total_weight = 0
        weighted_prob = 0
        
        for model_name, (pred, conf) in results.items():
            if conf > 0:  # 只考虑有效预测
                prob = conf if pred == 1 else 1 - conf
                weighted_prob += prob
                total_weight += 1
        
        if total_weight == 0:
            return 0, 0.5
        
        final_prob = weighted_prob / total_weight
        final_pred = int(final_prob > 0.5)
        final_conf = final_prob if final_pred == 1 else 1 - final_prob
        
        return final_pred, final_conf
    
    def _select_and_load_model(self):
        """让用户选择并加载模型"""
        print("Qwen3微博情感分析预测系统")
        print("="*40)
        print("请选择要使用的模型:")
        print("\n方法选择:")
        print("  1. Embedding + 分类头 (推理快速，显存占用少)")
        print("  2. LoRA微调 (效果更好，显存占用较多)")
        
        method_choice = None
        while method_choice not in ['1', '2']:
            method_choice = input("\n请选择方法 (1/2): ").strip()
            if method_choice not in ['1', '2']:
                print("无效选择，请输入 1 或 2")
        
        method_type = "embedding" if method_choice == '1' else "lora"
        method_name = "Embedding + 分类头" if method_choice == '1' else "LoRA微调"
        
        print(f"\n已选择: {method_name}")
        print("\n模型大小选择:")
        print("  1. 0.6B - 轻量级，推理快速")
        print("  2. 4B  - 中等规模，性能均衡") 
        print("  3. 8B  - 大规模，性能最佳")
        
        size_choice = None
        while size_choice not in ['1', '2', '3']:
            size_choice = input("\n请选择模型大小 (1/2/3): ").strip()
            if size_choice not in ['1', '2', '3']:
                print("无效选择，请输入 1、2 或 3")
        
        size_map = {'1': '0.6B', '2': '4B', '3': '8B'}
        model_size = size_map[size_choice]
        
        print(f"已选择: Qwen3-{method_name}-{model_size}")
        print("正在加载模型...")
        
        try:
            self.load_model(method_type, model_size)
            print(f"模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请检查模型文件是否存在，或先进行训练")
    
    def interactive_predict(self):
        """交互式预测模式"""
        if len(self.models) == 0:
            # 让用户选择要加载的模型
            self._select_and_load_model()
            if len(self.models) == 0:
                print("没有加载任何模型，退出预测")
                return
        
        print("\n" + "="*60)
        print("Qwen3微博情感分析预测系统")
        print("="*60)
        print("已加载模型:")
        for model_info in self.models.values():
            print(f"   - {model_info['display_name']}")
        print("\n命令提示:")
        print("   输入 'q' 退出程序")
        print("   输入 'switch' 切换模型")  
        print("   输入 'models' 查看已加载模型")
        print("   输入 'compare' 比较所有模型性能")
        print("-"*60)
        
        while True:
            try:
                text = input("\n请输入要分析的微博内容: ").strip()
                
                if text.lower() == 'q':
                    print("感谢使用，再见！")
                    break
                
                if text.lower() == 'models':
                    print("已加载模型:")
                    for model_info in self.models.values():
                        print(f"   - {model_info['display_name']}")
                    continue
                
                if text.lower() == 'switch':
                    print("切换模型...")
                    self.models.clear()  # 清空当前模型
                    self._select_and_load_model()
                    if len(self.models) > 0:
                        print("模型切换成功!")
                        for model_info in self.models.values():
                            print(f"   当前模型: {model_info['display_name']}")
                    continue
                
                if text.lower() == 'compare':
                    test_text = input("请输入要比较的文本: ")
                    self._compare_models(test_text)
                    continue
                
                if not text:
                    print("请输入有效内容")
                    continue
                
                # 预测
                results = self.predict_single(text)
                
                print(f"\n原文: {text}")
                print("预测结果:")
                
                # 按模型类型和大小排序显示
                sorted_results = sorted(results.items())
                for model_name, (pred, conf) in sorted_results:
                    sentiment = "正面" if pred == 1 else "负面"
                    print(f"   {model_name:20}: {sentiment} (置信度: {conf:.4f})")
                
                # 只显示单个模型的预测结果（不进行集成）
                
            except KeyboardInterrupt:
                print("\n\n程序被中断，再见！")
                break
            except Exception as e:
                print(f"预测过程中出现错误: {e}")
    
    def _compare_models(self, text: str):
        """比较不同模型的性能"""
        print(f"\n模型性能比较 - 文本: {text}")
        print("-" * 60)
        
        results = self.predict_single(text)
        
        embedding_models = []
        lora_models = []
        
        for model_name, (pred, conf) in results.items():
            sentiment = "正面" if pred == 1 else "负面"
            if "Embedding" in model_name:
                embedding_models.append((model_name, sentiment, conf))
            elif "Lora" in model_name:
                lora_models.append((model_name, sentiment, conf))
        
        if embedding_models:
            print("Embedding + 分类头方法:")
            for name, sentiment, conf in embedding_models:
                print(f"   {name}: {sentiment} ({conf:.4f})")
        
        if lora_models:
            print("LoRA微调方法:")
            for name, sentiment, conf in lora_models:
                print(f"   {name}: {sentiment} ({conf:.4f})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Qwen3微博情感分析统一预测接口')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='模型文件目录')
    parser.add_argument('--model_type', type=str, choices=['embedding', 'lora'],
                        help='指定模型类型')
    parser.add_argument('--model_size', type=str, choices=['0.6B', '4B', '8B'],
                        help='指定模型大小')
    parser.add_argument('--text', type=str,
                        help='直接预测指定文本')
    parser.add_argument('--interactive', action='store_true', default=True,
                        help='交互式预测模式（默认）')
    parser.add_argument('--ensemble', action='store_true',
                        help='使用集成预测')
    parser.add_argument('--load_all', action='store_true',
                        help='加载所有可用模型')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = Qwen3UniversalPredictor()
    
    # 加载模型
    if args.load_all:
        # 加载所有模型
        predictor.load_all_models(args.model_dir)
    elif args.model_type and args.model_size:
        # 加载指定模型
        predictor.load_model(args.model_type, args.model_size)
    # 如果没有指定模型，交互式模式会让用户选择
    
    # 如果指定了文本，直接预测
    if args.text:
        if args.ensemble and len(predictor.models) > 1:
            pred, conf = predictor.ensemble_predict(args.text)
            sentiment = "正面" if pred == 1 else "负面"
            print(f"文本: {args.text}")
            print(f"集成预测: {sentiment} (置信度: {conf:.4f})")
        else:
            results = predictor.predict_single(args.text)
            print(f"文本: {args.text}")
            for model_name, (pred, conf) in results.items():
                sentiment = "正面" if pred == 1 else "负面"
                print(f"{model_name}: {sentiment} (置信度: {conf:.4f})")
    else:
        # 进入交互式模式
        predictor.interactive_predict()


if __name__ == "__main__":
    main()