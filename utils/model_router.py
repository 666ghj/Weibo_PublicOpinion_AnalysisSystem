import os
import json
import logging
import re
from collections import defaultdict
import random
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

logger = logging.getLogger('model_router')
logger.setLevel(logging.INFO)

class ModelRouter:
    """
    模型路由器 - 自动根据内容类型选择最优的AI模型
    
    功能:
    1. 根据内容类型和任务需求，自动选择最合适的AI模型
    2. 支持多种模型供应商和模型类型
    3. 考虑性能、成本和准确度等因素进行智能路由
    4. 学习和适应用户偏好和使用模式
    5. 提供标准化的API接口，支持私有模型集成
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRouter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 支持的模型定义
        self.models = {
            # OpenAI 模型
            'gpt-4o-latest': {
                'provider': 'openai',
                'capabilities': {
                    'text_analysis': 0.95,
                    'sentiment_analysis': 0.92,
                    'keyword_extraction': 0.90,
                    'summarization': 0.93,
                    'classification': 0.89,
                    'chinese_text': 0.88
                },
                'cost_per_1k': 0.01,
                'max_tokens': 128000,
                'avg_latency': 2.5,  # 秒
                'requires_api_key': 'OPENAI_API_KEY'
            },
            'gpt-4o-mini': {
                'provider': 'openai',
                'capabilities': {
                    'text_analysis': 0.85,
                    'sentiment_analysis': 0.82,
                    'keyword_extraction': 0.80,
                    'summarization': 0.84,
                    'classification': 0.81,
                    'chinese_text': 0.79
                },
                'cost_per_1k': 0.00015,
                'max_tokens': 4000,
                'avg_latency': 1.2,
                'requires_api_key': 'OPENAI_API_KEY'
            },
            'gpt-3.5-turbo': {
                'provider': 'openai',
                'capabilities': {
                    'text_analysis': 0.75,
                    'sentiment_analysis': 0.72,
                    'keyword_extraction': 0.70,
                    'summarization': 0.77,
                    'classification': 0.73,
                    'chinese_text': 0.65
                },
                'cost_per_1k': 0.0015,
                'max_tokens': 16000,
                'avg_latency': 0.8,
                'requires_api_key': 'OPENAI_API_KEY'
            },
            
            # Claude 模型
            'claude-3.5-sonnet': {
                'provider': 'anthropic',
                'capabilities': {
                    'text_analysis': 0.90,
                    'sentiment_analysis': 0.91,
                    'keyword_extraction': 0.85,
                    'summarization': 0.92,
                    'classification': 0.89,
                    'chinese_text': 0.80
                },
                'cost_per_1k': 0.015,
                'max_tokens': 200000,
                'avg_latency': 2.8,
                'requires_api_key': 'ANTHROPIC_API_KEY'
            },
            'claude-3.5-haiku': {
                'provider': 'anthropic',
                'capabilities': {
                    'text_analysis': 0.84,
                    'sentiment_analysis': 0.83,
                    'keyword_extraction': 0.79,
                    'summarization': 0.85,
                    'classification': 0.80,
                    'chinese_text': 0.72
                },
                'cost_per_1k': 0.0025,
                'max_tokens': 200000,
                'avg_latency': 1.5,
                'requires_api_key': 'ANTHROPIC_API_KEY'
            },
            
            # DeepSeek 模型
            'deepseek-chat': {
                'provider': 'deepseek',
                'capabilities': {
                    'text_analysis': 0.82,
                    'sentiment_analysis': 0.79,
                    'keyword_extraction': 0.77,
                    'summarization': 0.80,
                    'classification': 0.77,
                    'chinese_text': 0.90  # 特别好中文
                },
                'cost_per_1k': 0.002,
                'max_tokens': 4000,
                'avg_latency': 1.0,
                'requires_api_key': 'DEEPSEEK_API_KEY'
            },
            'deepseek-reasoner': {
                'provider': 'deepseek',
                'capabilities': {
                    'text_analysis': 0.87,
                    'sentiment_analysis': 0.75,
                    'keyword_extraction': 0.76,
                    'summarization': 0.78,
                    'classification': 0.85,
                    'chinese_text': 0.88
                },
                'cost_per_1k': 0.003,
                'max_tokens': 4000,
                'avg_latency': 1.8,
                'requires_api_key': 'DEEPSEEK_API_KEY'
            }
        }
        
        # 任务类型定义
        self.task_types = {
            'sentiment_analysis': {
                'description': '情感分析',
                'key_capabilities': ['sentiment_analysis', 'text_analysis'],
                'example_prompt': '分析以下文本的情感倾向（积极、消极或中性）'
            },
            'topic_classification': {
                'description': '话题分类',
                'key_capabilities': ['classification', 'text_analysis'],
                'example_prompt': '将以下文本分类到最合适的话题类别'
            },
            'keyword_extraction': {
                'description': '关键词提取',
                'key_capabilities': ['keyword_extraction', 'text_analysis'],
                'example_prompt': '从以下文本中提取5个最重要的关键词'
            },
            'text_summarization': {
                'description': '文本摘要',
                'key_capabilities': ['summarization', 'text_analysis'],
                'example_prompt': '为以下文本生成一个简短的摘要'
            },
            'comprehensive_analysis': {
                'description': '综合分析',
                'key_capabilities': ['text_analysis', 'sentiment_analysis', 'keyword_extraction', 'summarization'],
                'example_prompt': '对以下文本进行全面分析，包括情感、关键词和主要观点'
            }
        }
        
        # 用户偏好和使用历史
        self.usage_history = defaultdict(list)
        
        # 模型可用性缓存
        self.available_models = {}
        
        # 更新模型可用性
        self._update_available_models()
        
        self._initialized = True
        logger.info("模型路由器初始化完成")
    
    def _update_available_models(self):
        """更新模型可用性"""
        self.available_models = {}
        
        for model_id, model_info in self.models.items():
            # 检查API密钥是否可用
            api_key_env = model_info.get('requires_api_key')
            if api_key_env and os.getenv(api_key_env):
                self.available_models[model_id] = model_info
        
        if not self.available_models:
            logger.warning("未找到可用的模型，请检查API密钥配置")
        else:
            logger.info(f"找到 {len(self.available_models)} 个可用模型")
    
    def detect_content_type(self, text: str) -> Dict[str, float]:
        """
        检测内容类型和特征
        
        参数:
            text: 要分析的文本
            
        返回:
            内容类型特征字典，键为特征名称，值为权重
        """
        features = {
            'chinese_text': 0.0,
            'length': 0.0,
            'complexity': 0.0
        }
        
        if not text:
            return features
        
        # 检测中文比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        
        # 文本长度评分 (归一化至0-1)
        length_score = min(1.0, len(text) / 10000)
        
        # 文本复杂度简单估计
        # 基于句子长度、词汇多样性等
        sentences = re.split(r'[.!?。！？]', text)
        avg_sentence_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        unique_words = len(set(re.findall(r'\w+', text.lower())))
        total_words = len(re.findall(r'\w+', text.lower()))
        
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        complexity_score = (avg_sentence_len / 50 + lexical_diversity) / 2
        complexity_score = min(1.0, complexity_score)
        
        features['chinese_text'] = chinese_ratio
        features['length'] = length_score
        features['complexity'] = complexity_score
        
        return features
    
    def select_model(self, text: str, task_type: str, 
                     optimize_for: str = 'balanced', 
                     exclude_models: List[str] = None) -> str:
        """
        为给定文本和任务选择最合适的模型
        
        参数:
            text: 要处理的文本
            task_type: 任务类型，如 'sentiment_analysis'
            optimize_for: 优化目标，可选值：'cost'(成本), 'performance'(性能), 'balanced'(平衡)
            exclude_models: 要排除的模型列表
            
        返回:
            选择的模型ID
        """
        if not self.available_models:
            logger.error("没有可用的模型，请检查API密钥配置")
            return None
        
        if task_type not in self.task_types:
            logger.warning(f"未知的任务类型: {task_type}，使用默认任务类型: 'comprehensive_analysis'")
            task_type = 'comprehensive_analysis'
        
        # 获取内容特征
        content_features = self.detect_content_type(text)
        
        # 获取任务关键能力
        task_capabilities = self.task_types[task_type]['key_capabilities']
        
        # 计算每个模型的得分
        model_scores = {}
        exclude_models = exclude_models or []
        
        for model_id, model_info in self.available_models.items():
            if model_id in exclude_models:
                continue
                
            # 基于任务能力的得分
            capability_score = 0
            for capability in task_capabilities:
                capability_score += model_info['capabilities'].get(capability, 0)
            
            capability_score /= len(task_capabilities)
            
            # 基于内容特征的得分调整
            content_score = 1.0
            
            # 如果有大量中文，增加中文能力的权重
            if content_features['chinese_text'] > 0.5:
                chinese_capability = model_info['capabilities'].get('chinese_text', 0)
                content_score *= (1.0 + chinese_capability) / 2
            
            # 如果文本很长，检查模型的最大token限制
            if content_features['length'] > 0.7:
                max_tokens = model_info.get('max_tokens', 4000)
                if max_tokens < 10000:
                    content_score *= 0.7  # 长文本降低短上下文模型的分数
            
            # 如果文本很复杂，可能需要更强大的模型
            if content_features['complexity'] > 0.7:
                # 假设能力得分更高的模型更能处理复杂文本
                content_score *= (1.0 + capability_score) / 2
            
            # 根据优化目标调整最终得分
            final_score = capability_score * content_score
            
            if optimize_for == 'cost':
                # 成本越低，分数越高
                cost_factor = 1 - min(1.0, model_info.get('cost_per_1k', 0) / 0.03)
                final_score = final_score * 0.3 + cost_factor * 0.7
            elif optimize_for == 'performance':
                # 能力得分权重更高
                final_score = capability_score * 0.8 + content_score * 0.2
            # balanced 是默认值，不需要额外调整
            
            model_scores[model_id] = final_score
        
        if not model_scores:
            logger.warning("没有符合条件的可用模型")
            return list(self.available_models.keys())[0]
        
        # 选择得分最高的模型
        selected_model = max(model_scores, key=model_scores.get)
        
        # 记录使用历史
        self.usage_history[task_type].append({
            'model': selected_model,
            'timestamp': datetime.now().timestamp(),
            'score': model_scores[selected_model],
            'optimize_for': optimize_for
        })
        
        logger.info(f"为任务 '{task_type}' 选择了模型: {selected_model} (得分: {model_scores[selected_model]:.4f})")
        return selected_model
    
    def get_model_info(self, model_id: str) -> Dict:
        """获取模型信息"""
        if model_id in self.models:
            return self.models[model_id]
        return None
    
    def get_available_models(self, refresh: bool = False) -> Dict[str, Dict]:
        """获取所有可用的模型"""
        if refresh:
            self._update_available_models()
        return self.available_models
    
    def get_model_by_provider(self, provider: str, optimize_for: str = 'balanced') -> str:
        """根据提供商获取推荐模型"""
        provider_models = {
            model_id: info for model_id, info in self.available_models.items()
            if info['provider'] == provider
        }
        
        if not provider_models:
            logger.warning(f"未找到提供商 '{provider}' 的可用模型")
            return None
            
        if optimize_for == 'cost':
            # 选择成本最低的模型
            return min(provider_models.items(), key=lambda x: x[1].get('cost_per_1k', float('inf')))[0]
        elif optimize_for == 'performance':
            # 选择性能最好的模型，简单取所有能力的平均值
            return max(provider_models.items(), 
                      key=lambda x: sum(x[1]['capabilities'].values()) / len(x[1]['capabilities']))[0]
        else:
            # 平衡模式，综合考虑成本和性能
            scores = {}
            for model_id, info in provider_models.items():
                perf_score = sum(info['capabilities'].values()) / len(info['capabilities'])
                cost_score = 1 - min(1.0, info.get('cost_per_1k', 0) / 0.03)
                scores[model_id] = perf_score * 0.5 + cost_score * 0.5
            
            return max(scores, key=scores.get)
    
    def get_task_types(self) -> Dict[str, Dict]:
        """获取支持的任务类型"""
        return self.task_types
    
    def register_custom_model(self, model_id: str, model_info: Dict[str, Any]) -> bool:
        """
        注册自定义模型
        
        参数:
            model_id: 模型唯一标识符
            model_info: 模型信息字典，包含以下字段：
                - provider: 提供商名称
                - capabilities: 能力评分字典
                - cost_per_1k: 每千token的成本
                - max_tokens: 最大token限制
                - avg_latency: 平均延迟（秒）
                - requires_api_key: API密钥环境变量名
                
        返回:
            是否注册成功
        """
        # 验证必要字段
        required_fields = ['provider', 'capabilities', 'cost_per_1k', 'max_tokens']
        for field in required_fields:
            if field not in model_info:
                logger.error(f"注册模型失败: 缺少必要字段 '{field}'")
                return False
        
        # 验证能力评分
        if not isinstance(model_info['capabilities'], dict):
            logger.error("注册模型失败: 'capabilities' 必须是字典")
            return False
            
        # 添加模型
        self.models[model_id] = model_info
        
        # 更新可用模型列表
        self._update_available_models()
        
        logger.info(f"成功注册自定义模型: {model_id}")
        return True

# 创建全局模型路由器实例
model_router = ModelRouter()

def select_model(text, task_type, optimize_for='balanced', exclude_models=None):
    """选择最合适的模型"""
    return model_router.select_model(text, task_type, optimize_for, exclude_models)

def get_available_models(refresh=False):
    """获取所有可用的模型"""
    return model_router.get_available_models(refresh)

def get_model_by_provider(provider, optimize_for='balanced'):
    """根据提供商获取推荐模型"""
    return model_router.get_model_by_provider(provider, optimize_for)

def get_task_types():
    """获取支持的任务类型"""
    return model_router.get_task_types()

def register_custom_model(model_id, model_info):
    """注册自定义模型"""
    return model_router.register_custom_model(model_id, model_info)

# 示例用法
if __name__ == "__main__":
    # 示例文本
    chinese_text = """
    近日，人工智能技术的发展引发广泛关注。
    专家指出，大型语言模型在自然语言处理领域取得了显著进展，
    但同时也带来了诸多伦理和安全问题。对此，业界呼吁加强监管，
    确保人工智能的发展能够造福人类社会。
    """
    
    english_text = """
    Recent developments in artificial intelligence technology have drawn widespread attention.
    Experts point out that large language models have made significant progress in the field of natural language processing,
    but also bring many ethical and security issues. In response, the industry calls for strengthened regulation
    to ensure that the development of artificial intelligence can benefit human society.
    """
    
    # 测试模型选择
    print("中文文本任务测试:")
    model_for_chinese = select_model(chinese_text, 'sentiment_analysis')
    print(f"选择的模型: {model_for_chinese}")
    
    print("\n英文文本任务测试:")
    model_for_english = select_model(english_text, 'sentiment_analysis')
    print(f"选择的模型: {model_for_english}")
    
    print("\n成本优化测试:")
    model_for_cost = select_model(chinese_text, 'sentiment_analysis', optimize_for='cost')
    print(f"选择的模型: {model_for_cost}")
    
    print("\n性能优化测试:")
    model_for_perf = select_model(chinese_text, 'sentiment_analysis', optimize_for='performance')
    print(f"选择的模型: {model_for_perf}")
    
    # 测试API提供商
    print("\n根据提供商获取模型:")
    for provider in ['openai', 'anthropic', 'deepseek']:
        model = get_model_by_provider(provider)
        if model:
            print(f"{provider}: {model}")
        else:
            print(f"{provider}: 无可用模型") 