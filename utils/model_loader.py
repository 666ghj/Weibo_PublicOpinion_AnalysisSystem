import os
import sys
import pickle
import marshal
import types
import logging
import torch
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger('model_loader')
logger.setLevel(logging.INFO)

def load_sentiment_model(model_path, device=None):
    """
    加载情感分析模型
    
    参数:
        model_path: 模型文件路径
        device: 设备（可忽略，marshal模型不依赖设备）
    
    返回:
        加载好的模型对象
    """
    try:
        logger.info(f"加载情感分析模型: {model_path}")
        
        if model_path.endswith('.marshal') or model_path.endswith('.marshal.3'):
            with open(model_path, 'rb') as f:
                model_data = marshal.load(f)
                
            # 将marshal数据转换为可调用的函数对象
            sentiment_func = types.FunctionType(model_data, globals(), "sentiment_func")
            logger.info("情感分析模型加载成功")
            return sentiment_func
        else:
            raise ValueError(f"不支持的情感模型格式: {model_path}")
    except Exception as e:
        logger.error(f"加载情感分析模型失败: {e}")
        raise

def load_bert_ctm_model(model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载BERT-CTM模型
    
    参数:
        model_dir: 模型目录
        device: 计算设备
    
    返回:
        包含模型和分词器的字典
    """
    try:
        logger.info(f"加载BERT-CTM模型: {model_dir}")
        
        sys.path.append('model_pro')
        from BERT_CTM import BERT_CTM
        from transformers import BertTokenizer
        
        # 加载模型
        model_path = os.path.join(model_dir, 'final_model.pt') if not model_dir.endswith('.pt') else model_dir
        model = BERT_CTM()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 加载分词器
        tokenizer_path = os.path.join(os.path.dirname(model_dir), 'bert_model')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        logger.info("BERT-CTM模型加载成功")
        return {
            'model': model,
            'tokenizer': tokenizer,
            'device': device
        }
    except Exception as e:
        logger.error(f"加载BERT-CTM模型失败: {e}")
        raise

def load_bcat_model(model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载BCAT模型
    
    参数:
        model_dir: 模型目录
        device: 计算设备
    
    返回:
        包含模型和分词器的字典
    """
    try:
        logger.info(f"加载BCAT模型: {model_dir}")
        
        sys.path.append('model_pro')
        from BCAT import BCAT
        from transformers import BertTokenizer
        
        # 加载模型配置
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 初始化模型
        model = BCAT(**config)
        
        # 加载模型权重
        model_path = os.path.join(model_dir, 'model.pt')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 加载分词器
        tokenizer_path = os.path.join(model_dir, 'tokenizer')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        logger.info("BCAT模型加载成功")
        return {
            'model': model,
            'tokenizer': tokenizer,
            'device': device,
            'config': config
        }
    except Exception as e:
        logger.error(f"加载BCAT模型失败: {e}")
        raise

def load_topic_classifier(model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载话题分类模型
    
    参数:
        model_dir: 模型目录
        device: 计算设备
    
    返回:
        包含模型、分词器和标签映射的字典
    """
    try:
        logger.info(f"加载话题分类模型: {model_dir}")
        
        # 尝试加载transformers模型
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # 加载模型
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.to(device)
            model.eval()
            
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # 加载标签映射
            labels_path = os.path.join(model_dir, 'labels.json')
            if os.path.exists(labels_path):
                with open(labels_path, 'r', encoding='utf-8') as f:
                    labels_map = json.load(f)
            else:
                # 尝试从config中读取标签
                if hasattr(model.config, 'id2label'):
                    labels_map = model.config.id2label
                else:
                    labels_map = {}
            
            logger.info("话题分类模型加载成功 (transformers)")
            return {
                'model': model,
                'tokenizer': tokenizer,
                'labels_map': labels_map,
                'device': device
            }
        except Exception as e:
            logger.warning(f"使用transformers加载失败，尝试其他方法: {e}")
        
        # 尝试加载PyTorch模型
        model_path = os.path.join(model_dir, 'model.pt')
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=device)
            
            # 加载分词器
            tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
            else:
                tokenizer = None
            
            # 加载标签映射
            labels_path = os.path.join(model_dir, 'labels.json')
            if os.path.exists(labels_path):
                with open(labels_path, 'r', encoding='utf-8') as f:
                    labels_map = json.load(f)
            else:
                labels_map = {}
            
            logger.info("话题分类模型加载成功 (PyTorch)")
            return {
                'model': model,
                'tokenizer': tokenizer,
                'labels_map': labels_map,
                'device': device
            }
        
        raise ValueError(f"无法加载模型: {model_dir}")
    except Exception as e:
        logger.error(f"加载话题分类模型失败: {e}")
        raise

def load_echarts_optimizer():
    """
    加载ECharts优化器，用于提升大数据渲染性能
    
    返回:
        ECharts优化器对象
    """
    try:
        class EChartsOptimizer:
            def __init__(self):
                self.chunk_size = 1000  # 分块大小
                logger.info("ECharts优化器初始化成功")
            
            def optimize_option(self, option):
                """优化ECharts配置，提升大数据渲染性能"""
                if not option:
                    return option
                
                # 深拷贝以避免修改原始对象
                import copy
                option = copy.deepcopy(option)
                
                # 添加渐进式渲染
                if 'progressive' not in option:
                    option['progressive'] = 300  # 每帧渲染的数据点数量
                
                if 'progressiveThreshold' not in option:
                    option['progressiveThreshold'] = 5000  # 启动渐进式渲染的阈值
                
                if 'series' in option and isinstance(option['series'], list):
                    for series in option['series']:
                        # 对大数据系列应用优化
                        if 'data' in series and isinstance(series['data'], list) and len(series['data']) > 5000:
                            # 大数据采样
                            if series.get('type') in ['scatter', 'line']:
                                self._optimize_large_data_series(series)
                
                return option
            
            def _optimize_large_data_series(self, series):
                """优化大数据系列"""
                # 添加大数据优化选项
                series['large'] = True
                series['largeThreshold'] = 2000
                
                # 按需设置抽样
                if len(series['data']) > 50000:
                    # 对非常大的数据集进行抽样
                    step = max(1, len(series['data']) // 50000)
                    series['data'] = series['data'][::step]
                    series['sampling'] = 'average'
                
                return series
            
            def chunk_process_data(self, data, process_func):
                """分块处理大数据"""
                result = []
                for i in range(0, len(data), self.chunk_size):
                    chunk = data[i:i + self.chunk_size]
                    result.extend(process_func(chunk))
                return result
        
        return EChartsOptimizer()
    except Exception as e:
        logger.error(f"加载ECharts优化器失败: {e}")
        return None

# 导出所有加载函数
__all__ = [
    'load_sentiment_model',
    'load_bert_ctm_model',
    'load_bcat_model',
    'load_topic_classifier',
    'load_echarts_optimizer'
] 