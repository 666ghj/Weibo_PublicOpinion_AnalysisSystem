import os
import time
import threading
import logging
import gc
import torch
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta

logger = logging.getLogger('model_manager')
logger.setLevel(logging.INFO)

class ModelManager:
    """
    模型管理器 - 实现模型预加载和按需卸载技术
    
    功能：
    1. 预加载经常使用的模型，减少加载等待时间
    2. 使用LRU (Least Recently Used) 策略管理内存中加载的模型
    3. 支持模型的异步加载和监控
    4. 自动检测并释放长时间未使用的模型内存
    5. 提供模型使用统计
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
            
        # 已加载模型的缓存，使用OrderedDict实现LRU
        self.loaded_models = OrderedDict()
        # 模型使用统计
        self.model_stats = {}
        # 模型预热配置
        self.preload_config = {}
        # 最大内存占用（GB）
        self.max_memory_usage = float(os.getenv('MAX_MODEL_MEMORY_USAGE', '4.0'))
        # 模型加载中的锁
        self.loading_locks = {}
        # 模型卸载超时（分钟）
        self.unload_timeout = int(os.getenv('MODEL_UNLOAD_TIMEOUT', '30'))
        
        # 启动模型监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_models, daemon=True)
        self.monitor_thread.start()
        
        self.initialized = True
        logger.info(f"模型管理器初始化完成，最大内存占用: {self.max_memory_usage}GB")
    
    def register_model(self, model_id, model_path, preload=False, model_size_gb=0.5, 
                      load_function=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        注册模型，可选设置为预加载
        
        参数:
            model_id: 模型唯一标识符
            model_path: 模型路径
            preload: 是否预加载
            model_size_gb: 模型估计大小（GB）
            load_function: 自定义加载函数，签名为 load_function(model_path, device) -> model
            device: 加载模型的设备
        """
        self.preload_config[model_id] = {
            'model_path': model_path,
            'preload': preload,
            'model_size_gb': model_size_gb,
            'load_function': load_function,
            'device': device
        }
        
        self.model_stats[model_id] = {
            'load_count': 0,
            'use_count': 0,
            'total_load_time': 0,
            'last_used': None,
            'avg_load_time': 0
        }
        
        if preload:
            logger.info(f"模型 {model_id} 已注册并标记为预加载")
            # 启动预加载线程
            threading.Thread(target=self._preload_model, args=(model_id,), daemon=True).start()
        else:
            logger.info(f"模型 {model_id} 已注册")
        
        return True
    
    def get_model(self, model_id):
        """
        获取模型，如果未加载则加载
        
        参数:
            model_id: 模型唯一标识符
            
        返回:
            加载好的模型对象
        """
        if model_id not in self.preload_config:
            raise ValueError(f"模型 {model_id} 未注册")
            
        # 更新最后使用时间
        self.model_stats[model_id]['last_used'] = datetime.now()
        self.model_stats[model_id]['use_count'] += 1
        
        # 检查模型是否已加载
        if model_id in self.loaded_models:
            # 将模型移至OrderedDict末尾，表示最近使用
            model = self.loaded_models.pop(model_id)
            self.loaded_models[model_id] = model
            logger.debug(f"使用已加载的模型: {model_id}")
            return model
            
        # 获取模型加载锁，防止并发加载同一模型
        if model_id not in self.loading_locks:
            self.loading_locks[model_id] = threading.Lock()
            
        # 加锁加载模型
        with self.loading_locks[model_id]:
            # 再次检查模型是否已被其他线程加载
            if model_id in self.loaded_models:
                return self.loaded_models[model_id]
                
            # 检查是否有足够内存
            self._ensure_memory_available(self.preload_config[model_id]['model_size_gb'])
            
            # 加载模型
            start_time = time.time()
            model = self._load_model(model_id)
            load_time = time.time() - start_time
            
            # 更新统计
            self.model_stats[model_id]['load_count'] += 1
            self.model_stats[model_id]['total_load_time'] += load_time
            self.model_stats[model_id]['avg_load_time'] = (
                self.model_stats[model_id]['total_load_time'] / 
                self.model_stats[model_id]['load_count']
            )
            
            logger.info(f"模型 {model_id} 加载完成，耗时: {load_time:.2f}秒")
            
            # 存储模型
            self.loaded_models[model_id] = model
            return model
    
    def unload_model(self, model_id):
        """
        手动卸载模型
        
        参数:
            model_id: 模型唯一标识符
        """
        if model_id in self.loaded_models:
            logger.info(f"手动卸载模型: {model_id}")
            del self.loaded_models[model_id]
            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False
    
    def get_model_stats(self):
        """获取所有模型的使用统计"""
        result = {}
        for model_id, stats in self.model_stats.items():
            is_loaded = model_id in self.loaded_models
            result[model_id] = {
                **stats,
                'is_loaded': is_loaded,
                'preload': self.preload_config[model_id]['preload'],
                'model_size_gb': self.preload_config[model_id]['model_size_gb'],
                'device': self.preload_config[model_id]['device'],
            }
        return result
    
    def preload_all(self):
        """预加载所有标记为预加载的模型"""
        for model_id, config in self.preload_config.items():
            if config['preload'] and model_id not in self.loaded_models:
                threading.Thread(target=self._preload_model, args=(model_id,), daemon=True).start()
    
    def _preload_model(self, model_id):
        """预加载单个模型的内部方法"""
        try:
            logger.info(f"开始预加载模型: {model_id}")
            # 确保有足够内存
            self._ensure_memory_available(self.preload_config[model_id]['model_size_gb'])
            
            # 加载模型
            start_time = time.time()
            model = self._load_model(model_id)
            load_time = time.time() - start_time
            
            # 更新统计
            self.model_stats[model_id]['load_count'] += 1
            self.model_stats[model_id]['total_load_time'] += load_time
            self.model_stats[model_id]['avg_load_time'] = (
                self.model_stats[model_id]['total_load_time'] / 
                self.model_stats[model_id]['load_count']
            )
            
            # 存储模型
            self.loaded_models[model_id] = model
            logger.info(f"模型 {model_id} 预加载完成，耗时: {load_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"预加载模型 {model_id} 失败: {e}")
    
    def _load_model(self, model_id):
        """加载模型的内部方法"""
        config = self.preload_config[model_id]
        
        if config['load_function'] is not None:
            # 使用自定义加载函数
            return config['load_function'](config['model_path'], config['device'])
        
        # 默认加载逻辑 - 根据文件扩展名确定加载方式
        model_path = config['model_path']
        device = config['device']
        
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            # PyTorch模型
            return torch.load(model_path, map_location=device)
        elif model_path.endswith('.pkl'):
            # Pickle模型
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            # 尝试作为目录加载
            if os.path.isdir(model_path):
                # 如果是目录，尝试加载预训练模型
                try:
                    from transformers import AutoModel, AutoTokenizer
                    model = AutoModel.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    return {'model': model.to(device), 'tokenizer': tokenizer}
                except ImportError:
                    logger.error("transformers库未安装，无法加载预训练模型")
                    raise
                except Exception as e:
                    logger.error(f"加载预训练模型失败: {e}")
                    raise
            
            raise ValueError(f"无法确定如何加载模型: {model_path}")
    
    def _ensure_memory_available(self, required_gb):
        """确保有足够的内存来加载新模型"""
        # 如果当前没有加载的模型，直接返回
        if not self.loaded_models:
            return
            
        # 计算当前已加载模型的总内存
        current_usage = sum(
            self.preload_config[model_id]['model_size_gb'] 
            for model_id in self.loaded_models
        )
        
        # 如果添加新模型后超过限制，需要卸载一些模型
        while current_usage + required_gb > self.max_memory_usage and self.loaded_models:
            # 卸载最久未使用的模型（OrderedDict的首项）
            oldest_model_id, _ = next(iter(self.loaded_models.items()))
            # 检查是否是预加载且最近使用过的模型
            if (self.preload_config[oldest_model_id]['preload'] and
                self.model_stats[oldest_model_id]['last_used'] and
                (datetime.now() - self.model_stats[oldest_model_id]['last_used']) < 
                timedelta(minutes=self.unload_timeout)):
                # 跳过预加载且最近使用过的模型
                # 将该模型移至OrderedDict末尾
                model = self.loaded_models.pop(oldest_model_id)
                self.loaded_models[oldest_model_id] = model
                # 如果所有模型都是预加载的且最近使用过，允许超过限制
                if len(self.loaded_models) <= 1:
                    break
                continue
                
            # 卸载模型并更新内存使用
            model_size = self.preload_config[oldest_model_id]['model_size_gb']
            del self.loaded_models[oldest_model_id]
            current_usage -= model_size
            logger.info(f"自动卸载模型以释放内存: {oldest_model_id} ({model_size}GB)")
            
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _monitor_models(self):
        """监控并管理模型的内部线程方法"""
        while True:
            try:
                # 检查长时间未使用的非预加载模型
                current_time = datetime.now()
                for model_id in list(self.loaded_models.keys()):
                    if (not self.preload_config[model_id]['preload'] and
                        self.model_stats[model_id]['last_used'] and
                        (current_time - self.model_stats[model_id]['last_used']) > 
                        timedelta(minutes=self.unload_timeout)):
                        # 卸载长时间未使用的非预加载模型
                        logger.info(f"卸载长时间未使用的模型: {model_id}")
                        del self.loaded_models[model_id]
                        # 强制垃圾回收
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                # 每5分钟检查一次
                time.sleep(300)
            except Exception as e:
                logger.error(f"模型监控线程出错: {e}")
                time.sleep(300)

# 创建全局模型管理器实例
model_manager = ModelManager()

# 注册示例函数
def register_sentiment_model():
    """注册情感分析模型示例"""
    from utils.model_loader import load_sentiment_model  # 假设您有一个加载情感模型的函数
    
    try:
        model_path = os.path.join('model', 'sentiment.marshal.3')
        model_manager.register_model(
            model_id='sentiment_basic',
            model_path=model_path,
            preload=True,
            model_size_gb=0.2,
            load_function=load_sentiment_model
        )
        return True
    except Exception as e:
        logger.error(f"注册情感分析模型失败: {e}")
        return False

def register_bert_model():
    """注册BERT模型示例"""
    try:
        model_path = os.path.join('model_pro', 'bert_model')
        model_manager.register_model(
            model_id='bert_classifier',
            model_path=model_path,
            preload=True,
            model_size_gb=0.8
        )
        return True
    except Exception as e:
        logger.error(f"注册BERT模型失败: {e}")
        return False

# 自动注册常用模型（在导入时执行）
try:
    register_sentiment_model()
    register_bert_model()
except Exception as e:
    logger.error(f"自动注册模型失败: {e}") 