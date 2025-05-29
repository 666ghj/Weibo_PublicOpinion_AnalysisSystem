import json
import os
import time
import shutil
from datetime import datetime, timedelta
import threading
import queue
from collections import OrderedDict
import pickle
import hashlib
import logging

logger = logging.getLogger('cache_manager')
logger.setLevel(logging.INFO)

class LRUCache:
    """实现LRU (Least Recently Used) 缓存策略"""
    
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        # 访问元素时，将其移至末尾，表示最近使用
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        # 如果键已存在，更新值并将其移至末尾
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
            return
        
        # 如果缓存已满，删除最久未使用的项（OrderedDict 的首项）
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        
        # 添加新项至末尾
        self.cache[key] = value
    
    def remove(self, key):
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        self.cache.clear()
    
    def __len__(self):
        return len(self.cache)
    
    def get_all_keys(self):
        return list(self.cache.keys())


class CacheManager:
    """两级缓存系统：内存LRU缓存 + 磁盘持久化缓存"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CacheManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, name="default", memory_capacity=1000, cache_duration=24,
                disk_cache_dir="cache", flush_interval=5):
        if hasattr(self, 'initialized'):
            return
        
        self.name = name
        self.memory_cache = LRUCache(memory_capacity)
        self.disk_cache_dir = os.path.join(disk_cache_dir, name)
        self.cache_duration = timedelta(hours=cache_duration)
        self.flush_interval = flush_interval  # 定时将内存缓存刷新到磁盘的间隔（分钟）
        self.cache_stats = {"hits": 0, "misses": 0, "disk_hits": 0}
        self.disk_queue = queue.Queue()
            self.initialized = True
            
            # 确保缓存目录存在
        os.makedirs(self.disk_cache_dir, exist_ok=True)
            
        # 启动缓存管理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_and_flush_task, daemon=True)
            self.cleanup_thread.start()
            
        # 启动磁盘写入线程
        self.disk_writer_thread = threading.Thread(target=self._disk_writer_task, daemon=True)
        self.disk_writer_thread.start()
        
        logger.info(f"初始化缓存管理器: {name}，内存容量: {memory_capacity}项，缓存时间: {cache_duration}小时")
    
    def _get_cache_key(self, key):
        """标准化缓存键"""
        if isinstance(key, str):
            return key
        return hashlib.md5(str(key).encode()).hexdigest()
    
    def _get_disk_path(self, key):
        """获取磁盘缓存路径"""
        safe_key = self._get_cache_key(key)
        return os.path.join(self.disk_cache_dir, f"{safe_key}.cache")
    
    def _is_cache_valid(self, timestamp):
        """检查缓存是否过期"""
        cache_time = datetime.fromtimestamp(timestamp)
        return datetime.now() - cache_time < self.cache_duration
    
    def get(self, key):
        """获取缓存数据，首先检查内存，然后检查磁盘"""
        cache_key = self._get_cache_key(key)
        
        # 1. 检查内存缓存
        cache_data = self.memory_cache.get(cache_key)
        if cache_data is not None:
            if self._is_cache_valid(cache_data['timestamp']):
                self.cache_stats["hits"] += 1
                logger.debug(f"内存缓存命中: {key}")
                return cache_data['data']
            else:
                # 过期缓存，从内存中删除
                self.memory_cache.remove(cache_key)
        
        # 2. 检查磁盘缓存
        disk_path = self._get_disk_path(cache_key)
        if os.path.exists(disk_path):
            try:
                with open(disk_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if self._is_cache_valid(cache_data['timestamp']):
                    # 从磁盘加载后，放入内存缓存
                    self.memory_cache.put(cache_key, cache_data)
                    self.cache_stats["disk_hits"] += 1
                    logger.debug(f"磁盘缓存命中: {key}")
                    return cache_data['data']
                else:
                    # 过期缓存，删除磁盘文件
                    os.remove(disk_path)
            except Exception as e:
                logger.warning(f"读取磁盘缓存失败: {key}, 错误: {e}")
        
        self.cache_stats["misses"] += 1
        logger.debug(f"缓存未命中: {key}")
        return None
    
    def set(self, key, data, immediate_disk_write=False):
        """设置缓存数据，同时更新内存和安排磁盘写入"""
        cache_key = self._get_cache_key(key)
        cache_data = {
            'data': data,
            'timestamp': datetime.now().timestamp()
        }
        
        # 更新内存缓存
        self.memory_cache.put(cache_key, cache_data)
        
        # 安排写入磁盘
        if immediate_disk_write:
            self._write_to_disk(cache_key, cache_data)
        else:
            self.disk_queue.put((cache_key, cache_data))
        
        logger.debug(f"缓存已设置: {key}")
        return True
    
    def invalidate(self, key):
        """使指定键的缓存失效"""
        cache_key = self._get_cache_key(key)
        
        # 从内存中删除
        self.memory_cache.remove(cache_key)
        
        # 从磁盘中删除
        disk_path = self._get_disk_path(cache_key)
        if os.path.exists(disk_path):
            try:
                os.remove(disk_path)
                logger.debug(f"缓存已失效: {key}")
            except Exception as e:
                logger.warning(f"删除磁盘缓存失败: {key}, 错误: {e}")
        
        return True
    
    def clear_all(self):
        """清除所有缓存"""
        # 清除内存缓存
        self.memory_cache.clear()
        
        # 清除磁盘缓存
        try:
            shutil.rmtree(self.disk_cache_dir)
            os.makedirs(self.disk_cache_dir, exist_ok=True)
            logger.info(f"所有缓存已清除: {self.name}")
        except Exception as e:
            logger.error(f"清除磁盘缓存失败: {e}")
        
        # 重置统计信息
        self.cache_stats = {"hits": 0, "misses": 0, "disk_hits": 0}
        
        return True
    
    def get_stats(self):
        """获取缓存统计信息"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        total_hits = self.cache_stats["hits"] + self.cache_stats["disk_hits"]
        
        memory_size = len(self.memory_cache)
        disk_size = len([f for f in os.listdir(self.disk_cache_dir) if f.endswith('.cache')])
        
        return {
            "name": self.name,
            "memory_items": memory_size,
            "disk_items": disk_size,
            "memory_hits": self.cache_stats["hits"],
            "disk_hits": self.cache_stats["disk_hits"],
            "misses": self.cache_stats["misses"],
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "two_level_hit_rate": (total_hits / total_requests * 100) if total_requests > 0 else 0
        }
    
    def _write_to_disk(self, cache_key, cache_data):
        """将缓存写入磁盘"""
        disk_path = self._get_disk_path(cache_key)
        try:
            with open(disk_path, 'wb') as f:
                pickle.dump(cache_data, f)
            return True
        except Exception as e:
            logger.warning(f"写入磁盘缓存失败: {cache_key}, 错误: {e}")
            return False
    
    def _disk_writer_task(self):
        """后台线程，负责将缓存写入磁盘"""
        while True:
            try:
                # 尝试从队列获取条目，超时后继续循环
                try:
                    cache_key, cache_data = self.disk_queue.get(timeout=1)
                    self._write_to_disk(cache_key, cache_data)
                    self.disk_queue.task_done()
                except queue.Empty:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"磁盘写入线程出错: {e}")
                time.sleep(5)  # 发生错误时等待一段时间
    
    def _cleanup_and_flush_task(self):
        """后台线程，负责清理过期缓存和定期刷新内存缓存到磁盘"""
        while True:
            try:
                # 1. 清理过期的内存缓存
                current_time = datetime.now()
                for key in self.memory_cache.get_all_keys():
                    cache_data = self.memory_cache.get(key)
                    if not self._is_cache_valid(cache_data['timestamp']):
                        self.memory_cache.remove(key)
                
                # 2. 清理过期的磁盘缓存
                for filename in os.listdir(self.disk_cache_dir):
                    if filename.endswith('.cache'):
                        filepath = os.path.join(self.disk_cache_dir, filename)
                        try:
                            with open(filepath, 'rb') as f:
                                cache_data = pickle.load(f)
                            if not self._is_cache_valid(cache_data['timestamp']):
                                os.remove(filepath)
                        except Exception as e:
                            # 清理损坏的缓存文件
                            logger.warning(f"读取缓存文件失败，将删除: {filepath}, 错误: {e}")
                            os.remove(filepath)
                
                # 3. 将内存缓存刷新到磁盘
                # 注意：这会重写已经写入磁盘的缓存，但确保内存和磁盘保持同步
                for key in self.memory_cache.get_all_keys():
                    cache_data = self.memory_cache.get(key)
                    self._write_to_disk(key, cache_data)
                
                # 每小时执行一次清理
                time.sleep(3600)
            except Exception as e:
                logger.error(f"缓存清理线程出错: {e}")
                time.sleep(3600)  # 发生错误时也等待一段时间


# 创建不同领域的缓存实例
prediction_cache = CacheManager(name="predictions", memory_capacity=500, cache_duration=24)
sentiment_cache = CacheManager(name="sentiment", memory_capacity=1000, cache_duration=12)
topic_cache = CacheManager(name="topics", memory_capacity=200, cache_duration=6)
user_data_cache = CacheManager(name="user_data", memory_capacity=300, cache_duration=48)

# 向后兼容的别名
PredictionCache = CacheManager
# 为保持向后兼容，我们保留原来的prediction_cache
prediction_cache_old = prediction_cache 