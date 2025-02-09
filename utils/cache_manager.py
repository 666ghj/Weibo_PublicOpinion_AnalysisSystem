import json
import os
import time
from datetime import datetime, timedelta
import threading
import queue

class PredictionCache:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PredictionCache, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.cache_dir = 'cache/predictions'
            self.cache_duration = timedelta(hours=24)  # 缓存24小时
            self.cache = {}
            self.cache_queue = queue.Queue()
            self.initialized = True
            
            # 确保缓存目录存在
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # 启动缓存清理线程
            self.cleanup_thread = threading.Thread(target=self._cleanup_old_cache, daemon=True)
            self.cleanup_thread.start()
            
            # 加载现有缓存
            self._load_cache()
    
    def _load_cache(self):
        """加载磁盘上的缓存文件"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        # 检查缓存是否过期
                        if self._is_cache_valid(cache_data['timestamp']):
                            topic = filename[:-5]  # 移除.json后缀
                            self.cache[topic] = cache_data
                        else:
                            # 删除过期缓存文件
                            os.remove(filepath)
        except Exception as e:
            print(f"加载缓存失败: {e}")
    
    def _cleanup_old_cache(self):
        """定期清理过期缓存的后台线程"""
        while True:
            try:
                # 检查并清理内存缓存
                current_time = datetime.now()
                expired_topics = []
                
                for topic, cache_data in self.cache.items():
                    if not self._is_cache_valid(cache_data['timestamp']):
                        expired_topics.append(topic)
                        
                # 删除过期缓存
                for topic in expired_topics:
                    del self.cache[topic]
                    cache_file = os.path.join(self.cache_dir, f"{topic}.json")
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                
                # 休眠1小时后再次检查
                time.sleep(3600)
            except Exception as e:
                print(f"清理缓存时出错: {e}")
                time.sleep(3600)  # 发生错误时也等待1小时
    
    def _is_cache_valid(self, timestamp):
        """检查缓存是否有效"""
        cache_time = datetime.fromtimestamp(timestamp)
        return datetime.now() - cache_time < self.cache_duration
    
    def get(self, topic):
        """获取话题的预测缓存"""
        if topic in self.cache and self._is_cache_valid(self.cache[topic]['timestamp']):
            return self.cache[topic]['prediction']
        return None
    
    def set(self, topic, prediction):
        """设置话题的预测缓存"""
        cache_data = {
            'prediction': prediction,
            'timestamp': datetime.now().timestamp()
        }
        
        # 更新内存缓存
        self.cache[topic] = cache_data
        
        # 异步保存到磁盘
        self.cache_queue.put((topic, cache_data))
        threading.Thread(target=self._save_cache_to_disk, daemon=True).start()
    
    def _save_cache_to_disk(self):
        """异步保存缓存到磁盘"""
        try:
            while not self.cache_queue.empty():
                topic, cache_data = self.cache_queue.get()
                cache_file = os.path.join(self.cache_dir, f"{topic}.json")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存到磁盘失败: {e}")

# 创建全局缓存实例
prediction_cache = PredictionCache() 