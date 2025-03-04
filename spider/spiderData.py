from spiderDataPackage.spiderNav import start as spiderNav
from spiderDataPackage.spiderContent import start as spiderContent
from spiderDataPackage.spiderComments import start as spiderComments
from spiderDataPackage.settings import navAddr
import os
import requests
import time
import random
import logging
from bs4 import BeautifulSoup
from datetime import datetime
from utils.logger import spider_logger as logging
from utils.db_manager import DatabaseManager
from cachetools import TTLCache, LRUCache
from typing import List, Dict, Any
import pandas as pd

def spiderData():
    if not os.path.exists(navAddr):
        print('正在爬取导航栏数据')
        spiderNav()
    print('正在爬取文章数据')
    spiderContent(9,1)
    print('正在爬取文章评论数据')
    spiderComments()

class SpiderData:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = 'https://s.weibo.com'
        self.db = DatabaseManager()
        
        # 初始化缓存
        self.data_cache = TTLCache(maxsize=1000, ttl=3600)  # 1小时TTL缓存
        self.html_cache = LRUCache(maxsize=100)  # LRU缓存最近的100个页面
        
        # 批量插入缓冲区
        self.insert_buffer = []
        self.buffer_size = 50  # 每50条数据批量插入一次
    
    def _get_cached_page(self, url: str) -> str:
        """获取缓存的页面内容"""
        return self.html_cache.get(url)
    
    def _cache_page(self, url: str, content: str):
        """缓存页面内容"""
        self.html_cache[url] = content
    
    def _get_cached_data(self, key: str) -> Dict[str, Any]:
        """获取缓存的数据"""
        return self.data_cache.get(key)
    
    def _cache_data(self, key: str, data: Dict[str, Any]):
        """缓存数据"""
        self.data_cache[key] = data
    
    def _flush_buffer(self):
        """将缓冲区数据批量插入数据库"""
        if not self.insert_buffer:
            return
        
        try:
            connection = self.db.get_connection()
            with connection.cursor() as cursor:
                # 使用pandas进行高效的批量插入
                df = pd.DataFrame(self.insert_buffer)
                
                # 构建批量插入SQL
                columns = ', '.join(df.columns)
                values = ', '.join(['%s'] * len(df.columns))
                sql = f"""
                INSERT INTO article ({columns})
                VALUES ({values})
                ON DUPLICATE KEY UPDATE
                forward_count = VALUES(forward_count),
                comment_count = VALUES(comment_count),
                like_count = VALUES(like_count),
                crawl_time = VALUES(crawl_time)
                """
                
                # 执行批量插入
                cursor.executemany(sql, df.values.tolist())
                connection.commit()
                
                logging.info(f"成功批量插入 {len(self.insert_buffer)} 条数据")
                self.insert_buffer.clear()
                
        except Exception as e:
            logging.error(f"批量插入数据失败: {e}")
            if connection:
                connection.rollback()
    
    def crawl_topic(self, topic: str, depth: int = 3, interval: int = 5,
                    max_retries: int = 3, timeout: int = 30):
        """爬取指定话题的微博内容"""
        # 参数验证
        if not isinstance(depth, int) or depth < 1 or depth > 10:
            raise ValueError("爬取深度必须在1-10页之间")
        if not isinstance(interval, int) or interval < 3 or interval > 30:
            raise ValueError("请求间隔必须在3-30秒之间")
        if not isinstance(max_retries, int) or max_retries < 1 or max_retries > 5:
            raise ValueError("最大重试次数必须在1-5次之间")
        if not isinstance(timeout, int) or timeout < 10 or timeout > 60:
            raise ValueError("请求超时时间必须在10-60秒之间")
        
        logging.info(f"开始爬取话题: {topic}, 参数: depth={depth}, interval={interval}, max_retries={max_retries}, timeout={timeout}")
        
        for page in range(1, depth + 1):
            retries = 0
            while retries < max_retries:
                try:
                    url = f"{self.base_url}/weibo?q={topic}&page={page}"
                    
                    # 检查缓存
                    cached_content = self._get_cached_page(url)
                    if cached_content:
                        self._parse_page(cached_content)
                        logging.info(f"使用缓存数据: {topic} 第 {page} 页")
                        break
                    
                    response = requests.get(url, headers=self.headers, timeout=timeout)
                    
                    if response.status_code == 200:
                        # 缓存页面内容
                        self._cache_page(url, response.text)
                        self._parse_page(response.text)
                        logging.info(f"成功爬取话题 {topic} 第 {page} 页")
                        break
                    else:
                        logging.warning(f"请求失败，状态码: {response.status_code}")
                        retries += 1
                
                except requests.RequestException as e:
                    logging.error(f"请求异常: {e}")
                    retries += 1
                
                if retries < max_retries:
                    sleep_time = interval * (1 + random.random())
                    logging.info(f"等待 {sleep_time:.2f} 秒后重试...")
                    time.sleep(sleep_time)
            
            if retries == max_retries:
                logging.error(f"话题 {topic} 第 {page} 页爬取失败，已达到最大重试次数")
                continue
            
            # 在页面之间添加随机延迟
            if page < depth:
                sleep_time = interval * (1 + random.random())
                logging.info(f"等待 {sleep_time:.2f} 秒后继续...")
                time.sleep(sleep_time)
        
        # 最后刷新缓冲区
        self._flush_buffer()
    
    def _parse_page(self, html_content: str):
        """解析页面内容并保存数据"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            weibo_items = soup.find_all('div', class_='card-wrap')
            
            for item in weibo_items:
                try:
                    # 提取微博内容
                    content = item.find('p', class_='txt')
                    if not content:
                        continue
                    
                    # 提取用户信息
                    user_info = item.find('a', class_='name')
                    if not user_info:
                        continue
                    
                    # 提取发布时间
                    time_info = item.find('p', class_='from')
                    
                    # 提取互动数据
                    actions = item.find_all('li', class_='action')
                    
                    # 构建数据字典
                    weibo_data = {
                        'content': content.text.strip(),
                        'user_name': user_info.text.strip(),
                        'publish_time': time_info.text.strip() if time_info else '',
                        'forward_count': self._extract_number(actions[0].text) if len(actions) > 0 else 0,
                        'comment_count': self._extract_number(actions[1].text) if len(actions) > 1 else 0,
                        'like_count': self._extract_number(actions[2].text) if len(actions) > 2 else 0,
                        'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # 添加到插入缓冲区
                    self.insert_buffer.append(weibo_data)
                    
                    # 如果缓冲区达到阈值，执行批量插入
                    if len(self.insert_buffer) >= self.buffer_size:
                        self._flush_buffer()
                    
                except Exception as e:
                    logging.error(f"解析微博项时出错: {e}")
                    continue
            
        except Exception as e:
            logging.error(f"解析页面时出错: {e}")
    
    def _extract_number(self, text: str) -> int:
        """从文本中提取数字"""
        try:
            return int(''.join(filter(str.isdigit, text)))
        except ValueError:
            return 0

if __name__ == '__main__':
    spiderData()