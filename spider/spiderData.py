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
    
    def crawl_topic(self, topic, depth=3, interval=5, max_retries=3, timeout=30):
        """
        爬取指定话题的微博内容
        
        :param topic: 要爬取的话题
        :param depth: 爬取深度（页数）
        :param interval: 请求间隔时间（秒）
        :param max_retries: 最大重试次数
        :param timeout: 请求超时时间（秒）
        """
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
                    response = requests.get(url, headers=self.headers, timeout=timeout)
                    
                    if response.status_code == 200:
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
    
    def _parse_page(self, html_content):
        """
        解析页面内容并保存数据
        
        :param html_content: 页面HTML内容
        """
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
                    
                    # 保存到数据库
                    self._save_to_database(weibo_data)
                    
                except Exception as e:
                    logging.error(f"解析微博项时出错: {e}")
                    continue
            
        except Exception as e:
            logging.error(f"解析页面时出错: {e}")
    
    def _extract_number(self, text):
        """
        从文本中提取数字
        
        :param text: 包含数字的文本
        :return: 提取的数字，如果没有找到则返回0
        """
        try:
            return int(''.join(filter(str.isdigit, text)))
        except ValueError:
            return 0
    
    def _save_to_database(self, data):
        """
        将数据保存到数据库
        
        :param data: 要保存的数据字典
        """
        connection = None
        try:
            connection = self.db.get_connection()
            
            with connection.cursor() as cursor:
                # 插入文章数据
                sql = """
                INSERT INTO article (content, user_name, publish_time, forward_count, 
                                   comment_count, like_count, crawl_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    data['content'],
                    data['user_name'],
                    data['publish_time'],
                    data['forward_count'],
                    data['comment_count'],
                    data['like_count'],
                    data['crawl_time']
                ))
                
                connection.commit()
                logging.info(f"成功保存微博数据: {data['content'][:30]}...")
                
        except Exception as e:
            logging.error(f"保存数据时出错: {e}")
            if connection:
                connection.rollback()

if __name__ == '__main__':
    spiderData()