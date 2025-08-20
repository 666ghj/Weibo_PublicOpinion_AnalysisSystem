#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BroadTopicExtraction模块 - 数据库管理器
只负责新闻数据和话题分析的存储和查询
"""

import sys
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional
import pymysql
from pymysql.cursors import DictCursor

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    raise ImportError("无法导入config.py配置文件")

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        """初始化数据库管理器"""
        self.connection = None
        self.connect()
    
    def connect(self):
        """连接数据库"""
        try:
            self.connection = pymysql.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                database=config.DB_NAME,
                charset=config.DB_CHARSET,
                autocommit=True,
                cursorclass=DictCursor
            )
            print(f"成功连接到数据库: {config.DB_NAME}")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("数据库连接已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ==================== 新闻数据操作 ====================
    
    def save_daily_news(self, news_data: List[Dict], crawl_date: date = None) -> int:
        """
        保存每日新闻数据，如果当天已有数据则覆盖
        
        Args:
            news_data: 新闻数据列表
            crawl_date: 爬取日期，默认为今天
        
        Returns:
            保存的新闻数量
        """
        if not crawl_date:
            crawl_date = date.today()
        
        current_timestamp = int(datetime.now().timestamp())
        
        try:
            cursor = self.connection.cursor()
            
            # 先删除当天所有的新闻记录（覆盖模式）
            delete_query = "DELETE FROM daily_news WHERE crawl_date = %s"
            deleted_count = cursor.execute(delete_query, (crawl_date,))
            if deleted_count > 0:
                print(f"覆盖模式：删除了当天已有的 {deleted_count} 条新闻记录")
            
            # 批量插入新记录
            saved_count = 0
            for news_item in news_data:
                try:
                    # 简化的新闻ID生成
                    news_id = f"{news_item.get('source', 'unknown')}_{news_item.get('id', news_item.get('rank', 0))}"
                    
                    # 插入新记录
                    insert_query = """
                        INSERT INTO daily_news (
                            news_id, source_platform, title, url, crawl_date, 
                            rank_position, add_ts
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (
                        news_id,
                        news_item.get('source', 'unknown'),
                        news_item.get('title', ''),
                        news_item.get('url', ''),
                        crawl_date,
                        news_item.get('rank', None),
                        current_timestamp
                    ))
                    saved_count += 1
                    
                except Exception as e:
                    print(f"保存单条新闻失败: {e}")
                    continue
            
            print(f"成功保存 {saved_count} 条新闻记录")
            return saved_count
            
        except Exception as e:
            print(f"保存新闻数据失败: {e}")
            return 0
    
    def get_daily_news(self, crawl_date: date = None) -> List[Dict]:
        """
        获取每日新闻数据
        
        Args:
            crawl_date: 爬取日期，默认为今天
        
        Returns:
            新闻列表
        """
        if not crawl_date:
            crawl_date = date.today()
        
        query = """
            SELECT * FROM daily_news 
            WHERE crawl_date = %s 
            ORDER BY rank_position ASC
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query, (crawl_date,))
        return cursor.fetchall()
    
    # ==================== 话题数据操作 ====================
    
    def save_daily_topics(self, keywords: List[str], summary: str, extract_date: date = None) -> bool:
        """
        保存每日话题分析
        
        Args:
            keywords: 话题关键词列表
            summary: 新闻分析总结
            extract_date: 提取日期，默认为今天
        
        Returns:
            是否保存成功
        """
        if not extract_date:
            extract_date = date.today()
        
        current_timestamp = int(datetime.now().timestamp())
        
        try:
            cursor = self.connection.cursor()
            
            # 检查今天是否已有记录
            check_query = "SELECT id FROM daily_topics WHERE extract_date = %s"
            cursor.execute(check_query, (extract_date,))
            existing = cursor.fetchone()
            
            keywords_json = json.dumps(keywords, ensure_ascii=False)
            
            if existing:
                # 更新现有记录
                update_query = """
                    UPDATE daily_topics 
                    SET keywords = %s, summary = %s, add_ts = %s
                    WHERE extract_date = %s
                """
                cursor.execute(update_query, (keywords_json, summary, current_timestamp, extract_date))
                print(f"更新了 {extract_date} 的话题分析")
            else:
                # 插入新记录
                insert_query = """
                    INSERT INTO daily_topics (extract_date, keywords, summary, add_ts)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(insert_query, (extract_date, keywords_json, summary, current_timestamp))
                print(f"保存了 {extract_date} 的话题分析")
            
            return True
            
        except Exception as e:
            print(f"保存话题分析失败: {e}")
            return False
    
    def get_daily_topics(self, extract_date: date = None) -> Optional[Dict]:
        """
        获取每日话题分析
        
        Args:
            extract_date: 提取日期，默认为今天
        
        Returns:
            话题分析数据，如果不存在返回None
        """
        if not extract_date:
            extract_date = date.today()
        
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM daily_topics WHERE extract_date = %s"
            cursor.execute(query, (extract_date,))
            result = cursor.fetchone()
            
            if result:
                # 解析关键词JSON
                result['keywords'] = json.loads(result['keywords'])
                return result
            else:
                return None
                
        except Exception as e:
            print(f"获取话题分析失败: {e}")
            return None
    
    def get_recent_topics(self, days: int = 7) -> List[Dict]:
        """
        获取最近几天的话题分析
        
        Args:
            days: 天数
        
        Returns:
            话题分析列表
        """
        try:
            cursor = self.connection.cursor()
            query = """
                SELECT * FROM daily_topics 
                WHERE extract_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                ORDER BY extract_date DESC
            """
            cursor.execute(query, (days,))
            results = cursor.fetchall()
            
            # 解析每个结果的关键词JSON
            for result in results:
                result['keywords'] = json.loads(result['keywords'])
            
            return results
            
        except Exception as e:
            print(f"获取最近话题分析失败: {e}")
            return []
    
    # ==================== 统计查询 ====================
    
    def get_summary_stats(self, days: int = 7) -> Dict:
        """获取统计摘要"""
        try:
            cursor = self.connection.cursor()
            
            # 新闻统计
            news_query = """
                SELECT 
                    crawl_date,
                    COUNT(*) as news_count,
                    COUNT(DISTINCT source_platform) as platforms_count
                FROM daily_news 
                WHERE crawl_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                GROUP BY crawl_date
                ORDER BY crawl_date DESC
            """
            cursor.execute(news_query, (days,))
            news_stats = cursor.fetchall()
            
            # 话题统计
            topics_query = """
                SELECT 
                    extract_date,
                    keywords,
                    CHAR_LENGTH(summary) as summary_length
                FROM daily_topics 
                WHERE extract_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                ORDER BY extract_date DESC
            """
            cursor.execute(topics_query, (days,))
            topics_stats = cursor.fetchall()
            
            return {
                'news_stats': news_stats,
                'topics_stats': topics_stats
            }
            
        except Exception as e:
            print(f"获取统计摘要失败: {e}")
            return {'news_stats': [], 'topics_stats': []}

if __name__ == "__main__":
    # 测试数据库管理器
    with DatabaseManager() as db:
        # 测试获取新闻
        news = db.get_daily_news()
        print(f"今日新闻数量: {len(news)}")
        
        # 测试获取话题
        topics = db.get_daily_topics()
        if topics:
            print(f"今日话题关键词: {topics['keywords']}")
        else:
            print("今日暂无话题分析")
        
        print("简化数据库管理器测试完成！")
