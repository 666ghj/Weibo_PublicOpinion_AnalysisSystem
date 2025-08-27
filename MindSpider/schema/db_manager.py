#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MindSpider AI爬虫项目 - 数据库管理工具
提供数据库状态查看、数据统计、清理等功能
"""

import os
import sys
import pymysql
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    print("错误: 无法导入config.py配置文件")
    sys.exit(1)

class DatabaseManager:
    def __init__(self):
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
                autocommit=True
            )
            print(f"成功连接到数据库: {config.DB_NAME}")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            sys.exit(1)
    
    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
    
    def show_tables(self):
        """显示所有表"""
        print("\n" + "=" * 60)
        print("数据库表列表")
        print("=" * 60)
        
        cursor = self.connection.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        if not tables:
            print("数据库中没有表")
            return
        
        # 分类显示表
        mindspider_tables = []
        mediacrawler_tables = []
        
        for table in tables:
            table_name = table[0]
            if table_name in ['daily_news', 'daily_topics', 'topic_news_relation', 'crawling_tasks']:
                mindspider_tables.append(table_name)
            else:
                mediacrawler_tables.append(table_name)
        
        print("MindSpider核心表:")
        for table in mindspider_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  - {table:<25} ({count:>6} 条记录)")
        
        print("\nMediaCrawler平台表:")
        for table in mediacrawler_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  - {table:<25} ({count:>6} 条记录)")
            except:
                print(f"  - {table:<25} (查询失败)")
    
    def show_statistics(self):
        """显示数据统计"""
        print("\n" + "=" * 60)
        print("数据统计")
        print("=" * 60)
        
        cursor = self.connection.cursor()
        
        try:
            # 新闻统计
            cursor.execute("SELECT COUNT(*) FROM daily_news")
            news_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT crawl_date) FROM daily_news")
            news_days = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT source_platform) FROM daily_news")
            platforms = cursor.fetchone()[0]
            
            print(f"新闻数据:")
            print(f"  - 总新闻数: {news_count}")
            print(f"  - 覆盖天数: {news_days}")
            print(f"  - 新闻平台: {platforms}")
            
            # 话题统计
            cursor.execute("SELECT COUNT(*) FROM daily_topics")
            topic_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT extract_date) FROM daily_topics")
            topic_days = cursor.fetchone()[0]
            
            print(f"\n话题数据:")
            print(f"  - 总话题数: {topic_count}")
            print(f"  - 提取天数: {topic_days}")
            
            # 爬取任务统计
            cursor.execute("SELECT COUNT(*) FROM crawling_tasks")
            task_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT task_status, COUNT(*) FROM crawling_tasks GROUP BY task_status")
            task_status = cursor.fetchall()
            
            print(f"\n爬取任务:")
            print(f"  - 总任务数: {task_count}")
            for status, count in task_status:
                print(f"  - {status}: {count}")
            
            # 爬取内容统计
            print(f"\n平台内容统计:")
            platform_tables = {
                'xhs_note': '小红书',
                'douyin_aweme': '抖音',
                'kuaishou_video': '快手',
                'bilibili_video': 'B站',
                'weibo_note': '微博',
                'tieba_note': '贴吧',
                'zhihu_content': '知乎'
            }
            
            for table, platform in platform_tables.items():
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  - {platform}: {count}")
                except:
                    print(f"  - {platform}: 表不存在")
                    
        except Exception as e:
            print(f"统计查询失败: {e}")
    
    def show_recent_data(self, days=7):
        """显示最近几天的数据"""
        print(f"\n" + "=" * 60)
        print(f"最近{days}天的数据")
        print("=" * 60)
        
        cursor = self.connection.cursor()
        
        # 最近的新闻
        cursor.execute("""
            SELECT crawl_date, COUNT(*) as news_count, COUNT(DISTINCT source_platform) as platforms
            FROM daily_news 
            WHERE crawl_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
            GROUP BY crawl_date 
            ORDER BY crawl_date DESC
        """, (days,))
        
        news_data = cursor.fetchall()
        if news_data:
            print("每日新闻统计:")
            for date, count, platforms in news_data:
                print(f"  {date}: {count} 条新闻, {platforms} 个平台")
        
        # 最近的话题
        cursor.execute("""
            SELECT extract_date, COUNT(*) as topic_count
            FROM daily_topics 
            WHERE extract_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
            GROUP BY extract_date 
            ORDER BY extract_date DESC
        """, (days,))
        
        topic_data = cursor.fetchall()
        if topic_data:
            print("\n每日话题统计:")
            for date, count in topic_data:
                print(f"  {date}: {count} 个话题")
    
    def cleanup_old_data(self, days=90, dry_run=True):
        """清理旧数据"""
        print(f"\n" + "=" * 60)
        print(f"清理{days}天前的数据 ({'预览模式' if dry_run else '执行模式'})")
        print("=" * 60)
        
        cursor = self.connection.cursor()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 检查要删除的数据
        cleanup_queries = [
            ("daily_news", f"SELECT COUNT(*) FROM daily_news WHERE crawl_date < '{cutoff_date.date()}'"),
            ("daily_topics", f"SELECT COUNT(*) FROM daily_topics WHERE extract_date < '{cutoff_date.date()}'"),
            ("crawling_tasks", f"SELECT COUNT(*) FROM crawling_tasks WHERE scheduled_date < '{cutoff_date.date()}'")
        ]
        
        for table, query in cleanup_queries:
            cursor.execute(query)
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"  {table}: {count} 条记录将被删除")
                if not dry_run:
                    delete_query = query.replace("SELECT COUNT(*)", "DELETE")
                    cursor.execute(delete_query)
                    print(f"    已删除 {count} 条记录")
            else:
                print(f"  {table}: 无需清理")
        
        if dry_run:
            print("\n这是预览模式，没有实际删除数据。使用 --execute 参数执行实际清理。")

def main():
    parser = argparse.ArgumentParser(description="MindSpider数据库管理工具")
    parser.add_argument("--tables", action="store_true", help="显示所有表")
    parser.add_argument("--stats", action="store_true", help="显示数据统计")
    parser.add_argument("--recent", type=int, default=7, help="显示最近N天的数据 (默认7天)")
    parser.add_argument("--cleanup", type=int, help="清理N天前的数据")
    parser.add_argument("--execute", action="store_true", help="执行实际清理操作")
    
    args = parser.parse_args()
    
    # 如果没有参数，显示所有信息
    if not any([args.tables, args.stats, args.recent != 7, args.cleanup]):
        args.tables = True
        args.stats = True
    
    db_manager = DatabaseManager()
    
    try:
        if args.tables:
            db_manager.show_tables()
        
        if args.stats:
            db_manager.show_statistics()
        
        if args.recent != 7 or not any([args.tables, args.stats, args.cleanup]):
            db_manager.show_recent_data(args.recent)
        
        if args.cleanup:
            db_manager.cleanup_old_data(args.cleanup, dry_run=not args.execute)
    
    finally:
        db_manager.close()

if __name__ == "__main__":
    main()
