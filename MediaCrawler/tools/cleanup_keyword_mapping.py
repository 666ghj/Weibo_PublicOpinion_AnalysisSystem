#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理关键词映射表重复数据的工具脚本
"""

import asyncio
import sys
import os
from typing import List, Dict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from db import AsyncMysqlDB, AsyncSqliteDB
from var import media_crawler_db_var


# 所有关键词映射表
KEYWORD_MAPPING_TABLES = [
    "xhs_note_keyword_map",
    "douyin_aweme_keyword_map", 
    "weibo_note_keyword_map",
    "tieba_note_keyword_map",
    "kuaishou_video_keyword_map",
    "bilibili_video_keyword_map",
    "zhihu_content_keyword_map"
]


async def check_duplicate_data():
    """检查重复数据"""
    print("🔍 正在检查关键词映射表中的重复数据...")
    
    async_db_conn = media_crawler_db_var.get()
    total_duplicates = 0
    
    for table in KEYWORD_MAPPING_TABLES:
        try:
            if isinstance(async_db_conn, AsyncMysqlDB):
                # MySQL查询重复数据
                sql = f"""
                SELECT keyword, COUNT(*) as count 
                FROM {table} 
                GROUP BY keyword 
                HAVING COUNT(*) > 1
                ORDER BY count DESC
                """
            else:
                # SQLite查询重复数据
                sql = f"""
                SELECT keyword, COUNT(*) as count 
                FROM {table} 
                GROUP BY keyword 
                HAVING COUNT(*) > 1
                ORDER BY count DESC
                """
            
            rows = await async_db_conn.query(sql)
            
            if rows:
                table_duplicates = sum(row['count'] - 1 for row in rows)  # 减1是因为每组保留一条
                total_duplicates += table_duplicates
                print(f"📊 {table}: 发现 {len(rows)} 个重复关键词，共 {table_duplicates} 条重复记录")
                
                # 显示前5个最严重的重复
                for i, row in enumerate(rows[:5]):
                    print(f"   - '{row['keyword']}': {row['count']} 条记录")
                if len(rows) > 5:
                    print(f"   ... 还有 {len(rows) - 5} 个重复关键词")
            else:
                print(f"✅ {table}: 无重复数据")
                
        except Exception as e:
            print(f"❌ 检查表 {table} 时出错: {e}")
    
    print(f"\n📈 总计发现 {total_duplicates} 条重复记录")
    return total_duplicates


async def clean_duplicate_data(confirm: bool = False):
    """清理重复数据"""
    if not confirm:
        print("⚠️  这是一个危险操作，将删除重复的关键词映射记录！")
        response = input("确认要继续吗？(输入 'yes' 确认): ")
        if response.lower() != 'yes':
            print("❌ 操作已取消")
            return
    
    print("🧹 开始清理重复数据...")
    
    async_db_conn = media_crawler_db_var.get()
    total_deleted = 0
    
    for table in KEYWORD_MAPPING_TABLES:
        try:
            if isinstance(async_db_conn, AsyncMysqlDB):
                # MySQL删除重复记录，保留最早的一条
                if table == "tieba_note_keyword_map":
                    # 贴吧表的note_id字段是VARCHAR(255)
                    sql = f"""
                    DELETE t1 FROM {table} t1
                    INNER JOIN {table} t2 
                    WHERE t1.note_id = t2.note_id 
                    AND t1.keyword = t2.keyword 
                    AND t1.created_at > t2.created_at
                    """
                else:
                    # 其他表的ID字段是VARCHAR(64)
                    sql = f"""
                    DELETE t1 FROM {table} t1
                    INNER JOIN {table} t2 
                    WHERE t1.{get_id_field(table)} = t2.{get_id_field(table)}
                    AND t1.keyword = t2.keyword 
                    AND t1.created_at > t2.created_at
                    """
            else:
                # SQLite删除重复记录
                sql = f"""
                DELETE FROM {table} 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM {table} 
                    GROUP BY {get_id_field(table)}, keyword
                )
                """
            
            deleted_count = await async_db_conn.execute(sql)
            total_deleted += deleted_count
            print(f"✅ {table}: 删除了 {deleted_count} 条重复记录")
            
        except Exception as e:
            print(f"❌ 清理表 {table} 时出错: {e}")
    
    print(f"\n🎉 清理完成！总共删除了 {total_deleted} 条重复记录")


def get_id_field(table_name: str) -> str:
    """获取表的ID字段名"""
    id_mapping = {
        "xhs_note_keyword_map": "note_id",
        "douyin_aweme_keyword_map": "aweme_id",
        "weibo_note_keyword_map": "note_id", 
        "tieba_note_keyword_map": "note_id",
        "kuaishou_video_keyword_map": "video_id",
        "bilibili_video_keyword_map": "video_id",
        "zhihu_content_keyword_map": "content_id"
    }
    return id_mapping.get(table_name, "id")


async def drop_keyword_tables(confirm: bool = False):
    """删除所有关键词映射表"""
    if not confirm:
        print("⚠️  这将永久删除所有关键词映射表！")
        response = input("确认要删除所有关键词映射表吗？(输入 'DELETE_ALL' 确认): ")
        if response != 'DELETE_ALL':
            print("❌ 操作已取消")
            return
    
    print("🗑️  开始删除关键词映射表...")
    
    async_db_conn = media_crawler_db_var.get()
    
    for table in KEYWORD_MAPPING_TABLES:
        try:
            sql = f"DROP TABLE IF EXISTS {table}"
            await async_db_conn.execute(sql)
            print(f"✅ 已删除表: {table}")
        except Exception as e:
            print(f"❌ 删除表 {table} 时出错: {e}")
    
    print("🎉 所有关键词映射表已删除完成！")


async def show_table_stats():
    """显示表统计信息"""
    print("📊 关键词映射表统计信息:")
    
    async_db_conn = media_crawler_db_var.get()
    
    for table in KEYWORD_MAPPING_TABLES:
        try:
            # 检查表是否存在
            if isinstance(async_db_conn, AsyncMysqlDB):
                check_sql = f"SHOW TABLES LIKE '{table}'"
            else:
                check_sql = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
            
            exists = await async_db_conn.query(check_sql)
            
            if not exists:
                print(f"❌ {table}: 表不存在")
                continue
            
            # 获取记录总数
            count_sql = f"SELECT COUNT(*) as count FROM {table}"
            count_result = await async_db_conn.query(count_sql)
            total_count = count_result[0]['count'] if count_result else 0
            
            # 获取唯一关键词数量
            unique_sql = f"SELECT COUNT(DISTINCT keyword) as count FROM {table}"
            unique_result = await async_db_conn.query(unique_sql)
            unique_count = unique_result[0]['count'] if unique_result else 0
            
            print(f"📋 {table}: {total_count} 条记录, {unique_count} 个唯一关键词")
            
        except Exception as e:
            print(f"❌ 查询表 {table} 时出错: {e}")


async def main():
    """主函数"""
    print("🛠️  关键词映射表清理工具")
    print("=" * 50)
    
    # 初始化数据库连接
    if config.SAVE_DATA_OPTION == "db":
        from db import init_mysql_db
        await init_mysql_db()
    elif config.SAVE_DATA_OPTION == "sqlite":
        from db import init_sqlite_db
        await init_sqlite_db()
    else:
        print("❌ 不支持的数据库类型，请设置 SAVE_DATA_OPTION 为 'db' 或 'sqlite'")
        return
    
    while True:
        print("\n请选择操作:")
        print("1. 检查重复数据")
        print("2. 清理重复数据")
        print("3. 显示表统计信息")
        print("4. 删除所有关键词映射表")
        print("5. 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == "1":
            await check_duplicate_data()
        elif choice == "2":
            await clean_duplicate_data()
        elif choice == "3":
            await show_table_stats()
        elif choice == "4":
            await drop_keyword_tables()
        elif choice == "5":
            print("👋 再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")


if __name__ == "__main__":
    asyncio.run(main())
