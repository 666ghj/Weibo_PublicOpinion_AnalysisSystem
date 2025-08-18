# -*- coding: utf-8 -*-
import asyncio
import aiomysql
import sys
import os

# 将项目根目录添加到Python路径中，以便导入config模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import db_config

# 需要清理的表列表
TABLES_TO_CLEAN = [
    "bilibili_video",
    "douyin_aweme",
    "kuaishou_video",
    "weibo_note",
    "xhs_note",
    "tieba_note",
    "zhihu_content",
]

async def cleanup_data():
    """
    连接到数据库，查找并删除 source_keyword 字段中包含中文逗号的记录。
    """
    conn = None
    try:
        # 连接数据库
        conn = await aiomysql.connect(
            host=db_config.MYSQL_DB_HOST,
            port=db_config.MYSQL_DB_PORT,
            user=db_config.MYSQL_DB_USER,
            password=db_config.MYSQL_DB_PWD,
            db=db_config.MYSQL_DB_NAME,
            autocommit=True
        )
        cursor = await conn.cursor()

        total_to_delete = 0
        counts = {}

        print("正在检查 `source_keyword` 字段中包含中文逗号的数据...")
        for table in TABLES_TO_CLEAN:
            try:
                # 检查 `source_keyword` 列是否存在
                await cursor.execute(f"SHOW COLUMNS FROM `{table}` LIKE 'source_keyword'")
                if await cursor.fetchone():
                    query = f"SELECT COUNT(*) FROM `{table}` WHERE `source_keyword` LIKE '%，%'"
                    await cursor.execute(query)
                    count = (await cursor.fetchone())[0]
                    if count > 0:
                        print(f"在表 '{table}' 中找到 {count} 条要删除的记录。")
                        counts[table] = count
                        total_to_delete += count
                else:
                    print(f"表 '{table}' 中没有 'source_keyword' 列，已跳过。")
            except Exception as e:
                print(f"检查表 {table} 时出错: {e}")

        if total_to_delete == 0:
            print("在 `source_keyword` 字段中未找到包含中文逗号的记录，无需任何操作。")
            return

        # 请求用户确认
        confirm = input(f"总共要删除 {total_to_delete} 条记录。您确定要继续吗？ (yes/no): ").lower()

        if confirm == 'yes':
            print("开始删除...")
            for table, count in counts.items():
                if count > 0:
                    try:
                        delete_query = f"DELETE FROM `{table}` WHERE `source_keyword` LIKE '%，%'"
                        await cursor.execute(delete_query)
                        print(f"已从 '{table}' 表中删除 {cursor.rowcount} 条记录。")
                    except Exception as e:
                        print(f"从表 {table} 删除时出错: {e}")
            print("清理完成。")
        else:
            print("用户取消了删除操作。")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    asyncio.run(cleanup_data())