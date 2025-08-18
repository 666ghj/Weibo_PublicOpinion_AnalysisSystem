# 声明：本代码仅供学习和研究目的使用。使用者应遵守以下原则：
# 1. 不得用于任何商业用途。
# 2. 使用时应遵守目标平台的使用条款和robots.txt规则。
# 3. 不得进行大规模爬取或对平台造成运营干扰。
# 4. 应合理控制请求频率，避免给目标平台带来不必要的负担。
# 5. 不得用于任何非法或不当的用途。
#
# 详细许可条款请参阅项目根目录下的LICENSE文件。
# 使用本代码即表示您同意遵守上述原则和LICENSE中的所有条款。


# -*- coding: utf-8 -*-
# @Author  : relakkes@gmail.com
# @Time    : 2024/4/6 15:30
# @Desc    : sql接口集合

from typing import Dict, List, Union

from async_db import AsyncMysqlDB
from async_sqlite_db import AsyncSqliteDB
from var import media_crawler_db_var


async def query_content_by_content_id(content_id: str) -> Dict:
    """
    查询一条内容记录（xhs的帖子 ｜ 抖音的视频 ｜ 微博 ｜ 快手视频 ...）
    Args:
        content_id:

    Returns:

    """
    async_db_conn: Union[AsyncMysqlDB, AsyncSqliteDB] = media_crawler_db_var.get()
    sql: str = f"select * from xhs_note where note_id = '{content_id}'"
    rows: List[Dict] = await async_db_conn.query(sql)
    if len(rows) > 0:
        return rows[0]
    return dict()


async def add_new_content(content_item: Dict) -> int:
    """
    新增一条内容记录（xhs的帖子 ｜ 抖音的视频 ｜ 微博 ｜ 快手视频 ...）
    Args:
        content_item:

    Returns:

    """
    async_db_conn: Union[AsyncMysqlDB, AsyncSqliteDB] = media_crawler_db_var.get()
    last_row_id: int = await async_db_conn.item_to_table("xhs_note", content_item)
    return last_row_id


async def update_content_by_content_id(content_id: str, content_item: Dict) -> int:
    """
    更新一条记录（xhs的帖子 ｜ 抖音的视频 ｜ 微博 ｜ 快手视频 ...）
    Args:
        content_id:
        content_item:

    Returns:

    """
    async_db_conn: Union[AsyncMysqlDB, AsyncSqliteDB] = media_crawler_db_var.get()
    effect_row: int = await async_db_conn.update_table("xhs_note", content_item, "note_id", content_id)
    return effect_row



async def query_comment_by_comment_id(comment_id: str) -> Dict:
    """
    查询一条评论内容
    Args:
        comment_id:

    Returns:

    """
    async_db_conn: Union[AsyncMysqlDB, AsyncSqliteDB] = media_crawler_db_var.get()
    sql: str = f"select * from xhs_note_comment where comment_id = '{comment_id}'"
    rows: List[Dict] = await async_db_conn.query(sql)
    if len(rows) > 0:
        return rows[0]
    return dict()


async def add_new_comment(comment_item: Dict) -> int:
    """
    新增一条评论记录
    Args:
        comment_item:

    Returns:

    """
    async_db_conn: Union[AsyncMysqlDB, AsyncSqliteDB] = media_crawler_db_var.get()
    last_row_id: int = await async_db_conn.item_to_table("xhs_note_comment", comment_item)
    return last_row_id


async def update_comment_by_comment_id(comment_id: str, comment_item: Dict) -> int:
    """
    更新增一条评论记录
    Args:
        comment_id:
        comment_item:

    Returns:

    """
    async_db_conn: Union[AsyncMysqlDB, AsyncSqliteDB] = media_crawler_db_var.get()
    effect_row: int = await async_db_conn.update_table("xhs_note_comment", comment_item, "comment_id", comment_id)
    return effect_row


async def query_creator_by_user_id(user_id: str) -> Dict:
    """
    查询一条创作者记录
    Args:
        user_id:

    Returns:

    """
    async_db_conn: Union[AsyncMysqlDB, AsyncSqliteDB] = media_crawler_db_var.get()
    sql: str = f"select * from xhs_creator where user_id = '{user_id}'"
    rows: List[Dict] = await async_db_conn.query(sql)
    if len(rows) > 0:
        return rows[0]
    return dict()


async def add_new_creator(creator_item: Dict) -> int:
    """
    新增一条创作者信息
    Args:
        creator_item:

    Returns:

    """
    async_db_conn: Union[AsyncMysqlDB, AsyncSqliteDB] = media_crawler_db_var.get()
    last_row_id: int = await async_db_conn.item_to_table("xhs_creator", creator_item)
    return last_row_id


async def update_creator_by_user_id(user_id: str, creator_item: Dict) -> int:
    """
    更新一条创作者信息
    Args:
        user_id:
        creator_item:

    Returns:

    """
    async_db_conn: Union[AsyncMysqlDB, AsyncSqliteDB] = media_crawler_db_var.get()
    effect_row: int = await async_db_conn.update_table("xhs_creator", creator_item, "user_id", user_id)
    return effect_row


async def add_or_ignore_note_keyword_map(note_id: str, keyword: str) -> None:
    """
    添加或忽略笔记关键词映射记录
    Args:
        note_id: 笔记ID
        keyword: 关键词

    Returns:

    """
    from tools import utils

    async_db_conn: Union[AsyncMysqlDB, AsyncSqliteDB] = media_crawler_db_var.get()

    # 检查是否已存在相同的映射记录
    check_sql = "SELECT COUNT(*) as count FROM xhs_note_keyword_map WHERE note_id = ? AND keyword = ?"
    rows = await async_db_conn.query(check_sql, (note_id, keyword))

    if rows and rows[0]['count'] > 0:
        # 记录已存在，忽略
        return

    # 插入新记录
    insert_sql = """
        INSERT INTO xhs_note_keyword_map (note_id, keyword, created_at)
        VALUES (?, ?, ?)
    """
    current_timestamp = utils.get_current_timestamp()
    await async_db_conn.execute(insert_sql, (note_id, keyword, current_timestamp))