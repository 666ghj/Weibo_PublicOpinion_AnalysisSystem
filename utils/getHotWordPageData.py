from utils.getPublicData import *
from utils.query import query
from utils.logger import app_logger as logging

def getHotWordLen(hotWord):
    """
    获取指定热词的出现次数
    """
    try:
        # 修改SQL，统计文章和评论中的热词出现次数
        sql = """
        SELECT 
            (SELECT COUNT(*) FROM article WHERE topic = %s) +
            (SELECT COUNT(*) FROM comments WHERE content LIKE %s)
        AS total_count
        """
        result = query(sql, query_type="select", params=[hotWord, f'%{hotWord}%'])
        
        if not result or len(result) == 0:
            logging.warning(f"未找到热词 '{hotWord}' 出现次数")
            return 0
            
        return result[0]['total_count'] if 'total_count' in result[0] else 0
    except Exception as e:
        logging.error(f"获取热词 '{hotWord}' 出现次数出错: {e}")
        return 0

def getHotWordPageCreatedAtCharData(hotWord):
    """
    获取特定热词随时间的分布数据
    """
    try:
        # 修改SQL，从文章和评论中统计热词随时间分布
        sql = """
        SELECT 
            SUBSTRING(created_at, 1, 10) as date,
            COUNT(*) as count
        FROM 
            (
                SELECT created_at FROM article WHERE topic = %s AND created_at IS NOT NULL
                UNION ALL
                SELECT created_at FROM comments WHERE content LIKE %s AND created_at IS NOT NULL
            ) as combined_data
        GROUP BY 
            SUBSTRING(created_at, 1, 10)
        ORDER BY 
            date
        LIMIT 30
        """
        result = query(sql, query_type="select", params=[hotWord, f'%{hotWord}%'])
        
        if not result or len(result) == 0:
            logging.warning(f"热词 '{hotWord}' 时间分布数据为空")
            # 返回过去7天作为默认数据
            from datetime import datetime, timedelta
            today = datetime.now()
            dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            dates.reverse()
            return dates, [0] * 7
        
        X = [row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']) for row in result]
        Y = [row['count'] for row in result]
        
        logging.info(f"成功获取热词 '{hotWord}' 时间分布数据，共{len(X)}个数据点")
        return X, Y
    except Exception as e:
        logging.error(f"获取热词 '{hotWord}' 时间分布数据出错: {e}")
        # 返回默认数据，避免页面显示错误
        from datetime import datetime, timedelta
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
        dates.reverse()
        return dates, [0] * 7

def getCommentFilterData(hotWord):
    """
    根据热词获取相关评论数据
    """
    try:
        sql = """
        SELECT 
            articleId, created_at, likes_counts, region, 
            content, authorName, authorGender, authorAddress, authorAvatar
        FROM 
            comments
        WHERE 
            topic = %s OR content LIKE %s
        ORDER BY 
            likes_counts DESC
        LIMIT 20
        """
        result = query(sql, query_type="select", params=[hotWord, f'%{hotWord}%'])
        
        if not result or len(result) == 0:
            logging.warning(f"热词 '{hotWord}' 相关评论数据为空")
            return []
            
        # 将查询结果转换为列表格式
        comments_list = []
        for row in result:
            comment = [
                row['articleId'],
                row.get('created_at', ''),
                row.get('likes_counts', 0),
                row.get('region', '未知'),
                row.get('content', ''),
                row.get('authorName', '匿名用户'),
                row.get('authorGender', 'm'),
                row.get('authorAddress', ''),
                row.get('authorAvatar', '')
            ]
            comments_list.append(comment)
        
        logging.info(f"成功获取热词 '{hotWord}' 相关评论数据，共{len(comments_list)}条评论")
        return comments_list
    except Exception as e:
        logging.error(f"获取热词 '{hotWord}' 相关评论数据出错: {e}")
        return []

def getTopicLen(topic):
    """
    获取指定话题的出现次数
    """
    try:
        # 文章中的话题数量
        article_sql = "SELECT COUNT(*) FROM article WHERE topic = %s"
        article_result = query(article_sql, (topic,), query_type="select")
        
        # 评论中的话题数量
        comment_sql = "SELECT COUNT(*) FROM comment WHERE topic = %s"
        comment_result = query(comment_sql, (topic,), query_type="select")
        
        # 获取计数结果
        article_count = article_result[0][0] if article_result and article_result[0] else 0
        comment_count = comment_result[0][0] if comment_result and comment_result[0] else 0
        
        # 返回总数
        return article_count + comment_count
    except Exception as e:
        logging.error(f"获取话题长度失败: {e}", exc_info=True)
        return 0  # 返回0表示出错或未找到

def getCommentFilterDataTopic(topic):
    """
    获取指定话题的评论数据
    """
    try:
        sql = """
        SELECT c.id, c.created_at, c.likes_counts, c.author_id, c.content, 
               c.author_name, c.gender, c.article_id, a.title, c.topic
        FROM comment c
        LEFT JOIN article a ON c.article_id = a.id
        WHERE c.topic = %s
        LIMIT 100
        """
        
        result = query(sql, (topic,), query_type="select")
        return result if result else []
    except Exception as e:
        logging.error(f"获取评论数据失败: {e}", exc_info=True)
        return []  # 返回空列表表示出错或未找到