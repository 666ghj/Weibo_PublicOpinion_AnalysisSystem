from utils.query import query
from utils.getPublicData import *
from model2.model_use import *

articleList = getAllArticleData()
commentList = getAllCommentsData()

def column_exists(table_name, column_name):
    """检查列是否存在于表中"""
    sql = "SHOW COLUMNS FROM {} LIKE %s".format(table_name)
    params = [column_name]
    result = query(sql, params, type='select')
    return len(result) > 0

def add_label_column(table_name):
    """为表添加label列"""
    if not column_exists(table_name, 'label'):
        sql = f"ALTER TABLE {table_name} ADD COLUMN label TEXT NULL"
        params = []
        query(sql, params)

def drop_label_column(table_name):
    """删除表中的label列"""
    sql = f"ALTER TABLE {table_name} DROP COLUMN label"
    params = []
    query(sql, params)

def topicdefine():
    """根据文章和评论内容进行标注"""
    label_article = []
    label_comments = []
    
    # 为文章添加标签
    for x in articleList:
        label_article.append((x[0], predict_topic(x[5])))
    
    # 为评论添加标签
    for x in commentList:
        label_comments.append((x[5], x[8], predict_topic(x[4])))
    
    return label_article, label_comments

def update_data():
    """更新文章和评论的label列"""
    label_article, label_comments = topicdefine()
    
    # 先为文章和评论添加label列
    add_label_column('article')
    add_label_column('comments')

    try:
        # 使用事务保证批量更新操作的原子性
        # 处理文章的label更新
        for row in label_article:
            id, label = row
            sql = "UPDATE article SET label = %s WHERE id = %s"
            params = [str(label), str(id)]
            query(sql, params)
        
        # 处理评论的label更新
        for row in label_comments:
            id, image, label = row
            sql = "UPDATE comments SET label = %s WHERE authorName=%s AND authorAvatar = %s"
            params = [str(label), str(id), str(image)]
            query(sql, params)
    except Exception as e:
        print(f"更新数据时发生错误: {e}")
        # 如果发生错误，可以选择在此处进行回滚操作（取决于数据库支持的功能）
    finally:
        # 提交事务
        pass

if __name__ == '__main__':
    # 删除label列的操作已经被注释掉，如需删除可取消注释
    # drop_label_column('article')
    # drop_label_column('comments')
    update_data()
