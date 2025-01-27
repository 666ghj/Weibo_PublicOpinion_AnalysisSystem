import os
import pandas as pd
from sqlalchemy import create_engine
from getpass import getpass
from utils.logger import spider_logger as logging

# 假设 articleAddr 和 commentsAddr 是绝对路径或相对于脚本的路径
from spiderDataPackage.settings import articleAddr, commentsAddr

def get_db_connection_interactive():
    """
    通过终端交互获取数据库连接参数，若按回车则使用默认值。
    返回 SQLAlchemy 的数据库引擎。
    """
    print("请依次输入数据库连接信息（直接按回车使用默认值）：")
    
    host = input(" 1. 主机 (默认: localhost): ") or "localhost"
    port_str = input(" 2. 端口 (默认: 3306): ") or "3306"
    try:
        port = int(port_str)
    except ValueError:
        logging.warning("端口号无效，使用默认端口 3306。")
        port = 3306
    
    user = input(" 3. 用户名 (默认: root): ") or "root"
    password = getpass(" 4. 密码 (默认: 12345678): ") or "12345678"
    db_name = input(" 5. 数据库名 (默认: Weibo_PublicOpinion_AnalysisSystem): ") or "Weibo_PublicOpinion_AnalysisSystem"
    
    # 构建数据库连接字符串
    connection_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}?charset=utf8mb4"
    
    try:
        engine = create_engine(connection_str)
        # 测试连接
        with engine.connect() as connection:
            logging.info(f"成功连接到数据库: {user}@{host}:{port}/{db_name}")
        return engine
    except Exception as e:
        logging.error(f"无法连接到数据库: {e}")
        exit(1)

def saveData(engine):
    """
    从数据库和CSV文件读取数据，合并后去重并保存回数据库。
    最后删除CSV文件。
    """
    try:
        # 读取旧数据
        oldArticle = pd.read_sql('SELECT * FROM article', engine)
        oldComment = pd.read_sql('SELECT * FROM comments', engine)
        logging.info("成功从数据库读取旧的文章和评论数据。")
        
        # 读取新数据
        newArticle = pd.read_csv(articleAddr)
        newComment = pd.read_csv(commentsAddr)
        logging.info("成功从CSV文件读取新的文章和评论数据。")
        
        # 合并数据
        mergeArticle = pd.concat([newArticle, oldArticle], ignore_index=True, sort=False)
        mergeComment = pd.concat([newComment, oldComment], ignore_index=True, sort=False)
        logging.info("成功合并新旧文章和评论数据。")
        
        # 去重
        mergeArticle.drop_duplicates(subset='id', keep='last', inplace=True)
        mergeComment.drop_duplicates(subset='content', keep='last', inplace=True)
        logging.info("成功去除重复的文章和评论数据。")
        
        # 保存回数据库
        mergeArticle.to_sql('article', con=engine, if_exists='replace', index=False)
        mergeComment.to_sql('comments', con=engine, if_exists='replace', index=False)
        logging.info("成功将合并后的数据保存回数据库。")
        
    except pd.errors.EmptyDataError as e:
        logging.error(f"读取CSV文件时出错: {e}")
    except Exception as e:
        logging.error(f"保存数据时出错: {e}")
    else:
        # 删除CSV文件
        try:
            os.remove(articleAddr)
            os.remove(commentsAddr)
            logging.info("成功删除CSV文件。")
        except Exception as e:
            logging.warning(f"删除CSV文件时出错: {e}")

def main():
    # 获取数据库连接
    engine = get_db_connection_interactive()
    
    # 保存数据
    saveData(engine)
    
    # 关闭引擎（可选，因为SQLAlchemy引擎会自动管理连接池）
    engine.dispose()
    logging.info("数据库连接已关闭。")

if __name__ == '__main__':
    main()
