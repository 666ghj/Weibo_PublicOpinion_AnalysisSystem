import pymysql
import os
from utils.logger import app_logger as logging
from getpass import getpass

# 全局连接对象
conn = None
cursor = None

def get_db_connection_interactive():
    """
    通过终端交互获取数据库连接参数，若按回车则使用默认值。
    返回一个连接对象。
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
    password = getpass(" 4. 密码 (默认: 123456): ") or "123456"
    db_name = input(" 5. 数据库名 (默认: Weibo_PublicOpinion_AnalysisSystem): ") or "Weibo_PublicOpinion_AnalysisSystem"
    
    logging.info(f"尝试连接到数据库: {user}@{host}:{port}/{db_name}")
    
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor  # 返回字典格式
        )
        logging.info("数据库连接成功。")
        return connection
    except pymysql.MySQLError as e:
        logging.error(f"数据库连接失败: {e}")
        return None

def get_db_connection():
    """
    从环境变量或默认值获取数据库连接参数
    """
    # 从环境变量获取数据库连接信息，如果没有则使用默认值
    host = os.getenv('DB_HOST', 'localhost')
    port_str = os.getenv('DB_PORT', '3306')
    try:
        port = int(port_str)
    except ValueError:
        logging.warning("端口号无效，使用默认端口 3306。")
        port = 3306
    
    user = os.getenv('DB_USER', 'root')
    password = os.getenv('DB_PASSWORD', '123456')
    db_name = os.getenv('DB_NAME', 'Weibo_PublicOpinion_AnalysisSystem')
    
    logging.info(f"尝试连接到数据库: {user}@{host}:{port}/{db_name}")
    
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor  # 返回字典格式
        )
        logging.info("数据库连接成功。")
        return connection
    except pymysql.MySQLError as e:
        logging.error(f"数据库连接失败: {e}")
        return None

# 初始化数据库连接
def initialize_connection(interactive=False):
    """
    初始化数据库连接
    
    :param interactive: 是否使用交互式方式获取连接参数
    :return: 是否成功连接
    """
    global conn, cursor
    try:
        if interactive:
            conn = get_db_connection_interactive()
        else:
            conn = get_db_connection()
            
        if conn:
            cursor = conn.cursor()
            return True
        return False
    except Exception as e:
        logging.critical(f"无法创建数据库连接: {e}")
        conn = None
        cursor = None
        return False

# 根据命令行参数决定是否使用交互式连接
def init_db_connection():
    """
    根据环境或用户选择初始化数据库连接
    """
    use_interactive = False
    
    # 检查是否需要交互式连接
    try:
        choice = input("是否使用交互式方式连接数据库? (y/n, 默认n): ").lower()
        if choice == 'y' or choice == 'yes':
            use_interactive = True
    except:
        pass
    
    return initialize_connection(interactive=use_interactive)

# 初始尝试连接
# 注意：如果代码被导入而不是直接运行，将使用默认方式连接
# 如果是直接运行该模块，则会询问用户是否使用交互式连接
if __name__ == '__main__':
    success = init_db_connection()
else:
    success = initialize_connection(interactive=False)

def ensure_connection():
    """确保数据库连接有效，如果无效则尝试重新连接"""
    global conn, cursor

    # 如果连接不存在，尝试重新连接
    if conn is None:
        logging.warning("数据库连接不存在，尝试重新连接")
        return initialize_connection()

    # 检查连接是否有效
    try:
        conn.ping(reconnect=True)
        return True
    except Exception as e:
        logging.warning(f"数据库连接已断开，尝试重新连接: {e}")
        return initialize_connection()

def query(sql, params=None, query_type="no_select"):
    """
    执行SQL查询或操作。
    
    :param sql: SQL语句
    :param params: SQL参数（可选）
    :param query_type: 查询类型，默认为 "no_select"
                       如果不是 "no_select"，则执行 fetch 操作
    :return: 如果是查询操作，返回数据列表；否则返回 None
    """
    global conn, cursor

    # 确保连接有效
    if not ensure_connection():
        logging.error("数据库连接未初始化，无法执行查询")
        return [] if query_type != "no_select" else None

    try:
        if params:
            params = tuple(params)
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        
        # 确保连接保持活跃
        conn.ping(reconnect=True)
        
        if query_type != "no_select":
            data_list = cursor.fetchall()
            conn.commit()
            logging.info("查询成功，已获取数据。")
            return data_list
        else:
            conn.commit()
            logging.info("操作成功，已提交事务。")
            return None
    except pymysql.MySQLError as e:
        logging.error(f"执行SQL时出错: {e}")
        # 修复此处：确保conn不为None再执行rollback
        if conn:
            try:
                conn.rollback()
            except Exception as rollback_error:
                logging.error(f"回滚事务时出错: {rollback_error}")
        return [] if query_type != "no_select" else None

def close_connection():
    """关闭数据库连接"""
    global conn, cursor
    if cursor:
        try:
            cursor.close()
        except:
            pass
        cursor = None

    if conn:
        try:
            conn.close()
        except:
            pass
        conn = None

def main():
    # 示例用法
    
    # 使用交互式连接方式
    if not conn or not cursor:
        initialize_connection(interactive=True)
    
    # 执行查询操作
    select_sql = "SELECT * FROM article LIMIT 5"
    articles = query(select_sql, query_type="select")
    if articles:
        for article in articles:
            print(article)
    else:
        print("没有找到文章或查询失败")
    
    # 执行插入操作（根据实际表结构修改）
    try:
        print("\n是否插入测试数据? (y/n)")
        choice = input().lower()
        if choice == 'y' or choice == 'yes':
            insert_sql = "INSERT INTO article (id, content) VALUES (%s, %s)"
            article_id = input("输入文章ID: ")
            article_content = input("输入文章内容: ")
            new_article = (article_id, article_content)
            query(insert_sql, params=new_article, query_type="no_select")
            print("插入操作完成。")
    except Exception as e:
        print(f"插入操作失败: {e}")
    
    # 关闭游标和连接
    close_connection()
    logging.info("数据库连接已关闭。")

if __name__ == '__main__':
    main()
