import getpass
import pymysql
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("database_operations.log"),
        logging.StreamHandler()
    ]
)

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
    password = getpass.getpass(" 4. 密码 (默认: 312517): ") or "312517"
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
        exit(1)

# 获取数据库连接
conn = get_db_connection_interactive()

# 获取游标
cursor = conn.cursor()

def query(sql, params=None, query_type="no_select"):
    """
    执行SQL查询或操作。
    
    :param sql: SQL语句
    :param params: SQL参数（可选）
    :param query_type: 查询类型，默认为 "no_select"
                       如果不是 "no_select"，则执行 fetch 操作
    :return: 如果是查询操作，返回数据列表；否则返回 None
    """
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
    except pymysql.MySQLError as e:
        logging.error(f"执行SQL时出错: {e}")
        conn.rollback()
        return None

def main():
    # 示例用法
    
    # 执行查询操作
    select_sql = "SELECT * FROM article LIMIT 5"
    articles = query(select_sql, query_type="select")
    if articles:
        for article in articles:
            print(article)
    
    # 执行插入操作（根据实际表结构修改）
    insert_sql = "INSERT INTO article (id, content) VALUES (%s, %s)"
    new_article = (12345, "这是一条新的文章内容。")
    result = query(insert_sql, params=new_article, query_type="no_select")
    if result is None:
        logging.info("插入操作完成。")
    
    # 关闭游标和连接
    cursor.close()
    conn.close()
    logging.info("数据库连接已关闭。")

if __name__ == '__main__':
    main()
