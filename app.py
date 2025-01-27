import os
import re
import getpass
import pymysql
import subprocess
from flask import Flask, session, request, redirect, render_template
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import utc
from datetime import datetime, timedelta
import time
from utils.logger import app_logger as logging

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
    password = getpass.getpass(" 4. 密码 (默认: 12345678): ") or "12345678"
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

def initialize_database(connection, sql_file_path):
    """
    执行 SQL 文件中的语句以初始化数据库。
    
    :param connection: 已建立的数据库连接
    :param sql_file_path: SQL 文件的路径
    """
    try:
        with open(sql_file_path, 'r', encoding='utf8') as file:
            sql_commands = file.read()
        
        with connection.cursor() as cursor:
            for statement in sql_commands.split(';'):
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
            connection.commit()
        logging.info("数据库初始化成功。")
    except FileNotFoundError:
        logging.error(f"SQL 文件未找到: {sql_file_path}")
        exit(1)
    except pymysql.MySQLError as e:
        logging.error(f"执行 SQL 时出错: {e}")
        connection.rollback()
        exit(1)
    except Exception as e:
        logging.error(f"初始化数据库时出错: {e}")
        connection.rollback()
        exit(1)

def prompt_first_run():
    """
    询问用户是否首次运行，需要初始化数据库。
    
    :return: Boolean，True 表示需要初始化数据库
    """
    while True:
        choice = input("是否首次运行该项目，需要初始化数据库？(Y/n): ").strip().lower()
        if choice in ['y', 'yes', '']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("请输入 Y 或 N。")

# 初始化 Flask 应用
app = Flask(__name__)
app.secret_key = 'this is secret_key you know ?'  # 设置 Flask 的密钥，用于 session 加密

# 导入蓝图
from views.page import page
from views.user import user
app.register_blueprint(page.pb)  # 注册页面蓝图
app.register_blueprint(user.ub)    # 注册用户蓝图

# 首页路由，清空 session
@app.route('/')
def hello_world():
    session.clear()  # 清空 session，用户退出登录
    return "Session Cleared"

# 中间件：处理请求前的逻辑
@app.before_request
def before_request():
    # 如果请求的是静态文件路径，允许访问
    if request.path.startswith('/static'):
        return
    
    # 如果请求的是登录或注册页面，不需要会话验证
    if request.path in ['/user/login', '/user/register']:
        return
    
    # 如果 session 中没有用户名，重定向到登录页面
    if not session.get('username'):
        return redirect('/user/login')

# 404 错误页面路由
@app.route('/<path:path>')
def catch_all(path):
    return render_template('404.html')  # 如果路径不存在，返回 404 页面

# 定义定时任务，运行爬虫脚本
def run_script():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
    spider_script = os.path.join(current_dir, 'spider', 'main.py')  # 爬虫脚本路径
    # cutComments_script = os.path.join(current_dir, 'utils', 'cutComments.py')  # 评论处理脚本路径
    # cipingTotal_script = os.path.join(current_dir, 'utils', 'cipingTotal.py')  # 评分处理脚本路径

    # 定义所有要运行的脚本
    scripts = [
        ("Spider Script", spider_script),
        # ("Cut Comments Script", cutComments_script),
        # ("Ciping Total Script", cipingTotal_script)
    ]

    # 执行所有脚本
    for script_name, script_path in scripts:
        try:
            logging.info(f"Running {script_name}...")
            subprocess.run(['python', script_path], check=True)  # 使用 subprocess 执行脚本
            logging.info(f"{script_name} finished successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"An error occurred while running {script_name}: {e}")

# 新增功能：动态调度爬虫脚本
def check_database_empty():
    """
    检查数据库中的指定表是否为空。
    
    :return: 如果表为空则返回 True，否则返回 False
    """
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM article")
            result = cursor.fetchone()
            count = result['count'] if result and 'count' in result else 0
            logging.info(f"数据库中共有 {count} 条记录。")
            return count == 0
    except pymysql.MySQLError as e:
        logging.error(f"检查数据库失败: {e}")
        return True  # 连接失败时假设数据库为空，以防止阻塞
    finally:
        if 'connection' in locals():
            connection.close()

def dynamic_crawl():
    """
    执行爬取任务并根据爬取耗时和获取的数据量动态调度下次爬取时间。
    """
    try:
        start_time = time.time()
        logging.info("开始爬取数据。")
        
        run_script()  # 执行爬虫脚本
        
        end_time = time.time()
        duration = end_time - start_time  # 爬取耗时
        
        # 获取爬取后数据库中记录的数量作为数据量
        try:
            connection = pymysql.connect(**DB_CONFIG)
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM article")
                result = cursor.fetchone()
                data_fetched = result['count'] if result and 'count' in result else 0
                logging.info(f"爬取完成，耗时 {duration:.2f} 秒，数据库中共有 {data_fetched} 条记录。")
        except pymysql.MySQLError as e:
            logging.error(f"获取数据量失败: {e}")
            data_fetched = 0
        finally:
            if 'connection' in locals():
                connection.close()
        
        # 根据爬取耗时和数据量调整下次爬取时间
        base_interval = 5 * 60 * 60  # 5小时的基础时间间隔（秒）
        
        if duration > 3600:  # 爬取耗时超过1小时
            next_interval = base_interval + duration
            logging.info(f"检测到长时间爬取。下次爬取将在 {next_interval/3600:.2f} 小时后执行。")
        elif data_fetched < 50:  # 获取的数据量少于50条
            next_interval = base_interval / 2
            logging.info(f"获取数据量较少。下次爬取将在 {next_interval/60:.2f} 分钟后执行。")
        else:
            next_interval = base_interval
            logging.info(f"标准爬取完成。下次爬取将在 {next_interval/3600:.2f} 小时后执行。")
        
        # 安排下次爬取任务
        scheduler.add_job(dynamic_crawl, 'date', run_date=datetime.now() + timedelta(seconds=next_interval), id='dynamic_crawl')
    
    except Exception as e:
        logging.error(f"动态爬取过程中发生错误: {e}")

# 数据库配置，用于动态调度功能
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '12345678',
    'database': 'Weibo_PublicOpinion_AnalysisSystem',
    'port': 3306,
    'charset': 'utf8mb4'
}

# 主程序入口
if __name__ == '__main__':
    # 检测是否需要初始化数据库
    if prompt_first_run():
        # 获取数据库连接
        connection = get_db_connection_interactive()
        
        # 执行数据库初始化
        sql_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'createTables.sql')
        initialize_database(connection, sql_file)
        
        # 关闭数据库连接
        connection.close()
        logging.info("数据库连接已关闭。")
    
    # 设置定时任务，动态执行爬虫脚本
    scheduler = BackgroundScheduler(timezone=utc)  # 创建后台任务调度器
    scheduler.start()  # 启动调度器
    
    # 初始化调度：如果数据库为空，立即爬取；否则，按照基础时间间隔安排首次爬取
    if check_database_empty():
        logging.info("数据库为空。立即开始初始爬取。")
        dynamic_crawl()
    else:
        logging.info("数据库已有数据。安排首次爬取。")
        base_interval = 5 * 60 * 60  # 5小时
        scheduler.add_job(dynamic_crawl, 'date', run_date=datetime.now() + timedelta(seconds=base_interval), id='dynamic_crawl')
    
    try:
        app.run()  # 启动 Flask 应用
    finally:
        scheduler.shutdown()  # 确保在应用关闭时关闭调度器

# 设置日志记录，捕获应用的请求信息
@app.before_request
def log_request_info():
    # 记录每次请求的信息，便于调试和监控
    logging.info(f"Request: {request.method} {request.path}")  # 记录请求的方式（GET/POST）和路径
