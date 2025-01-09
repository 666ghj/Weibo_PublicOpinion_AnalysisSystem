import os
import re
import logging
import getpass
import pymysql
import subprocess
from flask import Flask, session, request, redirect, render_template
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import utc

# 初始化日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
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
            print(f"Running {script_name}...")  # 打印运行开始的信息
            subprocess.run(['python', script_path], check=True)  # 使用 subprocess 执行脚本
            print(f"{script_name} finished successfully.")  # 打印脚本成功完成的消息
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running {script_name}: {e}")  # 打印错误信息

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
    
    # 设置定时任务，定期执行爬虫脚本
    scheduler = BackgroundScheduler(timezone=utc)  # 创建后台任务调度器
    scheduler.add_job(run_script, 'interval', hours=5)  # 每5小时执行一次爬虫脚本
    scheduler.start()  # 启动调度器

    try:
        app.run()  # 启动 Flask 应用
    finally:
        scheduler.shutdown()  # 确保在应用关闭时关闭调度器

# 设置日志记录，捕获应用的请求信息
@app.before_request
def log_request_info():
    # 记录每次请求的信息，便于调试和监控
    logging.info(f"Request: {request.method} {request.path}")  # 记录请求的方式（GET/POST）和路径
