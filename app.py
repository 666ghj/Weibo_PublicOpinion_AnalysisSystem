import os
import getpass
import pymysql
import subprocess
from flask import Flask, session, request, redirect
from apscheduler.schedulers.background import BackgroundScheduler
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python < 3.9
from datetime import datetime, timedelta
import secrets
from dotenv import load_dotenv
from utils.logger import app_logger as logging
from utils.db_pool import DatabasePool
from utils.error_handlers import register_error_handlers
from middleware.security import set_secure_headers, log_request_info, require_https

# 加载环境变量
load_dotenv()

def get_db_connection_interactive():
    """
    通过终端交互获取数据库连接参数，若按回车则使用默认值。
    返回一个连接对象。
    """
    print("请依次输入数据库连接信息（直接按回车使用默认值）：")
    
    host = input(" 1. 主机 (默认: localhost): ") or os.getenv('DB_HOST', 'localhost')
    port_str = input(" 2. 端口 (默认: 3306): ") or os.getenv('DB_PORT', '3306')
    try:
        port = int(port_str)
    except ValueError:
        logging.warning("端口号无效，使用默认端口 3306。")
        port = 3306
    
    user = input(" 3. 用户名 (默认: root): ") or os.getenv('DB_USER', 'root')
    password = getpass.getpass(" 4. 密码: ") or os.getenv('DB_PASSWORD', '')
    db_name = input(" 5. 数据库名 (默认: Weibo_PublicOpinion_AnalysisSystem): ") or os.getenv('DB_NAME', 'Weibo_PublicOpinion_AnalysisSystem')
    
    logging.info(f"尝试连接到数据库: {user}@{host}:{port}/{db_name}")
    
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            ssl={'ssl': {'ca': os.getenv('DB_SSL_CA')}} if os.getenv('DB_SSL_CA') else None
        )
        logging.info("数据库连接成功。")
        return connection
    except pymysql.MySQLError as e:
        logging.error(f"数据库连接失败: {e}")
        raise

# 初始化 Flask 应用
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

# 导入蓝图
from views.page import page
from views.user import user
from views.spider_control import spider_bp
from views.workflow_api import workflow_bp, workflow_api_bp
app.register_blueprint(page.pb)
app.register_blueprint(user.ub)
app.register_blueprint(spider_bp)
app.register_blueprint(workflow_bp)
app.register_blueprint(workflow_api_bp)

# 注册错误处理器
register_error_handlers(app)

# 首页路由
@app.route('/')
@require_https()
def hello_world():
    session.clear()
    return redirect('/user/login')

# 请求前中间件
@app.before_request
def before_request():
    # 记录请求信息
    log_request_info()
    
    # 如果请求的是静态文件路径，允许访问
    if request.path.startswith('/static'):
        return

    # 如果请求的是登录或注册页面，不需要会话验证
    if request.path in ['/user/login', '/user/register']:
        return

    # 验证会话
    if not session.get('username'):
        return redirect('/user/login')
    
    # 验证会话完整性
    if 'client_info' not in session:
        session.clear()
        return redirect('/user/login')
        
    # 验证客户端信息
    current_client = {
        'ip': request.remote_addr,
        'user_agent': str(request.user_agent)
    }
    stored_client = session.get('client_info', {})
    
    if (current_client['ip'] != stored_client.get('ip') or 
        current_client['user_agent'] != stored_client.get('user_agent')):
        session.clear()
        return redirect('/user/login')

# 响应后中间件
@app.after_request
def after_request(response):
    return set_secure_headers(response)

# 数据库配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'Weibo_PublicOpinion_AnalysisSystem'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'charset': 'utf8mb4',
    'ssl': {'ca': os.getenv('DB_SSL_CA')} if os.getenv('DB_SSL_CA') else None
}

if __name__ == '__main__':
    # 检测是否需要初始化数据库
    try:
        if os.getenv('INITIALIZE_DB', 'false').lower() == 'true':
            connection = get_db_connection_interactive()
            sql_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'createTables.sql')
            initialize_database(connection, sql_file)
            connection.close()
            logging.info("数据库初始化完成。")
    except Exception as e:
        logging.error(f"数据库初始化失败: {e}")
        exit(1)

    # 初始化数据库连接池
    try:
        DatabasePool.initialize(DB_CONFIG)
    except Exception as e:
        logging.error(f"数据库连接池初始化失败: {e}")
        exit(1)

    # 设置定时任务
    try:
        scheduler = BackgroundScheduler(timezone=ZoneInfo("UTC"))
        scheduler.start()

        if check_database_empty():
            logging.info("数据库为空。立即开始初始爬取。")
            dynamic_crawl()
        else:
            logging.info("数据库已有数据。安排首次爬取。")
            base_interval = int(os.getenv('CRAWL_INTERVAL', '18000'))  # 默认5小时
            scheduler.add_job(
                dynamic_crawl, 
                'date', 
                run_date=datetime.now() + timedelta(seconds=base_interval), 
                id='dynamic_crawl'
            )

        # 启动应用
        app.run(
            host=os.getenv('FLASK_HOST', '127.0.0.1'),
            port=int(os.getenv('FLASK_PORT', '5000')),
            ssl_context='adhoc' if os.getenv('ENABLE_HTTPS', 'false').lower() == 'true' else None
        )
    except Exception as e:
        logging.error(f"应用启动失败: {e}")
        if 'scheduler' in locals():
            scheduler.shutdown()
        DatabasePool.close()
        exit(1)
    finally:
        if 'scheduler' in locals():
            scheduler.shutdown()
        DatabasePool.close()
