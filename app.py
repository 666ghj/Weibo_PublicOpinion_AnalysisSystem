import os
import re
import getpass
import pymysql
import subprocess
from flask import Flask, session, request, redirect, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import utc
from datetime import datetime, timedelta
import time
from utils.logger import app_logger as logging
from utils.db_manager import DatabaseManager
import secrets
from dotenv import load_dotenv
from functools import wraps
import bleach

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

def sanitize_input(text):
    """清理用户输入，防止XSS攻击"""
    if text is None:
        return None
    return bleach.clean(str(text), strip=True)

def set_secure_headers(response):
    """设置安全响应头"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

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
from views.workflow_api import workflow_bp
app.register_blueprint(page.pb)
app.register_blueprint(user.ub)
app.register_blueprint(spider_bp)
app.register_blueprint(workflow_bp)  # 注册工作流蓝图

# 首页路由
@app.route('/')
def hello_world():
    session.clear()
    return redirect('/user/login')

# 请求前中间件
@app.before_request
def before_request():
    # 检查是否是HTTPS
    if not request.is_secure and not app.debug:
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

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

# 错误处理
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', 
                          error_code=500, 
                          error_title='服务器错误', 
                          error_message='服务器遇到了一个问题，请稍后再试。',
                          error_i18n_key='serverError'), 500

@app.errorhandler(403)
def forbidden_error(error):
    return render_template('error.html', 
                          error_code=403, 
                          error_title='禁止访问', 
                          error_message='您没有权限访问此页面。',
                          error_i18n_key='forbidden'), 403

@app.errorhandler(400)
def bad_request_error(error):
    return render_template('error.html', 
                          error_code=400, 
                          error_title='错误请求', 
                          error_message='服务器无法理解您的请求。',
                          error_i18n_key='badRequest'), 400

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

# 初始化数据库管理器
DatabaseManager.initialize(DB_CONFIG)

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

    # 设置定时任务
    try:
        scheduler = BackgroundScheduler(timezone=utc)
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
        exit(1)
    finally:
        if 'scheduler' in locals():
            scheduler.shutdown()

# 请求日志记录
@app.before_request
def log_request_info():
    # 记录请求信息，但排除敏感数据
    sanitized_headers = dict(request.headers)
    if 'Authorization' in sanitized_headers:
        sanitized_headers['Authorization'] = '[FILTERED]'
    if 'Cookie' in sanitized_headers:
        sanitized_headers['Cookie'] = '[FILTERED]'

    logging.info(
        f"Request: {request.method} {request.path}\n"
        f"Remote IP: {request.remote_addr}\n"
        f"Headers: {sanitized_headers}"
    )
