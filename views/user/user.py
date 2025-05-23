import time
import hashlib
from flask import Blueprint, redirect, render_template, request, Flask, session, current_app, make_response
from datetime import datetime, timedelta
import re
from utils.query import query
from utils.errorResponse import errorResponse
from utils.logger import app_logger as logging
from functools import wraps
import secrets
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
import json
import bleach
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import html
import os
from utils.db_manager import DatabaseManager
import pymysql

# 创建Argon2密码哈希器
ph = PasswordHasher()

# Redis连接
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# 创建限流器
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

ub = Blueprint('user',
               __name__,
               url_prefix='/user',
               template_folder='templates')

def sanitize_input(text):
    """清理用户输入，防止XSS攻击"""
    if text is None:
        return None
    return bleach.clean(str(text), strip=True)

def validate_csrf_token():
    """验证CSRF令牌"""
    # 开发环境中可以跳过验证
    if os.environ.get('FLASK_ENV') == 'development':
        return True
        
    token = request.form.get('csrf_token')
    stored_token = session.get('csrf_token')
    if not token or not stored_token or token != stored_token:
        return False
    return True

def get_client_info():
    """获取客户端信息"""
    return {
        'ip': request.remote_addr,
        'user_agent': str(request.user_agent.string),
        'platform': str(request.user_agent.platform),
        'browser': str(request.user_agent.browser),
    }

def is_suspicious_ip(ip):
    """检查IP是否可疑"""
    key = f"login_attempts:{ip}"
    attempts = redis_client.get(key)
    if attempts and int(attempts) >= 5:  # 5次失败尝试
        return True
    return False

def record_failed_attempt(ip):
    """记录失败的登录尝试"""
    key = f"login_attempts:{ip}"
    pipe = redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, 1800)  # 30分钟后重置
    pipe.execute()

def clear_login_attempts(ip):
    """清除登录尝试记录"""
    redis_client.delete(f"login_attempts:{ip}")

def set_secure_headers(response):
    """设置安全响应头"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect('/user/login')
        
        # 验证会话完整性
        if 'client_info' not in session or 'session_id' not in session:
            session.clear()
            return redirect('/user/login')
            
        # 验证客户端信息
        current_client = get_client_info()
        stored_client = session['client_info']
        
        if (current_client['ip'] != stored_client['ip'] or 
            current_client['user_agent'] != stored_client['user_agent']):
            session.clear()
            return redirect('/user/login')
            
        # 验证会话ID
        stored_session_id = redis_client.get(f"session:{session['username']}")
        if not stored_session_id or stored_session_id != session['session_id']:
            session.clear()
            return redirect('/user/login')
            
        return f(*args, **kwargs)
    return decorated_function

def hash_password(password: str) -> str:
    """
    使用Argon2id算法哈希密码
    :param password: 用户输入的密码
    :return: 哈希后的密码
    """
    return ph.hash(password)

def verify_password(stored_hash: str, password: str) -> bool:
    """
    验证密码
    :param stored_hash: 存储的密码哈希
    :param password: 用户输入的密码
    :return: 是否匹配
    """
    try:
        return ph.verify(stored_hash, password)
    except VerifyMismatchError:
        return False

def validate_password(password: str) -> bool:
    """
    验证密码强度
    """
    if len(password) < 12:  # 增加最小长度要求
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    # 检查常见密码模式
    common_patterns = ['password', '123456', 'qwerty']
    if any(pattern in password.lower() for pattern in common_patterns):
        return False
    return True

@ub.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    """处理用户登录请求"""
    # 在函数开始处导入本地request对象，避免依赖全局request
    from flask import request as local_req
    
    if local_req.method == 'GET':
        # 生成CSRF令牌并保存到会话
        if 'csrf_token' not in session:
            session['csrf_token'] = secrets.token_hex(32)
            
        response = make_response(render_template('login_and_register.html'))
        # 直接设置安全头，不调用可能引起问题的函数
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'SAMEORIGIN',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'"
        }
        for header, value in headers.items():
            response.headers[header] = value
        return response

    try:
        # CSRF验证
        csrf_valid = False
        if os.environ.get('FLASK_ENV') == 'development':
            csrf_valid = True
        else:
            token = local_req.form.get('csrf_token')
            stored_token = session.get('csrf_token')
            if token and stored_token and token == stored_token:
                csrf_valid = True
                
        if local_req.method == 'POST' and not csrf_valid:
            logging.warning("CSRF验证失败")
            return render_template('error.html', 
                                 error_code=400, 
                                 error_title="请求错误", 
                                 error_message="无效的请求", 
                                 error_i18n_key="error_invalid_request")

        client_ip = local_req.remote_addr
        
        # 检查IP是否可疑
        key = f"login_attempts:{client_ip}"
        attempts = redis_client.get(key)
        if attempts and int(attempts) >= 5:
            logging.warning(f"可疑IP尝试登录: {client_ip}")
            return render_template('error.html', 
                                 error_code=429, 
                                 error_title="请求过于频繁", 
                                 error_message="由于多次失败尝试，请30分钟后再试", 
                                 error_i18n_key="error_too_many_attempts")

        # 获取并清理用户输入
        username = local_req.form.get('username', '')
        if username:
            username = bleach.clean(str(username), strip=True)
            
        password = local_req.form.get('password', '')
        
        if not username or not password:
            logging.warning("登录失败：用户名或密码为空")
            return render_template('error.html', 
                                 error_code=400, 
                                 error_title="登录错误", 
                                 error_message="用户名和密码不能为空", 
                                 error_i18n_key="error_credential_required")
        
        # 查询用户信息
        try:
            conn = pymysql.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD', ''),
                database=os.getenv('DB_NAME', 'Weibo_PublicOpinion_AnalysisSystem'),
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            
            with conn.cursor() as cursor:
                cursor.execute("SELECT password, status FROM user WHERE username = %s", (username,))
                result = cursor.fetchone()
                
                if result:
                    stored_password = result['password']
                    status = result['status']
                    
                    if status != 'active':
                        logging.warning(f"已禁用的账户尝试登录: {username}")
                        return render_template('error.html', 
                                             error_code=403, 
                                             error_title="账户已禁用", 
                                             error_message="您的账户已被禁用", 
                                             error_i18n_key="error_account_disabled")
                    
                    # 验证密码
                    password_valid = False
                    try:
                        password_valid = ph.verify(stored_password, password)
                    except VerifyMismatchError:
                        password_valid = False
                        
                    if password_valid:
                        session.clear()
                        
                        # 生成会话信息
                        session_id = secrets.token_hex(32)
                        
                        # 获取客户端信息
                        client_info = {
                            'ip': local_req.remote_addr,
                            'user_agent': str(local_req.user_agent),
                            'platform': str(local_req.user_agent.platform) if local_req.user_agent.platform else 'unknown',
                            'browser': str(local_req.user_agent.browser) if local_req.user_agent.browser else 'unknown',
                        }
                        
                        # 存储会话信息
                        session['username'] = username
                        session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        session['csrf_token'] = secrets.token_hex(32)
                        session['client_info'] = client_info
                        session['session_id'] = session_id
                        session.permanent = True
                        current_app.permanent_session_lifetime = timedelta(hours=2)
                        
                        # 在Redis中存储会话ID
                        redis_client.setex(
                            f"session:{username}",
                            int(current_app.permanent_session_lifetime.total_seconds()),
                            session_id
                        )
                        
                        # 清除失败尝试记录
                        redis_client.delete(f"login_attempts:{client_ip}")
                        
                        # 记录登录历史
                        try:
                            cursor.execute(
                                """
                                INSERT INTO login_history 
                                (username, login_time, ip_address, user_agent, success, attempt_count) 
                                VALUES (%s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    username,
                                    datetime.now(),
                                    client_info['ip'],
                                    client_info['user_agent'],
                                    True,
                                    redis_client.get(f"login_attempts:{client_ip}") or 0
                                )
                            )
                            conn.commit()
                        except Exception as e:
                            logging.warning(f"记录登录历史失败: {e}")
                            # 不阻止登录流程继续
                        
                        logging.info(f"用户 {username} 登录成功")
                        return redirect('/page/home')
            
            # 记录失败尝试
            pipe = redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, 1800)  # 30分钟后重置
            pipe.execute()
            
            logging.warning(f"登录失败：用户名或密码错误")
            return render_template('error.html', 
                                 error_code=401, 
                                 error_title="登录失败", 
                                 error_message="用户名或密码错误", 
                                 error_i18n_key="error_invalid_credentials")
                
        except Exception as e:
            logging.error(f"数据库操作失败: {e}")
            return render_template('error.html', 
                                error_code=500, 
                                error_title="服务器错误", 
                                error_message="登录失败，请稍后重试", 
                                error_i18n_key="error_server")
        finally:
            if 'conn' in locals() and conn:
                conn.close()
            
    except Exception as e:
        logging.error(f"登录过程发生错误: {e}")
        return render_template('error.html', 
                             error_code=500, 
                             error_title="服务器错误", 
                             error_message="登录失败，请稍后重试", 
                             error_i18n_key="error_server")

@ub.route('/register', methods=['GET', 'POST'])
@limiter.limit("3 per hour")
def register():
    """处理用户注册请求"""
    try:
        # 导入本地request对象，不要被任何其他代码影响
        from flask import request as local_req
        
        # GET请求处理
        if local_req.method == 'GET':
            # 生成CSRF令牌并保存到会话
            if 'csrf_token' not in session:
                session['csrf_token'] = secrets.token_hex(32)
            response = make_response(render_template('login_and_register.html'))
            headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'SAMEORIGIN',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'Content-Security-Policy': "default-src 'self'"
            }
            for header, value in headers.items():
                response.headers[header] = value
            return response
        
        # POST请求处理
        if local_req.method == 'POST':
            # 直接获取表单数据
            username = local_req.form.get('username', '')
            password = local_req.form.get('password', '')
            email = local_req.form.get('email', '')
            
            # 清理数据防止XSS
            if username:
                username = bleach.clean(str(username), strip=True)
            if email:
                email = bleach.clean(str(email), strip=True)
            
            # 记录注册尝试
            logging.info(f"收到注册请求: username={username}, email={email}")
            
            # 基本输入验证
            if not username:
                # 直接返回错误，不使用errorResponse
                return render_template('error.html', 
                                     error_code=400, 
                                     error_title="请求错误", 
                                     error_message="用户名不能为空", 
                                     error_i18n_key="error_user_required")
            
            if not password:
                return render_template('error.html', 
                                     error_code=400, 
                                     error_title="请求错误", 
                                     error_message="密码不能为空", 
                                     error_i18n_key="error_password_required")
            
            if not email:
                return render_template('error.html', 
                                     error_code=400, 
                                     error_title="请求错误", 
                                     error_message="邮箱不能为空", 
                                     error_i18n_key="error_email_required")
            
            # 格式验证
            if not re.match(r'^[a-zA-Z0-9_]{4,20}$', username):
                return render_template('error.html', 
                                     error_code=400, 
                                     error_title="输入错误", 
                                     error_message="用户名只能包含字母、数字和下划线，长度4-20位", 
                                     error_i18n_key="error_username_format")
            
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                return render_template('error.html', 
                                     error_code=400, 
                                     error_title="输入错误", 
                                     error_message="邮箱格式不正确", 
                                     error_i18n_key="error_email_format")
            
            # 密码强度验证
            if len(password) < 12:
                return render_template('error.html', 
                                     error_code=400, 
                                     error_title="密码太弱", 
                                     error_message="密码长度至少12位", 
                                     error_i18n_key="error_password_length")
            
            if not re.search(r"[A-Z]", password):
                return render_template('error.html', 
                                     error_code=400, 
                                     error_title="密码太弱", 
                                     error_message="密码必须包含大写字母", 
                                     error_i18n_key="error_password_uppercase")
            
            if not re.search(r"[a-z]", password):
                return render_template('error.html', 
                                     error_code=400, 
                                     error_title="密码太弱", 
                                     error_message="密码必须包含小写字母", 
                                     error_i18n_key="error_password_lowercase")
            
            if not re.search(r"\d", password):
                return render_template('error.html', 
                                     error_code=400, 
                                     error_title="密码太弱", 
                                     error_message="密码必须包含数字", 
                                     error_i18n_key="error_password_digit")
            
            if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
                return render_template('error.html', 
                                     error_code=400, 
                                     error_title="密码太弱", 
                                     error_message="密码必须包含特殊字符", 
                                     error_i18n_key="error_password_special")
            
            # 数据库操作
            try:
                conn = pymysql.connect(
                    host=os.getenv('DB_HOST', 'localhost'),
                    user=os.getenv('DB_USER', 'root'),
                    password=os.getenv('DB_PASSWORD', ''),
                    database=os.getenv('DB_NAME', 'Weibo_PublicOpinion_AnalysisSystem'),
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor
                )
                
                with conn.cursor() as cursor:
                    # 检查用户名是否存在
                    cursor.execute(
                        "SELECT COUNT(*) as count FROM user WHERE LOWER(username) = LOWER(%s)",
                        (username.lower(),)
                    )
                    if cursor.fetchone()['count'] > 0:
                        return render_template('error.html', 
                                             error_code=400, 
                                             error_title="注册失败", 
                                             error_message="该用户名已被注册", 
                                             error_i18n_key="error_username_exists")
                    
                    # 检查邮箱是否存在
                    cursor.execute(
                        "SELECT COUNT(*) as count FROM user WHERE LOWER(email) = LOWER(%s)",
                        (email.lower(),)
                    )
                    if cursor.fetchone()['count'] > 0:
                        return render_template('error.html', 
                                             error_code=400, 
                                             error_title="注册失败", 
                                             error_message="该邮箱已被注册", 
                                             error_i18n_key="error_email_exists")
                    
                    # 哈希密码
                    hashed_password = ph.hash(password)
                    
                    # 插入新用户
                    current_time = datetime.now()
                    cursor.execute(
                        """
                        INSERT INTO user(username, password, email, status, createTime, last_password_change) 
                        VALUES(%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            username, 
                            hashed_password,
                            email,
                            'active',
                            current_time,
                            current_time
                        )
                    )
                    
                    # 记录注册历史（如果表存在）
                    try:
                        client_info = {
                            'ip': local_req.remote_addr,
                            'user_agent': str(local_req.user_agent),
                            'platform': str(local_req.user_agent.platform) if local_req.user_agent.platform else 'unknown',
                            'browser': str(local_req.user_agent.browser) if local_req.user_agent.browser else 'unknown',
                        }
                        
                        cursor.execute(
                            """
                            INSERT INTO register_history 
                            (username, register_time, ip_address, user_agent, email) 
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (
                                username,
                                current_time,
                                client_info['ip'],
                                client_info['user_agent'],
                                email
                            )
                        )
                    except Exception as e:
                        # 如果register_history表不存在，忽略错误
                        logging.warning(f"注册历史记录失败: {e}")
                    
                    conn.commit()
                    logging.info(f"新用户注册成功: {username}")
                
                # 重定向到登录页
                return redirect('/user/login')
            
            except Exception as e:
                logging.error(f"数据库操作失败: {e}")
                return render_template('error.html', 
                                       error_code=500, 
                                       error_title="注册失败", 
                                       error_message="服务器内部错误，请稍后重试", 
                                       error_i18n_key="error_server")
            finally:
                if 'conn' in locals() and conn:
                    conn.close()
    
    except Exception as e:
        logging.error(f"注册过程发生错误: {e}")
        return render_template('error.html', 
                               error_code=500, 
                               error_title="注册失败", 
                               error_message="服务器内部错误，请稍后重试", 
                               error_i18n_key="error_server")

@ub.route('/logout')
@login_required
def logout():
    """用户登出"""
    try:
        # 避免依赖装饰器，直接在函数内检查登录状态
        if 'username' not in session:
            return redirect('/user/login')
            
        username = session.get('username')
        client_info = session.get('client_info', {})
        
        try:
            # 记录登出历史
            conn = pymysql.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD', ''),
                database=os.getenv('DB_NAME', 'Weibo_PublicOpinion_AnalysisSystem'),
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO logout_history 
                    (username, logout_time, ip_address, user_agent, session_id) 
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        username,
                        datetime.now(),
                        client_info.get('ip'),
                        client_info.get('user_agent'),
                        session.get('session_id')
                    )
                )
                conn.commit()
        except Exception as e:
            logging.warning(f"记录登出历史失败: {e}")
            # 不阻止登出流程继续
        finally:
            if 'conn' in locals() and conn:
                conn.close()
        
        # 删除Redis中的会话
        redis_client.delete(f"session:{username}")
        
        session.clear()
        logging.info(f"用户 {username} 成功登出")
        
        response = make_response(redirect('/user/login'))
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'SAMEORIGIN',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'"
        }
        for header, value in headers.items():
            response.headers[header] = value
        return response
    except Exception as e:
        logging.error(f"登出过程发生错误: {e}")
        response = make_response(redirect('/user/login'))
        return response
