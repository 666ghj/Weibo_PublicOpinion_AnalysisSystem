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
    if request.method == 'GET':
        response = make_response(render_template('login_and_register.html'))
        return set_secure_headers(response)

    try:
        if request.method == 'POST' and not validate_csrf_token():
            logging.warning("CSRF验证失败")
            return errorResponse('无效的请求')

        client_ip = request.remote_addr
        
        if is_suspicious_ip(client_ip):
            logging.warning(f"可疑IP尝试登录: {client_ip}")
            return errorResponse('由于多次失败尝试，请30分钟后再试')

        username = sanitize_input(request.form.get('username'))
        password = request.form.get('password')  # 密码不需要sanitize
        
        if not username or not password:
            logging.warning("登录失败：用户名或密码为空")
            return errorResponse('用户名和密码不能为空')
        
        # 查询用户信息
        sql = "SELECT password, status FROM user WHERE username = %s"
        result = query(sql, [username], "select")
        
        if result:
            stored_password = result[0]['password']
            status = result[0]['status']
            
            if status != 'active':
                logging.warning(f"已禁用的账户尝试登录: {username}")
                return errorResponse('账户已被禁用')
            
            if verify_password(stored_password, password):
                session.clear()
                session.regenerate()
                
                # 生成唯一会话ID
                session_id = secrets.token_hex(32)
                client_info = get_client_info()
                
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
                
                clear_login_attempts(client_ip)
                
                # 记录登录历史
                login_history_sql = '''
                    INSERT INTO login_history 
                    (username, login_time, ip_address, user_agent, success, attempt_count) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                '''
                query(login_history_sql, [
                    username,
                    datetime.now(),
                    client_info['ip'],
                    client_info['user_agent'],
                    True,
                    redis_client.get(f"login_attempts:{client_ip}") or 0
                ])
                
                logging.info(f"用户 {username} 登录成功")
                response = make_response(redirect('/page/home'))
                return set_secure_headers(response)
        
        record_failed_attempt(client_ip)
        logging.warning(f"登录失败：用户名或密码错误")
        return errorResponse('用户名或密码错误')
            
    except Exception as e:
        logging.error(f"登录过程发生错误: {e}")
        return errorResponse('登录失败，请稍后重试')

@ub.route('/register', methods=['GET', 'POST'])
@limiter.limit("3 per hour")
def register():
    if request.method == 'GET':
        response = make_response(render_template('login_and_register.html'))
        return set_secure_headers(response)
    
    try:
        if request.method == 'POST' and not validate_csrf_token():
            logging.warning("CSRF验证失败")
            return errorResponse('无效的请求')

        username = sanitize_input(request.form.get('username'))
        password = request.form.get('password')
        email = sanitize_input(request.form.get('email'))

        if not username or not password or not email:
            return errorResponse('用户名、密码和邮箱不能为空')

        # 验证用户名格式
        if not re.match(r'^[a-zA-Z0-9_]{4,20}$', username):
            return errorResponse('用户名只能包含字母、数字和下划线，长度4-20位')

        # 验证邮箱格式
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return errorResponse('邮箱格式不正确')

        # 验证密码强度
        if not validate_password(password):
            return errorResponse('密码必须包含大小写字母、数字和特殊字符，且长度至少12位')

        try:
            # 检查用户名和邮箱是否存在
            check_sql = """
                SELECT 
                    (SELECT COUNT(*) FROM user WHERE LOWER(username) = LOWER(%s)) as username_count,
                    (SELECT COUNT(*) FROM user WHERE LOWER(email) = LOWER(%s)) as email_count
            """
            result = query(check_sql, [username.lower(), email.lower()], "select")
            
            if result[0]['username_count'] > 0:
                return errorResponse('该用户名已被注册')
                
            if result[0]['email_count'] > 0:
                return errorResponse('该邮箱已被注册')

            # 哈希密码
            hashed_password = hash_password(password)
            
            # 插入新用户
            insert_sql = '''
                INSERT INTO user(username, password, email, status, createTime, last_password_change) 
                VALUES(%s, %s, %s, %s, %s, %s)
            '''
            current_time = datetime.now()
            query(insert_sql, [
                username, 
                hashed_password,
                email,
                'active',
                current_time,
                current_time
            ])
            
            # 记录注册信息
            client_info = get_client_info()
            register_history_sql = '''
                INSERT INTO register_history 
                (username, register_time, ip_address, user_agent, email) 
                VALUES (%s, %s, %s, %s, %s)
            '''
            query(register_history_sql, [
                username,
                current_time,
                client_info['ip'],
                client_info['user_agent'],
                email
            ])
            
            logging.info(f"新用户注册成功: {username}")
            response = make_response(redirect('/user/login'))
            return set_secure_headers(response)
            
        except Exception as e:
            logging.error(f"注册过程发生错误: {e}")
            return errorResponse('注册失败，请稍后重试')

    except Exception as e:
        logging.error(f"注册过程发生错误: {e}")
        return errorResponse('注册失败，请稍后重试')

@ub.route('/logout')
@login_required
def logout():
    """用户登出"""
    try:
        username = session.get('username')
        client_info = session.get('client_info', {})
        
        # 记录登出历史
        logout_history_sql = '''
            INSERT INTO logout_history 
            (username, logout_time, ip_address, user_agent, session_id) 
            VALUES (%s, %s, %s, %s, %s)
        '''
        query(logout_history_sql, [
            username,
            datetime.now(),
            client_info.get('ip'),
            client_info.get('user_agent'),
            session.get('session_id')
        ])
        
        # 删除Redis中的会话
        redis_client.delete(f"session:{username}")
        
        session.clear()
        logging.info(f"用户 {username} 成功登出")
        response = make_response(redirect('/user/login'))
        return set_secure_headers(response)
    except Exception as e:
        logging.error(f"登出过程发生错误: {e}")
        response = make_response(redirect('/user/login'))
        return set_secure_headers(response)
