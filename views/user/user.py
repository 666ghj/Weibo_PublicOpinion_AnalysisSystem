import time
import hashlib
from flask import Blueprint, redirect, render_template, request, Flask, session, current_app
from datetime import datetime, timedelta
import re
from utils.query import query
from utils.errorResponse import errorResponse
from utils.logger import app_logger as logging
from functools import wraps
import secrets

ub = Blueprint('user',
               __name__,
               url_prefix='/user',
               template_folder='templates')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect('/user/login')
        return f(*args, **kwargs)
    return decorated_function

# 密码加密函数
def hash_password(password: str, salt: str = None) -> tuple:
    """
    使用 SHA256 对密码进行加盐哈希
    :param password: 用户输入的密码
    :param salt: 可选的盐值
    :return: (哈希后的密码, 盐值)
    """
    if not salt:
        salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256()
    hash_obj.update(salt.encode('utf-8'))
    hash_obj.update(password.encode('utf-8'))
    return hash_obj.hexdigest(), salt

def validate_password(password: str) -> bool:
    """
    验证密码强度
    """
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True

@ub.route('/login', methods=['GET', 'POST'])
def login():
    """
    处理用户登录请求
    """
    if request.method == 'GET':
        return render_template('login_and_register.html')

    try:
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            logging.warning("登录失败：用户名或密码为空")
            return render_template('login_and_register.html', msg='用户名和密码不能为空')
        
        # 查询用户和盐值
        sql = "SELECT password, salt FROM user WHERE username = %s"
        result = query(sql, [username], "select")
        
        if result:
            stored_password = result[0]['password']
            salt = result[0]['salt']
            
            # 验证密码
            hashed_input, _ = hash_password(password, salt)
            
            if hashed_input == stored_password:
                session.clear()
                session['username'] = username
                session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                session['csrf_token'] = secrets.token_hex(32)
                session.permanent = True
                current_app.permanent_session_lifetime = timedelta(hours=2)
                
                logging.info(f"用户 {username} 登录成功")
                return redirect('/page/home')
        
        # 使用相同的响应防止用户枚举
        logging.warning(f"登录失败：用户名或密码错误")
        return render_template('login_and_register.html', msg='用户名或密码错误')
            
    except Exception as e:
        logging.error(f"登录过程发生错误: {e}")
        return render_template('login_and_register.html', msg='登录失败，请稍后重试')

@ub.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('login_and_register.html')
    
    try:
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            return errorResponse('用户名和密码不能为空')

        # 验证用户名格式
        if not re.match(r'^[a-zA-Z0-9_]{4,20}$', username):
            return errorResponse('用户名只能包含字母、数字和下划线，长度4-20位')

        # 验证密码强度
        if not validate_password(password):
            return errorResponse('密码必须包含大小写字母、数字和特殊字符，且长度至少8位')

        # 使用事务处理竞态条件
        try:
            # 检查用户名是否存在
            check_sql = "SELECT COUNT(*) as count FROM user WHERE username = %s"
            result = query(check_sql, [username], "select")
            
            if result[0]['count'] > 0:
                return errorResponse('该用户名已被注册')

            # 生成密码哈希和盐值
            hashed_password, salt = hash_password(password)
            
            # 插入新用户
            insert_sql = '''
                INSERT INTO user(username, password, salt, createTime) 
                VALUES(%s, %s, %s, %s)
            '''
            current_time = datetime.now().strftime('%Y-%m-%d')
            query(insert_sql, [username, hashed_password, salt, current_time])
            
            logging.info(f"新用户注册成功: {username}")
            return redirect('/user/login')
            
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
        session.clear()
        logging.info(f"用户 {username} 成功登出")
        return redirect('/user/login')
    except Exception as e:
        logging.error(f"登出过程发生错误: {e}")
        return redirect('/user/login')
