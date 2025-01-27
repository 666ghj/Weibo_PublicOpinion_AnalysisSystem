import time
import hashlib
from flask import Blueprint, redirect, render_template, request, Flask, session

from utils.query import query
from utils.errorResponse import errorResponse
from utils.logger import app_logger as logging

ub = Blueprint('user',
               __name__,
               url_prefix='/user',
               template_folder='templates')

# 密码加密函数
def hash_password(password: str, salt: str = 'XiaoXueQi2024') -> str:
    """
    使用 SHA256 对密码进行加盐哈希
    :param password: 用户输入的密码
    :param salt: 加盐值，默认值为 'XiaoXueQi2024'
    :return: 哈希后的密码
    """
    hash_with_salt = hashlib.sha256(salt.encode('utf-8'))
    hash_with_salt.update(password.encode('utf-8'))
    return hash_with_salt.hexdigest()
  
@ub.route('/login', methods=['GET', 'POST'])
def login():
    """
    处理用户登录请求
    :return: 登录页面或重定向到主页
    """
    if request.method == 'GET':
        return render_template('login_and_register.html')  # 显示登录页面

    try:
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            logging.warning("登录失败：用户名或密码为空")
            return render_template('login_and_register.html', msg='用户名和密码不能为空')
        
        # 查询用户
        sql = "SELECT * FROM user WHERE username = %s AND password = %s"
        result = query(sql, [username, password], "select")
        
        if result:
            session['username'] = username
            logging.info(f"用户 {username} 登录成功")
            return redirect('/page/home')
        else:
            logging.warning(f"用户 {username} 登录失败：用户名或密码错误")
            return render_template('login_and_register.html', msg='用户名或密码错误')
            
    except Exception as e:
        logging.error(f"登录过程发生错误: {e}")
        return render_template('login_and_register.html', msg='登录失败，请稍后重试')


@ub.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('login_and_register.html')
    else:

        def filter_fn(user):
            return request.form['username'] in user

        users = query('select * from user', [], 'select')
        filter_list = list(filter(filter_fn, users))
        if len(filter_list):
            return errorResponse('该用户名已被注册')
        else:
            time_tuple = time.localtime(time.time())
            hash_with_salt = hashlib.sha256('XiaoXueQi2024'.encode('utf-8'))
            hash_with_salt.update(request.form['password'].encode('utf-8'))
            query(
                '''
                insert into user(username,password,createTime) values(%s,%s,%s)
            ''', [
                    request.form['username'],
                    hash_with_salt.hexdigest(),
                    str(time_tuple[0]) + '-' + str(time_tuple[1]) + '-' +
                    str(time_tuple[2])
                ])

        return redirect('/user/login')


@ub.route('/logOut')
def logOut():
    session.clear()
    return redirect('/user/login')

@ub.route('/user/logout')
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
