import time
from flask import Blueprint, redirect, render_template, request,Flask, session

from utils.query import query
from utils.errResp import errorResponse


ub = Blueprint('user',__name__,url_prefix='/user',template_folder='templates')

@ub.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        def filter_fn(user):
            return request.form['username'] in user and request.form['password'] in user
        users = query('select * from user', [], 'select')
        login_success = list(filter(filter_fn,users))
        if not len(login_success):return errorResponse('账号或密码错误')

        session['username'] = request.form['username']
        return redirect('/page/home')
    
@ub.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        if request.form['password'] != request.form['checkPassword']:return errorResponse('两次密码不符合')
        def filter_fn(user):
            return request.form['username'] in user

        users = query('select * from user',[],'select')
        filter_list = list(filter(filter_fn,users))
        if len(filter_list):
            return errorResponse('该用户名已被注册')
        else:
            time_tuple = time.localtime(time.time())
            query('''
                insert into user(username,password,createTime) values(%s,%s,%s)
            ''',[request.form['username'],request.form['password'],str(time_tuple[0]) + '-' + str(time_tuple[1]) + '-' + str(time_tuple[2])])

        return redirect('/user/login')