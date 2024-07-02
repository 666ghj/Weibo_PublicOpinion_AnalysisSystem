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