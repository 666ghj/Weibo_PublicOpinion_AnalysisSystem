from flask import Flask,session,request,redirect,render_template
import re
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import os
from pytz import utc
import logging

app = Flask(__name__)
app.secret_key = 'this is secret_key you know ?'

from views.page import page
from views.user import user
app.register_blueprint(page.pb)
app.register_blueprint(user.ub)

@app.route('/')
def hello_world():  # put application's code here
    return session.clear()

"""
@app.before_request
def before_reuqest():
    pat = re.compile(r'^/static')
    if re.search(pat,request.path):return
    elif request.path == '/user/login' or request.path == '/user/register':return
    elif session.get('username'):return
    return redirect('/user/login')
"""
#中间件代码逻辑可以优化，以减少重复的 return 语句，并提高可读性：
@app.before_request
def before_request():
    # 静态文件路径允许直接访问
    if request.path.startswith('/static'):
        return
    
    # 登录和注册页面无需验证会话
    if request.path in ['/user/login', '/user/register']:
        return
    
    # 验证用户是否登录
    if not session.get('username'):
        return redirect('/user/login')

@app.route('/<path:path>')
def catch_all(path):
    return render_template('404.html')

def run_script():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spider_script = os.path.join(current_dir, 'spider', 'main.py')
    # cutComments_script = os.path.join(current_dir, 'utils', 'cutComments.py')
    # cipingTotal_script = os.path.join(current_dir, 'utils', 'cipingTotal.py')

    scripts = [
        ("Spider Script", spider_script),
        # ("Cut Comments Script", cutComments_script),
        # ("Ciping Total Script", cipingTotal_script)
    ]

    for script_name, script_path in scripts:
        try:
            print(f"Running {script_name}...")
            subprocess.run(['python', script_path], check=True)
            print(f"{script_name} finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running {script_name}: {e}")


if __name__ == '__main__':
    scheduler = BackgroundScheduler(timezone=utc)
    scheduler.add_job(run_script, 'interval', hours=5)
    scheduler.start()

    try:
        app.run()
    finally:
        scheduler.shutdown()

#为了更好地调试和监控，建议为应用添加日志记录，捕获用户请求和错误：
logging.basicConfig(level=logging.INFO)
@app.before_request
def log_request_info():
    logging.info(f"Request: {request.method} {request.path}")
