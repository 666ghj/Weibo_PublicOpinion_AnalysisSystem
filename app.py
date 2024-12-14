from flask import Flask, session, request, redirect, render_template
import re
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import os
from pytz import utc
import logging

# 初始化 Flask 应用
app = Flask(__name__)
app.secret_key = 'this is secret_key you know ?'  # 设置 Flask 的密钥，用于 session 加密

# 导入蓝图
from views.page import page
from views.user import user
app.register_blueprint(page.pb)  # 注册页面蓝图
app.register_blueprint(user.ub)  # 注册用户蓝图

# 首页路由，清空 session
@app.route('/')
def hello_world():
    return session.clear()  # 清空 session，用户退出登录

"""
@app.before_request
def before_reuqest():
    pat = re.compile(r'^/static')  # 正则匹配静态文件路径
    if re.search(pat, request.path):  # 如果是静态文件，直接返回
        return
    elif request.path == '/user/login' or request.path == '/user/register':  # 登录或注册页面无需验证
        return
    elif session.get('username'):  # 如果 session 中有用户名，则允许继续
        return
    return redirect('/user/login')  # 否则重定向到登录页面
"""

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
    # 设置定时任务，定期执行爬虫脚本
    scheduler = BackgroundScheduler(timezone=utc)  # 创建后台任务调度器
    scheduler.add_job(run_script, 'interval', hours=5)  # 每5小时执行一次爬虫脚本
    scheduler.start()  # 启动调度器

    try:
        app.run()  # 启动 Flask 应用
    finally:
        scheduler.shutdown()  # 确保在应用关闭时关闭调度器

# 设置日志记录，捕获应用的请求信息
logging.basicConfig(level=logging.INFO)  # 配置日志记录，设置日志级别为 INFO

@app.before_request
def log_request_info():
    # 记录每次请求的信息，便于调试和监控
    logging.info(f"Request: {request.method} {request.path}")  # 记录请求的方式（GET/POST）和路径
