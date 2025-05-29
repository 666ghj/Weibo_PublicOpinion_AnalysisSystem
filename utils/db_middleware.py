from flask import Flask, request, g
from utils.logger import app_logger as logging
from utils.query import ensure_connection

def db_middleware(app):
    @app.before_request
    def check_db_connection():
        if request.path.startswith('/static'):
            # 静态资源不需要数据库连接
            return
            
        if 'db_checked' not in g:
            # 在每个请求开始时确保数据库连接可用
            if not ensure_connection():
                logging.error("无法建立数据库连接，请求可能无法正常处理")
            g.db_checked = True 