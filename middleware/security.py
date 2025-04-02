from flask import request, redirect
from functools import wraps
import bleach
from utils.logger import app_logger as logging

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
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
    return response

def require_https():
    """强制HTTPS中间件"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_secure and not request.is_localhost:
                url = request.url.replace('http://', 'https://', 1)
                return redirect(url, code=301)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def log_request_info():
    """请求日志记录中间件"""
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