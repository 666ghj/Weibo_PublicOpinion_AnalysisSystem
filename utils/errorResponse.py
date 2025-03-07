from flask import render_template, jsonify
import bleach
import re

def sanitize_error_message(message):
    """
    清理和验证错误消息
    """
    if not message:
        return "发生未知错误"
        
    # 移除任何敏感信息
    message = re.sub(r'(password|token|key|secret)=[\w\-]+', r'\1=[FILTERED]', str(message))
    
    # 清理HTML和特殊字符
    message = bleach.clean(message, strip=True)
    
    # 限制消息长度
    return message[:200] if len(message) > 200 else message

def errorResponse(errorMsg, status_code=400):
    """
    统一的错误响应处理
    :param errorMsg: 错误消息
    :param status_code: HTTP状态码
    :return: 错误响应
    """
    safe_message = sanitize_error_message(errorMsg)
    
    if 'application/json' in request.headers.get('Accept', ''):
        return jsonify({
            'success': False,
            'error': safe_message
        }), status_code
    
    return render_template(
        'error.html',
        errorMsg=safe_message,
        status_code=status_code
    ), status_code