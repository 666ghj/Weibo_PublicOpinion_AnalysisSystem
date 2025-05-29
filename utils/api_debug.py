import logging
import json
import re
from datetime import datetime
import os

logger = logging.getLogger('api_debug')
logger.setLevel(logging.DEBUG)

# 确保日志目录存在
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 创建一个文件处理器
log_file = os.path.join(log_dir, f'api_debug_{datetime.now().strftime("%Y%m%d")}.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_api_request(api_name, request_data, **kwargs):
    """记录API请求数据"""
    try:
        # 过滤掉敏感信息
        filtered_data = filter_sensitive_info(request_data)
        log_message = f"API请求 - {api_name}\n"
        log_message += f"请求数据: {json.dumps(filtered_data, ensure_ascii=False, indent=2)}\n"
        
        for key, value in kwargs.items():
            if isinstance(value, (dict, list)):
                log_message += f"{key}: {json.dumps(value, ensure_ascii=False, indent=2)}\n"
            else:
                log_message += f"{key}: {value}\n"
                
        logger.debug(log_message)
    except Exception as e:
        logger.error(f"记录API请求失败: {e}")

def log_api_response(api_name, response_data, elapsed_time=None, **kwargs):
    """记录API响应数据"""
    try:
        # 过滤掉敏感信息
        filtered_data = filter_sensitive_info(response_data)
        log_message = f"API响应 - {api_name}\n"
        
        if elapsed_time is not None:
            log_message += f"耗时: {elapsed_time:.2f}秒\n"
        
        # 处理不同类型的响应数据
        if isinstance(response_data, (dict, list)):
            log_message += f"响应数据: {json.dumps(filtered_data, ensure_ascii=False, indent=2)}\n"
        else:
            # 如果是字符串等其他类型
            log_message += f"响应数据: {filtered_data}\n"
        
        for key, value in kwargs.items():
            if isinstance(value, (dict, list)):
                log_message += f"{key}: {json.dumps(value, ensure_ascii=False, indent=2)}\n"
            else:
                log_message += f"{key}: {value}\n"
                
        logger.debug(log_message)
    except Exception as e:
        logger.error(f"记录API响应失败: {e}")

def filter_sensitive_info(data):
    """过滤敏感信息，如API密钥、密码等"""
    if isinstance(data, dict):
        filtered_data = {}
        for key, value in data.items():
            # 如果是敏感字段
            if any(sensitive in key.lower() for sensitive in ['key', 'token', 'password', 'secret']):
                if isinstance(value, str) and len(value) > 8:
                    filtered_data[key] = value[:4] + '***' + value[-4:]
                else:
                    filtered_data[key] = '***'
            elif isinstance(value, (dict, list)):
                filtered_data[key] = filter_sensitive_info(value)
            else:
                filtered_data[key] = value
        return filtered_data
    elif isinstance(data, list):
        return [filter_sensitive_info(item) for item in data]
    else:
        # 如果是字符串，尝试过滤掉可能嵌入的敏感信息
        if isinstance(data, str):
            # 过滤API密钥格式
            data = re.sub(r'(sk-|api-|key-)([a-zA-Z0-9]{5,})([a-zA-Z0-9]{4})', r'\1***\3', data)
        return data 