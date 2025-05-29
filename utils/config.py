import os
from dotenv import load_dotenv

# 确保只加载一次环境变量
if not os.getenv('ENV_LOADED'):
    load_dotenv()
    os.environ['ENV_LOADED'] = 'true'

def get_database_config():
    """获取数据库配置"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', '123456'),
        'database': os.getenv('DB_NAME', 'Weibo_PublicOpinion_AnalysisSystem'),
        'charset': 'utf8mb4'
    } 