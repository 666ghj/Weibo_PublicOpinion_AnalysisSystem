import pymysql
from pymysql.cursors import DictCursor

class DatabaseManager:
    _instance = None
    _connection = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config):
        """初始化数据库配置"""
        cls._config = config

    @classmethod
    def get_connection(cls):
        """获取数据库连接"""
        if cls._connection is None or not cls._connection.open:
            if cls._config is None:
                raise ValueError("数据库未初始化，请先调用initialize方法设置配置")
            cls._connection = pymysql.connect(
                **cls._config,
                cursorclass=DictCursor
            )
        return cls._connection

    @classmethod
    def close(cls):
        """关闭数据库连接"""
        if cls._connection and cls._connection.open:
            cls._connection.close()
            cls._connection = None 