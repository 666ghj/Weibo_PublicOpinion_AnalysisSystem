import pymysql
from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB
from utils.logger import app_logger as logging

class DatabasePool:
    _pool = None
    
    @classmethod
    def initialize(cls, db_config):
        """初始化数据库连接池"""
        try:
            cls._pool = PooledDB(
                creator=pymysql,
                maxconnections=10,
                mincached=2,
                maxcached=5,
                maxshared=3,
                blocking=True,
                maxusage=None,
                setsession=[],
                ping=0,
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                charset=db_config['charset'],
                cursorclass=DictCursor,
                ssl=db_config.get('ssl')
            )
            logging.info("数据库连接池初始化成功")
        except Exception as e:
            logging.error(f"数据库连接池初始化失败: {e}")
            raise
    
    @classmethod
    def get_connection(cls):
        """获取数据库连接"""
        if cls._pool is None:
            raise Exception("数据库连接池未初始化")
        return cls._pool.connection()
    
    @classmethod
    def close(cls):
        """关闭数据库连接池"""
        if cls._pool:
            cls._pool._pool.close()
            cls._pool = None
            logging.info("数据库连接池已关闭") 