from flask import Blueprint, jsonify, request, render_template
import json
import os
from datetime import datetime
import threading
from queue import Queue
import asyncio
import websockets
import logging
from spider.spiderData import SpiderData
from openai import OpenAI
from anthropic import Anthropic
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
import pymysql
import psycopg2
from pymongo import MongoClient
from cryptography.fernet import Fernet
import base64
import re

# 创建蓝图
spider_bp = Blueprint('spider', __name__)

# 创建日志记录器
logger = logging.getLogger('spider_control')
logger.setLevel(logging.INFO)

# 加密密钥
ENCRYPTION_KEY = Fernet.generate_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

# 存储WebSocket连接的集合
websocket_connections = set()

# 创建消息队列
message_queue = Queue()

# 创建线程池
thread_pool = ThreadPoolExecutor(max_workers=3)

# 创建异步事件循环
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# 默认配置
DEFAULT_CONFIG = {
    'crawlDepth': 3,
    'interval': 5,
    'maxRetries': 3,
    'timeout': 30,
    'maxConcurrent': 2,
    'requestsPerMinute': 60
}

def encrypt_data(data):
    """加密敏感数据"""
    if not data:
        return None
    return cipher_suite.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data):
    """解密敏感数据"""
    if not encrypted_data:
        return None
    return cipher_suite.decrypt(encrypted_data.encode()).decode()

@spider_bp.route('/api/spider/test-db', methods=['POST'])
def test_db_connection():
    """测试数据库连接"""
    try:
        data = request.get_json()
        db_type = data.get('type')
        host = data.get('host')
        port = data.get('port')
        db_name = data.get('name')
        user = data.get('user')
        password = data.get('password')

        if not all([db_type, host, port, db_name, user, password]):
            return jsonify({
                'success': False,
                'message': '请提供完整的数据库配置信息'
            })

        try:
            if db_type == 'mysql':
                connection = pymysql.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database=db_name
                )
                connection.close()
            elif db_type == 'postgresql':
                connection = psycopg2.connect(
                    host=host,
                    port=port,
                    database=db_name,
                    user=user,
                    password=password
                )
                connection.close()
            elif db_type == 'mongodb':
                client = MongoClient(
                    host=host,
                    port=port,
                    username=user,
                    password=password,
                    authSource=db_name
                )
                client.server_info()  # 测试连接
                client.close()
            else:
                return jsonify({
                    'success': False,
                    'message': '不支持的数据库类型'
                })

            return jsonify({
                'success': True,
                'message': '数据库连接测试成功'
            })

        except Exception as e:
            logger.error(f"数据库连接测试失败: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'数据库连接失败: {str(e)}'
            })

    except Exception as e:
        logger.error(f"处理数据库测试请求时出错: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

class SpiderWorker:
    def __init__(self, topics, parameters):
        self.topics = topics
        self.parameters = parameters
        self.total_topics = len(topics)
        self.completed_topics = 0
        self.spider = SpiderData()
        self.message_buffer = []
        self.message_buffer_size = 10
        self.semaphore = asyncio.Semaphore(parameters.get('maxConcurrent', DEFAULT_CONFIG['maxConcurrent']))
        self.rate_limiter = asyncio.Semaphore(parameters.get('requestsPerMinute', DEFAULT_CONFIG['requestsPerMinute']))
        self.accounts = parameters.get('accounts', [])
        self.current_account_index = 0
        self.account_lock = asyncio.Lock()
        
        # 添加筛选条件
        self.filters = parameters.get('filters', {})
        self.interaction_filters = self.filters.get('interaction', {})
        self.regex_filters = self.filters.get('regex', [])
        self.filter_options = self.filters.get('options', {})
        
        # 初始化正则表达式
        self.compiled_regex = []
        for regex_filter in self.regex_filters:
            try:
                pattern = regex_filter['pattern']
                if pattern:
                    self.compiled_regex.append({
                        'regex': re.compile(pattern),
                        'target': regex_filter['target'],
                        'inverse': regex_filter['inverse']
                    })
            except re.error as e:
                logger.error(f"正则表达式编译失败: {pattern}, 错误: {e}")

    def get_next_account(self):
        """获取下一个可用账号"""
        with self.account_lock:
            if not self.accounts:
                raise ValueError("没有可用的账号")
            
            account = self.accounts[self.current_account_index]
            self.current_account_index = (self.current_account_index + 1) % len(self.accounts)
            return account
    
    async def acquire_rate_limit(self):
        """获取速率限制令牌"""
        await self.rate_limiter.acquire()
        asyncio.create_task(self.release_rate_limit())
    
    async def release_rate_limit(self):
        """释放速率限制令牌"""
        await asyncio.sleep(60)  # 1分钟后释放
        self.rate_limiter.release()
    
    async def send_message(self, message):
        """异步发送消息，使用缓冲区优化"""
        self.message_buffer.append(message)
        if len(self.message_buffer) >= self.message_buffer_size:
            await self.flush_messages()
    
    async def flush_messages(self):
        """刷新消息缓冲区"""
        if not self.message_buffer:
            return
        
        try:
            await broadcast_message(self.message_buffer)
            self.message_buffer.clear()
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def crawl_single_topic(self, topic):
        """爬取单个话题"""
        try:
            await self.acquire_rate_limit()
            
            # 获取当前要使用的账号
            account = self.get_next_account()
            
            await self.send_message({
                'type': 'log',
                'message': f'使用账号 {account["username"]} 开始爬取话题: {topic}'
            })
            
            filtered_count = 0
            total_count = 0
            
            async with self.semaphore:
                # 创建一个回调函数来处理爬取的数据
                def process_post(post):
                    nonlocal filtered_count, total_count
                    total_count += 1
                    
                    # 应用筛选条件
                    if self.apply_filters(post):
                        filtered_count += 1
                        return True
                    return False
                
                # 调用爬虫并传入回调函数
                await asyncio.get_event_loop().run_in_executor(
                    thread_pool,
                    lambda: self.spider.crawl_topic(
                        topic,
                        self.parameters['crawlDepth'],
                        self.parameters['interval'],
                        self.parameters['maxRetries'],
                        self.parameters['timeout'],
                        account['cookie'],
                        process_post  # 传入回调函数
                    )
                )
            
            self.completed_topics += 1
            progress = int((self.completed_topics / self.total_topics) * 100)
            
            await self.send_message({
                'type': 'progress',
                'value': progress
            })
            
            # 发送筛选统计信息
            await self.send_message({
                'type': 'log',
                'message': f'话题 {topic} 爬取完成，共爬取 {total_count} 条微博，符合筛选条件 {filtered_count} 条'
            })
            
        except Exception as e:
            logger.error(f"爬取话题 {topic} 失败: {e}")
            await self.send_message({
                'type': 'log',
                'message': f'爬取话题 {topic} 时出错: {str(e)}'
            })
            raise
    
    async def run(self):
        """运行爬虫任务"""
        try:
            tasks = [self.crawl_single_topic(topic) for topic in self.topics]
            await asyncio.gather(*tasks)
            await self.flush_messages()
            
            await self.send_message({
                'type': 'log',
                'message': '所有话题爬取完成'
            })
            
        except Exception as e:
            logger.error(f"爬虫任务执行出错: {e}")
            await self.send_message({
                'type': 'log',
                'message': f'爬虫任务执行出错: {str(e)}'
            })
        finally:
            await self.flush_messages()

    def apply_filters(self, post):
        """
        应用筛选条件到单条微博
        
        Args:
            post: 微博数据字典
            
        Returns:
            bool: 是否通过筛选
        """
        try:
            # 1. 检查互动数据
            if not self._check_interaction_metrics(post):
                return False
                
            # 2. 检查正则匹配
            if not self._check_regex_filters(post):
                return False
                
            # 3. 检查高级选项
            if not self._check_advanced_options(post):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"应用筛选条件时出错: {e}")
            return False
    
    def _check_interaction_metrics(self, post):
        """检查互动指标是否满足条件"""
        try:
            # 获取互动指标的最小值要求
            min_likes = self.interaction_filters.get('minLikes', 0)
            min_comments = self.interaction_filters.get('minComments', 0)
            min_reposts = self.interaction_filters.get('minReposts', 0)
            min_reads = self.interaction_filters.get('minReads', 0)
            
            # 检查是否满足所有条件
            if post.get('like_count', 0) < min_likes:
                return False
            if post.get('comment_count', 0) < min_comments:
                return False
            if post.get('forward_count', 0) < min_reposts:
                return False
            if post.get('read_count', 0) < min_reads:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"检查互动指标时出错: {e}")
            return False
    
    def _check_regex_filters(self, post):
        """检查正则表达式匹配"""
        try:
            for regex_filter in self.compiled_regex:
                regex = regex_filter['regex']
                target = regex_filter['target']
                inverse = regex_filter['inverse']
                
                # 获取目标文本
                if target == 'content':
                    text = post.get('content', '')
                elif target == 'author':
                    text = post.get('user_name', '')
                elif target == 'location':
                    text = post.get('location', '')
                else:
                    continue
                
                # 执行匹配
                match = bool(regex.search(text))
                
                # 如果是反向匹配，取反结果
                if inverse:
                    match = not match
                
                # 如果不满足条件，返回False
                if not match:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查正则匹配时出错: {e}")
            return False
    
    def _check_advanced_options(self, post):
        """检查高级筛选选项"""
        try:
            # 检查是否只要原创内容
            if self.filter_options.get('originalOnly') and not post.get('is_original', False):
                return False
            
            # 检查是否必须包含媒体
            if self.filter_options.get('withMediaOnly') and not post.get('has_media', False):
                return False
            
            # 检查是否只要认证用户
            if self.filter_options.get('verifiedOnly') and not post.get('user_verified', False):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查高级选项时出错: {e}")
            return False

async def broadcast_message(messages):
    """广播消息到所有WebSocket连接"""
    if not websocket_connections:
        return
    
    for websocket in websocket_connections.copy():
        try:
            if isinstance(messages, list):
                for message in messages:
                    await websocket.send(json.dumps(message))
            else:
                await websocket.send(json.dumps(messages))
        except websockets.exceptions.ConnectionClosed:
            websocket_connections.remove(websocket)
        except Exception as e:
            logger.error(f"发送WebSocket消息失败: {e}")
            websocket_connections.remove(websocket)

@spider_bp.route('/spider/control')
def spider_control():
    """渲染爬虫控制页面"""
    return render_template('spider_control.html')

@spider_bp.route('/api/spider/start', methods=['POST'])
async def start_spider():
    """启动爬虫任务"""
    try:
        data = request.get_json()
        topics = data.get('topics', [])
        parameters = {**DEFAULT_CONFIG, **data.get('parameters', {})}
        accounts = data.get('accounts', [])
        
        if not topics:
            return jsonify({
                'success': False,
                'message': '请选择至少一个话题'
            })
        
        if not accounts:
            return jsonify({
                'success': False,
                'message': '请配置至少一个账号'
            })
        
        # 处理账号Cookie的加密存储
        for account in accounts:
            if account.get('saveCookie'):
                account['cookie'] = encrypt_data(account['cookie'])
        
        # 将账号信息添加到参数中
        parameters['accounts'] = accounts
        
        # 创建爬虫工作器
        worker = SpiderWorker(topics, parameters)
        
        # 在事件循环中运行爬虫任务
        asyncio.create_task(worker.run())
        
        return jsonify({
            'success': True,
            'message': '爬虫任务已启动'
        })
        
    except Exception as e:
        logger.error(f"启动爬虫任务失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@spider_bp.route('/api/spider/save-config', methods=['POST'])
def save_spider_config():
    """保存爬虫配置"""
    try:
        config = request.get_json()
        if save_config(config):
            return jsonify({
                'success': True,
                'message': '配置保存成功'
            })
        else:
            return jsonify({
                'success': False,
                'message': '配置保存失败'
            })
    except Exception as e:
        logger.error(f"保存配置失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@spider_bp.websocket('/ws/spider-status')
async def spider_status_socket(websocket):
    """WebSocket连接处理"""
    try:
        websocket_connections.add(websocket)
        logging.info("新的WebSocket连接已建立")
        
        try:
            while True:
                # 等待消息，保持连接活跃
                message = await websocket.receive()
                if message is None:
                    break
        except websockets.exceptions.ConnectionClosed:
            logging.info("WebSocket连接已关闭")
        finally:
            websocket_connections.remove(websocket)
            logging.info("WebSocket连接已移除")
    except Exception as e:
        logger.error(f"WebSocket连接处理失败: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

def get_ai_client():
    """获取可用的AI客户端"""
    # 按优先级尝试不同的AI服务
    if os.getenv('ANTHROPIC_API_KEY'):
        return {
            'type': 'anthropic',
            'client': Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        }
    elif os.getenv('OPENAI_API_KEY'):
        return {
            'type': 'openai',
            'client': OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        }
    else:
        raise ValueError("未找到可用的AI API密钥")

def parse_ai_response(response_text):
    """解析AI响应中的JSON配置"""
    try:
        # 查找JSON内容
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("未找到有效的JSON配置")
        
        json_str = response_text[start:end]
        config = json.loads(json_str)
        
        # 验证配置格式
        if not isinstance(config.get('topics'), list):
            raise ValueError("配置必须包含话题列表")
        
        parameters = config.get('parameters', {})
        if not all(key in parameters for key in ['crawlDepth', 'interval', 'maxRetries', 'timeout']):
            raise ValueError("配置缺少必要的参数")
        
        # 提取建议文本（JSON之前的部分）
        suggestion = response_text[:start].strip()
        
        return config, suggestion
    except Exception as e:
        raise ValueError(f"解析AI响应失败: {str(e)}")

@spider_bp.route('/api/spider/ai-config', methods=['POST'])
def generate_ai_config():
    """使用AI生成爬虫配置"""
    try:
        prompt = request.json.get('prompt', '')
        if not prompt:
            return jsonify({
                'success': False,
                'message': '请提供爬虫需求描述'
            })
        
        # 构建AI提示
        system_prompt = """你是一个专业的爬虫配置助手。请根据用户的自然语言描述，生成合适的微博爬虫配置。

配置应包含以下内容：
1. 要爬取的话题列表
2. 爬虫参数配置
   - 爬取深度(crawlDepth)：1-10页
   - 间隔时间(interval)：3-30秒
   - 重试次数(maxRetries)：1-5次
   - 超时时间(timeout)：10-60秒
   - 最大并行数(maxConcurrent)：1-5
   - 每分钟请求数限制(requestsPerMinute)：30-120

3. 内容筛选条件
   a) 互动数据筛选（设为0表示不启用）
      - 最小点赞数(minLikes)
      - 最小评论数(minComments)
      - 最小转发数(minReposts)
      - 最小阅读数(minReads)
   
   b) 正则表达式筛选（数组，可以有多个规则）
      - pattern: 正则表达式模式
      - target: 匹配目标（content/author/location）
      - inverse: 是否反向匹配（true/false）
   
   c) 高级筛选选项（布尔值）
      - originalOnly: 是否只要原创内容
      - withMediaOnly: 是否必须包含媒体
      - verifiedOnly: 是否只要认证用户

请先用通俗易懂的语言解释你的配置建议，然后在最后提供一个JSON格式的具体配置。
所有数值参数必须是整数，并且在指定范围内。

示例输出格式：
根据您的需求，我建议...

{
    "topics": ["话题1", "话题2"],
    "parameters": {
        "crawlDepth": 5,
        "interval": 5,
        "maxRetries": 3,
        "timeout": 30,
        "maxConcurrent": 2,
        "requestsPerMinute": 60
    },
    "filters": {
        "interaction": {
            "minLikes": 1000,
            "minComments": 100,
            "minReposts": 50,
            "minReads": 10000
        },
        "regex": [
            {
                "pattern": "关键词",
                "target": "content",
                "inverse": false
            }
        ],
        "options": {
            "originalOnly": true,
            "withMediaOnly": false,
            "verifiedOnly": true
        }
    }
}"""

        # 获取AI客户端
        ai = get_ai_client()
        
        try:
            if ai['type'] == 'anthropic':
                response = ai['client'].messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = response.content[0].text
            else:  # OpenAI
                response = ai['client'].chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = response.choices[0].message.content
            
            # 解析AI响应
            config, suggestion = parse_ai_response(response_text)
            
            return jsonify({
                'success': True,
                'config': config,
                'suggestion': suggestion
            })
            
        except Exception as e:
            logger.error(f"AI服务调用失败: {e}")
            return jsonify({
                'success': False,
                'message': f"AI配置生成失败: {str(e)}"
            })
            
    except Exception as e:
        logger.error(f"生成配置失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@spider_bp.route('/api/spider/validate-account', methods=['POST'])
async def validate_account():
    """验证微博账号"""
    try:
        data = request.get_json()
        cookie = data.get('cookie')

        if not cookie:
            return jsonify({
                'success': False,
                'message': 'Cookie不能为空'
            })

        # 创建测试请求
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Cookie': cookie,
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                # 尝试访问微博API
                async with session.get('https://weibo.com/ajax/profile/info', headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data', {}).get('user', {}):
                            return jsonify({
                                'success': True,
                                'message': '账号验证成功'
                            })
                    
                    return jsonify({
                        'success': False,
                        'message': 'Cookie无效或已过期'
                    })
        except Exception as e:
            logger.error(f"验证账号时发生错误: {e}")
            return jsonify({
                'success': False,
                'message': f'验证过程出错: {str(e)}'
            })

    except Exception as e:
        logger.error(f"处理账号验证请求时出错: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }) 