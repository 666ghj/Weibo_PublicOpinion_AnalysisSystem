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

# 创建蓝图
spider_bp = Blueprint('spider', __name__)

# 创建日志记录器
logger = logging.getLogger('spider_control')
logger.setLevel(logging.INFO)

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
    'maxConcurrent': 2
}

# 限流装饰器
@sleep_and_retry
@limits(calls=100, period=60)  # 每分钟最多100个请求
def rate_limited_request():
    pass

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
            rate_limited_request()
            
            await self.send_message({
                'type': 'log',
                'message': f'开始爬取话题: {topic}'
            })
            
            async with self.semaphore:
                await asyncio.get_event_loop().run_in_executor(
                    thread_pool,
                    self.spider.crawl_topic,
                    topic,
                    self.parameters['crawlDepth'],
                    self.parameters['interval'],
                    self.parameters['maxRetries'],
                    self.parameters['timeout']
                )
            
            self.completed_topics += 1
            progress = int((self.completed_topics / self.total_topics) * 100)
            
            await self.send_message({
                'type': 'progress',
                'value': progress
            })
            
            await self.send_message({
                'type': 'log',
                'message': f'话题 {topic} 爬取完成'
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
        
        if not topics:
            return jsonify({
                'success': False,
                'message': '请选择至少一个话题'
            })
        
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
2. 爬虫参数（爬取深度、间隔时间、重试次数、超时时间）

请先用通俗易懂的语言解释你的配置建议，然后在最后提供一个JSON格式的具体配置。
注意：
- 爬取深度(crawlDepth)范围：1-10页
- 间隔时间(interval)范围：3-30秒
- 重试次数(maxRetries)范围：1-5次
- 超时时间(timeout)范围：10-60秒
- 所有参数都必须是整数

示例输出格式：
根据您的需求，我建议...

{
    "topics": ["话题1", "话题2"],
    "parameters": {
        "crawlDepth": 5,
        "interval": 5,
        "maxRetries": 3,
        "timeout": 30
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