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

# 创建蓝图
spider_bp = Blueprint('spider', __name__)

# 创建日志记录器
logger = logging.getLogger('spider_control')
logger.setLevel(logging.INFO)

# 存储WebSocket连接的集合
websocket_connections = set()

# 创建消息队列
message_queue = Queue()

# 默认配置
DEFAULT_CONFIG = {
    'crawlDepth': 3,
    'interval': 5,
    'maxRetries': 3,
    'timeout': 30
}

def load_config():
    """加载爬虫配置"""
    config_path = os.path.join(os.path.dirname(__file__), '../spider/config.json')
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
    return DEFAULT_CONFIG

def save_config(config):
    """保存爬虫配置"""
    config_path = os.path.join(os.path.dirname(__file__), '../spider/config.json')
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        return False

async def broadcast_message(message):
    """广播消息到所有WebSocket连接"""
    if not websocket_connections:
        return
    
    for websocket in websocket_connections.copy():
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            websocket_connections.remove(websocket)
        except Exception as e:
            logger.error(f"发送WebSocket消息失败: {e}")
            websocket_connections.remove(websocket)

def spider_worker(topics, parameters):
    """爬虫工作线程"""
    total_topics = len(topics)
    completed_topics = 0
    
    try:
        spider = SpiderData()
        
        for topic in topics:
            try:
                # 更新进度
                progress = int((completed_topics / total_topics) * 100)
                asyncio.run(broadcast_message({
                    'type': 'progress',
                    'value': progress
                }))
                
                # 发送开始爬取的日志
                asyncio.run(broadcast_message({
                    'type': 'log',
                    'message': f'开始爬取话题: {topic}'
                }))
                
                # 执行爬取
                spider.crawl_topic(
                    topic=topic,
                    depth=parameters['crawlDepth'],
                    interval=parameters['interval'],
                    max_retries=parameters['maxRetries'],
                    timeout=parameters['timeout']
                )
                
                completed_topics += 1
                
                # 发送完成爬取的日志
                asyncio.run(broadcast_message({
                    'type': 'log',
                    'message': f'话题 {topic} 爬取完成'
                }))
                
            except Exception as e:
                # 发送错误日志
                asyncio.run(broadcast_message({
                    'type': 'log',
                    'message': f'爬取话题 {topic} 时出错: {str(e)}'
                }))
        
        # 更新最终进度
        asyncio.run(broadcast_message({
            'type': 'progress',
            'value': 100
        }))
        
        # 发送完成消息
        asyncio.run(broadcast_message({
            'type': 'log',
            'message': '所有话题爬取完成'
        }))
        
    except Exception as e:
        # 发送错误日志
        asyncio.run(broadcast_message({
            'type': 'log',
            'message': f'爬虫任务执行出错: {str(e)}'
        }))

@spider_bp.route('/spider/control')
def spider_control():
    """渲染爬虫控制页面"""
    return render_template('spider_control.html')

@spider_bp.route('/api/spider/start', methods=['POST'])
def start_spider():
    """启动爬虫任务"""
    try:
        data = request.get_json()
        topics = data.get('topics', [])
        parameters = data.get('parameters', DEFAULT_CONFIG)
        
        if not topics:
            return jsonify({
                'success': False,
                'message': '请选择至少一个话题'
            })
        
        # 启动爬虫线程
        thread = threading.Thread(
            target=spider_worker,
            args=(topics, parameters),
            daemon=True
        )
        thread.start()
        
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
async def spider_status_socket():
    """WebSocket连接处理"""
    try:
        websocket = websockets.WebSocketServerProtocol()
        websocket_connections.add(websocket)
        
        try:
            while True:
                # 保持连接活跃
                await websocket.ping()
                await asyncio.sleep(30)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            websocket_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket连接处理失败: {e}")

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