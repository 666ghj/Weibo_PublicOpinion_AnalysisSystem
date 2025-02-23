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