"""
Flask主应用 - 统一管理三个Streamlit应用
"""

import os
import sys
import subprocess
import time
import json
import threading
from datetime import datetime
from queue import Queue, Empty
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import signal
import atexit
import requests
import logging
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Dedicated-to-creating-a-concise-and-versatile-public-opinion-analysis-platform'
socketio = SocketIO(app, cors_allowed_origins="*")

# 设置UTF-8编码环境
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 创建日志目录
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

# 初始化ForumEgine的forum.log文件
def init_forum_log():
    """初始化forum.log文件"""
    try:
        forum_log_file = LOG_DIR / "forum.log"
        if not forum_log_file.exists():
            with open(forum_log_file, 'w', encoding='utf-8') as f:
                start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== ForumEgine 系统初始化 - {start_time} ===\n")
            print(f"ForumEgine: forum.log 已初始化")
    except Exception as e:
        print(f"ForumEgine: 初始化forum.log失败: {e}")

# 初始化forum.log
init_forum_log()

# 启动ForumEgine智能监控
def start_forum_engine():
    """启动ForumEgine论坛"""
    try:
        from ForumEgine.monitor import start_forum_monitoring
        print("ForumEgine: 启动论坛...")
        success = start_forum_monitoring()
        if not success:
            print("ForumEgine: 论坛启动失败")
    except Exception as e:
        print(f"ForumEgine: 启动论坛失败: {e}")

# 停止ForumEgine智能监控
def stop_forum_engine():
    """停止ForumEgine论坛"""
    try:
        from ForumEgine.monitor import stop_forum_monitoring
        print("ForumEgine: 停止论坛...")
        stop_forum_monitoring()
        print("ForumEgine: 论坛已停止")
    except Exception as e:
        print(f"ForumEgine: 停止论坛失败: {e}")

# 启动ForumEgine
start_forum_engine()

# 全局变量存储进程信息
processes = {
    'insight': {'process': None, 'port': 8501, 'status': 'stopped', 'output': [], 'log_file': None},
    'media': {'process': None, 'port': 8502, 'status': 'stopped', 'output': [], 'log_file': None},
    'query': {'process': None, 'port': 8503, 'status': 'stopped', 'output': [], 'log_file': None},
    'forum': {'process': None, 'port': None, 'status': 'stopped', 'output': [], 'log_file': None}
}

# 输出队列
output_queues = {
    'insight': Queue(),
    'media': Queue(),
    'query': Queue(),
    'forum': Queue()
}

def write_log_to_file(app_name, line):
    """将日志写入文件"""
    try:
        log_file_path = LOG_DIR / f"{app_name}.log"
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
            f.flush()
    except Exception as e:
        print(f"Error writing log for {app_name}: {e}")

def read_log_from_file(app_name, tail_lines=None):
    """从文件读取日志"""
    try:
        log_file_path = LOG_DIR / f"{app_name}.log"
        if not log_file_path.exists():
            return []
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n\r') for line in lines if line.strip()]
            
            if tail_lines:
                return lines[-tail_lines:]
            return lines
    except Exception as e:
        print(f"Error reading log for {app_name}: {e}")
        return []

def read_process_output(process, app_name):
    """读取进程输出并写入文件"""
    import select
    import sys
    
    while True:
        try:
            if process.poll() is not None:
                # 进程结束，读取剩余输出
                remaining_output = process.stdout.read()
                if remaining_output:
                    lines = remaining_output.decode('utf-8', errors='replace').split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            formatted_line = f"[{timestamp}] {line}"
                            write_log_to_file(app_name, formatted_line)
                            socketio.emit('console_output', {
                                'app': app_name,
                                'line': formatted_line
                            })
                break
            
            # 使用非阻塞读取
            if sys.platform == 'win32':
                # Windows下使用不同的方法
                output = process.stdout.readline()
                if output:
                    line = output.decode('utf-8', errors='replace').strip()
                    if line:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        formatted_line = f"[{timestamp}] {line}"
                        
                        # 写入日志文件
                        write_log_to_file(app_name, formatted_line)
                        
                        # 发送到前端
                        socketio.emit('console_output', {
                            'app': app_name,
                            'line': formatted_line
                        })
                else:
                    # 没有输出时短暂休眠
                    time.sleep(0.1)
            else:
                # Unix系统使用select
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if ready:
                    output = process.stdout.readline()
                    if output:
                        line = output.decode('utf-8', errors='replace').strip()
                        if line:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            formatted_line = f"[{timestamp}] {line}"
                            
                            # 写入日志文件
                            write_log_to_file(app_name, formatted_line)
                            
                            # 发送到前端
                            socketio.emit('console_output', {
                                'app': app_name,
                                'line': formatted_line
                            })
                            
        except Exception as e:
            error_msg = f"Error reading output for {app_name}: {e}"
            print(error_msg)
            write_log_to_file(app_name, f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
            break

def start_streamlit_app(app_name, script_path, port):
    """启动Streamlit应用"""
    try:
        if processes[app_name]['process'] is not None:
            return False, "应用已经在运行"
        
        # 检查文件是否存在
        if not os.path.exists(script_path):
            return False, f"文件不存在: {script_path}"
        
        # 清空之前的日志文件
        log_file_path = LOG_DIR / f"{app_name}.log"
        if log_file_path.exists():
            log_file_path.unlink()
        
        # 创建启动日志
        start_msg = f"[{datetime.now().strftime('%H:%M:%S')}] 启动 {app_name} 应用..."
        write_log_to_file(app_name, start_msg)
        
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            script_path,
            '--server.port', str(port),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            # '--logger.level', 'debug',  # 增加日志详细程度
            '--logger.level', 'info',
            '--server.enableCORS', 'false'
        ]
        
        # 设置环境变量确保UTF-8编码和减少缓冲
        env = os.environ.copy()
        env.update({
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONUTF8': '1',
            'LANG': 'en_US.UTF-8',
            'LC_ALL': 'en_US.UTF-8',
            'PYTHONUNBUFFERED': '1',  # 禁用Python缓冲
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
        })
        
        # 使用当前工作目录而不是脚本目录
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,  # 无缓冲
            universal_newlines=False,
            cwd=os.getcwd(),
            env=env,
            encoding=None,  # 让我们手动处理编码
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        
        processes[app_name]['process'] = process
        processes[app_name]['status'] = 'starting'
        processes[app_name]['output'] = []
        
        # 启动输出读取线程
        output_thread = threading.Thread(
            target=read_process_output,
            args=(process, app_name),
            daemon=True
        )
        output_thread.start()
        
        return True, f"{app_name} 应用启动中..."
        
    except Exception as e:
        error_msg = f"启动失败: {str(e)}"
        write_log_to_file(app_name, f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
        return False, error_msg

def stop_streamlit_app(app_name):
    """停止Streamlit应用"""
    try:
        if processes[app_name]['process'] is None:
            return False, "应用未运行"
        
        process = processes[app_name]['process']
        process.terminate()
        
        # 等待进程结束
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        
        processes[app_name]['process'] = None
        processes[app_name]['status'] = 'stopped'
        
        return True, f"{app_name} 应用已停止"
        
    except Exception as e:
        return False, f"停止失败: {str(e)}"

def check_app_status():
    """检查应用状态"""
    for app_name, info in processes.items():
        if app_name == 'forum':
            # ForumEngine特殊处理 - 检查是否有监控活动
            try:
                from ForumEgine.monitor import is_monitoring_active
                if is_monitoring_active():
                    info['status'] = 'running'
                else:
                    info['status'] = 'stopped'
            except Exception:
                info['status'] = 'stopped'
        else:
            # 普通Streamlit应用的状态检查
            if info['process'] is not None:
                if info['process'].poll() is None:
                    # 进程仍在运行，检查端口是否可访问
                    try:
                        response = requests.get(f"http://localhost:{info['port']}", timeout=2)
                        if response.status_code == 200:
                            info['status'] = 'running'
                        else:
                            info['status'] = 'starting'
                    except requests.exceptions.RequestException:
                        info['status'] = 'starting'
                    except Exception:
                        info['status'] = 'starting'
                else:
                    # 进程已结束
                    info['process'] = None
                    info['status'] = 'stopped'

def wait_for_app_startup(app_name, max_wait_time=30):
    """等待应用启动完成"""
    import time
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        info = processes[app_name]
        if info['process'] is None:
            return False, "进程已停止"
        
        if info['process'].poll() is not None:
            return False, "进程启动失败"
        
        try:
            response = requests.get(f"http://localhost:{info['port']}", timeout=2)
            if response.status_code == 200:
                info['status'] = 'running'
                return True, "启动成功"
        except:
            pass
        
        time.sleep(1)
    
    return False, "启动超时"

def cleanup_processes():
    """清理所有进程"""
    for app_name in processes:
        stop_streamlit_app(app_name)

# 注册清理函数
atexit.register(cleanup_processes)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """获取所有应用状态"""
    check_app_status()
    return jsonify({
        app_name: {
            'status': info['status'],
            'port': info['port'],
            'output_lines': len(info['output'])
        }
        for app_name, info in processes.items()
    })

@app.route('/api/start/<app_name>')
def start_app(app_name):
    """启动指定应用"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': '未知应用'})
    
    script_paths = {
        'insight': 'SingleEngineApp/insight_engine_streamlit_app.py',
        'media': 'SingleEngineApp/media_engine_streamlit_app.py',
        'query': 'SingleEngineApp/query_engine_streamlit_app.py'
    }
    
    success, message = start_streamlit_app(
        app_name, 
        script_paths[app_name], 
        processes[app_name]['port']
    )
    
    
    if success:
        # 等待应用启动
        startup_success, startup_message = wait_for_app_startup(app_name, 15)
        if not startup_success:
            message += f" 但启动检查失败: {startup_message}"
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop/<app_name>')
def stop_app(app_name):
    """停止指定应用"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': '未知应用'})
    
    success, message = stop_streamlit_app(app_name)
    return jsonify({'success': success, 'message': message})

@app.route('/api/output/<app_name>')
def get_output(app_name):
    """获取应用输出"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': '未知应用'})
    
    if app_name == 'forum':
        # ForumEngine特殊处理 - 读取forum.log
        try:
            forum_log_file = LOG_DIR / "forum.log"
            if forum_log_file.exists():
                with open(forum_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    output_lines = [line.rstrip('\n\r') for line in lines if line.strip()]
            else:
                output_lines = ["[系统] forum.log 文件不存在"]
        except Exception as e:
            output_lines = [f"[错误] 读取forum.log失败: {str(e)}"]
    else:
        # 从文件读取完整日志
        output_lines = read_log_from_file(app_name)
    
    return jsonify({
        'success': True,
        'output': output_lines
    })

@app.route('/api/test_log/<app_name>')
def test_log(app_name):
    """测试日志写入功能"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': '未知应用'})
    
    # 写入测试消息
    test_msg = f"[{datetime.now().strftime('%H:%M:%S')}] 测试日志消息 - {datetime.now()}"
    write_log_to_file(app_name, test_msg)
    
    # 通过Socket.IO发送
    socketio.emit('console_output', {
        'app': app_name,
        'line': test_msg
    })
    
    return jsonify({
        'success': True,
        'message': f'测试消息已写入 {app_name} 日志'
    })

@app.route('/api/forum/start')
def start_forum_monitoring_api():
    """手动启动ForumEgine论坛"""
    try:
        from ForumEgine.monitor import start_forum_monitoring
        success = start_forum_monitoring()
        if success:
            return jsonify({'success': True, 'message': 'ForumEgine论坛已启动'})
        else:
            return jsonify({'success': False, 'message': 'ForumEgine论坛启动失败'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'启动论坛失败: {str(e)}'})

@app.route('/api/forum/stop')
def stop_forum_monitoring_api():
    """手动停止ForumEgine论坛"""
    try:
        from ForumEgine.monitor import stop_forum_monitoring
        stop_forum_monitoring()
        return jsonify({'success': True, 'message': 'ForumEgine论坛已停止'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'停止论坛失败: {str(e)}'})

@app.route('/api/forum/log')
def get_forum_log():
    """获取ForumEgine的forum.log内容"""
    try:
        from ForumEgine.monitor import get_forum_log
        log_content = get_forum_log()
        return jsonify({
            'success': True,
            'log_lines': log_content,
            'total_lines': len(log_content)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'读取forum.log失败: {str(e)}'})

@app.route('/api/forum/dialogue')
def get_forum_dialogue():
    """获取ForumEgine的对话格式数据"""
    try:
        forum_log_file = LOG_DIR / "forum.log"
        if not forum_log_file.exists():
            return jsonify({
                'success': True,
                'messages': [],
                'total_messages': 0
            })
        
        messages = []
        with open(forum_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line or line.startswith('==='):
                continue
                
            # 解析日志格式: [时间] [类型] 内容
            if '] [' in line:
                try:
                    # 提取时间
                    time_start = line.find('[') + 1
                    time_end = line.find(']')
                    timestamp = line[time_start:time_end]
                    
                    # 提取类型
                    remaining = line[time_end+1:].strip()
                    if remaining.startswith('[') and '] ' in remaining:
                        type_end = remaining.find('] ')
                        msg_type = remaining[1:type_end]
                        content = remaining[type_end+2:]
                        
                        # 映射消息类型到显示名称
                        type_mapping = {
                            'QUERY': 'Query Agent',
                            'MEDIA': 'Media Agent', 
                            'INSIGHT': 'Insight Agent',
                            'SYSTEM': '系统'
                        }
                        
                        speaker = type_mapping.get(msg_type, msg_type)
                        
                        messages.append({
                            'timestamp': timestamp,
                            'speaker': speaker,
                            'content': content,
                            'type': msg_type.lower()
                        })
                except Exception as e:
                    # 如果解析失败，作为系统消息处理
                    messages.append({
                        'timestamp': '',
                        'speaker': '系统',
                        'content': line,
                        'type': 'system'
                    })
        
        return jsonify({
            'success': True,
            'messages': messages,
            'total_messages': len(messages)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'解析forum.log失败: {str(e)}'})

@app.route('/api/search', methods=['POST'])
def search():
    """统一搜索接口"""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'success': False, 'message': '搜索查询不能为空'})
    
    # ForumEgine论坛已经在后台运行，会自动检测搜索活动
    # print("ForumEgine: 搜索请求已收到，论坛将自动检测日志变化")
    
    # 检查哪些应用正在运行
    check_app_status()
    running_apps = [name for name, info in processes.items() if info['status'] == 'running']
    
    if not running_apps:
        return jsonify({'success': False, 'message': '没有运行中的应用'})
    
    # 向运行中的应用发送搜索请求
    results = {}
    api_ports = {'insight': 8601, 'media': 8602, 'query': 8603}
    
    for app_name in running_apps:
        try:
            api_port = api_ports[app_name]
            # 调用Streamlit应用的API端点
            response = requests.post(
                f"http://localhost:{api_port}/api/search",
                json={'query': query},
                timeout=10
            )
            if response.status_code == 200:
                results[app_name] = response.json()
            else:
                results[app_name] = {'success': False, 'message': 'API调用失败'}
        except Exception as e:
            results[app_name] = {'success': False, 'message': str(e)}
    
    # 搜索完成后可以选择停止监控，或者让它继续运行以捕获后续的处理日志
    # 这里我们让监控继续运行，用户可以通过其他接口手动停止
    
    return jsonify({
        'success': True,
        'query': query,
        'results': results
    })

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    emit('status', 'Connected to Flask server')

@socketio.on('request_status')
def handle_status_request():
    """请求状态更新"""
    check_app_status()
    emit('status_update', {
        app_name: {
            'status': info['status'],
            'port': info['port']
        }
        for app_name, info in processes.items()
    })

if __name__ == '__main__':
    # 启动时自动启动所有Streamlit应用
    print("正在启动Streamlit应用...")
    
    # 先停止ForumEgine监控器，避免文件占用冲突
    print("停止ForumEgine监控器以避免文件冲突...")
    stop_forum_engine()
    
    script_paths = {
        'insight': 'SingleEngineApp/insight_engine_streamlit_app.py',
        'media': 'SingleEngineApp/media_engine_streamlit_app.py',
        'query': 'SingleEngineApp/query_engine_streamlit_app.py'
    }
    
    for app_name, script_path in script_paths.items():
        print(f"检查文件: {script_path}")
        if os.path.exists(script_path):
            print(f"启动 {app_name}...")
            success, message = start_streamlit_app(app_name, script_path, processes[app_name]['port'])
            print(f"{app_name}: {message}")
            
            if success:
                print(f"等待 {app_name} 启动完成...")
                startup_success, startup_message = wait_for_app_startup(app_name, 30)
                print(f"{app_name} 启动检查: {startup_message}")
        else:
            print(f"错误: {script_path} 不存在")
    
    start_forum_engine()
    
    print("启动Flask服务器...")
    
    try:
        # 启动Flask应用
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n正在关闭应用...")
        cleanup_processes()
