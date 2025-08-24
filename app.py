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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'weibo_analysis_system_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量存储进程信息
processes = {
    'insight': {'process': None, 'port': 8501, 'status': 'stopped', 'output': []},
    'media': {'process': None, 'port': 8502, 'status': 'stopped', 'output': []},
    'query': {'process': None, 'port': 8503, 'status': 'stopped', 'output': []}
}

# 输出队列
output_queues = {
    'insight': Queue(),
    'media': Queue(),
    'query': Queue()
}

def read_process_output(process, app_name):
    """读取进程输出并放入队列"""
    while True:
        try:
            if process.poll() is not None:
                break
            
            output = process.stdout.readline()
            if output:
                line = output.decode('utf-8', errors='ignore').strip()
                if line:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    formatted_line = f"[{timestamp}] {line}"
                    
                    # 添加到输出列表（保持最近100行）
                    processes[app_name]['output'].append(formatted_line)
                    if len(processes[app_name]['output']) > 100:
                        processes[app_name]['output'].pop(0)
                    
                    # 发送到前端
                    socketio.emit('console_output', {
                        'app': app_name,
                        'line': formatted_line
                    })
        except Exception as e:
            print(f"Error reading output for {app_name}: {e}")
            break

def start_streamlit_app(app_name, script_path, port):
    """启动Streamlit应用"""
    try:
        if processes[app_name]['process'] is not None:
            return False, "应用已经在运行"
        
        # 检查文件是否存在
        if not os.path.exists(script_path):
            return False, f"文件不存在: {script_path}"
        
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            script_path,
            '--server.port', str(port),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            '--logger.level', 'info'
        ]
        
        # 使用当前工作目录而不是脚本目录
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False,
            cwd=os.getcwd()
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
        return False, f"启动失败: {str(e)}"

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
    
    return jsonify({
        'success': True,
        'output': processes[app_name]['output']
    })

@app.route('/api/search', methods=['POST'])
def search():
    """统一搜索接口"""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'success': False, 'message': '搜索查询不能为空'})
    
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
    
    print("所有应用启动完成，启动Flask服务器...")
    
    try:
        # 启动Flask应用
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n正在关闭应用...")
        cleanup_processes()
