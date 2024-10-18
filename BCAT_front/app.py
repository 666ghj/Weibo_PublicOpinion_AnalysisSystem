import os
import subprocess
import threading
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/'  # 上传文件的保存目录
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 文件大小限制为16MB
app.secret_key = 'secret_key'  # 用于Flash消息的密钥
ALLOWED_EXTENSIONS = {'csv'}  # 允许的文件扩展名
processing_status = {}  # 全局字典用于存储处理状态和统计信息

def allowed_file(filename):
    """检查文件是否是允许的类型"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    """显示文件上传表单"""
    return render_template('main.html')

@app.route('/status/<filename>')
def check_status(filename):
    """检查文件处理状态，并返回状态和统计信息"""
    status_info = processing_status.get(filename, {'status': 'processing', 'stats': None})
    return json.dumps(status_info)

@app.route('/waiting/<filename>')
def waiting_page(filename):
    """显示等待页面，并传递文件名"""
    return render_template('waiting.html', filename=filename)

@app.route('/upload-success')
def upload_success():
    """文件处理成功页面"""
    filename = request.args.get('filename')
    stats = processing_status.get(filename, {}).get('stats', {})
    return render_template('success.html', stats=stats)

@app.route('/upload-failure')
def upload_failure():
    """文件处理失败页面"""
    filename = request.args.get('filename')
    stats = processing_status.get(filename, {}).get('stats', {})
    return render_template('failure.html', stats=stats)

def handle_file_processing(filepath, filename):
    """异步处理文件并根据统计结果设置处理状态"""
    try:
        script_path = r'E:\ICTfront\BCAT\using_example.py'  # 请根据实际路径更新
        stats_output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'stats_{filename}.json')

        # 执行外部脚本，传递文件路径和统计信息文件路径作为参数
        result = subprocess.run(
            ['python', script_path, filepath, stats_output_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        print(f"脚本标准输出: {result.stdout}")
        print(f"脚本标准错误: {result.stderr}")

        if result.returncode == 0:
            # 读取统计信息
            with open(stats_output_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)

            # 获取“不良”标签的占比
            bad_percentage = float(stats.get("不良", {}).get("percentage", "0%").strip('%'))

            if bad_percentage > 5.0:
                # 失败占比超过5%，标记为失败
                processing_status[filename] = {
                    'status': 'failure',
                    'stats': stats
                }
            else:
                # 成功
                processing_status[filename] = {
                    'status': 'success',
                    'stats': stats
                }
        else:
            # 脚本执行失败
            processing_status[filename] = {
                'status': 'failure',
                'stats': None
            }
    except Exception as e:
        print(f"运行脚本时出错: {str(e)}")
        processing_status[filename] = {
            'status': 'failure',
            'stats': None
        }

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和启动异步处理"""
    if 'file' not in request.files:
        flash('没有文件部分', 'error')
        return redirect(url_for('upload_form'))

    file = request.files['file']

    if file.filename == '':
        flash('未选择文件', 'error')
        return redirect(url_for('upload_form'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        filepath = os.path.abspath(filepath)  # 转换为绝对路径

        try:
            file.save(filepath)
            print(f'文件已保存到 {filepath}')

            # 初始化处理状态
            processing_status[filename] = {'status': 'processing', 'stats': None}

            # 启动后台线程处理文件
            thread = threading.Thread(target=handle_file_processing, args=(filepath, filename))
            thread.start()

            # 重定向到等待页面，并传递文件名以跟踪状态
            return redirect(url_for('waiting_page', filename=filename))
        except Exception as e:
            flash(f'文件上传失败: {str(e)}', 'error')
            return redirect(url_for('upload_failure'))
    else:
        flash('文件类型不允许', 'error')
        return redirect(url_for('upload_form'))