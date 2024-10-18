from flask import Flask, render_template, request

import json
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/'  # 上传文件的保存目录
ALLOWED_EXTENSIONS = {'csv'}  # 允许的文件扩展名
processing_status = {}  # 全局字典用于存储处理状态和统计信息

@app.route('/')
def upload_form():
    """显示文件上传表单"""
    return render_template('main.html')

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