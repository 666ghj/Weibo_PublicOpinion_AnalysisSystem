from flask import render_template

@app.route('/')
def upload_form():
    """显示文件上传表单"""
    return render_template('main.html')