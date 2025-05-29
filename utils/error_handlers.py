from flask import render_template
from utils.logger import app_logger as logging

def register_error_handlers(app):
    """注册错误处理器"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        logging.warning(f"404错误: {request.url}")
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        logging.error(f"500错误: {error}")
        return render_template('error.html', 
                             error_code=500, 
                             error_title='服务器错误', 
                             error_message='服务器遇到了一个问题，请稍后再试。',
                             error_i18n_key='serverError'), 500

    @app.errorhandler(403)
    def forbidden_error(error):
        logging.warning(f"403错误: {request.url}")
        return render_template('error.html', 
                             error_code=403, 
                             error_title='禁止访问', 
                             error_message='您没有权限访问此页面。',
                             error_i18n_key='forbidden'), 403

    @app.errorhandler(400)
    def bad_request_error(error):
        logging.warning(f"400错误: {error}")
        return render_template('error.html', 
                             error_code=400, 
                             error_title='错误请求', 
                             error_message='服务器无法理解您的请求。',
                             error_i18n_key='badRequest'), 400

    @app.errorhandler(Exception)
    def handle_exception(error):
        logging.error(f"未处理的异常: {error}")
        return render_template('error.html',
                             error_code=500,
                             error_title='系统错误',
                             error_message='系统发生了一个未预期的错误。',
                             error_i18n_key='unexpectedError'), 500 