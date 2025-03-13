import os
import sys
import json
import getpass
import secrets
import logging
import platform
import socket
import hashlib
import base64
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import pymysql
from dotenv import load_dotenv, set_key, find_dotenv

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('init_wizard')

class InitWizard:
    """
    初始化向导 - 简化系统的初始配置流程，并提供安全加固功能
    """
    
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        
        # 配置项
        self.config = {
            # 数据库配置
            'db': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '3306')),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', ''),
                'database': os.getenv('DB_NAME', 'Weibo_PublicOpinion_AnalysisSystem'),
                'ssl': bool(os.getenv('DB_SSL', 'false').lower() == 'true')
            },
            # Flask应用配置
            'app': {
                'host': os.getenv('FLASK_HOST', '127.0.0.1'),
                'port': int(os.getenv('FLASK_PORT', '5000')),
                'secret_key': os.getenv('FLASK_SECRET_KEY', ''),
                'enable_https': bool(os.getenv('ENABLE_HTTPS', 'false').lower() == 'true'),
                'debug': bool(os.getenv('FLASK_DEBUG', 'false').lower() == 'true')
            },
            # API密钥配置
            'api_keys': {
                'openai': os.getenv('OPENAI_API_KEY', ''),
                'anthropic': os.getenv('ANTHROPIC_API_KEY', ''),
                'deepseek': os.getenv('DEEPSEEK_API_KEY', '')
            },
            # 安全配置
            'security': {
                'enable_rate_limit': bool(os.getenv('ENABLE_RATE_LIMIT', 'true').lower() == 'true'),
                'enable_ip_blocking': bool(os.getenv('ENABLE_IP_BLOCKING', 'true').lower() == 'true'),
                'enable_sensitive_data_filter': bool(os.getenv('ENABLE_SENSITIVE_DATA_FILTER', 'true').lower() == 'true'),
                'enable_mutual_auth': bool(os.getenv('ENABLE_MUTUAL_AUTH', 'false').lower() == 'true'),
                'min_password_length': int(os.getenv('MIN_PASSWORD_LENGTH', '8')),
                'session_timeout': int(os.getenv('SESSION_TIMEOUT', '120')),  # 分钟
            },
            # 爬虫配置
            'crawler': {
                'interval': int(os.getenv('CRAWL_INTERVAL', '18000')),  # 秒
                'max_retries': int(os.getenv('CRAWL_MAX_RETRIES', '3')),
                'timeout': int(os.getenv('CRAWL_TIMEOUT', '30')),
                'max_concurrent': int(os.getenv('CRAWL_MAX_CONCURRENT', '2')),
                'user_agent': os.getenv('CRAWL_USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            },
            # 系统配置
            'system': {
                'initialized': bool(os.getenv('SYSTEM_INITIALIZED', 'false').lower() == 'true'),
                'version': os.getenv('SYSTEM_VERSION', '2.0.0'),
                'log_level': os.getenv('LOG_LEVEL', 'INFO'),
                'data_dir': os.getenv('DATA_DIR', 'data'),
                'temp_dir': os.getenv('TEMP_DIR', 'temp'),
                'cache_dir': os.getenv('CACHE_DIR', 'cache'),
                'max_model_memory': float(os.getenv('MAX_MODEL_MEMORY_USAGE', '4.0')),  # GB
            }
        }
        
        # 安全选项
        self.security_options = {
            'rate_limit': {
                'name': '请求速率限制',
                'description': '防止API被滥用，限制单个IP的请求频率',
                'default': True
            },
            'ip_blocking': {
                'name': 'IP黑名单',
                'description': '阻止可疑IP访问系统',
                'default': True
            },
            'sensitive_data_filter': {
                'name': '敏感信息过滤',
                'description': '自动识别并屏蔽输出内容中的敏感信息（如手机号、邮箱等）',
                'default': True
            },
            'mutual_auth': {
                'name': '双向认证',
                'description': '要求API调用方提供有效证书，增强API安全性（需要HTTPS）',
                'default': False
            }
        }

    def start(self):
        """启动初始化向导"""
        self._print_welcome()
        
        if self.config['system']['initialized']:
            print("\n系统已经初始化过。您想重新配置吗? [y/N]: ", end='')
            choice = input().strip().lower()
            if choice != 'y':
                print("初始化向导已退出。如需重新配置，请设置环境变量 SYSTEM_INITIALIZED=false 或删除 .env 文件。")
                return
        
        # 主配置流程
        try:
            self._configure_database()
            self._configure_app()
            self._configure_api_keys()
            self._configure_security()
            self._configure_crawler()
            self._configure_system()
            
            # 保存配置
            self._save_config()
            
            # 应用安全措施
            self._apply_security_measures()
            
            print("\n✅ 初始化完成！系统已成功配置。")
            print("您现在可以运行 python app.py 启动应用。")
            
        except KeyboardInterrupt:
            print("\n\n初始化向导已取消。配置未保存。")
        except Exception as e:
            logger.error(f"初始化过程中发生错误: {e}")
            print(f"\n❌ 初始化失败: {e}")
            print("请检查错误并重试。")

    def _print_welcome(self):
        """打印欢迎信息"""
        print("\n" + "="*80)
        print(" "*20 + "微博舆情分析预测系统 - 初始化向导 v2.0")
        print("="*80)
        print("\n欢迎使用微博舆情分析预测系统！此向导将引导您完成系统的初始配置。")
        print("按Ctrl+C可随时退出向导。")
        print("\n系统信息:")
        print(f"  • 操作系统: {platform.system()} {platform.release()}")
        print(f"  • Python版本: {platform.python_version()}")
        print(f"  • 主机名: {socket.gethostname()}")
        print(f"  • 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n让我们开始配置吧！每个选项都有默认值，直接按回车即可使用默认值。")
        print("-"*80)

    def _configure_database(self):
        """配置数据库连接"""
        print("\n📦 数据库配置")
        print("-"*50)
        
        # 询问数据库连接信息
        self.config['db']['host'] = self._prompt(
            "数据库主机", self.config['db']['host'])
        
        port_str = self._prompt(
            "数据库端口", str(self.config['db']['port']))
        try:
            self.config['db']['port'] = int(port_str)
        except ValueError:
            print(f"端口号无效，使用默认值 {self.config['db']['port']}")
        
        self.config['db']['user'] = self._prompt(
            "数据库用户名", self.config['db']['user'])
        
        # 密码使用getpass以避免明文显示
        default_pass = '*' * len(self.config['db']['password']) if self.config['db']['password'] else ''
        password = getpass.getpass(f"数据库密码 [{default_pass}]: ")
        if password:
            self.config['db']['password'] = password
        
        self.config['db']['database'] = self._prompt(
            "数据库名", self.config['db']['database'])
        
        ssl_str = self._prompt(
            "使用SSL连接 (true/false)", str(self.config['db']['ssl']).lower())
        self.config['db']['ssl'] = ssl_str.lower() == 'true'
        
        # 测试数据库连接
        print("\n正在测试数据库连接...")
        try:
            self._test_db_connection()
            print("✅ 数据库连接成功！")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            retry = input("是否重新配置数据库连接? [Y/n]: ").strip().lower()
            if retry != 'n':
                return self._configure_database()
            else:
                print("跳过数据库连接测试，但配置可能不正确。")

    def _configure_app(self):
        """配置Flask应用"""
        print("\n🚀 应用配置")
        print("-"*50)
        
        self.config['app']['host'] = self._prompt(
            "监听地址 (0.0.0.0表示所有网络接口)", self.config['app']['host'])
        
        port_str = self._prompt(
            "监听端口", str(self.config['app']['port']))
        try:
            self.config['app']['port'] = int(port_str)
        except ValueError:
            print(f"端口号无效，使用默认值 {self.config['app']['port']}")
        
        # 自动生成密钥
        if not self.config['app']['secret_key']:
            self.config['app']['secret_key'] = secrets.token_hex(32)
            print(f"已自动生成应用密钥: {self.config['app']['secret_key'][:8]}...")
        else:
            regenerate = input("应用密钥已存在。是否重新生成? [y/N]: ").strip().lower()
            if regenerate == 'y':
                self.config['app']['secret_key'] = secrets.token_hex(32)
                print(f"已重新生成应用密钥: {self.config['app']['secret_key'][:8]}...")
        
        https_str = self._prompt(
            "启用HTTPS (true/false)", str(self.config['app']['enable_https']).lower())
        self.config['app']['enable_https'] = https_str.lower() == 'true'
        
        debug_str = self._prompt(
            "启用调试模式 (true/false, 生产环境建议false)", str(self.config['app']['debug']).lower())
        self.config['app']['debug'] = debug_str.lower() == 'true'

    def _configure_api_keys(self):
        """配置API密钥"""
        print("\n🔑 API密钥配置")
        print("-"*50)
        print("系统支持多个大语言模型，至少需要配置一个API密钥。")
        
        # 配置OpenAI API密钥
        has_openai = self._prompt(
            "是否配置OpenAI API密钥? (y/n)", "y" if self.config['api_keys']['openai'] else "n")
        if has_openai.lower() == 'y':
            self.config['api_keys']['openai'] = self._prompt(
                "OpenAI API密钥", self.config['api_keys']['openai'])
            
        # 配置Anthropic API密钥
        has_anthropic = self._prompt(
            "是否配置Anthropic (Claude) API密钥? (y/n)", "y" if self.config['api_keys']['anthropic'] else "n")
        if has_anthropic.lower() == 'y':
            self.config['api_keys']['anthropic'] = self._prompt(
                "Anthropic API密钥", self.config['api_keys']['anthropic'])
            
        # 配置DeepSeek API密钥
        has_deepseek = self._prompt(
            "是否配置DeepSeek API密钥? (y/n)", "y" if self.config['api_keys']['deepseek'] else "n")
        if has_deepseek.lower() == 'y':
            self.config['api_keys']['deepseek'] = self._prompt(
                "DeepSeek API密钥", self.config['api_keys']['deepseek'])
        
        # 检查是否至少配置了一个API密钥
        if not (self.config['api_keys']['openai'] or self.config['api_keys']['anthropic'] or self.config['api_keys']['deepseek']):
            print("⚠️ 警告: 您未配置任何API密钥，系统的AI分析功能将不可用。")
            confirm = input("是否继续? [Y/n]: ").strip().lower()
            if confirm == 'n':
                return self._configure_api_keys()

    def _configure_security(self):
        """配置安全设置"""
        print("\n🔒 安全配置")
        print("-"*50)
        
        for key, option in self.security_options.items():
            current_value = self.config['security'][f'enable_{key}']
            print(f"\n{option['name']}: {option['description']}")
            enable_str = self._prompt(
                f"启用{option['name']} (true/false)", str(current_value).lower())
            self.config['security'][f'enable_{key}'] = enable_str.lower() == 'true'
        
        # 密码安全策略
        min_len_str = self._prompt(
            "最小密码长度 (推荐不低于8)", str(self.config['security']['min_password_length']))
        try:
            self.config['security']['min_password_length'] = int(min_len_str)
            if self.config['security']['min_password_length'] < 6:
                print("⚠️ 警告: 短密码容易被暴力破解，建议设置更长的密码。")
        except ValueError:
            print(f"无效输入，使用默认值 {self.config['security']['min_password_length']}")
        
        # 会话超时设置
        timeout_str = self._prompt(
            "会话超时时间 (分钟)", str(self.config['security']['session_timeout']))
        try:
            self.config['security']['session_timeout'] = int(timeout_str)
        except ValueError:
            print(f"无效输入，使用默认值 {self.config['security']['session_timeout']}")

    def _configure_crawler(self):
        """配置爬虫设置"""
        print("\n🕷️ 爬虫配置")
        print("-"*50)
        
        interval_str = self._prompt(
            "爬取间隔 (秒)", str(self.config['crawler']['interval']))
        try:
            self.config['crawler']['interval'] = int(interval_str)
        except ValueError:
            print(f"无效输入，使用默认值 {self.config['crawler']['interval']}")
        
        retries_str = self._prompt(
            "最大重试次数", str(self.config['crawler']['max_retries']))
        try:
            self.config['crawler']['max_retries'] = int(retries_str)
        except ValueError:
            print(f"无效输入，使用默认值 {self.config['crawler']['max_retries']}")
        
        timeout_str = self._prompt(
            "超时时间 (秒)", str(self.config['crawler']['timeout']))
        try:
            self.config['crawler']['timeout'] = int(timeout_str)
        except ValueError:
            print(f"无效输入，使用默认值 {self.config['crawler']['timeout']}")
        
        concurrent_str = self._prompt(
            "最大并发数", str(self.config['crawler']['max_concurrent']))
        try:
            self.config['crawler']['max_concurrent'] = int(concurrent_str)
        except ValueError:
            print(f"无效输入，使用默认值 {self.config['crawler']['max_concurrent']}")
        
        self.config['crawler']['user_agent'] = self._prompt(
            "User-Agent", self.config['crawler']['user_agent'])

    def _configure_system(self):
        """配置系统设置"""
        print("\n⚙️ 系统配置")
        print("-"*50)
        
        # 日志级别
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        current_level = self.config['system']['log_level']
        print(f"可选日志级别: {', '.join(log_levels)}")
        log_level = self._prompt("日志级别", current_level).upper()
        if log_level in log_levels:
            self.config['system']['log_level'] = log_level
        else:
            print(f"无效的日志级别，使用默认值 {current_level}")
        
        # 数据目录
        data_dir = self._prompt("数据目录", self.config['system']['data_dir'])
        if data_dir:
            self.config['system']['data_dir'] = data_dir
            os.makedirs(data_dir, exist_ok=True)
            print(f"已创建数据目录: {data_dir}")
        
        # 缓存目录
        cache_dir = self._prompt("缓存目录", self.config['system']['cache_dir'])
        if cache_dir:
            self.config['system']['cache_dir'] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            print(f"已创建缓存目录: {cache_dir}")
        
        # 临时目录
        temp_dir = self._prompt("临时文件目录", self.config['system']['temp_dir'])
        if temp_dir:
            self.config['system']['temp_dir'] = temp_dir
            os.makedirs(temp_dir, exist_ok=True)
            print(f"已创建临时文件目录: {temp_dir}")
        
        # 模型内存限制
        memory_str = self._prompt(
            "最大模型内存使用量 (GB)", str(self.config['system']['max_model_memory']))
        try:
            self.config['system']['max_model_memory'] = float(memory_str)
        except ValueError:
            print(f"无效输入，使用默认值 {self.config['system']['max_model_memory']}")
        
        # 标记系统已初始化
        self.config['system']['initialized'] = True

    def _save_config(self):
        """保存配置到.env文件"""
        print("\n正在保存配置...")
        
        # 构建.env文件内容
        env_content = [
            "# 微博舆情分析预测系统配置文件",
            f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "# 数据库配置",
            f"DB_HOST={self.config['db']['host']}",
            f"DB_PORT={self.config['db']['port']}",
            f"DB_USER={self.config['db']['user']}",
            f"DB_PASSWORD={self.config['db']['password']}",
            f"DB_NAME={self.config['db']['database']}",
            f"DB_SSL={str(self.config['db']['ssl']).lower()}",
            "",
            "# 应用配置",
            f"FLASK_HOST={self.config['app']['host']}",
            f"FLASK_PORT={self.config['app']['port']}",
            f"FLASK_SECRET_KEY={self.config['app']['secret_key']}",
            f"ENABLE_HTTPS={str(self.config['app']['enable_https']).lower()}",
            f"FLASK_DEBUG={str(self.config['app']['debug']).lower()}",
            "",
            "# API密钥",
            f"OPENAI_API_KEY={self.config['api_keys']['openai']}",
            f"ANTHROPIC_API_KEY={self.config['api_keys']['anthropic']}",
            f"DEEPSEEK_API_KEY={self.config['api_keys']['deepseek']}",
            "",
            "# 安全配置",
            f"ENABLE_RATE_LIMIT={str(self.config['security']['enable_rate_limit']).lower()}",
            f"ENABLE_IP_BLOCKING={str(self.config['security']['enable_ip_blocking']).lower()}",
            f"ENABLE_SENSITIVE_DATA_FILTER={str(self.config['security']['enable_sensitive_data_filter']).lower()}",
            f"ENABLE_MUTUAL_AUTH={str(self.config['security']['enable_mutual_auth']).lower()}",
            f"MIN_PASSWORD_LENGTH={self.config['security']['min_password_length']}",
            f"SESSION_TIMEOUT={self.config['security']['session_timeout']}",
            "",
            "# 爬虫配置",
            f"CRAWL_INTERVAL={self.config['crawler']['interval']}",
            f"CRAWL_MAX_RETRIES={self.config['crawler']['max_retries']}",
            f"CRAWL_TIMEOUT={self.config['crawler']['timeout']}",
            f"CRAWL_MAX_CONCURRENT={self.config['crawler']['max_concurrent']}",
            f"CRAWL_USER_AGENT={self.config['crawler']['user_agent']}",
            "",
            "# 系统配置",
            f"SYSTEM_INITIALIZED={str(self.config['system']['initialized']).lower()}",
            f"SYSTEM_VERSION={self.config['system']['version']}",
            f"LOG_LEVEL={self.config['system']['log_level']}",
            f"DATA_DIR={self.config['system']['data_dir']}",
            f"TEMP_DIR={self.config['system']['temp_dir']}",
            f"CACHE_DIR={self.config['system']['cache_dir']}",
            f"MAX_MODEL_MEMORY_USAGE={self.config['system']['max_model_memory']}",
        ]
        
        # 写入.env文件
        with open('.env', 'w') as f:
            f.write('\n'.join(env_content))
        
        print("✅ 配置已保存到 .env 文件")
        
        # 创建备份
        backup_path = f".env.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2('.env', backup_path)
        print(f"✅ 配置备份已保存到 {backup_path}")

    def _test_db_connection(self):
        """测试数据库连接"""
        connection = pymysql.connect(
            host=self.config['db']['host'],
            port=self.config['db']['port'],
            user=self.config['db']['user'],
            password=self.config['db']['password'],
            database=self.config['db']['database'],
            charset='utf8mb4',
            ssl={'ssl': {'ca': None}} if self.config['db']['ssl'] else None
        )
        connection.close()

    def _apply_security_measures(self):
        """应用安全措施"""
        print("\n正在应用安全措施...")
        
        # 创建相关目录
        security_dir = os.path.join(self.config['system']['data_dir'], 'security')
        os.makedirs(security_dir, exist_ok=True)
        
        # 设置文件权限
        try:
            # 仅在类Unix系统上设置文件权限
            if platform.system() != "Windows":
                os.chmod('.env', 0o600)  # 只有所有者可读写
                print("✅ 已设置.env文件权限为600 (只有所有者可读写)")
        except Exception as e:
            logger.warning(f"设置文件权限失败: {e}")
        
        # 生成密钥对（如果启用了双向认证）
        if self.config['security']['enable_mutual_auth']:
            cert_dir = os.path.join(security_dir, 'certs')
            os.makedirs(cert_dir, exist_ok=True)
            
            try:
                # 检查是否有OpenSSL可用
                subprocess.run(['openssl', 'version'], check=True, capture_output=True)
                
                # 生成自签名证书
                key_file = os.path.join(cert_dir, 'server.key')
                cert_file = os.path.join(cert_dir, 'server.crt')
                
                if not os.path.exists(key_file) or not os.path.exists(cert_file):
                    print("正在生成SSL证书...")
                    subprocess.run([
                        'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
                        '-keyout', key_file, '-out', cert_file,
                        '-days', '365', '-nodes',
                        '-subj', '/CN=localhost'
                    ], check=True)
                    print(f"✅ SSL证书已生成: {cert_file}")
            except subprocess.CalledProcessError:
                print("⚠️ OpenSSL不可用，无法生成SSL证书。如需使用HTTPS，请手动配置证书。")
            except Exception as e:
                logger.warning(f"生成SSL证书失败: {e}")
        
        # 创建敏感信息过滤器配置
        if self.config['security']['enable_sensitive_data_filter']:
            filter_config = {
                'enabled': True,
                'patterns': {
                    'phone': r'\b1[3-9]\d{9}\b',
                    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    'id_card': r'\b[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b',
                    'credit_card': r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b',
                    'address': r'(北京|上海|广州|深圳|天津|重庆|南京|杭州|武汉|成都|西安)市.*?(路|街|道|巷).*?(号)'
                },
                'replacements': {
                    'phone': '***********',
                    'email': '******@*****',
                    'id_card': '******************',
                    'credit_card': '****************',
                    'address': '[地址已隐藏]'
                }
            }
            
            filter_path = os.path.join(security_dir, 'sensitive_filter.json')
            with open(filter_path, 'w', encoding='utf-8') as f:
                json.dump(filter_config, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 敏感信息过滤器配置已保存到 {filter_path}")
        
        # 创建IP黑名单文件
        if self.config['security']['enable_ip_blocking']:
            blacklist_path = os.path.join(security_dir, 'ip_blacklist.txt')
            if not os.path.exists(blacklist_path):
                with open(blacklist_path, 'w') as f:
                    f.write("# 每行一个IP地址\n")
                print(f"✅ IP黑名单文件已创建: {blacklist_path}")

    def _prompt(self, prompt, default=""):
        """提示用户输入，如果用户直接按回车则返回默认值"""
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
        else:
            user_input = input(f"{prompt}: ").strip()
        
        return user_input if user_input else default


if __name__ == "__main__":
    wizard = InitWizard()
    wizard.start() 