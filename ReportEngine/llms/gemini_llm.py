"""
Report Engine Gemini LLM实现
使用Gemini 2.5-pro中转API进行文本生成
"""

import os
import sys
from typing import Optional, Dict, Any
from openai import OpenAI
from .base import BaseLLM

DEFAULT_GEMINI_BASE_URL = "https://www.chataiapi.com/v1"

# 导入根目录的config
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    import config
except ImportError:
    config = None

# 添加utils目录到Python路径并导入重试模块
try:
    if root_dir:
        utils_dir = os.path.join(root_dir, 'utils')
        if utils_dir not in sys.path:
            sys.path.append(utils_dir)
    from retry_helper import with_retry, with_graceful_retry, LLM_RETRY_CONFIG, RetryConfig
    # 创建动态重试配置生成函数
    def create_report_retry_config(config=None):
        """创建ReportEngine专用的重试配置，适应7分钟平均生成时间"""
        return RetryConfig(
            max_retries=config.max_retries if config and hasattr(config, 'max_retries') else 8,
            initial_delay=8.0,      # 初始延迟增加到8秒，适应长时间生成
            backoff_factor=2.0,     # 保持2倍退避
            max_delay=config.max_retry_delay if config and hasattr(config, 'max_retry_delay') else 180.0
        )
    # 创建默认配置用于模块导入时的向后兼容
    REPORT_LLM_RETRY_CONFIG = create_report_retry_config()
except ImportError:
    # 如果无法导入重试模块，使用空装饰器避免报错
    def with_retry(config):
        def decorator(func):
            return func
        return decorator
    LLM_RETRY_CONFIG = None
    REPORT_LLM_RETRY_CONFIG = None


class GeminiLLM(BaseLLM):
    """Report Engine Gemini LLM实现类"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, base_url: Optional[str] = None, config=None):
        """
        初始化Gemini客户端
        
        Args:
            api_key: Gemini API密钥，如果不提供则从config或环境变量读取
            model_name: 模型名称，默认使用gemini-2.5-pro
            base_url: Gemini API基础地址
            config: 配置对象，用于获取超时设置
        """
        if api_key is None:
            # 优先从根目录config读取
            if config and hasattr(config, 'GEMINI_API_KEY'):
                api_key = config.GEMINI_API_KEY
            else:
                # 备选方案：从环境变量读取
                api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                raise ValueError("Gemini API Key未找到！请在config.py中设置GEMINI_API_KEY或设置环境变量")
        
        super().__init__(api_key, model_name)
        
        # 存储配置对象
        self.config = config
        
        # 从配置获取超时时间，默认15分钟（适应7分钟平均生成时间）
        timeout = config.api_timeout if config and hasattr(config, 'api_timeout') else 900.0

        self.base_url = (
            base_url
            or (getattr(self.config, 'gemini_base_url', None) if self.config else None)
            or os.getenv('GEMINI_BASE_URL')
            or DEFAULT_GEMINI_BASE_URL
        )

        # 创建针对此实例的重试配置
        self.retry_config = create_report_retry_config(config)

        # 初始化OpenAI客户端，使用Gemini的中转endpoint
        # 专门为报告生成设置长超时（15分钟），适应7分钟平均生成时间
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout
        )
        
        self.default_model = model_name or self.get_default_model()
    
    def get_default_model(self) -> str:
        """获取默认模型名称"""
        return "gemini-2.5-pro"
    
    def _make_api_call(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        内部API调用方法
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            **kwargs: 其他参数
            
        Returns:
            API响应内容
        """
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 设置默认参数
        params = {
            "model": self.default_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 50000),
            "stream": False
        }
        
        # 调用API
        response = self.client.chat.completions.create(**params)
        
        # 提取回复内容
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            return self.validate_response(content)
        else:
            return ""

    def invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        调用Gemini API生成回复（带动态重试配置）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            **kwargs: 其他参数，如temperature、max_tokens等
            
        Returns:
            Gemini生成的回复文本
        """
        import time
        
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = self._make_api_call(system_prompt, user_prompt, **kwargs)
                if attempt > 0:
                    print(f"Report Engine Gemini API在第 {attempt + 1} 次尝试后成功")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.retry_config.max_retries:
                    print(f"Report Engine Gemini API在 {self.retry_config.max_retries + 1} 次尝试后仍然失败")
                    print(f"最终错误: {str(e)}")
                    raise e
                
                # 计算延迟时间
                delay = min(
                    self.retry_config.initial_delay * (self.retry_config.backoff_factor ** attempt),
                    self.retry_config.max_delay
                )
                
                print(f"Report Engine Gemini API第 {attempt + 1} 次尝试失败: {str(e)}")
                print(f"将在 {delay:.1f} 秒后进行第 {attempt + 2} 次尝试...")
                
                time.sleep(delay)
        
        # 这里不应该到达，但作为安全网
        if last_exception:
            raise last_exception
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "provider": "Gemini",
            "model": self.default_model,
            "api_base": self.base_url,
            "purpose": "Report Generation"
        }
