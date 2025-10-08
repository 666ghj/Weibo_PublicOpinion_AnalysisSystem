"""
Gemini LLM实现
使用Gemini 2.5-pro中转API进行文本生成
"""

import os
import sys
from typing import Optional, Dict, Any
from openai import OpenAI
from .base import BaseLLM

DEFAULT_GEMINI_BASE_URL = "https://www.chataiapi.com/v1"

# 添加utils目录到Python路径并导入重试模块
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    utils_dir = os.path.join(root_dir, 'utils')
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    from retry_helper import with_retry, with_graceful_retry, LLM_RETRY_CONFIG
except ImportError:
    # 如果无法导入重试模块，使用空装饰器避免报错
    def with_retry(config):
        def decorator(func):
            return func
        return decorator
    LLM_RETRY_CONFIG = None


class GeminiLLM(BaseLLM):
    """Gemini LLM实现类"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化Gemini客户端
        
        Args:
            api_key: Gemini API密钥，如果不提供则从环境变量读取
            model_name: 模型名称，默认使用gemini-2.5-pro
            base_url: Gemini API基础地址
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API Key未找到！请设置GEMINI_API_KEY环境变量或在初始化时提供")
        
        super().__init__(api_key, model_name)
        
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL") or DEFAULT_GEMINI_BASE_URL
        
        # 初始化OpenAI客户端，使用Gemini的中转endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.default_model = model_name or self.get_default_model()
    
    def get_default_model(self) -> str:
        """获取默认模型名称"""
        return "gemini-2.5-pro"
    
    @with_retry(LLM_RETRY_CONFIG)
    def invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        调用Gemini API生成回复
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            **kwargs: 其他参数，如temperature、max_tokens等
            
        Returns:
            Gemini生成的回复文本
        """
        try:
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
                "max_tokens": kwargs.get("max_tokens", 30000),  # 提高到30000以支持一万字报告
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
                
        except Exception as e:
            print(f"Gemini API调用错误: {str(e)}")
            raise e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "provider": "Gemini",
            "model": self.default_model,
            "api_base": self.base_url
        }
