"""
LLM基础抽象类
定义所有LLM实现需要遵循的接口标准
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLM(ABC):
    """LLM基础抽象类"""
    
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            model_name: 模型名称，如果不指定则使用默认模型
        """
        self.api_key = api_key
        self.model_name = model_name
        
    @abstractmethod
    def invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        调用LLM生成回复
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            **kwargs: 其他参数，如temperature、max_tokens等
            
        Returns:
            LLM生成的回复文本
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """
        获取默认模型名称
        
        Returns:
            默认模型名称
        """
        pass
    
    def validate_response(self, response: str) -> str:
        """
        验证和清理响应内容
        
        Args:
            response: LLM原始响应
            
        Returns:
            清理后的响应内容
        """
        if response is None:
            return ""
        return response.strip()
