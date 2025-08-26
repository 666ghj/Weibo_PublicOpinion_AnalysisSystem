"""
Report Engine LLM基类
定义所有LLM实现的基础接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLLM(ABC):
    """LLM基类"""
    
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            model_name: 模型名称
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
            **kwargs: 其他参数
            
        Returns:
            生成的回复文本
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前模型信息
        
        Returns:
            模型信息字典
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
            response: 原始响应
            
        Returns:
            清理后的响应
        """
        if not response:
            return ""
        
        # 移除多余的空白字符
        response = response.strip()
        
        # 确保响应不为空
        if not response:
            return "抱歉，生成的内容为空。"
        
        return response
    
    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量（简单实现）
        
        Args:
            text: 输入文本
            
        Returns:
            估算的token数量
        """
        # 简单估算：中文字符按1.5个token计算，英文单词按1个token计算
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_words = len(text.split()) - chinese_chars
        
        return int(chinese_chars * 1.5 + english_words)
