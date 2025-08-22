"""
LLM调用模块
支持多种大语言模型的统一接口
"""

from .base import BaseLLM
from .deepseek import DeepSeekLLM
from .openai_llm import OpenAILLM

__all__ = ["BaseLLM", "DeepSeekLLM", "OpenAILLM"]
