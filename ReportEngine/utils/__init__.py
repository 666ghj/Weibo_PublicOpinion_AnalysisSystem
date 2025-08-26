"""
Report Engine工具模块
包含配置管理
"""

from .config import Config, load_config

__all__ = [
    "Config", 
    "load_config"
]
