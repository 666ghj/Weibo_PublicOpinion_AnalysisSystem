"""
Report Engine提示词模块
定义报告生成各个阶段使用的系统提示词
"""

from .prompts import (
    SYSTEM_PROMPT_TEMPLATE_SELECTION,
    SYSTEM_PROMPT_HTML_GENERATION,
    output_schema_template_selection,
    input_schema_html_generation
)

__all__ = [
    "SYSTEM_PROMPT_TEMPLATE_SELECTION",
    "SYSTEM_PROMPT_HTML_GENERATION", 
    "output_schema_template_selection",
    "input_schema_html_generation"
]
