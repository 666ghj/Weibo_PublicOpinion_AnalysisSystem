"""
搜索节点实现
负责生成搜索查询和反思查询
"""

import json
from typing import Dict, Any
from json.decoder import JSONDecodeError

from .base_node import BaseNode
from ..prompts import SYSTEM_PROMPT_FIRST_SEARCH, SYSTEM_PROMPT_REFLECTION
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_json_tags,
    extract_clean_response
)


class FirstSearchNode(BaseNode):
    """为段落生成首次搜索查询的节点"""
    
    def __init__(self, llm_client):
        """
        初始化首次搜索节点
        
        Args:
            llm_client: LLM客户端
        """
        super().__init__(llm_client, "FirstSearchNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                return "title" in data and "content" in data
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            return "title" in input_data and "content" in input_data
        return False
    
    def run(self, input_data: Any, **kwargs) -> Dict[str, str]:
        """
        调用LLM生成搜索查询和理由
        
        Args:
            input_data: 包含title和content的字符串或字典
            **kwargs: 额外参数
            
        Returns:
            包含search_query和reasoning的字典
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("输入数据格式错误，需要包含title和content字段")
            
            # 准备输入数据
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            self.log_info("正在生成首次搜索查询")
            
            # 调用LLM
            response = self.llm_client.invoke(SYSTEM_PROMPT_FIRST_SEARCH, message)
            
            # 处理响应
            processed_response = self.process_output(response)
            
            self.log_info(f"生成搜索查询: {processed_response.get('search_query', 'N/A')}")
            return processed_response
            
        except Exception as e:
            self.log_error(f"生成首次搜索查询失败: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> Dict[str, str]:
        """
        处理LLM输出，提取搜索查询和推理
        
        Args:
            output: LLM原始输出
            
        Returns:
            包含search_query和reasoning的字典
        """
        try:
            # 清理响应文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # 解析JSON
            try:
                result = json.loads(cleaned_output)
            except JSONDecodeError:
                # 使用更强大的提取方法
                result = extract_clean_response(cleaned_output)
                if "error" in result:
                    raise ValueError("JSON解析失败")
            
            # 验证和清理结果
            search_query = result.get("search_query", "")
            reasoning = result.get("reasoning", "")
            
            if not search_query:
                raise ValueError("未找到搜索查询")
            
            return {
                "search_query": search_query,
                "reasoning": reasoning
            }
            
        except Exception as e:
            self.log_error(f"处理输出失败: {str(e)}")
            # 返回默认查询
            return {
                "search_query": "相关主题研究",
                "reasoning": "由于解析失败，使用默认搜索查询"
            }


class ReflectionNode(BaseNode):
    """反思段落并生成新搜索查询的节点"""
    
    def __init__(self, llm_client):
        """
        初始化反思节点
        
        Args:
            llm_client: LLM客户端
        """
        super().__init__(llm_client, "ReflectionNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "paragraph_latest_state"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "paragraph_latest_state"]
            return all(field in input_data for field in required_fields)
        return False
    
    def run(self, input_data: Any, **kwargs) -> Dict[str, str]:
        """
        调用LLM反思并生成搜索查询
        
        Args:
            input_data: 包含title、content和paragraph_latest_state的字符串或字典
            **kwargs: 额外参数
            
        Returns:
            包含search_query和reasoning的字典
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("输入数据格式错误，需要包含title、content和paragraph_latest_state字段")
            
            # 准备输入数据
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            self.log_info("正在进行反思并生成新搜索查询")
            
            # 调用LLM
            response = self.llm_client.invoke(SYSTEM_PROMPT_REFLECTION, message)
            
            # 处理响应
            processed_response = self.process_output(response)
            
            self.log_info(f"反思生成搜索查询: {processed_response.get('search_query', 'N/A')}")
            return processed_response
            
        except Exception as e:
            self.log_error(f"反思生成搜索查询失败: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> Dict[str, str]:
        """
        处理LLM输出，提取搜索查询和推理
        
        Args:
            output: LLM原始输出
            
        Returns:
            包含search_query和reasoning的字典
        """
        try:
            # 清理响应文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # 解析JSON
            try:
                result = json.loads(cleaned_output)
            except JSONDecodeError:
                # 使用更强大的提取方法
                result = extract_clean_response(cleaned_output)
                if "error" in result:
                    raise ValueError("JSON解析失败")
            
            # 验证和清理结果
            search_query = result.get("search_query", "")
            reasoning = result.get("reasoning", "")
            
            if not search_query:
                raise ValueError("未找到搜索查询")
            
            return {
                "search_query": search_query,
                "reasoning": reasoning
            }
            
        except Exception as e:
            self.log_error(f"处理输出失败: {str(e)}")
            # 返回默认查询
            return {
                "search_query": "深度研究补充信息",
                "reasoning": "由于解析失败，使用默认反思搜索查询"
            }
