"""
总结节点实现
负责根据搜索结果生成和更新段落内容
"""

import json
from typing import Dict, Any, List
from json.decoder import JSONDecodeError

from .base_node import StateMutationNode
from ..state.state import State
from ..prompts import SYSTEM_PROMPT_FIRST_SUMMARY, SYSTEM_PROMPT_REFLECTION_SUMMARY
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_json_tags,
    extract_clean_response,
    format_search_results_for_prompt
)


class FirstSummaryNode(StateMutationNode):
    """根据搜索结果生成段落首次总结的节点"""
    
    def __init__(self, llm_client):
        """
        初始化首次总结节点
        
        Args:
            llm_client: LLM客户端
        """
        super().__init__(llm_client, "FirstSummaryNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "search_query", "search_results"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "search_query", "search_results"]
            return all(field in input_data for field in required_fields)
        return False
    
    def run(self, input_data: Any, **kwargs) -> str:
        """
        调用LLM生成段落总结
        
        Args:
            input_data: 包含title、content、search_query和search_results的数据
            **kwargs: 额外参数
            
        Returns:
            段落总结内容
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("输入数据格式错误")
            
            # 准备输入数据
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            self.log_info("正在生成首次段落总结")
            
            # 调用LLM
            response = self.llm_client.invoke(SYSTEM_PROMPT_FIRST_SUMMARY, message)
            
            # 处理响应
            processed_response = self.process_output(response)
            
            self.log_info("成功生成首次段落总结")
            return processed_response
            
        except Exception as e:
            self.log_error(f"生成首次总结失败: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> str:
        """
        处理LLM输出，提取段落总结
        
        Args:
            output: LLM原始输出
            
        Returns:
            段落总结内容
        """
        try:
            # 清理响应文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # 解析JSON
            try:
                result = json.loads(cleaned_output)
            except JSONDecodeError:
                # 如果不是JSON格式，直接返回清理后的文本
                return cleaned_output
            
            # 提取段落内容
            if isinstance(result, dict):
                paragraph_content = result.get("paragraph_latest_state", "")
                if paragraph_content:
                    return paragraph_content
            
            # 如果提取失败，返回原始清理后的文本
            return cleaned_output
            
        except Exception as e:
            self.log_error(f"处理输出失败: {str(e)}")
            return "段落总结生成失败"
    
    def mutate_state(self, input_data: Any, state: State, paragraph_index: int, **kwargs) -> State:
        """
        更新段落的最新总结到状态
        
        Args:
            input_data: 输入数据
            state: 当前状态
            paragraph_index: 段落索引
            **kwargs: 额外参数
            
        Returns:
            更新后的状态
        """
        try:
            # 生成总结
            summary = self.run(input_data, **kwargs)
            
            # 更新状态
            if 0 <= paragraph_index < len(state.paragraphs):
                state.paragraphs[paragraph_index].research.latest_summary = summary
                self.log_info(f"已更新段落 {paragraph_index} 的首次总结")
            else:
                raise ValueError(f"段落索引 {paragraph_index} 超出范围")
            
            state.update_timestamp()
            return state
            
        except Exception as e:
            self.log_error(f"状态更新失败: {str(e)}")
            raise e


class ReflectionSummaryNode(StateMutationNode):
    """根据反思搜索结果更新段落总结的节点"""
    
    def __init__(self, llm_client):
        """
        初始化反思总结节点
        
        Args:
            llm_client: LLM客户端
        """
        super().__init__(llm_client, "ReflectionSummaryNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "search_query", "search_results", "paragraph_latest_state"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "search_query", "search_results", "paragraph_latest_state"]
            return all(field in input_data for field in required_fields)
        return False
    
    def run(self, input_data: Any, **kwargs) -> str:
        """
        调用LLM更新段落内容
        
        Args:
            input_data: 包含完整反思信息的数据
            **kwargs: 额外参数
            
        Returns:
            更新后的段落内容
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("输入数据格式错误")
            
            # 准备输入数据
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            self.log_info("正在生成反思总结")
            
            # 调用LLM
            response = self.llm_client.invoke(SYSTEM_PROMPT_REFLECTION_SUMMARY, message)
            
            # 处理响应
            processed_response = self.process_output(response)
            
            self.log_info("成功生成反思总结")
            return processed_response
            
        except Exception as e:
            self.log_error(f"生成反思总结失败: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> str:
        """
        处理LLM输出，提取更新后的段落内容
        
        Args:
            output: LLM原始输出
            
        Returns:
            更新后的段落内容
        """
        try:
            # 清理响应文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # 解析JSON
            try:
                result = json.loads(cleaned_output)
            except JSONDecodeError:
                # 如果不是JSON格式，直接返回清理后的文本
                return cleaned_output
            
            # 提取更新后的段落内容
            if isinstance(result, dict):
                updated_content = result.get("updated_paragraph_latest_state", "")
                if updated_content:
                    return updated_content
            
            # 如果提取失败，返回原始清理后的文本
            return cleaned_output
            
        except Exception as e:
            self.log_error(f"处理输出失败: {str(e)}")
            return "反思总结生成失败"
    
    def mutate_state(self, input_data: Any, state: State, paragraph_index: int, **kwargs) -> State:
        """
        将更新后的总结写入状态
        
        Args:
            input_data: 输入数据
            state: 当前状态
            paragraph_index: 段落索引
            **kwargs: 额外参数
            
        Returns:
            更新后的状态
        """
        try:
            # 生成更新后的总结
            updated_summary = self.run(input_data, **kwargs)
            
            # 更新状态
            if 0 <= paragraph_index < len(state.paragraphs):
                state.paragraphs[paragraph_index].research.latest_summary = updated_summary
                state.paragraphs[paragraph_index].research.increment_reflection()
                self.log_info(f"已更新段落 {paragraph_index} 的反思总结")
            else:
                raise ValueError(f"段落索引 {paragraph_index} 超出范围")
            
            state.update_timestamp()
            return state
            
        except Exception as e:
            self.log_error(f"状态更新失败: {str(e)}")
            raise e
