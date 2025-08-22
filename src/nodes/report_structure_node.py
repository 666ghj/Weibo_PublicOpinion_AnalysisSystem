"""
报告结构生成节点
负责根据查询生成报告的整体结构
"""

import json
from typing import Dict, Any, List
from json.decoder import JSONDecodeError

from .base_node import StateMutationNode
from ..state.state import State
from ..prompts import SYSTEM_PROMPT_REPORT_STRUCTURE
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_json_tags,
    extract_clean_response
)


class ReportStructureNode(StateMutationNode):
    """生成报告结构的节点"""
    
    def __init__(self, llm_client, query: str):
        """
        初始化报告结构节点
        
        Args:
            llm_client: LLM客户端
            query: 用户查询
        """
        super().__init__(llm_client, "ReportStructureNode")
        self.query = query
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        return isinstance(self.query, str) and len(self.query.strip()) > 0
    
    def run(self, input_data: Any = None, **kwargs) -> List[Dict[str, str]]:
        """
        调用LLM生成报告结构
        
        Args:
            input_data: 输入数据（这里不使用，使用初始化时的query）
            **kwargs: 额外参数
            
        Returns:
            报告结构列表
        """
        try:
            self.log_info(f"正在为查询生成报告结构: {self.query}")
            
            # 调用LLM
            response = self.llm_client.invoke(SYSTEM_PROMPT_REPORT_STRUCTURE, self.query)
            
            # 处理响应
            processed_response = self.process_output(response)
            
            self.log_info(f"成功生成 {len(processed_response)} 个段落结构")
            return processed_response
            
        except Exception as e:
            self.log_error(f"生成报告结构失败: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> List[Dict[str, str]]:
        """
        处理LLM输出，提取报告结构
        
        Args:
            output: LLM原始输出
            
        Returns:
            处理后的报告结构列表
        """
        try:
            # 清理响应文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # 解析JSON
            try:
                report_structure = json.loads(cleaned_output)
            except JSONDecodeError:
                # 使用更强大的提取方法
                report_structure = extract_clean_response(cleaned_output)
                if "error" in report_structure:
                    raise ValueError("JSON解析失败")
            
            # 验证结构
            if not isinstance(report_structure, list):
                raise ValueError("报告结构应该是一个列表")
            
            # 验证每个段落
            validated_structure = []
            for i, paragraph in enumerate(report_structure):
                if not isinstance(paragraph, dict):
                    continue
                
                title = paragraph.get("title", f"段落 {i+1}")
                content = paragraph.get("content", "")
                
                validated_structure.append({
                    "title": title,
                    "content": content
                })
            
            return validated_structure
            
        except Exception as e:
            self.log_error(f"处理输出失败: {str(e)}")
            # 返回默认结构
            return [
                {
                    "title": "概述",
                    "content": f"对'{self.query}'的总体概述和背景介绍"
                },
                {
                    "title": "详细分析", 
                    "content": f"深入分析'{self.query}'的相关内容"
                }
            ]
    
    def mutate_state(self, input_data: Any = None, state: State = None, **kwargs) -> State:
        """
        将报告结构写入状态
        
        Args:
            input_data: 输入数据
            state: 当前状态，如果为None则创建新状态
            **kwargs: 额外参数
            
        Returns:
            更新后的状态
        """
        if state is None:
            state = State()
        
        try:
            # 生成报告结构
            report_structure = self.run(input_data, **kwargs)
            
            # 设置查询和报告标题
            state.query = self.query
            if not state.report_title:
                state.report_title = f"关于'{self.query}'的深度研究报告"
            
            # 添加段落到状态
            for paragraph_data in report_structure:
                state.add_paragraph(
                    title=paragraph_data["title"],
                    content=paragraph_data["content"]
                )
            
            self.log_info(f"已将 {len(report_structure)} 个段落添加到状态中")
            return state
            
        except Exception as e:
            self.log_error(f"状态更新失败: {str(e)}")
            raise e
