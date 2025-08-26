"""
模板选择节点
根据查询内容和可用模板选择最合适的报告模板
"""

import os
import json
from typing import Dict, Any, List, Optional

from .base_node import BaseNode
from ..prompts import SYSTEM_PROMPT_TEMPLATE_SELECTION


class TemplateSelectionNode(BaseNode):
    """模板选择处理节点"""
    
    def __init__(self, llm_client, template_dir: str = "ReportEngine/report_template"):
        """
        初始化模板选择节点
        
        Args:
            llm_client: LLM客户端
            template_dir: 模板目录路径
        """
        super().__init__(llm_client, "TemplateSelectionNode")
        self.template_dir = template_dir
        
    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行模板选择
        
        Args:
            input_data: 包含查询和报告内容的字典
                - query: 原始查询
                - reports: 三个子agent的报告列表
                - forum_logs: 论坛日志内容
                
        Returns:
            选择的模板信息
        """
        self.log_info("开始模板选择...")
        
        query = input_data.get('query', '')
        reports = input_data.get('reports', [])
        forum_logs = input_data.get('forum_logs', '')
        
        # 获取可用模板
        available_templates = self._get_available_templates()
        
        if not available_templates:
            self.log_info("未找到预设模板，使用内置默认模板")
            return self._get_fallback_template()
        
        # 首先尝试简单关键词匹配
        simple_match = self._simple_keyword_matching(query, available_templates)
        if simple_match:
            self.log_info(f"通过关键词匹配选择模板: {simple_match['template_name']}")
            return simple_match
        
        # 如果关键词匹配失败，尝试LLM选择
        try:
            llm_result = self._llm_template_selection(query, reports, forum_logs, available_templates)
            if llm_result:
                return llm_result
        except Exception as e:
            self.log_error(f"LLM模板选择失败: {str(e)}")
        
        # 所有方法都失败，使用默认的社会热点事件模板
        default_template = self._get_default_social_event_template(available_templates)
        if default_template:
            return default_template
        
        # 最后备选方案
        return self._get_fallback_template()
    
    def _simple_keyword_matching(self, query: str, available_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """基于关键词的简单模板匹配"""
        query_lower = query.lower()
        
        # 关键词映射
        keyword_mapping = {
            '企业': ['企业品牌'],
            '品牌': ['企业品牌'],
            '声誉': ['企业品牌'],
            '市场': ['市场竞争'],
            '竞争': ['市场竞争'],
            '格局': ['市场竞争'],
            '政策': ['政策', '行业'],
            '行业': ['政策', '行业'],
            '动态': ['政策', '行业'],
            '突发': ['突发事件', '危机'],
            '危机': ['突发事件', '危机'],
            '公关': ['突发事件', '危机'],
            '日常': ['日常', '定期'],
            '定期': ['日常', '定期'],
            '监测': ['日常', '定期'],
            '热点': ['社会公共热点'],
            '社会': ['社会公共热点'],
            '事件': ['社会公共热点'],
        }
        
        # 检查查询中的关键词
        for keyword, template_keywords in keyword_mapping.items():
            if keyword in query_lower:
                # 查找匹配的模板
                for template in available_templates:
                    for template_keyword in template_keywords:
                        if template_keyword in template['name']:
                            return {
                                'template_name': template['name'],
                                'template_content': template['content'],
                                'selection_reason': f'基于关键词"{keyword}"匹配选择'
                            }
        
        return None
    
    def _llm_template_selection(self, query: str, reports: List[Any], forum_logs: str, 
                              available_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """使用LLM进行模板选择"""
        self.log_info("尝试使用LLM进行模板选择...")
        
        # 构建模板列表
        template_list = "\n".join([f"- {t['name']}: {t['description']}" for t in available_templates])
        
        user_message = f"""查询内容: {query}

报告数量: {len(reports)} 个分析引擎报告
论坛日志: {'有' if forum_logs else '无'}

可用模板:
{template_list}

请选择最合适的模板。"""
        
        # 调用LLM
        response = self.llm_client.invoke(SYSTEM_PROMPT_TEMPLATE_SELECTION, user_message)
        
        # 检查响应是否为空
        if not response or not response.strip():
            self.log_error("LLM返回空响应")
            return None
        
        self.log_info(f"LLM原始响应: {response[:200]}...")
        
        # 尝试解析JSON响应
        try:
            # 清理响应文本
            cleaned_response = self._clean_llm_response(response)
            result = json.loads(cleaned_response)
            
            # 验证选择的模板是否存在
            selected_template_name = result.get('template_name', '')
            for template in available_templates:
                if template['name'] == selected_template_name or selected_template_name in template['name']:
                    self.log_info(f"LLM选择模板: {selected_template_name}")
                    return {
                        'template_name': template['name'],
                        'template_content': template['content'],
                        'selection_reason': result.get('selection_reason', 'LLM智能选择')
                    }
            
            self.log_error(f"LLM选择的模板不存在: {selected_template_name}")
            return None
            
        except json.JSONDecodeError as e:
            self.log_error(f"JSON解析失败: {str(e)}")
            # 尝试从文本响应中提取模板信息
            return self._extract_template_from_text(response, available_templates)
    
    def _clean_llm_response(self, response: str) -> str:
        """清理LLM响应"""
        # 移除可能的markdown代码块标记
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        
        # 移除前后空白
        response = response.strip()
        
        return response
    
    def _extract_template_from_text(self, response: str, available_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """从文本响应中提取模板信息"""
        self.log_info("尝试从文本响应中提取模板信息")
        
        # 查找响应中是否包含模板名称
        for template in available_templates:
            template_name_variants = [
                template['name'],
                template['name'].replace('.md', ''),
                template['name'].replace('模板', ''),
            ]
            
            for variant in template_name_variants:
                if variant in response:
                    self.log_info(f"在响应中找到模板: {template['name']}")
                    return {
                        'template_name': template['name'],
                        'template_content': template['content'],
                        'selection_reason': '从文本响应中提取'
                    }
        
        return None
    
    def _get_available_templates(self) -> List[Dict[str, Any]]:
        """获取可用的模板列表"""
        templates = []
        
        if not os.path.exists(self.template_dir):
            self.log_error(f"模板目录不存在: {self.template_dir}")
            return templates
        
        # 查找所有markdown模板文件
        for filename in os.listdir(self.template_dir):
            if filename.endswith('.md'):
                template_path = os.path.join(self.template_dir, filename)
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    template_name = filename.replace('.md', '')
                    description = self._extract_template_description(template_name)
                    
                    templates.append({
                        'name': template_name,
                        'path': template_path,
                        'content': content,
                        'description': description
                    })
                except Exception as e:
                    self.log_error(f"读取模板文件失败 {filename}: {str(e)}")
        
        return templates
    
    def _extract_template_description(self, template_name: str) -> str:
        """根据模板名称生成描述"""
        if '企业品牌' in template_name:
            return "适用于企业品牌声誉和形象分析"
        elif '市场竞争' in template_name:
            return "适用于市场竞争格局和对手分析"
        elif '日常' in template_name or '定期' in template_name:
            return "适用于日常监测和定期汇报"
        elif '政策' in template_name or '行业' in template_name:
            return "适用于政策影响和行业动态分析"
        elif '热点' in template_name or '社会' in template_name:
            return "适用于社会热点和公共事件分析"
        elif '突发' in template_name or '危机' in template_name:
            return "适用于突发事件和危机公关"
        
        return "通用报告模板"
    
    def _get_default_social_event_template(self, available_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """获取默认的社会热点事件分析模板"""
        # 查找社会热点事件分析模板
        for template in available_templates:
            if '社会公共热点事件' in template['name'] or '热点' in template['name']:
                self.log_info(f"使用默认模板: {template['name']}")
                return {
                    'template_name': template['name'],
                    'template_content': template['content'],
                    'selection_reason': '默认使用社会热点事件分析模板'
                }
        return None
    
    def _get_fallback_template(self) -> Dict[str, Any]:
        """获取备用默认模板（空模板，让LLM自行发挥）"""
        self.log_info("未找到合适模板，使用空模板让LLM自行发挥")
        
        return {
            'template_name': '自由发挥模板',
            'template_content': '',
            'selection_reason': '未找到合适的预设模板，让LLM根据内容自行设计报告结构'
        }
