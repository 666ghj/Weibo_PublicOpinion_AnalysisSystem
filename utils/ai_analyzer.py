import openai
import json
from typing import List, Dict
import os
from datetime import datetime
from utils.logger import app_logger as logging

class AIAnalyzer:
    def __init__(self):
        # 从环境变量获取API密钥
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("请设置OPENAI_API_KEY环境变量")
        
        openai.api_key = self.api_key
        
        # 不同深度的分析提示词
        self.prompt_templates = {
            'basic': """你是一个专业的舆情分析助手。请对每条消息进行基础的情感分析。
请按以下JSON格式返回：
{
    "analysis_results": [
        {
            "message_id": "消息ID",
            "sentiment": "情感倾向 (积极/消极/中性)",
            "sentiment_score": "情感分数 (0-1)",
            "keywords": ["关键词1", "关键词2"],
            "key_points": "简要概述",
            "influence_analysis": "基础影响分析",
            "risk_level": "风险等级 (低/中/高)",
            "timestamp": "分析时间戳"
        }
    ]
}""",
            'standard': """你是一个专业的舆情分析助手。请对每条消息进行标准深度的分析。
请按以下JSON格式返回：
{
    "analysis_results": [
        {
            "message_id": "消息ID",
            "sentiment": "情感倾向 (积极/消极/中性)",
            "sentiment_score": "情感分数 (0-1)",
            "keywords": ["关键词1", "关键词2", "关键词3"],
            "key_points": "核心观点概述",
            "influence_analysis": "潜在影响分析",
            "risk_level": "风险等级 (低/中/高)",
            "timestamp": "分析时间戳"
        }
    ]
}""",
            'deep': """你是一个专业的舆情分析助手。请对每条消息进行深度分析。
请按以下JSON格式返回：
{
    "analysis_results": [
        {
            "message_id": "消息ID",
            "sentiment": "情感倾向 (积极/消极/中性)",
            "sentiment_score": "情感分数 (0-1)",
            "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
            "key_points": "详细的核心观点分析",
            "influence_analysis": "深度影响分析，包括短期和长期影响",
            "risk_factors": ["风险因素1", "风险因素2", "风险因素3"],
            "risk_level": "风险等级 (低/中/高)",
            "suggestions": ["建议1", "建议2", "建议3"],
            "timestamp": "分析时间戳"
        }
    ]
}"""
        }

    async def analyze_messages(self, messages: List[Dict], batch_size: int = 50, 
                             model_type: str = "gpt-3.5-turbo", 
                             analysis_depth: str = "standard") -> List[Dict]:
        """分析一批消息并返回分析结果"""
        try:
            all_results = []
            
            # 分批处理消息
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]
                formatted_messages = []
                for msg in batch:
                    formatted_messages.append(f"消息ID: {msg['id']}\n内容: {msg['content']}")
                
                messages_text = "\n---\n".join(formatted_messages)
                
                # 获取对应深度的提示词
                system_prompt = self.prompt_templates.get(analysis_depth, self.prompt_templates['standard'])
                
                # 调用OpenAI API
                response = await openai.ChatCompletion.acreate(
                    model=model_type,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"请分析以下消息:\n{messages_text}"}
                    ],
                    temperature=0.3,  # 降低随机性
                    max_tokens=2000 if analysis_depth != 'deep' else 3000,
                    n=1
                )
                
                try:
                    result = json.loads(response.choices[0].message.content)
                    if isinstance(result, dict) and 'analysis_results' in result:
                        all_results.extend(result['analysis_results'])
                    else:
                        logging.error(f"API返回格式不正确: {response.choices[0].message.content}")
                except json.JSONDecodeError as e:
                    logging.error(f"JSON解析失败: {e}")
                    continue
                
            return all_results
                
        except Exception as e:
            logging.error(f"AI分析过程出错: {e}")
            return []
    
    def format_analysis_for_display(self, analysis: Dict) -> Dict:
        """将分析结果格式化为前端显示格式"""
        base_result = {
            'id': analysis['message_id'],
            'sentiment': analysis['sentiment'],
            'sentiment_score': f"{float(analysis['sentiment_score']):.2%}",
            'keywords': ', '.join(analysis['keywords']),
            'key_points': analysis['key_points'],
            'influence': analysis['influence_analysis'],
            'risk_level': analysis['risk_level'],
            'analysis_time': datetime.fromtimestamp(
                float(analysis['timestamp'])
            ).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 如果是深度分析，添加额外信息
        if 'risk_factors' in analysis:
            base_result.update({
                'risk_factors': analysis['risk_factors'],
                'suggestions': analysis['suggestions']
            })
            
        return base_result

# 创建全局AI分析器实例
ai_analyzer = AIAnalyzer() 