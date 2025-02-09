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
        
        # 系统提示词，限制AI的输出格式
        self.system_prompt = """你是一个专业的舆情分析助手。你的任务是分析每条消息的情感倾向、关键词和潜在影响。
请严格按照以下JSON格式返回分析结果：
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
}
请确保每个字段都有值，并保持JSON格式的一致性。"""

    async def analyze_messages(self, messages: List[Dict]) -> List[Dict]:
        """分析一批消息并返回分析结果"""
        try:
            # 构建输入消息
            formatted_messages = []
            for msg in messages:
                formatted_messages.append(f"消息ID: {msg['id']}\n内容: {msg['content']}")
            
            messages_text = "\n---\n".join(formatted_messages)
            
            # 调用OpenAI API
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"请分析以下消息:\n{messages_text}"}
                ],
                temperature=0.3,  # 降低随机性
                max_tokens=2000,
                n=1
            )
            
            # 解析返回结果
            try:
                result = json.loads(response.choices[0].message.content)
                # 验证结果格式
                if not isinstance(result, dict) or 'analysis_results' not in result:
                    raise ValueError("AI返回格式不正确")
                return result['analysis_results']
            except json.JSONDecodeError:
                logging.error("AI返回结果解析失败")
                return []
                
        except Exception as e:
            logging.error(f"AI分析过程出错: {e}")
            return []
    
    def format_analysis_for_display(self, analysis: Dict) -> Dict:
        """将分析结果格式化为前端显示格式"""
        return {
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

# 创建全局AI分析器实例
ai_analyzer = AIAnalyzer() 