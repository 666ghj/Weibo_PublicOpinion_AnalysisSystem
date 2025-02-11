import openai
import anthropic
import json
from typing import List, Dict
import os
from datetime import datetime
from utils.logger import app_logger as logging

class AIAnalyzer:
    def __init__(self):
        # 从环境变量获取API密钥
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.claude_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not self.openai_key and not self.claude_key:
            raise ValueError("请至少设置一个API密钥 (OPENAI_API_KEY 或 ANTHROPIC_API_KEY)")
        
        if self.openai_key:
            openai.api_key = self.openai_key
        if self.claude_key:
            self.claude_client = anthropic.Anthropic(api_key=self.claude_key)
        
        # 支持的模型列表
        self.supported_models = {
            # OpenAI 模型
            'gpt-3.5-turbo': {'provider': 'openai', 'max_tokens': 2000, 'cost_per_1k': 0.0015},
            'gpt-3.5-turbo-16k': {'provider': 'openai', 'max_tokens': 16000, 'cost_per_1k': 0.003},
            'gpt-4': {'provider': 'openai', 'max_tokens': 8000, 'cost_per_1k': 0.03},
            'gpt-4-32k': {'provider': 'openai', 'max_tokens': 32000, 'cost_per_1k': 0.06},
            'gpt-4-turbo-preview': {'provider': 'openai', 'max_tokens': 128000, 'cost_per_1k': 0.01},
            
            # Claude 模型
            'claude-3-opus-20240229': {'provider': 'anthropic', 'max_tokens': 4000, 'cost_per_1k': 0.015},
            'claude-3-sonnet-20240229': {'provider': 'anthropic', 'max_tokens': 3000, 'cost_per_1k': 0.003},
            'claude-3-haiku-20240307': {'provider': 'anthropic', 'max_tokens': 2000, 'cost_per_1k': 0.0025},
            'claude-2.1': {'provider': 'anthropic', 'max_tokens': 100000, 'cost_per_1k': 0.008},
            'claude-2.0': {'provider': 'anthropic', 'max_tokens': 100000, 'cost_per_1k': 0.008},
            'claude-instant-1.2': {'provider': 'anthropic', 'max_tokens': 100000, 'cost_per_1k': 0.0015}
        }
        
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
            if model_type not in self.supported_models:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            model_info = self.supported_models[model_type]
            provider = model_info['provider']
            max_tokens = model_info['max_tokens']
            
            # 根据模型类型调整批处理大小
            adjusted_batch_size = min(batch_size, self._get_optimal_batch_size(model_type))
            if adjusted_batch_size != batch_size:
                logging.info(f"已将批处理大小从 {batch_size} 调整为 {adjusted_batch_size}")
            
            all_results = []
            total_cost = 0
            
            # 分批处理消息
            for i in range(0, len(messages), adjusted_batch_size):
                batch = messages[i:i + adjusted_batch_size]
                formatted_messages = []
                for msg in batch:
                    formatted_messages.append(f"消息ID: {msg['id']}\n内容: {msg['content']}")
                
                messages_text = "\n---\n".join(formatted_messages)
                system_prompt = self.prompt_templates.get(analysis_depth, self.prompt_templates['standard'])
                
                if provider == 'openai':
                    result = await self._analyze_with_openai(
                        messages_text, 
                        system_prompt, 
                        model_type, 
                        max_tokens
                    )
                else:  # anthropic
                    result = await self._analyze_with_claude(
                        messages_text, 
                        system_prompt, 
                        model_type, 
                        max_tokens
                    )
                
                if result:
                    all_results.extend(result)
                    # 计算本批次成本
                    batch_cost = self._calculate_cost(len(messages_text), model_type)
                    total_cost += batch_cost
                    logging.info(f"批次处理完成,成本: ${batch_cost:.4f}")
            
            logging.info(f"分析完成,总成本: ${total_cost:.4f}")
            return all_results
                
        except Exception as e:
            logging.error(f"AI分析过程出错: {e}")
            return []
    
    def _get_optimal_batch_size(self, model_type: str) -> int:
        """根据模型类型获取最优批处理大小"""
        model_info = self.supported_models[model_type]
        max_tokens = model_info['max_tokens']
        
        # 估算每条消息的平均token数(假设为200)
        avg_tokens_per_message = 200
        
        # 预留20%的token用于系统提示词和响应
        available_tokens = int(max_tokens * 0.8)
        
        # 计算最优批处理大小
        optimal_batch_size = max(1, min(100, available_tokens // avg_tokens_per_message))
        
        return optimal_batch_size
    
    def _calculate_cost(self, input_length: int, model_type: str) -> float:
        """计算API调用成本"""
        model_info = self.supported_models[model_type]
        cost_per_1k = model_info['cost_per_1k']
        
        # 估算token数(假设每4个字符约等于1个token)
        estimated_tokens = input_length // 4
        
        # 计算成本(美元)
        cost = (estimated_tokens / 1000) * cost_per_1k
        
        return cost
    
    async def _analyze_with_openai(self, messages_text: str, system_prompt: str, 
                                 model: str, max_tokens: int) -> List[Dict]:
        """使用OpenAI API进行分析"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下消息:\n{messages_text}"}
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                n=1,
                response_format={"type": "json_object"}  # 强制JSON响应格式
            )
            
            result = json.loads(response.choices[0].message.content)
            if isinstance(result, dict) and 'analysis_results' in result:
                return result['analysis_results']
            else:
                logging.error(f"OpenAI API返回格式不正确: {response.choices[0].message.content}")
                return []
                
        except Exception as e:
            logging.error(f"OpenAI API调用失败: {e}")
            return []
    
    async def _analyze_with_claude(self, messages_text: str, system_prompt: str, 
                                 model: str, max_tokens: int) -> List[Dict]:
        """使用Claude API进行分析"""
        try:
            response = await self.claude_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"请分析以下消息:\n{messages_text}"
                    }
                ]
            )
            
            result = json.loads(response.content[0].text)
            if isinstance(result, dict) and 'analysis_results' in result:
                return result['analysis_results']
            else:
                logging.error(f"Claude API返回格式不正确: {response.content[0].text}")
                return []
                
        except Exception as e:
            logging.error(f"Claude API调用失败: {e}")
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