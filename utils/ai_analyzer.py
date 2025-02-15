import openai
import anthropic
import json
from typing import List, Dict, Tuple, Any
import os
import asyncio
import math
from datetime import datetime
from utils.logger import app_logger as logging

class AIAnalyzer:
    def __init__(self):
        # 尝试从环境变量中获取API密钥，如果没有则主动询问配置
        self.openai_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_key:
            print("未检测到 OPENAI_API_KEY。")
            # 提示时允许按回车跳过输入
            self.openai_key = input("请输入 OPENAI_API_KEY (按回车键跳过输入): ").strip()
        
        self.claude_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.claude_key:
            print("未检测到 ANTHROPIC_API_KEY。")
            self.claude_key = input("请输入 ANTHROPIC_API_KEY (按回车键跳过输入): ").strip()
        
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.deepseek_key:
            print("未检测到 DEEPSEEK_API_KEY。")
            self.deepseek_key = input("请输入 DEEPSEEK_API_KEY (按回车键跳过输入): ").strip()
        
        # 如果不希望通过交互输入，也可以直接在此处配置（注释掉下面几行即可）
        # self.openai_key = "你的OpenAI_API_KEY"
        # self.claude_key = "你的ANTHROPIC_API_KEY"
        # self.deepseek_key = "你的DEEPSEEK_API_KEY"
        
        # 配置各API客户端
        if self.openai_key:
            openai.api_key = self.openai_key
        if self.claude_key:
            self.claude_client = anthropic.Anthropic(api_key=self.claude_key)
        if self.deepseek_key:
            self.deepseek_client = openai.OpenAI(
                api_key=self.deepseek_key,
                base_url="https://api.deepseek.com/v1"
            )
        
        # 支持的模型列表（增加了最新的 ChatGPT 和 Claude 模型）
        self.supported_models: Dict[str, Dict[str, Any]] = {
            # OpenAI 最新模型（ChatGPT系列）
            'gpt-4o-latest': {
                'provider': 'openai',
                'max_tokens': 128000,    # 支持大窗口
                'cost_per_1k': 0.01      # 参考价格（美元）
            },
            'gpt-4o-mini': {
                'provider': 'openai',
                'max_tokens': 4000,      # 轻量版，适合快速任务
                'cost_per_1k': 0.00015   # 成本大幅降低
            },
            # 旧版OpenAI模型
            'gpt-3.5-turbo': {'provider': 'openai', 'max_tokens': 2000, 'cost_per_1k': 0.0015},
            'gpt-3.5-turbo-16k': {'provider': 'openai', 'max_tokens': 16000, 'cost_per_1k': 0.003},
            'gpt-4': {'provider': 'openai', 'max_tokens': 8000, 'cost_per_1k': 0.03},
            'gpt-4-32k': {'provider': 'openai', 'max_tokens': 32000, 'cost_per_1k': 0.06},
            'gpt-4-turbo-preview': {'provider': 'openai', 'max_tokens': 128000, 'cost_per_1k': 0.01},
            
            # Anthropic 最新模型（Claude系列）
            'claude-3.5-sonnet-new': {
                'provider': 'anthropic',
                'max_tokens': 200000,    # 新版Claude 3.5 Sonnet
                'cost_per_1k': 0.015
            },
            'claude-3.5-haiku': {
                'provider': 'anthropic',
                'max_tokens': 200000,    # 最新Claude 3.5 Haiku
                'cost_per_1k': 0.0025
            },
            # 旧版Claude模型
            'claude-2.1': {'provider': 'anthropic', 'max_tokens': 100000, 'cost_per_1k': 0.008},
            'claude-2.0': {'provider': 'anthropic', 'max_tokens': 100000, 'cost_per_1k': 0.008},
            'claude-instant-1.2': {'provider': 'anthropic', 'max_tokens': 100000, 'cost_per_1k': 0.0015},
            
            # DeepSeek 模型
            'deepseek-chat': {'provider': 'deepseek', 'max_tokens': 4000, 'cost_per_1k': 0.002},
            'deepseek-reasoner': {'provider': 'deepseek', 'max_tokens': 4000, 'cost_per_1k': 0.003}
        }
        
        # 不同深度的分析提示词
        self.prompt_templates: Dict[str, str] = {
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
                                 analysis_depth: str = "standard",
                                 prefer_deepseek: bool = True) -> List[Dict]:
        """
        分析一批消息并返回分析结果。
        如果 DeepSeek API 可用且 prefer_deepseek 为 True，则优先使用 DeepSeek 模型。
        """
        try:
            # 优先使用 DeepSeek 模型以降低成本
            if prefer_deepseek and self.deepseek_key:
                if model_type not in ['deepseek-chat', 'deepseek-reasoner']:
                    logging.info("检测到 DeepSeek API, 优先使用 'deepseek-chat' 模型以降低成本。")
                    model_type = 'deepseek-chat'
            
            if model_type not in self.supported_models:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            model_info = self.supported_models[model_type]
            provider = model_info['provider']
            max_tokens = model_info['max_tokens']
            
            # 根据模型类型调整批处理大小
            optimal_batch_size = self._get_optimal_batch_size(model_type)
            adjusted_batch_size = min(batch_size, optimal_batch_size)
            if adjusted_batch_size != batch_size:
                logging.info(f"已将批处理大小从 {batch_size} 调整为 {adjusted_batch_size}")
            
            tasks = []
            total_cost = 0.0
            # 分批处理消息并异步调用分析任务
            for i in range(0, len(messages), adjusted_batch_size):
                batch = messages[i:i + adjusted_batch_size]
                system_prompt = self.prompt_templates.get(analysis_depth, self.prompt_templates['standard'])
                tasks.append(self._process_batch(batch, system_prompt, model_type, max_tokens, provider))
            
            # 并发执行所有批次任务
            results = await asyncio.gather(*tasks)
            
            all_results = []
            for batch_result, batch_cost in results:
                all_results.extend(batch_result)
                total_cost += batch_cost
            
            logging.info(f"分析完成, 总成本: ${total_cost:.4f}")
            return all_results
        except Exception as e:
            logging.error(f"AI分析过程出错: {e}", exc_info=True)
            return []
    
    async def _process_batch(self, batch: List[Dict], system_prompt: str, 
                             model_type: str, max_tokens: int, provider: str) -> Tuple[List[Dict], float]:
        """
        处理单个批次的消息，返回 (分析结果, 本批次成本)
        """
        try:
            formatted_messages = [
                f"消息ID: {msg.get('id')}\n内容: {msg.get('content')}" for msg in batch
            ]
            messages_text = "\n---\n".join(formatted_messages)
            
            if provider == 'openai':
                result = await self._analyze_with_openai(messages_text, system_prompt, model_type, max_tokens)
            elif provider == 'anthropic':
                result = await self._analyze_with_claude(messages_text, system_prompt, model_type, max_tokens)
            elif provider == 'deepseek':
                result = await self._analyze_with_deepseek(messages_text, system_prompt, model_type, max_tokens)
            else:
                logging.error(f"未知的API供应商: {provider}")
                return ([], 0.0)
            
            batch_cost = self._calculate_cost(len(messages_text), model_type)
            logging.info(f"批次处理完成, 成本: ${batch_cost:.4f}")
            return (result, batch_cost)
        except Exception as e:
            logging.error(f"处理批次时出错: {e}", exc_info=True)
            return ([], 0.0)
    
    def _get_optimal_batch_size(self, model_type: str) -> int:
        """根据模型类型获取最优批处理大小"""
        model_info = self.supported_models[model_type]
        max_tokens = model_info['max_tokens']
        
        # 估算每条消息的平均 token 数（假设为 200）
        avg_tokens_per_message = 200
        # 预留 20% 的 token 用于系统提示词和响应
        available_tokens = int(max_tokens * 0.8)
        optimal_batch_size = max(1, min(100, available_tokens // avg_tokens_per_message))
        return optimal_batch_size
    
    def _calculate_cost(self, input_length: int, model_type: str) -> float:
        """计算 API 调用成本"""
        model_info = self.supported_models[model_type]
        cost_per_1k = model_info['cost_per_1k']
        # 估算 token 数（假设每 4 个字符约等于 1 个 token）
        estimated_tokens = math.ceil(input_length / 4)
        cost = (estimated_tokens / 1000) * cost_per_1k
        return cost
    
    async def _analyze_with_openai(self, messages_text: str, system_prompt: str, 
                                   model: str, max_tokens: int) -> List[Dict]:
        """使用 OpenAI API 进行分析"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下消息:\n{messages_text}"}
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                n=1
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            if isinstance(result, dict) and 'analysis_results' in result:
                return result['analysis_results']
            else:
                logging.error(f"OpenAI API返回格式不正确: {content}")
                return []
        except Exception as e:
            logging.error(f"OpenAI API调用失败: {e}", exc_info=True)
            return []
    
    async def _analyze_with_claude(self, messages_text: str, system_prompt: str, 
                                   model: str, max_tokens: int) -> List[Dict]:
        """使用 Claude API 进行分析"""
        try:
            response = await self.claude_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": f"请分析以下消息:\n{messages_text}"}]
            )
            content = response.content[0].text
            result = json.loads(content)
            if isinstance(result, dict) and 'analysis_results' in result:
                return result['analysis_results']
            else:
                logging.error(f"Claude API返回格式不正确: {content}")
                return []
        except Exception as e:
            logging.error(f"Claude API调用失败: {e}", exc_info=True)
            return []
    
    async def _analyze_with_deepseek(self, messages_text: str, system_prompt: str, 
                                     model: str, max_tokens: int) -> List[Dict]:
        """使用 DeepSeek API 进行分析"""
        try:
            response = await self.deepseek_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下消息:\n{messages_text}"}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            if isinstance(result, dict) and 'analysis_results' in result:
                return result['analysis_results']
            else:
                logging.error(f"DeepSeek API返回格式不正确: {content}")
                return []
        except Exception as e:
            logging.error(f"DeepSeek API调用失败: {e}", exc_info=True)
            return []
    
    def format_analysis_for_display(self, analysis: Dict) -> Dict:
        """将分析结果格式化为前端显示格式"""
        base_result = {
            'id': analysis.get('message_id', ''),
            'sentiment': analysis.get('sentiment', ''),
            'sentiment_score': f"{float(analysis.get('sentiment_score', 0)):.2%}",
            'keywords': ', '.join(analysis.get('keywords', [])),
            'key_points': analysis.get('key_points', ''),
            'influence': analysis.get('influence_analysis', ''),
            'risk_level': analysis.get('risk_level', ''),
            'analysis_time': datetime.fromtimestamp(
                float(analysis.get('timestamp', 0))
            ).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 如果是深度分析，添加额外信息
        if 'risk_factors' in analysis:
            base_result.update({
                'risk_factors': analysis.get('risk_factors', []),
                'suggestions': analysis.get('suggestions', [])
            })
            
        return base_result

# 创建全局 AI 分析器实例
ai_analyzer = AIAnalyzer()

# 若需要直接配置或测试，可在此处编写测试代码
if __name__ == "__main__":
    # 示例：直接配置并调用分析器（可替换为实际测试代码）
    sample_messages = [
        {"id": "1", "content": "今天天气真好，我很开心。"},
        {"id": "2", "content": "经济形势不容乐观，风险较大。"}
    ]
    
    async def test():
        results = await ai_analyzer.analyze_messages(sample_messages, model_type="gpt-4o-latest", analysis_depth="standard")
        for res in results:
            print(ai_analyzer.format_analysis_for_display(res))
    
    asyncio.run(test())
