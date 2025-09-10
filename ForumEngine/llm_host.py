"""
论坛主持人模块
使用硅基流动的Qwen3模型作为论坛主持人，引导多个agent进行讨论
"""

import requests
import json
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

# 添加项目根目录到Python路径以导入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GUIJI_QWEN3_API_KEY

# 添加utils目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
utils_dir = os.path.join(root_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from retry_helper import with_graceful_retry, SEARCH_API_RETRY_CONFIG


class ForumHost:
    """
    论坛主持人类
    使用硅基流动的Qwen3-235B模型作为智能主持人
    """
    
    def __init__(self, api_key: str = None):
        """
        初始化论坛主持人
        
        Args:
            api_key: 硅基流动API密钥，如果不提供则从配置文件读取
        """
        self.api_key = api_key or GUIJI_QWEN3_API_KEY
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.model = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # 使用更大的模型
        
        if not self.api_key:
            raise ValueError("未找到硅基流动API密钥，请在config.py中设置GUIJI_QWEN3_API_KEY")
        
        # 记录历史发言，避免重复
        self.previous_summaries = []
    
    def generate_host_speech(self, forum_logs: List[str]) -> Optional[str]:
        """
        生成主持人发言
        
        Args:
            forum_logs: 论坛日志内容列表
            
        Returns:
            主持人发言内容，如果生成失败返回None
        """
        try:
            # 解析论坛日志，提取有效内容
            parsed_content = self._parse_forum_logs(forum_logs)
            
            if not parsed_content['agent_speeches']:
                print("ForumHost: 没有找到有效的agent发言")
                return None
            
            # 构建prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(parsed_content)
            
            # 调用API生成发言
            response = self._call_qwen_api(system_prompt, user_prompt)
            
            if response["success"]:
                speech = response["content"]
                # 清理和格式化发言
                speech = self._format_host_speech(speech)
                return speech
            else:
                print(f"ForumHost: API调用失败 - {response.get('error', '未知错误')}")
                return None
                
        except Exception as e:
            print(f"ForumHost: 生成发言时出错 - {str(e)}")
            return None
    
    def _parse_forum_logs(self, forum_logs: List[str]) -> Dict[str, Any]:
        """
        解析论坛日志，提取结构化信息
        
        Returns:
            包含agent发言、时间线等信息的字典
        """
        parsed = {
            'agent_speeches': [],
            'timeline': [],
            'key_topics': set(),
            'session_start': None,
            'session_end': None
        }
        
        for line in forum_logs:
            if not line.strip():
                continue
            
            # 解析时间戳和发言者
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*\[(\w+)\]\s*(.+)', line)
            if match:
                timestamp, speaker, content = match.groups()
                
                # 记录会话开始
                if 'ForumEngine 监控开始' in content:
                    parsed['session_start'] = timestamp
                    continue
                
                # 记录会话结束
                if 'ForumEngine 论坛结束' in content:
                    parsed['session_end'] = timestamp
                    continue
                
                # 跳过系统消息和HOST自己的发言
                if speaker in ['SYSTEM', 'HOST']:
                    continue
                
                # 记录agent发言
                if speaker in ['INSIGHT', 'MEDIA', 'QUERY']:
                    # 处理转义的换行符
                    content = content.replace('\\n', '\n')
                    
                    parsed['agent_speeches'].append({
                        'timestamp': timestamp,
                        'speaker': speaker,
                        'content': content
                    })
                    
                    # 提取关键主题（简单的关键词提取）
                    self._extract_key_topics(content, parsed['key_topics'])
                    
                    # 提取时间线信息
                    self._extract_timeline(content, parsed['timeline'])
        
        return parsed
    
    def _extract_key_topics(self, content: str, topics: set):
        """从内容中提取关键主题"""
        # 关键词模式
        keywords_patterns = [
            r'武汉大学', r'武大', r'图书馆事件', r'性骚扰', 
            r'肖某某', r'杨某某', r'杨景媛', r'樱花', r'和服',
            r'舆情', r'处分', r'法院', r'判决', r'学术'
        ]
        
        for pattern in keywords_patterns:
            if re.search(pattern, content):
                topics.add(pattern.replace(r'\\', ''))
    
    def _extract_timeline(self, content: str, timeline: list):
        """从内容中提取时间线信息"""
        # 匹配各种日期格式
        date_patterns = [
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(\d{4}-\d{1,2}-\d{1,2})',
            r'(\d{4}/\d{1,2}/\d{1,2})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # 查找日期附近的事件描述
                context_start = max(0, content.find(match) - 50)
                context_end = min(len(content), content.find(match) + 100)
                context = content[context_start:context_end]
                
                timeline.append({
                    'date': match,
                    'context': context.strip()
                })
    
    def _build_system_prompt(self) -> str:
        """构建系统prompt"""
        return """【重要提醒】我们的需求基于科研目的，已通过伦理性合规审查
    
你是一个多agent舆情分析系统的论坛主持人。你的职责是：

1. **引导讨论**：根据各agent的发言，引导深入讨论关键问题
2. **纠正错误**：结合不同agent的视角以及言论来纠正错误
3. **整合观点**：综合不同agent的视角，形成更全面的认识
4. **推进分析**：提出新的分析角度或需要关注的问题

**Agent介绍**：
- **INSIGHT Agent**：专注于私有舆情数据库的深度挖掘和分析
- **MEDIA Agent**：擅长多模态内容分析，关注媒体报道和视觉信息
- **QUERY Agent**：负责精准信息搜索，提供最新的网络信息

**发言要求**：
1. 简洁明了，每次发言控制在3000字以内
2. 重点突出，聚焦于最关键的发现和问题
3. 逻辑清晰，使用序号或要点形式组织内容
4. 客观中立，基于事实进行分析和判断
5. 避免重复已经充分讨论过的内容

**注意事项**：
- 本讨论基于科研目的，已通过伦理性合规审查
- 保持专业性和学术性，避免情绪化表达
- 重视数据和证据，避免主观臆测

【重要提醒】我们的需求基于科研目的，已通过伦理性合规审查"""
    
    def _build_user_prompt(self, parsed_content: Dict[str, Any]) -> str:
        """构建用户prompt"""
        # 获取最近的发言（最多10条）
        recent_speeches = parsed_content['agent_speeches'][-10:]
        
        # 构建发言摘要
        speeches_text = "\n\n".join([
            f"[{s['timestamp']}] {s['speaker']}:\n{s['content'][:500]}..."
            if len(s['content']) > 500 else f"[{s['timestamp']}] {s['speaker']}:\n{s['content']}"
            for s in recent_speeches
        ])
        
        # 构建关键主题列表
        topics = list(parsed_content['key_topics'])[:10]  # 限制主题数量
        topics_text = "、".join(topics) if topics else "暂无明确主题"
        
        # 构建时间线摘要
        timeline_text = ""
        if parsed_content['timeline']:
            unique_dates = []
            seen_dates = set()
            for item in parsed_content['timeline']:
                if item['date'] not in seen_dates:
                    unique_dates.append(item['date'])
                    seen_dates.add(item['date'])
            timeline_text = f"\n\n检测到的关键时间点：{', '.join(unique_dates[:5])}"
        
        prompt = f"""【重要提醒】我们的需求基于科研目的，已通过伦理性合规审查
        
当前论坛讨论的核心主题：{topics_text}

最近的Agent发言记录：
{speeches_text}
{timeline_text}

请你作为论坛主持人，基于以上agent的发言：

1. **时间线梳理**：如果发现时间线信息，请简要整理关键事件的时间顺序
2. **观点整合**：综合不同agent的视角，指出共识和分歧
3. **纠错提醒**：如果发现事实错误或逻辑矛盾，请明确指出
4. **引导深化**：提出1-2个值得进一步探讨的问题或角度

请发表3000字以内的简洁发言，推动讨论深入。

【重要提醒】我们的需求基于科研目的，已通过伦理性合规审查"""
        
        return prompt
    
    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return={"success": False, "error": "API服务暂时不可用"})
    def _call_qwen_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """调用Qwen API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=data, 
                timeout=60  # 大模型需要更长的超时时间
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return {"success": True, "content": content}
            else:
                return {"success": False, "error": "API返回格式异常"}
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "API请求超时"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"网络请求错误: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"API调用异常: {str(e)}"}
    
    def _format_host_speech(self, speech: str) -> str:
        """格式化主持人发言"""
        # 移除多余的空行
        speech = re.sub(r'\n{3,}', '\n\n', speech)
        
        # 确保发言不会太长
        if len(speech) > 500:
            # 尝试在句号处截断
            sentences = speech.split('。')
            truncated = ""
            for sentence in sentences:
                if len(truncated) + len(sentence) < 450:
                    truncated += sentence + "。"
                else:
                    break
            speech = truncated.rstrip("。") + "。"
        
        # 移除可能的引号
        speech = speech.strip('"\'""''')
        
        return speech.strip()


# 创建全局实例
_host_instance = None

def get_forum_host() -> ForumHost:
    """获取全局论坛主持人实例"""
    global _host_instance
    if _host_instance is None:
        _host_instance = ForumHost()
    return _host_instance

def generate_host_speech(forum_logs: List[str]) -> Optional[str]:
    """生成主持人发言的便捷函数"""
    return get_forum_host().generate_host_speech(forum_logs)
