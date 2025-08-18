# -*- coding: utf-8 -*-
"""
话题检测模块
支持BERTopic和其他话题检测模型
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import asyncio
import json
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class TopicDetector:
    """话题检测器基类"""
    
    def __init__(self, model_type: str = "bertopic"):
        self.model_type = model_type
        self.model = None
        self.is_loaded = False
        self.topics_cache = {}
    
    async def load_model(self):
        """加载模型"""
        try:
            if self.model_type == "bertopic":
                await self._load_bertopic_model()
            elif self.model_type == "keyword_extraction":
                await self._load_keyword_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.is_loaded = True
            logger.info(f"Topic detection model {self.model_type} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load topic detection model {self.model_type}: {e}")
            raise
    
    async def _load_bertopic_model(self):
        """加载BERTopic模型"""
        try:
            bertopic_path = project_root / "LLMTopicDetection_BERTopic"
            sys.path.insert(0, str(bertopic_path))
            
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            # 使用中文预训练模型
            embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            self.model = BERTopic(
                language="chinese",
                embedding_model=embedding_model,
                min_topic_size=5,
                nr_topics=20,
                verbose=False
            )
            
        except ImportError as e:
            logger.error(f"BERTopic dependencies not found: {e}")
            logger.info("Please install: pip install bertopic sentence-transformers")
            # 降级到关键词提取
            await self._load_keyword_model()
        except Exception as e:
            logger.error(f"Failed to load BERTopic model: {e}")
            # 降级到关键词提取
            await self._load_keyword_model()
    
    async def _load_keyword_model(self):
        """加载关键词提取模型（备用方案）"""
        try:
            import jieba
            import jieba.analyse
            
            # 设置jieba
            jieba.setLogLevel(logging.WARNING)
            
            # 加载停用词
            stopwords_path = project_root / "SensitiveStopWords" / "stopword.dic"
            self.stopwords = set()
            if stopwords_path.exists():
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    self.stopwords = set(line.strip() for line in f if line.strip())
            
            self.model = "keyword_extraction"
            self.model_type = "keyword_extraction"
            
        except ImportError:
            logger.error("jieba not found. Please install: pip install jieba")
            raise
    
    async def detect_topics(self, texts: List[str], num_topics: int = 10) -> Dict[str, Any]:
        """检测话题"""
        if not self.is_loaded:
            await self.load_model()
        
        try:
            if self.model_type == "bertopic":
                return await self._detect_bertopic(texts, num_topics)
            elif self.model_type == "keyword_extraction":
                return await self._detect_keywords(texts, num_topics)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Failed to detect topics: {e}")
            return {
                "topics": [],
                "topic_assignments": [],
                "error": str(e)
            }
    
    async def _detect_bertopic(self, texts: List[str], num_topics: int = 10) -> Dict[str, Any]:
        """使用BERTopic检测话题"""
        if len(texts) < 5:
            logger.warning("Too few texts for BERTopic, falling back to keyword extraction")
            return await self._detect_keywords(texts, num_topics)
        
        try:
            # 预处理文本
            processed_texts = [self._preprocess_text(text) for text in texts]
            processed_texts = [text for text in processed_texts if len(text) > 10]
            
            if len(processed_texts) < 5:
                return await self._detect_keywords(texts, num_topics)
            
            # 训练模型
            topics, probs = self.model.fit_transform(processed_texts)
            
            # 获取话题信息
            topic_info = self.model.get_topic_info()
            
            # 格式化结果
            result_topics = []
            for idx, row in topic_info.iterrows():
                if row['Topic'] != -1:  # 排除噪声话题
                    topic_words = self.model.get_topic(row['Topic'])
                    keywords = [word for word, score in topic_words[:10]]
                    
                    result_topics.append({
                        "topic_id": f"topic_{row['Topic']}",
                        "topic_name": f"话题{row['Topic']}",
                        "keywords": keywords,
                        "document_count": row['Count'],
                        "coherence_score": 0.0,  # BERTopic暂不提供
                        "description": f"包含关键词: {', '.join(keywords[:5])}"
                    })
            
            return {
                "topics": result_topics[:num_topics],
                "topic_assignments": topics.tolist(),
                "model_name": "bertopic",
                "model_version": "1.0"
            }
            
        except Exception as e:
            logger.error(f"BERTopic detection failed: {e}")
            return await self._detect_keywords(texts, num_topics)
    
    async def _detect_keywords(self, texts: List[str], num_topics: int = 10) -> Dict[str, Any]:
        """使用关键词提取检测话题"""
        import jieba.analyse
        from collections import Counter
        
        # 提取所有文本的关键词
        all_keywords = []
        text_keywords = []
        
        for text in texts:
            # 预处理
            processed_text = self._preprocess_text(text)
            
            # 提取关键词
            keywords = jieba.analyse.extract_tags(processed_text, topK=10, withWeight=True)
            keywords = [(word, weight) for word, weight in keywords if word not in self.stopwords and len(word) > 1]
            
            text_keywords.append([word for word, weight in keywords])
            all_keywords.extend([word for word, weight in keywords])
        
        # 统计关键词频率
        keyword_freq = Counter(all_keywords)
        top_keywords = keyword_freq.most_common(50)
        
        # 基于关键词共现创建话题
        topics = []
        used_keywords = set()
        
        for i in range(min(num_topics, len(top_keywords) // 3)):
            # 选择未使用的高频关键词作为话题核心
            core_keywords = []
            for keyword, freq in top_keywords:
                if keyword not in used_keywords and len(core_keywords) < 5:
                    core_keywords.append(keyword)
                    used_keywords.add(keyword)
            
            if core_keywords:
                topics.append({
                    "topic_id": f"topic_{i}",
                    "topic_name": f"话题{i+1}: {core_keywords[0]}",
                    "keywords": core_keywords,
                    "document_count": sum(1 for tk in text_keywords if any(k in tk for k in core_keywords)),
                    "coherence_score": 0.0,
                    "description": f"关键词聚类话题: {', '.join(core_keywords)}"
                })
        
        # 为每个文本分配话题
        topic_assignments = []
        for tk in text_keywords:
            best_topic = -1
            max_overlap = 0
            
            for i, topic in enumerate(topics):
                overlap = len(set(tk) & set(topic["keywords"]))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_topic = i
            
            topic_assignments.append(best_topic)
        
        return {
            "topics": topics,
            "topic_assignments": topic_assignments,
            "model_name": "keyword_extraction",
            "model_version": "1.0"
        }
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除@用户名
        text = re.sub(r'@[^\s]+', '', text)
        
        # 移除话题标签
        text = re.sub(r'#[^#]+#', '', text)
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class TopicDetectionService:
    """话题检测服务"""
    
    def __init__(self):
        self.detectors = {}
        self.default_detector = "bertopic"
    
    async def get_detector(self, model_type: str = None) -> TopicDetector:
        """获取检测器"""
        if model_type is None:
            model_type = self.default_detector
        
        if model_type not in self.detectors:
            self.detectors[model_type] = TopicDetector(model_type)
            await self.detectors[model_type].load_model()
        
        return self.detectors[model_type]
    
    async def detect_topics_from_posts(self, posts_data: List[Dict[str, Any]], model_type: str = None, num_topics: int = 10) -> Dict[str, Any]:
        """从帖子数据中检测话题"""
        detector = await self.get_detector(model_type)
        
        # 提取文本内容
        texts = []
        post_ids = []
        
        for post in posts_data:
            content = post.get("desc", "") or post.get("content", "")
            title = post.get("title", "")
            text = f"{title} {content}".strip()
            
            if text and len(text) > 10:
                texts.append(text)
                post_ids.append(post.get("note_id", ""))
        
        if not texts:
            return {
                "topics": [],
                "topic_assignments": [],
                "error": "No valid text content found"
            }
        
        # 检测话题
        result = await detector.detect_topics(texts, num_topics)
        
        # 添加帖子ID映射
        if "topic_assignments" in result:
            topic_post_mapping = {}
            for i, (post_id, topic_id) in enumerate(zip(post_ids, result["topic_assignments"])):
                if topic_id not in topic_post_mapping:
                    topic_post_mapping[topic_id] = []
                topic_post_mapping[topic_id].append(post_id)
            
            # 更新话题信息，添加相关帖子
            for topic in result.get("topics", []):
                topic_idx = int(topic["topic_id"].split("_")[1]) if "_" in topic["topic_id"] else -1
                topic["related_posts"] = topic_post_mapping.get(topic_idx, [])
        
        return result
    
    async def detect_topics_from_comments(self, comments_data: List[Dict[str, Any]], model_type: str = None, num_topics: int = 10) -> Dict[str, Any]:
        """从评论数据中检测话题"""
        detector = await self.get_detector(model_type)
        
        # 提取评论内容
        texts = []
        comment_ids = []
        
        for comment in comments_data:
            content = comment.get("content", "")
            if content and len(content) > 5:
                texts.append(content)
                comment_ids.append(comment.get("comment_id", ""))
        
        if not texts:
            return {
                "topics": [],
                "topic_assignments": [],
                "error": "No valid comment content found"
            }
        
        # 检测话题
        result = await detector.detect_topics(texts, num_topics)
        
        # 添加评论ID映射
        if "topic_assignments" in result:
            topic_comment_mapping = {}
            for i, (comment_id, topic_id) in enumerate(zip(comment_ids, result["topic_assignments"])):
                if topic_id not in topic_comment_mapping:
                    topic_comment_mapping[topic_id] = []
                topic_comment_mapping[topic_id].append(comment_id)
            
            # 更新话题信息，添加相关评论
            for topic in result.get("topics", []):
                topic_idx = int(topic["topic_id"].split("_")[1]) if "_" in topic["topic_id"] else -1
                topic["related_comments"] = topic_comment_mapping.get(topic_idx, [])
        
        return result

# 全局话题检测服务实例
topic_service = TopicDetectionService()
