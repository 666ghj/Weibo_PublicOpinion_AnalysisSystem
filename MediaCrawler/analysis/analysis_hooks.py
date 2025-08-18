# -*- coding: utf-8 -*-
"""
分析钩子模块
在数据存储时自动触发分析任务
"""

import asyncio
import logging
from typing import Dict, Any, List
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class AnalysisHooks:
    """分析钩子管理器"""
    
    def __init__(self):
        self.enabled = True
        self.auto_sentiment = True
        self.auto_topic = False  # 话题检测比较耗时，默认关闭
        self.batch_size = 10  # 批处理大小
        self.pending_posts = []
        self.pending_comments = []
        
        # 分析结果存储路径
        self.results_dir = Path("data/analysis_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def enable_auto_analysis(self, sentiment: bool = True, topic: bool = False):
        """启用自动分析"""
        self.auto_sentiment = sentiment
        self.auto_topic = topic
        logger.info(f"Auto analysis enabled - Sentiment: {sentiment}, Topic: {topic}")
    
    def disable_auto_analysis(self):
        """禁用自动分析"""
        self.enabled = False
        logger.info("Auto analysis disabled")
    
    async def on_post_stored(self, post_data: Dict[str, Any]):
        """帖子存储后的钩子"""
        if not self.enabled:
            return
        
        try:
            self.pending_posts.append(post_data)
            
            # 当达到批处理大小时，触发分析
            if len(self.pending_posts) >= self.batch_size:
                await self._process_pending_posts()
                
        except Exception as e:
            logger.error(f"Error in post storage hook: {e}")
    
    async def on_comment_stored(self, comment_data: Dict[str, Any]):
        """评论存储后的钩子"""
        if not self.enabled:
            return
        
        try:
            self.pending_comments.append(comment_data)
            
            # 当达到批处理大小时，触发分析
            if len(self.pending_comments) >= self.batch_size:
                await self._process_pending_comments()
                
        except Exception as e:
            logger.error(f"Error in comment storage hook: {e}")
    
    async def _process_pending_posts(self):
        """处理待分析的帖子"""
        if not self.pending_posts:
            return
        
        posts_to_process = self.pending_posts.copy()
        self.pending_posts.clear()
        
        logger.info(f"Processing {len(posts_to_process)} posts for analysis")
        
        try:
            # 导入分析服务
            from .sentiment_analyzer import sentiment_service
            from .topic_detector import topic_service
            
            results = {
                "timestamp": asyncio.get_event_loop().time(),
                "posts_count": len(posts_to_process),
                "sentiment_results": [],
                "topic_results": {}
            }
            
            # 情感分析
            if self.auto_sentiment:
                try:
                    for post in posts_to_process:
                        sentiment_result = await sentiment_service.analyze_post_sentiment(post)
                        results["sentiment_results"].append(sentiment_result)
                        
                        # 保存单个结果
                        await self._save_sentiment_result(sentiment_result, "post")
                        
                except Exception as e:
                    logger.error(f"Sentiment analysis failed for posts: {e}")
            
            # 话题检测
            if self.auto_topic:
                try:
                    topic_result = await topic_service.detect_topics_from_posts(posts_to_process)
                    results["topic_results"] = topic_result
                    
                    # 保存话题结果
                    await self._save_topic_result(topic_result, "posts")
                    
                except Exception as e:
                    logger.error(f"Topic detection failed for posts: {e}")
            
            # 保存批处理结果
            await self._save_batch_result(results, "posts")
            
        except ImportError:
            logger.warning("Analysis modules not available, skipping auto analysis")
        except Exception as e:
            logger.error(f"Error processing pending posts: {e}")
    
    async def _process_pending_comments(self):
        """处理待分析的评论"""
        if not self.pending_comments:
            return
        
        comments_to_process = self.pending_comments.copy()
        self.pending_comments.clear()
        
        logger.info(f"Processing {len(comments_to_process)} comments for analysis")
        
        try:
            # 导入分析服务
            from .sentiment_analyzer import sentiment_service
            from .topic_detector import topic_service
            
            results = {
                "timestamp": asyncio.get_event_loop().time(),
                "comments_count": len(comments_to_process),
                "sentiment_results": [],
                "topic_results": {}
            }
            
            # 情感分析
            if self.auto_sentiment:
                try:
                    for comment in comments_to_process:
                        sentiment_result = await sentiment_service.analyze_comment_sentiment(comment)
                        results["sentiment_results"].append(sentiment_result)
                        
                        # 保存单个结果
                        await self._save_sentiment_result(sentiment_result, "comment")
                        
                except Exception as e:
                    logger.error(f"Sentiment analysis failed for comments: {e}")
            
            # 话题检测（仅在评论数量足够时进行）
            if self.auto_topic and len(comments_to_process) >= 20:
                try:
                    topic_result = await topic_service.detect_topics_from_comments(comments_to_process)
                    results["topic_results"] = topic_result
                    
                    # 保存话题结果
                    await self._save_topic_result(topic_result, "comments")
                    
                except Exception as e:
                    logger.error(f"Topic detection failed for comments: {e}")
            
            # 保存批处理结果
            await self._save_batch_result(results, "comments")
            
        except ImportError:
            logger.warning("Analysis modules not available, skipping auto analysis")
        except Exception as e:
            logger.error(f"Error processing pending comments: {e}")
    
    async def _save_sentiment_result(self, result: Dict[str, Any], data_type: str):
        """保存情感分析结果"""
        try:
            sentiment_file = self.results_dir / f"sentiment_{data_type}_results.jsonl"
            
            # 追加写入JSONL格式
            with open(sentiment_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save sentiment result: {e}")
    
    async def _save_topic_result(self, result: Dict[str, Any], data_type: str):
        """保存话题检测结果"""
        try:
            topic_file = self.results_dir / f"topic_{data_type}_results.jsonl"
            
            # 追加写入JSONL格式
            with open(topic_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save topic result: {e}")
    
    async def _save_batch_result(self, result: Dict[str, Any], data_type: str):
        """保存批处理结果"""
        try:
            batch_file = self.results_dir / f"batch_{data_type}_results.jsonl"
            
            # 追加写入JSONL格式
            with open(batch_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save batch result: {e}")
    
    async def flush_pending(self):
        """强制处理所有待分析的数据"""
        if self.pending_posts:
            await self._process_pending_posts()
        
        if self.pending_comments:
            await self._process_pending_comments()
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        stats = {
            "enabled": self.enabled,
            "auto_sentiment": self.auto_sentiment,
            "auto_topic": self.auto_topic,
            "pending_posts": len(self.pending_posts),
            "pending_comments": len(self.pending_comments),
            "results_files": []
        }
        
        # 检查结果文件
        for file_path in self.results_dir.glob("*.jsonl"):
            try:
                file_size = file_path.stat().st_size
                line_count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                
                stats["results_files"].append({
                    "name": file_path.name,
                    "size": file_size,
                    "records": line_count
                })
            except Exception:
                pass
        
        return stats

# 全局分析钩子实例
analysis_hooks = AnalysisHooks()
