# -*- coding: utf-8 -*-
"""
情感分析模块
支持多种情感分析模型
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import asyncio
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """情感分析器基类"""
    
    def __init__(self, model_type: str = "multilingual"):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    async def load_model(self):
        """加载模型"""
        try:
            if self.model_type == "multilingual":
                await self._load_multilingual_model()
            elif self.model_type == "machine_learning":
                await self._load_ml_models()
            elif self.model_type == "qwen":
                await self._load_qwen_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.is_loaded = True
            logger.info(f"Sentiment model {self.model_type} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model {self.model_type}: {e}")
            raise
    
    async def _load_multilingual_model(self):
        """加载多语言情感分析模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            model_name = "tabularisai/multilingual-sentiment-analysis"
            model_dir = project_root / "WeiboMultilingualSentiment" / "model"
            
            # 尝试从本地加载，如果不存在则从HuggingFace下载
            if model_dir.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # 保存到本地
                model_dir.mkdir(parents=True, exist_ok=True)
                self.tokenizer.save_pretrained(str(model_dir))
                self.model.save_pretrained(str(model_dir))
            
            # 设置为评估模式
            self.model.eval()
            
        except ImportError:
            logger.error("transformers library not found. Please install: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load multilingual model: {e}")
            raise
    
    async def _load_ml_models(self):
        """加载机器学习模型"""
        try:
            ml_path = project_root / "WeiboSentiment_MachineLearning"
            sys.path.insert(0, str(ml_path))
            
            # 导入预测器
            from predict import SentimentPredictor
            self.model = SentimentPredictor()
            
        except ImportError as e:
            logger.error(f"Failed to import ML models: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            raise
    
    async def _load_qwen_model(self):
        """加载Qwen模型"""
        try:
            qwen_path = project_root / "WeiboSentiment_SmallQwen"
            sys.path.insert(0, str(qwen_path))
            
            from qwen3_lora_universal import Qwen3LoRAModel
            self.model = Qwen3LoRAModel(model_size="0.5B")
            
        except ImportError as e:
            logger.error(f"Failed to import Qwen model: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """分析文本情感"""
        if not self.is_loaded:
            await self.load_model()
        
        try:
            if self.model_type == "multilingual":
                return await self._analyze_multilingual(text)
            elif self.model_type == "machine_learning":
                return await self._analyze_ml(text)
            elif self.model_type == "qwen":
                return await self._analyze_qwen(text)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return {
                "sentiment_label": 2,  # 中性
                "confidence": 0.0,
                "sentiment_scores": {},
                "error": str(e)
            }
    
    async def _analyze_multilingual(self, text: str) -> Dict[str, Any]:
        """使用多语言模型分析情感"""
        import torch
        
        # 预处理文本
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # 情感标签映射
        sentiment_map = {
            0: "非常负面",
            1: "负面", 
            2: "中性",
            3: "正面",
            4: "非常正面"
        }
        
        # 各类别得分
        sentiment_scores = {
            "very_negative": predictions[0][0].item(),
            "negative": predictions[0][1].item(),
            "neutral": predictions[0][2].item(),
            "positive": predictions[0][3].item(),
            "very_positive": predictions[0][4].item()
        }
        
        return {
            "sentiment_label": predicted_class,
            "sentiment_text": sentiment_map[predicted_class],
            "confidence": confidence,
            "sentiment_scores": sentiment_scores,
            "model_name": "multilingual-sentiment-analysis",
            "model_version": "1.0"
        }
    
    async def _analyze_ml(self, text: str) -> Dict[str, Any]:
        """使用机器学习模型分析情感"""
        # 使用BERT模型作为默认
        result = self.model.predict_single(text, model_type='bert')
        
        return {
            "sentiment_label": result["prediction"],
            "sentiment_text": "正面" if result["prediction"] == 1 else "负面",
            "confidence": result["confidence"],
            "sentiment_scores": result.get("probabilities", {}),
            "model_name": "bert-sentiment",
            "model_version": "1.0"
        }
    
    async def _analyze_qwen(self, text: str) -> Dict[str, Any]:
        """使用Qwen模型分析情感"""
        result = self.model.predict_single(text)
        
        return {
            "sentiment_label": result,
            "sentiment_text": "正面" if result == 1 else "负面",
            "confidence": 0.8,  # Qwen模型暂时没有置信度
            "sentiment_scores": {},
            "model_name": "qwen-sentiment",
            "model_version": "1.0"
        }
    
    async def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量分析情感"""
        results = []
        for text in texts:
            result = await self.analyze_sentiment(text)
            results.append(result)
        return results

class SentimentAnalysisService:
    """情感分析服务"""
    
    def __init__(self):
        self.analyzers = {}
        self.default_analyzer = "multilingual"
    
    async def get_analyzer(self, model_type: str = None) -> SentimentAnalyzer:
        """获取分析器"""
        if model_type is None:
            model_type = self.default_analyzer
        
        if model_type not in self.analyzers:
            self.analyzers[model_type] = SentimentAnalyzer(model_type)
            await self.analyzers[model_type].load_model()
        
        return self.analyzers[model_type]
    
    async def analyze_post_sentiment(self, post_data: Dict[str, Any], model_type: str = None) -> Dict[str, Any]:
        """分析帖子情感"""
        analyzer = await self.get_analyzer(model_type)
        
        # 提取文本内容
        content = post_data.get("desc", "") or post_data.get("content", "")
        title = post_data.get("title", "")
        text = f"{title} {content}".strip()
        
        if not text:
            return {
                "post_id": post_data.get("note_id", ""),
                "error": "No text content found"
            }
        
        # 分析情感
        result = await analyzer.analyze_sentiment(text)
        result["post_id"] = post_data.get("note_id", "")
        result["text_content"] = text[:200] + "..." if len(text) > 200 else text
        
        return result
    
    async def analyze_comment_sentiment(self, comment_data: Dict[str, Any], model_type: str = None) -> Dict[str, Any]:
        """分析评论情感"""
        analyzer = await self.get_analyzer(model_type)
        
        content = comment_data.get("content", "")
        if not content:
            return {
                "comment_id": comment_data.get("comment_id", ""),
                "error": "No comment content found"
            }
        
        # 分析情感
        result = await analyzer.analyze_sentiment(content)
        result["comment_id"] = comment_data.get("comment_id", "")
        result["post_id"] = comment_data.get("note_id", "")
        result["text_content"] = content[:200] + "..." if len(content) > 200 else content
        
        return result

# 全局情感分析服务实例
sentiment_service = SentimentAnalysisService()
