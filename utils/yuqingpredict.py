from utils.getPublicData import *
from utils.predict import predict_future_values  # 统计学预测方法
from utils.query import query
import csv
import os
import datetime
import pandas as pd
from utils.ai_analyzer import ai_analyzer  # 导入AI分析器
import asyncio
import json
from utils.logger import app_logger as logging

async def getAIPrediction(topic, model_type="gpt-3.5-turbo"):
    """
    使用AI进行舆情预测分析
    
    参数:
        topic: 话题内容
        model_type: AI模型类型
    
    返回:
        预测结果字典
    """
    try:
        # 构建提示词
        prompt = f"""
        请对以下话题进行舆情预测分析:
        
        话题: {topic}
        
        请预测该话题的未来发展趋势、可能的舆论方向、风险程度和建议对策。
        按以下JSON格式返回:
        
        {{
          "sentiment": "情感倾向（积极/中性/消极）",
          "sentiment_score": "情感分数(0-1)",
          "trend_prediction": "未来发展趋势预测",
          "risk_level": "风险等级(低/中/高)",
          "key_factors": ["关键影响因素1", "关键影响因素2", "..."],
          "suggested_actions": ["建议对策1", "建议对策2", "..."],
          "confidence": "预测置信度(0-1)"
        }}
        """
        
        # 构建消息格式
        messages = [{"id": "topic-prediction", "content": prompt}]
        
        # 调用AI分析器
        results = await ai_analyzer.analyze_messages(
            messages=messages,
            batch_size=1,
            model_type=model_type,
            analysis_depth="deep"
        )
        
        if not results or len(results) == 0:
            logging.error("AI分析器返回空结果")
            return None
            
        # 提取预测结果
        result = results[0]
        
        # 确保返回正确的JSON格式
        try:
            if isinstance(result, str):
                prediction = json.loads(result)
            else:
                # 使用关键点作为预测结果
                prediction = {
                    "sentiment": result.get('sentiment', '中性'),
                    "sentiment_score": result.get('sentiment_score', 0.5),
                    "trend_prediction": result.get('key_points', '无法预测趋势'),
                    "risk_level": result.get('risk_level', '中'),
                    "key_factors": result.get('keywords', []),
                    "suggested_actions": ["保持关注", "定期评估", "适时回应"],
                    "confidence": 0.7
                }
                
            return prediction
        except Exception as e:
            logging.error(f"解析AI预测结果时出错: {e}")
            return None
    except Exception as e:
        logging.error(f"AI预测过程中出错: {e}")
        return None

def getTopicCreatedAtandpredictData(topic):
    """
    获取话题时间分布数据和预测数据 - 保留原有统计预测功能
    """
    try:
        # 查询文章数据
        article_sql = """
        SELECT created_at, topic FROM article 
        WHERE topic = %s AND created_at IS NOT NULL
        """
        article_results = query(article_sql, (topic,), query_type="select")
        
        # 查询评论数据
        comment_sql = """
        SELECT created_at, topic FROM comment 
        WHERE topic = %s AND created_at IS NOT NULL
        """
        comment_results = query(comment_sql, (topic,), query_type="select")
        
        # 合并时间分布数据
        createdAt = {}
        
        # 处理文章数据
        if article_results:
            for row in article_results:
                date = row.get('created_at') if isinstance(row, dict) else row[0]
                if date:
                    date_str = date.split(' ')[0] if ' ' in date else date  # 只保留日期部分
                    if date_str in createdAt:
                        createdAt[date_str] += 1
                    else:
                        createdAt[date_str] = 1
        
        # 处理评论数据
        if comment_results:
            for row in comment_results:
                date = row.get('created_at') if isinstance(row, dict) else row[0]
                if date:
                    date_str = date.split(' ')[0] if ' ' in date else date  # 只保留日期部分
                    if date_str in createdAt:
                        createdAt[date_str] += 1
                    else:
                        createdAt[date_str] = 1
                        
        # 如果没有数据，添加一些示例数据以避免空图表
        if not createdAt:
            today = datetime.datetime.now()
            for i in range(5):
                date = (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                createdAt[date] = i + 1

        # 使用统计预测方法
        predictions = predict_future_values(createdAt, forecast_days=5)

        # 合并历史数据和预测数据
        combined_data = {**createdAt, **predictions}
        combined_data = {k: combined_data[k] for k in sorted(combined_data, key=lambda date: datetime.datetime.strptime(date, "%Y-%m-%d"))}

        return list(combined_data.keys()), list(combined_data.values())
        
    except Exception as e:
        logging.error(f"获取话题时间分布和预测数据时出错: {e}", exc_info=True)
        # 返回示例数据作为fallback
        today = datetime.datetime.now()
        dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10, -5, -1)]
        values = list(range(1, 11)) + [None] * 5  # 最后5个为预测值，设为None
        
        # 使用简单算法为预测值生成一些有意义的数据
        last_values = values[5:10]
        avg = sum(last_values) / len(last_values)
        trend = (last_values[-1] - last_values[0]) / 4  # 计算趋势
        
        # 根据趋势生成预测值
        for i in range(5):
            pred_value = max(1, int(avg + trend * (i+1)))
            values[10+i] = pred_value
            
        return dates, values
