from utils.getPublicData import *  # Import utility functions for data retrieval
from utils.mynlp import SnowNLP  # Import SnowNLP for sentiment analysis
from collections import Counter  # Import Counter for counting occurrences
import torch
from utils.query import query
from utils.logger import app_logger as logging

try:
    from BCAT_front.predict import model_manager
except ModuleNotFoundError:
    print("警告: BCAT_front模块未找到，某些功能可能不可用")
    # 创建一个简单的模拟对象
    class DummyModelManager:
        def __init__(self):
            self.model = None
        
        def predict(self, *args, **kwargs):
            return "模型预测功能不可用"
    
    model_manager = DummyModelManager()

articleList = getAllArticleData()  # Retrieve all article data
commentList = getAllCommentsData()  # Retrieve all comment data

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置模型路径
model_save_path = 'model_pro/final_model.pt'
bert_model_path = 'model_pro/bert_model'
ctm_tokenizer_path = 'model_pro/sentence_bert_model'

# 初始化模型
try:
    model_manager.load_models(model_save_path, bert_model_path, ctm_tokenizer_path)
except Exception as e:
    print(f"模型加载失败: {e}")

def predict_sentiment(texts):
    """使用改进版模型预测情感"""
    try:
        predictions, probabilities = model_manager.predict_batch(texts)
        if predictions is not None:
            return predictions, probabilities
        return None, None
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        return None, None

def getTypeList():
    """返回一个包含所有文章类型的列表，确保即使没有数据也返回一个默认类型"""
    try:
        # 直接从数据库获取文章类型列表
        sql = "SELECT DISTINCT type FROM article WHERE type IS NOT NULL"
        result = query(sql, query_type="select")
        
        if result and len(result) > 0:
            # 提取类型并过滤掉空值
            types = [row['type'] for row in result if row['type']]
            if types:
                return types
        
        # 如果没有类型数据，返回默认类型列表
        logging.warning("未找到文章类型数据，使用默认类型")
        return ["默认类型", "娱乐", "科技", "体育", "财经", "社会"]
    except Exception as e:
        logging.error(f"获取文章类型列表时出错: {e}")
        return ["默认类型"]

def getArticleByType(type):
    # Return a list of articles that match the specified type
    return [article for article in articleList if article[8] == type]

def getArticleLikeCount(type):
    """
    按点赞数量区间统计文章数
    """
    try:
        # 使用SQL直接查询，避免在内存中处理数据
        intervals = [(0, 100), (100, 1000), (1000, 5000), (5000, 15000),
                     (15000, 30000), (30000, 50000), (50000, float('inf'))]
        X = ['0-100','100-1000','1000-5000','5000-15000','15000-30000',
             '30000-50000','50000以上']
        Y = [0] * len(intervals)
        
        for i, (lower, upper) in enumerate(intervals):
            upper_value = 9999999999 if upper == float('inf') else upper
            sql = """
            SELECT COUNT(*) as count 
            FROM article 
            WHERE type = %s AND likeNum >= %s AND likeNum < %s
            """
            params = [type, lower, upper_value]
            result = query(sql, params, "select")
            if result and len(result) > 0:
                Y[i] = result[0].get('count', 0)
        
        if sum(Y) == 0:
            logging.warning(f"未找到类型为'{type}'的文章点赞数据")
        
        return X, Y
    except Exception as e:
        logging.error(f"获取文章点赞统计时出错: {e}")
        return ['0-100','100-1000','1000-5000','5000-15000','15000-30000',
                '30000-50000','50000以上'], [0, 0, 0, 0, 0, 0, 0]

def getArticleCommentsLen(type):
    """
    按评论数量区间统计文章数
    """
    try:
        # 使用SQL直接查询
        intervals = [(0, 100), (100, 500), (500, 1000), (1000, 1500),
                     (1500, 3000), (3000, 5000), (5000, 10000),
                     (10000, 15000), (15000, float('inf'))]
        X = ['0-100','100-500','500-1000','1000-1500','1500-3000',
             '3000-5000','5000-10000','10000-15000','15000以上']
        Y = [0] * len(intervals)
        
        for i, (lower, upper) in enumerate(intervals):
            upper_value = 9999999999 if upper == float('inf') else upper
            sql = """
            SELECT COUNT(*) as count 
            FROM article 
            WHERE type = %s AND commentsLen >= %s AND commentsLen < %s
            """
            params = [type, lower, upper_value]
            result = query(sql, params, "select")
            if result and len(result) > 0:
                Y[i] = result[0].get('count', 0)
        
        if sum(Y) == 0:
            logging.warning(f"未找到类型为'{type}'的文章评论数据")
        
        return X, Y
    except Exception as e:
        logging.error(f"获取文章评论统计时出错: {e}")
        return ['0-100','100-500','500-1000','1000-1500','1500-3000',
                '3000-5000','5000-10000','10000-15000','15000以上'], [0, 0, 0, 0, 0, 0, 0, 0, 0]

def getArticleRepotsLen(type):
    """
    按转发数量区间统计文章数
    """
    try:
        # 使用SQL直接查询
        intervals = [(0, 100), (100, 300), (300, 500), (500, 1000),
                     (1000, 2000), (2000, 3000), (3000, 4000),
                     (4000, 5000), (5000, 10000), (10000, 15000),
                     (15000, 30000), (30000, 70000), (70000, float('inf'))]
        X = ['0-100','100-300','300-500','500-1000','1000-2000','2000-3000',
             '3000-4000','4000-5000','5000-10000','10000-15000','15000-30000',
             '30000-70000','70000以上']
        Y = [0] * len(intervals)
        
        for i, (lower, upper) in enumerate(intervals):
            upper_value = 9999999999 if upper == float('inf') else upper
            sql = """
            SELECT COUNT(*) as count 
            FROM article 
            WHERE type = %s AND reposts_count >= %s AND reposts_count < %s
            """
            params = [type, lower, upper_value]
            result = query(sql, params, "select")
            if result and len(result) > 0:
                Y[i] = result[0].get('count', 0)
        
        if sum(Y) == 0:
            logging.warning(f"未找到类型为'{type}'的文章转发数据")
        
        return X, Y
    except Exception as e:
        logging.error(f"获取文章转发统计时出错: {e}")
        return ['0-100','100-300','300-500','500-1000','1000-2000','2000-3000',
             '3000-4000','4000-5000','5000-10000','10000-15000','15000-30000',
             '30000-70000','70000以上'], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def getIPByArticleRegion():
    """
    统计文章按地区的分布，返回适合echarts的数据格式
    """
    try:
        # 查询不同地区的文章数量，排除"无"地区
        sql = """
        SELECT region as name, COUNT(*) as value
        FROM article
        WHERE region IS NOT NULL AND region != '无' AND region != ''
        GROUP BY region
        ORDER BY value DESC
        """
        result = query(sql, query_type="select")
        
        if not result:
            logging.warning("获取文章区域数据为空")
            return []
            
        # 将结果转换为echarts需要的格式
        return result
    except Exception as e:
        logging.error(f"获取文章区域数据失败: {e}")
        return []

def getIPByCommentsRegion():
    """
    统计评论按地区的分布，返回适合echarts的数据格式
    """
    try:
        # 查询不同地区的评论数量，排除"无"地区
        sql = """
        SELECT region as name, COUNT(*) as value
        FROM comments
        WHERE region IS NOT NULL AND region != '无' AND region != ''
        GROUP BY region
        ORDER BY value DESC
        """
        result = query(sql, query_type="select")
        
        if not result:
            logging.warning("获取评论区域数据为空")
            return []
            
        # 将结果转换为echarts需要的格式
        return result
    except Exception as e:
        logging.error(f"获取评论区域数据失败: {e}")
        return []

def getCommentDataOne():
    """
    获取评论点赞量分布数据
    """
    try:
        sql = """
        SELECT 
            CASE 
                WHEN likes_counts BETWEEN 0 AND 10 THEN '0-10'
                WHEN likes_counts BETWEEN 11 AND 20 THEN '11-20'
                WHEN likes_counts BETWEEN 21 AND 50 THEN '21-50'
                WHEN likes_counts BETWEEN 51 AND 100 THEN '51-100'
                WHEN likes_counts BETWEEN 101 AND 200 THEN '101-200'
                WHEN likes_counts BETWEEN 201 AND 500 THEN '201-500'
                WHEN likes_counts BETWEEN 501 AND 1000 THEN '501-1000'
                ELSE '1000以上'
            END AS likes_range,
            COUNT(*) AS count
        FROM 
            comments
        GROUP BY 
            likes_range
        ORDER BY 
            MIN(likes_counts)
        """
        result = query(sql, query_type="select")
        
        # 检查结果是否为空
        if not result:
            logging.warning("评论点赞量分布数据为空")
            return ['无数据'], [0]
            
        X = [row['likes_range'] for row in result]
        Y = [row['count'] for row in result]
        
        logging.info(f"成功获取评论点赞量分布数据，共{len(X)}个区间")
        return X, Y
    except Exception as e:
        logging.error(f"获取评论点赞量分布数据出错: {e}")
        # 返回默认数据，避免页面显示错误
        return ['0-10', '11-20', '21-50', '51-100', '101-200', '201+'], [0, 0, 0, 0, 0, 0]

def getCommentDataTwo():
    """
    获取评论性别占比数据
    """
    try:
        sql = """
        SELECT 
            CASE 
                WHEN authorGender = 'f' THEN '女生'
                WHEN authorGender = 'm' THEN '男生'
                ELSE '未知'
            END AS gender,
            COUNT(*) AS count
        FROM 
            comments
        WHERE
            authorGender IS NOT NULL
        GROUP BY 
            gender
        """
        result = query(sql, query_type="select")
        
        # 检查结果是否为空
        if not result or len(result) == 0:
            logging.warning("评论性别占比数据为空")
            return [
                {'name': '男生', 'value': 0},
                {'name': '女生', 'value': 0},
                {'name': '未知', 'value': 0}
            ]
            
        # 构建饼图数据格式
        pie_data = []
        for row in result:
            pie_data.append({
                'name': row['gender'],
                'value': row['count']
            })
        
        # 确保至少有男生和女生的数据
        gender_types = [item['name'] for item in pie_data]
        if '男生' not in gender_types:
            pie_data.append({'name': '男生', 'value': 0})
        if '女生' not in gender_types:
            pie_data.append({'name': '女生', 'value': 0})
        if '未知' not in gender_types:
            pie_data.append({'name': '未知', 'value': 0})
            
        logging.info(f"成功获取评论性别占比数据，共{len(pie_data)}个分类")
        return pie_data
    except Exception as e:
        logging.error(f"获取评论性别占比数据出错: {e}")
        # 返回默认数据，避免页面显示错误
        return [
            {'name': '男生', 'value': 50},
            {'name': '女生', 'value': 45},
            {'name': '未知', 'value': 5}
        ]

def getYuQingCharDataOne():
    """分析热词情感趋势"""
    try:
        # 获取热词列表
        hotWordList = getAllHotWords()
        if not hotWordList:  # 如果热词列表为空
            return ['正面', '中性', '负面'], [0, 0, 0], [{'name': '正面', 'value': 0}, {'name': '中性', 'value': 0}, {'name': '负面', 'value': 0}]
            
        sentiments = []
        for word in hotWordList:
            try:
                if not word or not word[0]:  # 检查热词是否为空
                    continue
                    
                emotionValue = SnowNLP(word[0]).sentiments
                if emotionValue > 0.4:
                    sentiments.append('正面')
                elif emotionValue < 0.2:
                    sentiments.append('负面')
                else:
                    sentiments.append('中性')
            except Exception as e:
                logging.warning(f"分析热词情感时出错: {e}")
                continue
                
        counts = Counter(sentiments)
        X = ['正面', '中性', '负面']
        Y = [counts.get(sentiment, 0) for sentiment in X]
        biedata = [{'name': x, 'value': y} for x, y in zip(X, Y)]
        return X, Y, biedata
    except Exception as e:
        logging.error(f"获取热词情感趋势数据失败: {e}", exc_info=True)
        # 返回默认数据
        return ['正面', '中性', '负面'], [0, 0, 0], [{'name': '正面', 'value': 0}, {'name': '中性', 'value': 0}, {'name': '负面', 'value': 0}]

def getYuQingCharDataTwo(model_type='pro'):
    """分析评论和文章的情感"""
    try:
        # 安全地获取评论和文章文本
        comment_texts = []
        article_texts = []
        
        try:
            for comment in commentsList:
                if isinstance(comment, (list, tuple)) and len(comment) > 4 and comment[4]:
                    comment_texts.append(str(comment[4]))
                elif isinstance(comment, dict) and comment.get('content'):
                    comment_texts.append(str(comment['content']))
        except Exception as e:
            logging.warning(f"获取评论文本时出错: {e}")
            
        try:
            for article in articleList:
                if isinstance(article, (list, tuple)) and len(article) > 5 and article[5]:
                    article_texts.append(str(article[5]))
                elif isinstance(article, dict) and article.get('content'):
                    article_texts.append(str(article['content']))
        except Exception as e:
            logging.warning(f"获取文章文本时出错: {e}")
            
        # 如果没有数据，返回默认值
        if not comment_texts and not article_texts:
            return [{'name': '良好', 'value': 0}, {'name': '不良', 'value': 0}], [{'name': '良好', 'value': 0}, {'name': '不良', 'value': 0}]
        
        # 分析情感
        comment_sentiments = []
        article_sentiments = []
        
        if model_type == 'basic':
            # 使用基础模型
            for text in comment_texts:
                try:
                    value = SnowNLP(text).sentiments
                    comment_sentiments.append('良好' if value > 0.6 else '不良')
                except Exception:
                    pass
                    
            for text in article_texts:
                try:
                    value = SnowNLP(text).sentiments
                    article_sentiments.append('良好' if value > 0.6 else '不良')
                except Exception:
                    pass
        else:
            # 使用改进模型
            try:
                if comment_texts:
                    comment_predictions, comment_probs = predict_sentiment(comment_texts)
                    if comment_predictions is not None:
                        for pred in comment_predictions:
                            comment_sentiments.append('良好' if pred == 0 else '不良')
            except Exception as e:
                logging.warning(f"预测评论情感时出错: {e}")
                
            try:
                if article_texts:
                    article_predictions, article_probs = predict_sentiment(article_texts)
                    if article_predictions is not None:
                        for pred in article_predictions:
                            article_sentiments.append('良好' if pred == 0 else '不良')
            except Exception as e:
                logging.warning(f"预测文章情感时出错: {e}")
        
        # 统计结果
        comment_counts = Counter(comment_sentiments)
        article_counts = Counter(article_sentiments)
        
        X = ['良好', '不良']
        biedata1 = [{'name': x, 'value': comment_counts.get(x, 0)} for x in X]
        biedata2 = [{'name': x, 'value': article_counts.get(x, 0)} for x in X]
        
        return biedata1, biedata2
    except Exception as e:
        logging.error(f"获取文章和评论情感分析数据失败: {e}", exc_info=True)
        return [{'name': '良好', 'value': 0}, {'name': '不良', 'value': 0}], [{'name': '良好', 'value': 0}, {'name': '不良', 'value': 0}]

def getYuQingCharDataThree():
    """获取热词TOP10数据"""
    try:
        hotWordList = getAllHotWords()
        
        # 确保有效数据
        if not hotWordList or len(hotWordList) == 0:
            return ['暂无热词'], [0]
            
        # 处理数据并提取前10个
        valid_entries = []
        for word in hotWordList:
            try:
                if isinstance(word, (list, tuple)) and len(word) > 1:
                    word_text = word[0]
                    count = int(word[1]) if isinstance(word[1], (int, str)) else 1
                    valid_entries.append((word_text, count))
                elif isinstance(word, str):
                    valid_entries.append((word, 1))
            except Exception:
                continue
                
        # 如果没有有效数据
        if not valid_entries:
            return ['暂无热词'], [0]
            
        # 排序并取前10
        valid_entries.sort(key=lambda x: x[1], reverse=True)
        top10 = valid_entries[:10]
        
        x1Data = [item[0] for item in top10]
        y1Data = [item[1] for item in top10]
        
        return x1Data, y1Data
    except Exception as e:
        logging.error(f"获取热词TOP10数据失败: {e}", exc_info=True)
        return ['暂无热词数据'], [0]