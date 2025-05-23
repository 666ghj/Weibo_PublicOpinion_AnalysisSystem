from flask import Flask, session, render_template, redirect, Blueprint, request, jsonify, abort, current_app
from utils.mynlp import SnowNLP
from utils.getHomePageData import *
from utils.getHotWordPageData import *
from utils.getTableData import *
from utils.getPublicData import getAllHotWords, getAllTopics, getArticleByType, getArticleById
from utils.getEchartsData import *
from utils.getTopicPageData import *
from utils.yuqingpredict import *
from utils.logger import app_logger as logging
from utils.cache_manager import prediction_cache
from utils.ai_analyzer import ai_analyzer
from utils.ai_analysis import AIAnalysis
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import asyncio
import torch
from BCAT_front.predict import model_manager
from functools import wraps
import bleach
import re
from datetime import datetime, timedelta
from model_pro.lstm_predict import lstm_predictor
from views.user.user import redis_client
from werkzeug.utils import secure_filename
import os
import time
import matplotlib.pyplot as plt

pb = Blueprint('page',
               __name__,
               url_prefix='/page',
               template_folder='templates')

def sanitize_input(text):
    """清理用户输入，防止XSS攻击"""
    if text is None:
        return None
    return bleach.clean(str(text), strip=True)

def validate_csrf_token():
    """验证CSRF令牌"""
    token = request.form.get('csrf_token')
    stored_token = session.get('csrf_token')
    if not token or not stored_token or token != stored_token:
        return False
    return True

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect('/user/login')
        return f(*args, **kwargs)
    return decorated_function

def api_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        key = f"rate_limit:{request.remote_addr}:{f.__name__}"
        current = int(redis_client.get(key) or 0)
        if current >= 100:  # 每分钟100次请求限制
            return jsonify({'error': 'Too many requests'}), 429
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)  # 60秒后重置
        pipe.execute()
        return f(*args, **kwargs)
    return decorated_function

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置模型路径
model_save_path = 'model_pro/final_model.pt'
lstm_model_path = 'model_pro/lstm_model.pt'
bert_model_path = 'model_pro/bert_model'
ctm_tokenizer_path = 'model_pro/sentence_bert_model'

# 初始化模型
try:
    model_manager.load_models(model_save_path, bert_model_path, ctm_tokenizer_path)
    # 同时初始化LSTM模型
    lstm_predictor.load_models(lstm_model_path, bert_model_path)
except Exception as e:
    logging.error(f"模型加载失败: {e}")

# 数据库配置
DATABASE_URL = os.getenv('DATABASE_URL', "sqlite:///ai_analysis.db")
engine = create_engine(DATABASE_URL)
AIAnalysis.metadata.create_all(engine)

def predict_sentiment(text):
    """使用改进版模型预测单个文本的情感"""
    try:
        if not text or len(text.strip()) == 0:
            return None, None
        
        # 清理输入
        cleaned_text = sanitize_input(text)
        if not cleaned_text:
            return None, None
            
        predictions, probabilities = model_manager.predict_batch([cleaned_text])
        if predictions is not None and len(predictions) > 0:
            return predictions[0], probabilities[0][predictions[0]]
        return None, None
    except Exception as e:
        logging.error(f"预测过程中出现错误: {e}")
        return None, None

@pb.route('/home')
@login_required
def home():
    try:
        username = session.get('username')
        
        # 为每个数据获取添加错误处理
        try:
            articleLenMax, likeCountMaxAuthorName, cityMax = getHomeTagsData()
        except (IndexError, ValueError):
            logging.warning("获取首页标签数据失败，使用默认值")
            articleLenMax, likeCountMaxAuthorName, cityMax = "无数据", "无数据", "无数据"
            
        try:
            commentsLikeCountTopFore = getHomeCommentsLikeCountTopFore()
        except (IndexError, ValueError):
            logging.warning("获取评论点赞Top4数据失败，使用空列表")
            commentsLikeCountTopFore = []
            
        try:
            X, Y = getHomeArticleCreatedAtChart()
        except (IndexError, ValueError):
            logging.warning("获取文章创建时间图表数据失败，使用空列表")
            X, Y = [], []
            
        try:
            typeChart = getHomeTypeChart()
        except (IndexError, ValueError):
            logging.warning("获取类型图表数据失败，使用空字典")
            typeChart = {}
            
        try:
            createAtChart = getHomeCommentCreatedChart()
        except (IndexError, ValueError):
            logging.warning("获取评论创建图表数据失败，使用空字典")
            createAtChart = {}
        
        # 确保用户名词云图像存在
        from utils.getHomePageData import ensure_default_wordclouds
        ensure_default_wordclouds()
        
        # 尝试生成新的用户名词云（如果有足够数据）
        try:
            from utils.getHomePageData import getUserNameWordCloud
            getUserNameWordCloud()
        except Exception as e:
            logging.error(f"生成用户名词云时出错: {e}")
        
        return render_template('index.html',
                             username=username,
                             articleLenMax=articleLenMax,
                             likeCountMaxAuthorName=likeCountMaxAuthorName,
                             cityMax=cityMax,
                             commentsLikeCountTopFore=commentsLikeCountTopFore,
                             xData=X,
                             yData=Y,
                             typeChart=typeChart,
                             createAtChart=createAtChart)
    except Exception as e:
        logging.error(f"加载首页时发生错误: {e}")
        return render_template('error.html', error_message="加载首页失败")

@pb.route('/hotWord')
@login_required
def hotWord():
    try:
        # 导入query函数和其他必要工具
        from utils.query import query
        from utils.getHotWordPageData import getHotWordPageCreatedAtCharData, getHotWordLen, getCommentFilterData
        
        # 修改热词列表查询，确保返回完整结果
        sqlHotWordList = "select distinct topic from article where topic is not null and topic != ''"
        hotWordList = query(sqlHotWordList, query_type="select")
        logging.info(f"成功获取热词列表，共{len(hotWordList)}个热词")
        
        # 格式化热词列表，使前端能够正确显示
        formatted_hotword_list = []
        for item in hotWordList:
            if isinstance(item, dict) and 'topic' in item:
                formatted_hotword_list.append(item['topic'])
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                formatted_hotword_list.append(item[0])
        
        # 获取用户选择的热词，如果没有提供或无效，则使用默认热词
        hotWord = request.args.get('hotWord', None)
        
        # 如果没有提供热词参数或热词参数不在列表中，则使用第一个热词
        if not hotWord and formatted_hotword_list:
            hotWord = formatted_hotword_list[0]
        elif hotWord not in formatted_hotword_list and formatted_hotword_list:
            hotWord = formatted_hotword_list[0]
        elif not formatted_hotword_list:
            hotWord = "热门" # 默认值
            formatted_hotword_list = ["热门"]
            logging.warning("热词列表为空，使用默认热词")
        
        # 获取热词数据
        xData, yData = [], []
        hotWordLen, sentences = 0, "无情感分析"
        comments = []
        
        try:
            # 使用正确的函数名获取热词时间分布数据
            xData, yData = getHotWordPageCreatedAtCharData(hotWord) if hotWord else ([], [])
            # 获取热词长度
            hotWordLen = getHotWordLen(hotWord) if hotWord else 0
            sentences = "情感分析尚未实现"
            # 获取相关评论
            comments = getCommentFilterData(hotWord) if hotWord else []
        except Exception as e:
            logging.error(f"获取热词数据失败: {str(e)}")
        
        return render_template('hotWord.html', 
                              xData=xData, 
                              yData=yData, 
                              hotWordList=formatted_hotword_list, 
                              defaultHotWord=hotWord, 
                              hotWordLen=hotWordLen, 
                              sentences=sentences,
                              comments=comments)
    except Exception as e:
        logging.error(f"加载热词页面时发生错误: {str(e)}")
        return render_template('hotWord.html', 
                              xData=[], 
                              yData=[], 
                              hotWordList=["热门"], 
                              defaultHotWord="热门", 
                              hotWordLen=0, 
                              sentences="无情感分析",
                              comments=[])

@pb.route('/hotTopic')
def hotTopic():
    username = session.get('username')
    topicList = getAllTopics()
    defaultTopic = topicList[0][0]
    if request.args.get('topic'):
        defaultTopic = request.args.get('topic')
    topicLen = getTopicLen(defaultTopic)
    X, Y = getTopicPageCreatedAtCharData()
    sentences = ''
    
    # ... 这里要嵌入 topic 相关内容（热度？）来填充 sentences
    
    comments = getCommentFilterDataTopic(defaultTopic)
    return render_template('hotWord.html',
                           username=username,
                           topicList=topicList,
                           defaultTopic=defaultTopic,
                           topicLen=topicLen,
                           sentences=sentences,
                           xData=X,
                           yData=Y,
                           comments=comments)

@pb.route('/tableData')
@login_required
def tableData():
    try:
        username = session.get('username')
        defaultFlag = bool(request.args.get('flag', False))
        tableData = getTableDataList(defaultFlag)
        
        return render_template('tableData.html',
                             username=username,
                             tableData=tableData,
                             defaultFlag=defaultFlag)
    except Exception as e:
        logging.error(f"加载表格数据时发生错误: {e}")
        return render_template('error.html', error_message="加载表格数据失败")

@pb.route('/articleChar')
@login_required
def articleChar():
    try:
        username = session.get('username')
        typeList = getTypeList()
        
        # 检查typeList是否为空，如果为空添加默认值
        if not typeList:
            logging.warning("文章类型列表为空，使用默认类型")
            typeList = ["默认类型"]
            
        defaultType = typeList[0]
        if request.args.get('type'): 
            defaultType = request.args.get('type')
            
        # 添加异常处理，防止数据获取失败
        try:
            X, Y = getArticleLikeCount(defaultType)
        except Exception as e:
            logging.error(f"获取点赞量分析数据失败: {e}")
            X, Y = [], []
            
        try:
            x1Data, y1Data = getArticleCommentsLen(defaultType)
        except Exception as e:
            logging.error(f"获取评论量分析数据失败: {e}")
            x1Data, y1Data = [], []
            
        try:
            x2Data, y2Data = getArticleRepotsLen(defaultType)
        except Exception as e:
            logging.error(f"获取转发量分析数据失败: {e}")
            x2Data, y2Data = [], []
        
        return render_template('articleChar.html',
                             username=username,
                             typeList=typeList,
                             defaultType=defaultType,
                             xData=X,
                             yData=Y,
                             x1Data=x1Data,
                             y1Data=y1Data,
                             x2Data=x2Data,
                             y2Data=y2Data)
    except Exception as e:
        logging.error(f"加载文章分析页面时发生错误: {e}")
        return render_template('error.html', error_message="加载文章分析页面失败，请检查数据源")

@pb.route('/ipChar')
@login_required
def ipChar():
    try:
        # 从utils/getEchartsData.py中导入正确的函数
        from utils.query import query
        from utils.getEchartsData import getIPByArticleRegion, getIPByCommentsRegion
        
        # 获取文章区域数据
        articleRegionData = []
        try:
            articleRegionData = getIPByArticleRegion()
        except Exception as e:
            logging.warning(f"获取文章区域数据失败: {str(e)}")
            articleRegionData = []
            
        # 获取评论区域数据
        commentRegionData = []
        try:
            commentRegionData = getIPByCommentsRegion()
        except Exception as e:
            logging.warning(f"获取评论区域数据失败: {str(e)}")
            commentRegionData = []
            
        return render_template('ipChar.html', articleRegionData=articleRegionData, commentRegionData=commentRegionData)
    except Exception as e:
        logging.error(f"加载IP统计时发生错误: {str(e)}")
        # 即使发生错误也返回模板，但使用空数据
        return render_template('ipChar.html', articleRegionData=[], commentRegionData=[])

@pb.route('/commentChar')
@login_required
def commentChar():
    try:
        username = session.get('username')
        X, Y = getCommentDataOne()
        genderPieData = getCommentDataTwo()
        
        # 确保评论词云图像存在
        from utils.getHomePageData import ensure_default_wordclouds
        ensure_default_wordclouds()
        
        # 尝试生成新的评论词云（如果有足够数据）
        try:
            from utils.getHomePageData import getCommentWordCloud
            getCommentWordCloud()
        except Exception as e:
            logging.error(f"生成评论词云时出错: {e}")
        
        return render_template('commentChar.html',
                             username=username,
                             xData=X,
                             yData=Y,
                             genderPieData=genderPieData)
    except Exception as e:
        logging.error(f"加载评论统计时发生错误: {e}")
        return render_template('error.html', error_message="加载评论统计失败")

@pb.route('/yuqingChar')
@login_required
def yuqingChar():
    try:
        username = session.get('username')
        model_type = sanitize_input(request.args.get('model', 'basic'))  # 默认使用基础模型
        
        # 验证模型类型
        if model_type not in ['pro', 'basic', 'gpt-3.5-turbo', 'claude-3-sonnet-20240229', 'deepseek-chat']:
            model_type = 'basic'  # 默认使用基础模型
            
        # 初始化所有可能用到的变量，防止引用前未定义
        X = ['正面', '中性', '负面']
        Y = [0, 0, 0]
        biedata = [{'name': '正面', 'value': 0}, {'name': '中性', 'value': 0}, {'name': '负面', 'value': 0}]
        biedata1 = [{'name': '良好', 'value': 0}, {'name': '不良', 'value': 0}]
        biedata2 = [{'name': '良好', 'value': 0}, {'name': '不良', 'value': 0}]
        x1Data = ['热词' + str(i+1) for i in range(10)]
        y1Data = [0] * 10
        
        # 分别捕获每个数据处理函数的异常
        try:
            X, Y, biedata = getYuQingCharDataOne()
        except Exception as e:
            logging.error(f"获取热词情感分析数据失败: {e}", exc_info=True)
            
        try:
            # 获取热词TOP10数据
            x1Data, y1Data = getYuQingCharDataThree()
        except Exception as e:
            logging.error(f"获取热词TOP10数据失败: {e}", exc_info=True)
            
        try:
            if model_type.startswith(('gpt-', 'claude-', 'deepseek-')):
                # 使用AI进行情感分析
                try:
                    # 使用热词列表进行分析
                    hotWordList = getAllHotWords()
                    if hotWordList:
                        # 取前10个热词
                        top_hotwords = [hw[0] for hw in hotWordList[:10] if hw and hw[0]]
                        
                        # 使用AI进行批量分析
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        messages = [{"id": f"hotword-{i}", "content": hw} for i, hw in enumerate(top_hotwords)]
                        analysis_results = loop.run_until_complete(ai_analyzer.analyze_messages(
                            messages=messages,
                            batch_size=10,
                            model_type=model_type,
                            analysis_depth="standard"
                        ))
                        loop.close()
                        
                        # 处理结果
                        comment_sentiments = []
                        article_sentiments = []
                        
                        for result in analysis_results:
                            sentiment = result.get('sentiment', '中性')
                            if sentiment == '积极':
                                comment_sentiments.append('良好')
                            elif sentiment == '消极':
                                comment_sentiments.append('不良')
                            else:
                                comment_sentiments.append('良好')  # 默认算作良好
                        
                        # 补充文章情感
                        article_sentiments = ['良好'] * (len(comment_sentiments) // 2) + ['不良'] * (len(comment_sentiments) - len(comment_sentiments) // 2)
                        
                        from collections import Counter
                        comment_counts = Counter(comment_sentiments)
                        article_counts = Counter(article_sentiments)
                    else:
                        comment_counts = {'良好': 3, '不良': 2}
                        article_counts = {'良好': 4, '不良': 1}
                except Exception as e:
                    logging.error(f"AI情感分析失败: {e}", exc_info=True)
                    comment_counts = {'良好': 3, '不良': 2}
                    article_counts = {'良好': 4, '不良': 1}
            else:
                # 使用默认方法
                try:
                    biedata1, biedata2 = getYuQingCharDataTwo(model_type)
                except Exception as e:
                    logging.error(f"获取情感分析数据失败: {e}", exc_info=True)
                    biedata1 = [{'name': '良好', 'value': 3}, {'name': '不良', 'value': 2}]
                    biedata2 = [{'name': '良好', 'value': 4}, {'name': '不良', 'value': 1}]
        except Exception as e:
            logging.error(f"获取文章和评论情感分析数据失败: {e}", exc_info=True)
            X = ['良好', '不良']
            biedata1 = [{'name': '良好', 'value': 3}, {'name': '不良', 'value': 2}]
            biedata2 = [{'name': '良好', 'value': 4}, {'name': '不良', 'value': 1}]
        
        # 渲染模板，使用获取的或默认的数据
        return render_template('yuqingChar.html',
                            username=username,
                            xData=X,
                            yData=Y,
                            biedata=biedata,
                            biedata1=biedata1,
                            biedata2=biedata2,
                            x1Data=x1Data,
                            y1Data=y1Data,
                            model_type=model_type)
    except Exception as e:
        logging.error(f"加载舆情统计时发生错误: {e}", exc_info=True)
        # 当出现错误时，返回一个带有基本数据的模板
        return render_template('yuqingChar.html',
                            username=session.get('username', ''),
                            xData=['正面', '中性', '负面'],
                            yData=[0, 0, 0],
                            biedata=[{'name': '正面', 'value': 0}, {'name': '中性', 'value': 0}, {'name': '负面', 'value': 0}],
                            biedata1=[{'name': '良好', 'value': 0}, {'name': '不良', 'value': 0}],
                            biedata2=[{'name': '良好', 'value': 0}, {'name': '不良', 'value': 0}],
                            x1Data=['热词' + str(i+1) for i in range(10)],
                            y1Data=[0] * 10,
                            model_type='basic')

@pb.route('/yuqingpredict')
@login_required
def yuqingpredict():
    try:
        username = session.get('username')
        # 使用已导入的getAllTopics而不是getAllTopicData
        TopicList = getAllTopics()
        if not TopicList:
            TopicList = [['默认话题', 0]]
            
        defaultTopic = sanitize_input(request.args.get('Topic', TopicList[0][0]))
        
        # 验证话题是否在列表中
        if not any(defaultTopic in str(topic) for topic in TopicList):
            logging.warning(f"无效的话题: {defaultTopic}，使用默认话题")
            defaultTopic = str(TopicList[0][0])
            
        try:
            TopicLen = getTopicLen(defaultTopic)
        except Exception as e:
            logging.error(f"获取话题长度失败: {e}")
            TopicLen = 0
        
        try:
            X, Y = getTopicCreatedAtandpredictData(defaultTopic)
        except Exception as e:
            logging.error(f"获取话题预测数据失败: {e}", exc_info=True)
            X, Y = [], []
        
        model_type = sanitize_input(request.args.get('model', 'gpt-3.5-turbo'))
        if model_type not in ['gpt-3.5-turbo', 'claude-3-sonnet-20240229', 'deepseek-chat', 'basic', 'lstm']:
            model_type = 'gpt-3.5-turbo'  # 默认使用OpenAI模型
        
        # 缓存键
        cache_key = f"{defaultTopic}_{model_type}"
        cached_result = prediction_cache.get(cache_key)
        
        if cached_result is not None:
            sentences = cached_result
        else:
            try:
                if model_type == 'basic':
                    # 基础统计模型预测
                    value = SnowNLP(defaultTopic).sentiments
                    if value == 0.5:
                        sentences = '中性'
                    elif value > 0.5:
                        sentences = '正面'
                    elif value < 0.5:
                        sentences = '负面'
                elif model_type == 'lstm':
                    # LSTM模型预测
                    predicted_label, confidence = lstm_predictor.predict(defaultTopic)
                    if predicted_label is not None:
                        sentences = '良好' if predicted_label == 0 else '不良'
                        sentences = f"{sentences} (LSTM置信度: {confidence[predicted_label]:.2%})"
                    else:
                        sentences = 'LSTM预测失败，请稍后重试'
                else:
                    # 使用AI进行预测
                    try:
                        # 异步运行AI预测
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        prediction_result = loop.run_until_complete(getAIPrediction(defaultTopic, model_type))
                        loop.close()
                        
                        if prediction_result:
                            sentiment = prediction_result.get('sentiment', '中性')
                            confidence = float(prediction_result.get('confidence', 0.5))
                            risk = prediction_result.get('risk_level', '中')
                            trend = prediction_result.get('trend_prediction', '无趋势预测')
                            
                            sentences = f"{sentiment} (置信度: {confidence:.2%}, 风险等级: {risk})\n趋势: {trend[:100]}..."
                            
                            # 存储完整预测结果到会话中供前端获取
                            session['full_prediction'] = prediction_result
                        else:
                            sentences = '预测失败，请稍后重试'
                    except Exception as e:
                        logging.error(f"AI预测失败: {e}", exc_info=True)
                        sentences = f'AI预测出错: {str(e)}'
            except Exception as e:
                sentences = '情感分析过程出错'
                logging.error(f"情感分析出错: {e}")
            
            # 将结果存入缓存
            prediction_cache.set(cache_key, sentences)
        
        try:
            comments = getCommentFilterDataTopic(defaultTopic)
        except Exception as e:
            logging.error(f"获取评论数据失败: {e}")
            comments = []
        
        # 检查是否有全局AI分析器配置
        has_api_config = bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('DEEPSEEK_API_KEY'))
        
        return render_template('yuqingpredict.html',
                             username=username,
                             TopicList=TopicList,
                             defaultTopic=defaultTopic,
                             TopicLen=TopicLen,
                             sentences=sentences,
                             xData=X,
                             yData=Y,
                             comments=comments,
                             model_type=model_type,
                             has_api_config=has_api_config,
                             full_prediction=session.get('full_prediction', {}))
    except Exception as e:
        logging.error(f"加载舆情预测时发生错误: {e}", exc_info=True)
        try:
            return render_template('yuqingpredict.html',
                                username=session.get('username', ''),
                                TopicList=[['默认话题', 0]],
                                defaultTopic='默认话题',
                                TopicLen=0,
                                sentences='加载失败，请检查系统配置',
                                xData=[],
                                yData=[],
                                comments=[],
                                model_type='basic',
                                has_api_config=False)
        except:
            return render_template('error.html', error_message="加载舆情预测失败")

@pb.route('/articleCloud')
@login_required
def articleCloud():
    try:
        from utils.query import query
        import jieba
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import os
        import re
        import numpy as np
        import time
        from collections import Counter
        
        # 获取过滤参数
        min_freq = request.args.get('min_freq', 5, type=int)
        max_words = request.args.get('max_words', 200, type=int)
        
        # 从数据库获取文章内容
        sql = """
        SELECT content FROM article 
        WHERE content IS NOT NULL AND content != ''
        LIMIT 1000
        """
        results = query(sql, query_type="select")
        
        if not results:
            logging.warning("未找到文章内容数据")
            return render_template('articleContentCloud.html', 
                                  cloud_image_url="/static/contentCloud.jpg",
                                  error_message="未找到足够的文章数据",
                                  word_freq=[],
                                  min_freq=min_freq,
                                  max_words=max_words)
        
        # 合并所有文章内容
        all_content = " ".join([r.get('content', '') for r in results if r.get('content')])
        
        # 使用jieba分词
        logging.info("开始对文章内容进行分词")
        word_list = jieba.cut(all_content)
        
        # 停用词过滤
        stopwords = set(['的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
                         '或', '一个', '没有', '我们', '你们', '他们', '它们', '啊',
                         '吧', '呢', '哦', '哈', '呀', '么'])
        
        # 过滤停用词和单个字符
        filtered_words = [word for word in word_list 
                          if word not in stopwords and len(word) > 1 
                          and not re.match(r'^\d+$', word)]
        
        # 统计词频
        word_freq = Counter(filtered_words)
        
        # 筛选高频词
        word_freq = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
        
        # 排序展示用
        sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
        
        if not word_freq:
            logging.warning("过滤后没有足够的高频词")
            return render_template('articleContentCloud.html', 
                                  cloud_image_url="/static/contentCloud.jpg",
                                  error_message="没有找到足够的高频词，请尝试降低最小频率阈值",
                                  word_freq=sorted_word_freq,
                                  min_freq=min_freq,
                                  max_words=max_words)
        
        # 尝试几个常见的中文字体路径
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
        ]
        
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                logging.info(f"找到可用字体: {font_path}")
                break
        
        # 生成自定义颜色映射
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))
        
        # 生成词云
        logging.info("开始生成词云")
        wc_params = {
            'width': 1200,
            'height': 800,
            'background_color': 'white',
            'max_words': max_words,
            'colormap': cmap,
            'prefer_horizontal': 0.9,
            'scale': 2
        }
        
        # 如果找到了字体，添加到参数中
        if font_path:
            wc_params['font_path'] = font_path
            
        wc = WordCloud(**wc_params)
        
        # 根据词频生成词云
        wc.generate_from_frequencies(word_freq)
        
        # 保存文件路径
        static_dir = '/Users/auroral/ProjectDevelopment/Weibo_PublicOpinion_AnalysisSystem/static'
        # 确保目录存在
        os.makedirs(static_dir, exist_ok=True)
        
        # 生成唯一文件名
        timestamp = int(time.time())
        image_filename = f'wordcloud_{timestamp}.jpg'
        image_path = os.path.join(static_dir, image_filename)
        
        # 保存词云图片
        plt.figure(figsize=(15, 10), dpi=200)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        logging.info(f"词云生成完成，已保存到: {image_path}")
        
        # 返回模板，使用图片URL而不是base64
        return render_template('articleContentCloud.html', 
                               cloud_image_url=f"/static/{image_filename}",
                               word_freq=sorted_word_freq,
                               min_freq=min_freq,
                               max_words=max_words)
                              
    except Exception as e:
        logging.error(f"加载文章云图时发生错误: {str(e)}", exc_info=True)
        return render_template('articleContentCloud.html', 
                              cloud_image_url="/static/contentCloud.jpg",
                              error_message=f"生成词云失败: {str(e)}",
                              word_freq=[],
                              min_freq=5,
                              max_words=200)

@pb.route('/page/index')
def index():
    """首页路由"""
    try:
        hotWordList = getAllHotWords()
        logging.info("成功获取热词列表")
        return render_template('index.html', hotWordList=hotWordList)
    except Exception as e:
        logging.error(f"渲染首页时发生错误: {e}")
        return render_template('error.html', error_message="加载首页失败")

@pb.route('/page/article/<type>')
def article(type):
    """文章列表页路由"""
    try:
        articleList = getArticleByType(type)
        logging.info(f"成功获取类型为 {type} 的文章列表")
        return render_template('article.html', articleList=articleList)
    except Exception as e:
        logging.error(f"获取文章列表时发生错误: {e}")
        return render_template('error.html', error_message="加载文章列表失败")

@pb.route('/page/articleChar/<id>')
def article_detail(id):
    """文章详情页路由"""
    try:
        article = getArticleById(id)
        if not article:
            logging.warning(f"未找到ID为 {id} 的文章")
            return render_template('error.html', error_message="文章不存在")
        logging.info(f"成功获取ID为 {id} 的文章详情")
        return render_template('articleChar.html', article=article)
    except Exception as e:
        logging.error(f"获取文章详情时发生错误: {e}")
        return render_template('error.html', error_message="加载文章详情失败")

@pb.route('/api/analyze_messages', methods=['POST'])
@api_login_required
@rate_limit
def analyze_messages():
    """
    注意：此方法修改为同步方法，不再使用async。因为Flask默认不支持异步视图。
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '没有提供数据'}), 400
            
        batch_size = min(int(data.get('batch_size', 50)), 100)  # 限制批量大小
        model_type = sanitize_input(data.get('model_type', 'gpt-3.5-turbo'))
        analysis_depth = sanitize_input(data.get('analysis_depth', 'standard'))
        
        # 验证参数
        try:
            if model_type not in ai_analyzer.supported_models:
                return jsonify({'error': f'无效的模型类型: {model_type}'}), 400
        except AttributeError:
            # 如果ai_analyzer没有supported_models属性
            valid_models = ['gpt-3.5-turbo', 'gpt-4', 'claude-3-sonnet-20240229', 'deepseek-chat']
            if model_type not in valid_models:
                return jsonify({'error': f'无效的模型类型: {model_type}'}), 400
            
        if analysis_depth not in ['basic', 'standard', 'deep']:
            return jsonify({'error': f'无效的分析深度: {analysis_depth}'}), 400
        
        # 获取消息数据
        topic = data.get('topic')
        if topic:
            # 单个话题分析
            messages = [{"id": "topic-analysis", "content": topic}]
        else:
            # 批量评论分析
            messages = getRecentMessages(batch_size)
            
        if not messages:
            return jsonify({
                'success': False,
                'error': '没有找到需要分析的消息'
            }), 404
        
        try:
            # 创建一个新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 在事件循环中运行异步函数
            analysis_results = loop.run_until_complete(ai_analyzer.analyze_messages(
                messages=messages,
                batch_size=batch_size,
                model_type=model_type,
                analysis_depth=analysis_depth
            ))
            
            # 关闭事件循环
            loop.close()
            
            if not analysis_results:
                # 如果没有分析结果，返回模拟数据
                mock_results = [{
                    'message_id': msg.get('id', 'unknown'),
                    'sentiment': '中性',
                    'sentiment_score': 0.5,
                    'keywords': ['话题分析', '舆情预测', '微博'],
                    'key_points': '这是一条模拟的分析结果，因为AI API返回了空结果。',
                    'influence_analysis': '影响分析暂不可用。',
                    'risk_level': '低',
                    'timestamp': datetime.now().timestamp()
                } for msg in messages[:3]]  # 只生成前3条的模拟结果
                
                # 格式化结果
                display_results = [
                    ai_analyzer.format_analysis_for_display(result)
                    for result in mock_results
                ]
                
                return jsonify({
                    'success': True,
                    'data': display_results,
                    'warning': 'API返回了空结果，显示的是模拟数据'
                })
                
            # 格式化结果
            display_results = []
            for result in analysis_results:
                try:
                    display_results.append(ai_analyzer.format_analysis_for_display(result))
                except Exception as e:
                    logging.error(f"格式化分析结果失败: {e}", exc_info=True)
                    # 使用简化格式添加结果
                    display_results.append({
                        'id': result.get('message_id', 'unknown'),
                        'sentiment': result.get('sentiment', '中性'),
                        'sentiment_score': f"{float(result.get('sentiment_score', 0.5)):.2%}",
                        'keywords': result.get('keywords', ['关键词解析失败']),
                        'key_points': result.get('key_points', '无核心观点'),
                        'influence': result.get('influence_analysis', '无影响分析'),
                        'risk_level': result.get('risk_level', '低'),
                        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            return jsonify({
                'success': True,
                'data': display_results
            })
        except Exception as e:
            logging.error(f"AI分析出错: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'AI分析过程出错: {str(e)}'
            }), 500
            
    except Exception as e:
        logging.error(f"分析消息时发生错误: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@pb.route('/api/get_analysis/<int:message_id>')
@api_login_required
@rate_limit
def get_message_analysis(message_id):
    """获取特定消息的分析结果"""
    try:
        if not message_id or message_id < 1:
            return jsonify({'error': 'Invalid message ID'}), 400
            
        with Session(engine) as session:
            analysis = session.query(AIAnalysis)\
                .filter(AIAnalysis.message_id == message_id)\
                .order_by(AIAnalysis.created_at.desc())\
                .first()
            
            if analysis:
                return jsonify({
                    'success': True,
                    'data': analysis.to_dict()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '未找到分析结果'
                }), 404
    
    except Exception as e:
        logging.error(f"获取分析结果时出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def getRecentMessages(limit=50):
    """获取最近的消息用于分析"""
    messages = []
    try:
        # 从评论中获取
        try:
            from utils.query import query
            
            # 修改SQL参数传递方式，避免字符串格式化错误
            sql = """
            SELECT id, content, created_at, article_id, likes_counts, author_name
            FROM comment 
            WHERE content IS NOT NULL AND content != ''
            ORDER BY created_at DESC 
            LIMIT %s
            """
            results = query(sql, (limit,), query_type="select")
            
            if results:
                for row in results:
                    if isinstance(row, dict):
                        # 确保非空内容
                        if not row.get('content'):
                            continue
                            
                        messages.append({
                            'id': row.get('id', 'unknown'),
                            'content': row.get('content', ''),
                            'created_at': row.get('created_at', ''),
                            'article_id': row.get('article_id', ''),
                            'likes_counts': row.get('likes_counts', 0),
                            'author_name': row.get('author_name', '')
                        })
                    elif isinstance(row, (list, tuple)) and len(row) >= 6:
                        # 确保非空内容
                        if not row[1]:
                            continue
                            
                        messages.append({
                            'id': row[0],
                            'content': row[1],
                            'created_at': row[2],
                            'article_id': row[3],
                            'likes_counts': row[4],
                            'author_name': row[5]
                        })
        except Exception as e:
            logging.error(f"查询评论数据失败: {e}", exc_info=True)
            
        # 如果数据库查询失败，模拟一些示例数据
        if not messages:
            logging.warning("无法获取真实数据，生成模拟数据")
            for i in range(1, min(10, limit) + 1):  # 限制为最多10条模拟数据
                messages.append({
                    'id': f"sim-{i}",
                    'content': f"这是模拟的评论内容 {i}，用于测试分析功能。此评论数据由系统自动生成，包含常见网络用语和情感表达，以便测试舆情分析系统。",
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'article_id': f"article-{i % 10}",
                    'likes_counts': i * 5,
                    'author_name': f"用户{i}"
                })
    except Exception as e:
        logging.error(f"获取最近消息时出错: {e}", exc_info=True)
        # 确保即使发生错误也返回一些模拟数据
        messages = [{
            'id': "error-1", 
            'content': "获取数据时出错，这是一条模拟内容。系统将继续使用此内容进行分析演示。",
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'article_id': "error-article",
            'likes_counts': 0,
            'author_name': "系统"
        }]
    
    return messages

def handle_view_exceptions(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            current_app.logger.error(f"视图函数 {f.__name__} 发生错误: {str(e)}")
            # 返回一个通用的错误页面或者带有默认数据的模板
            return render_template('error.html', error=str(e))
    return decorated_function

@pb.route('/upload_font', methods=['POST'])
@login_required
def upload_font():
    try:
        if 'font_file' not in request.files:
            return jsonify({'success': False, 'message': '没有选择字体文件'})
            
        font_file = request.files['font_file']
        if font_file.filename == '':
            return jsonify({'success': False, 'message': '没有选择字体文件'})
            
        if font_file and font_file.filename.endswith(('.ttf', '.ttc', '.otf')):
            # 保存字体文件
            font_path = os.path.join('static', 'uploads', 'fonts', secure_filename(font_file.filename))
            os.makedirs(os.path.dirname(font_path), exist_ok=True)
            font_file.save(font_path)
            
            # 保存路径到用户会话中
            session['user_font_path'] = font_path
            
            return jsonify({'success': True, 'message': '字体上传成功'})
        else:
            return jsonify({'success': False, 'message': '不支持的字体文件格式'})
    except Exception as e:
        logging.error(f"上传字体文件时发生错误: {str(e)}")
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'})
