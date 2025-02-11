from flask import Flask, session, render_template, redirect, Blueprint, request, jsonify
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
from models.ai_analysis import AIAnalysis
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import asyncio
import torch
from BCAT_front.predict import model_manager

pb = Blueprint('page',
               __name__,
               url_prefix='/page',
               template_folder='templates')

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
    logging.error(f"模型加载失败: {e}")

# 数据库配置
DATABASE_URL = "sqlite:///ai_analysis.db"
engine = create_engine(DATABASE_URL)
AIAnalysis.metadata.create_all(engine)

def predict_sentiment(text):
    """使用改进版模型预测单个文本的情感"""
    try:
        predictions, probabilities = model_manager.predict_batch([text])
        if predictions is not None and len(predictions) > 0:
            return predictions[0], probabilities[0][predictions[0]]
        return None, None
    except Exception as e:
        logging.error(f"预测过程中出现错误: {e}")
        return None, None

@pb.route('/home')
def home():
    username = session.get('username')
    articleLenMax, likeCountMaxAuthorName, cityMax = getHomeTagsData()
    commentsLikeCountTopFore = getHomeCommentsLikeCountTopFore()
    X, Y = getHomeArticleCreatedAtChart()
    typeChart = getHomeTypeChart()
    createAtChart = getHomeCommentCreatedChart()
    # getUserNameWordCloud()
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


@pb.route('/hotWord')
def hotWord():
    username = session.get('username')
    hotWordList = getAllHotWords()
    print(hotWordList)
    defaultHotWord = hotWordList[0][0]
    if request.args.get('hotWord'):
        defaultHotWord = request.args.get('hotWord')
    hotWordLen = getHotWordLen(defaultHotWord)
    X, Y = getHotWordPageCreatedAtCharData(defaultHotWord)
    sentences = ''
    value = SnowNLP(defaultHotWord).sentiments
    if value == 0.5:
        sentences = '中性'
    elif value > 0.5:
        sentences = '正面'
    elif value < 0.5:
        sentences = '负面'
    comments = getCommentFilterData(defaultHotWord)
    return render_template('hotWord.html',
                           username=username,
                           hotWordList=hotWordList,
                           defaultHotWord=defaultHotWord,
                           hotWordLen=hotWordLen,
                           sentences=sentences,
                           xData=X,
                           yData=Y,
                           comments=comments)


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
def tableData():
    username = session.get('username')
    defaultFlag = False
    if request.args.get('flag'): defaultFlag = True
    tableData = getTableDataList(defaultFlag)
    return render_template('tableData.html',
                           username=username,
                           tableData=tableData,
                           defaultFlag=defaultFlag)


@pb.route('/articleChar')
def articleChar():
    username = session.get('username')
    typeList = getTypeList()
    defaultType = typeList[0]
    if request.args.get('type'): defaultType = request.args.get('type')
    X, Y = getArticleLikeCount(defaultType)
    x1Data, y1Data = getArticleCommentsLen(defaultType)
    x2Data, y2Data = getArticleRepotsLen(defaultType)
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


@pb.route('/ipChar')
def ipChar():
    username = session.get('username')
    articleRegionData = getIPByArticleRegion()
    commentRegionData = getIPByCommentsRegion()
    return render_template('ipChar.html',
                           username=username,
                           articleRegionData=articleRegionData,
                           commentRegionData=commentRegionData)


@pb.route('/commentChar')
def commentChar():
    username = session.get('username')
    X, Y = getCommentDataOne()
    genderPieData = getCommentDataTwo()
    return render_template('commentChar.html',
                           username=username,
                           xData=X,
                           yData=Y,
                           genderPieData=genderPieData)


@pb.route('/yuqingChar')
def yuqingChar():
    username = session.get('username')
    # 获取模型选择参数
    model_type = request.args.get('model', 'pro')  # 默认使用改进模型
    
    X, Y, biedata = getYuQingCharDataOne()
    biedata1, biedata2 = getYuQingCharDataTwo(model_type)
    x1Data, y1Data = getYuQingCharDataThree()
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

@pb.route('/yuqingpredict')
def yuqingpredict():
    try:
        username = session.get('username')
        TopicList = getAllTopicData()
        defaultTopic = TopicList[0][0]
        if request.args.get('Topic'):
            defaultTopic = request.args.get('Topic')
        TopicLen = getTopicLen(defaultTopic)
        X, Y = getTopicCreatedAtandpredictData(defaultTopic)
        
        # 获取模型选择参数
        model_type = request.args.get('model', 'pro')  # 默认使用改进模型
        
        # 尝试从缓存获取预测结果
        cache_key = f"{defaultTopic}_{model_type}"
        cached_result = prediction_cache.get(cache_key)
        
        if cached_result is not None:
            sentences = cached_result
        else:
            if model_type == 'basic':
                # 使用基础模型（SnowNLP）
                value = SnowNLP(defaultTopic).sentiments
                if value == 0.5:
                    sentences = '中性'
                elif value > 0.5:
                    sentences = '正面'
                elif value < 0.5:
                    sentences = '负面'
            else:
                # 使用改进模型
                predicted_label, confidence = predict_sentiment(defaultTopic)
                if predicted_label is not None:
                    sentences = '良好' if predicted_label == 0 else '不良'
                    sentences = f"{sentences} (置信度: {confidence:.2%})"
                else:
                    sentences = '预测失败，请稍后重试'
                    logging.error(f"预测失败，话题: {defaultTopic}")
            
            # 将结果存入缓存
            prediction_cache.set(cache_key, sentences)
        
        comments = getCommentFilterDataTopic(defaultTopic)
        return render_template('yuqingpredict.html',
                               username=username,
                               hotWordList=TopicList,
                               defaultHotWord=defaultTopic,
                               hotWordLen=TopicLen,
                               sentences=sentences,
                               xData=X,
                               yData=Y,
                               comments=comments,
                               model_type=model_type)
    except Exception as e:
        logging.error(f"舆情预测页面渲染失败: {e}")
        return render_template('error.html', error_message="加载舆情预测页面失败，请稍后重试")


@pb.route('/articleCloud')
def articleCloud():
    username = session.get('username')
    return render_template('articleContentCloud.html', username=username)


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
def articleChar(id):
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
async def analyze_messages():
    try:
        # 获取请求参数
        data = request.get_json()
        batch_size = data.get('batch_size', 50)
        model_type = data.get('model_type', 'gpt-3.5-turbo')
        analysis_depth = data.get('analysis_depth', 'standard')
        
        # 获取最近的消息
        messages = getRecentMessages(batch_size)
        if not messages:
            return jsonify({
                'success': False,
                'error': '没有找到需要分析的消息'
            }), 404
        
        # 调用AI进行分析
        analysis_results = await ai_analyzer.analyze_messages(
            messages=messages,
            batch_size=batch_size,
            model_type=model_type,
            analysis_depth=analysis_depth
        )
        
        if not analysis_results:
            return jsonify({
                'success': False,
                'error': '分析过程中出现错误'
            }), 500
        
        # 保存到数据库
        with Session(engine) as session:
            for result in analysis_results:
                analysis = AIAnalysis(
                    message_id=result['message_id'],
                    sentiment=result['sentiment'],
                    sentiment_score=float(result['sentiment_score']),
                    keywords=result['keywords'],
                    key_points=result['key_points'],
                    influence_analysis=result['influence_analysis'],
                    risk_level=result['risk_level']
                )
                session.add(analysis)
            session.commit()
        
        # 格式化结果用于显示
        display_results = [
            ai_analyzer.format_analysis_for_display(result)
            for result in analysis_results
        ]
        
        return jsonify({
            'success': True,
            'data': display_results,
            'meta': {
                'total_messages': len(messages),
                'analyzed_messages': len(analysis_results),
                'batch_size': batch_size,
                'model_type': model_type,
                'analysis_depth': analysis_depth
            }
        })
    
    except Exception as e:
        logging.error(f"AI分析过程出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@pb.route('/api/get_analysis/<int:message_id>')
def get_message_analysis(message_id):
    """获取特定消息的分析结果"""
    try:
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
    """获取最近的消息"""
    # 这里需要根据你的数据库结构实现具体的查询逻辑
    messages = []
    try:
        # 示例查询逻辑
        with Session(engine) as session:
            results = session.execute(
                """
                SELECT id, content 
                FROM comments 
                ORDER BY created_at DESC 
                LIMIT :limit
                """,
                {'limit': limit}
            ).fetchall()
            
            messages = [
                {'id': row[0], 'content': row[1]}
                for row in results
            ]
    except Exception as e:
        logging.error(f"获取最近消息时出错: {e}")
    
    return messages
