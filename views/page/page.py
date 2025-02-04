from flask import Flask, session, render_template, redirect, Blueprint, request
from utils.mynlp import SnowNLP
from utils.getHomePageData import *
from utils.getHotWordPageData import *
from utils.getTableData import *
from utils.getPublicData import getAllHotWords, getAllTopics, getArticleByType, getArticleById
from utils.getEchartsData import *
from utils.getTopicPageData import *
from utils.yuqingpredict import *
from utils.logger import app_logger as logging
import torch
from model_pro.MHA import MultiHeadAttentionLayer
from model_pro.classifier import FinalClassifier
from model_pro.BERT_CTM import BERT_CTM_Model

pb = Blueprint('page',
               __name__,
               url_prefix='/page',
               template_folder='templates')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型（全局变量，避免重复加载）
model_save_path = 'model_pro/final_model.pt'
bert_model_path = 'model_pro/bert_model'
ctm_tokenizer_path = 'model_pro/sentence_bert_model'

try:
    classifier_model = torch.load(model_save_path, map_location=device)
    classifier_model.eval()
    attention_model = MultiHeadAttentionLayer(embed_size=768, num_heads=8)
    attention_model.to(device)
    attention_model.eval()
    bert_ctm_model = BERT_CTM_Model(
        bert_model_path=bert_model_path,
        ctm_tokenizer_path=ctm_tokenizer_path
    )
except Exception as e:
    print(f"模型加载失败: {e}")

def predict_sentiment(text):
    """使用改进版模型预测单个文本的情感"""
    try:
        # 获取文本嵌入
        embeddings = bert_ctm_model.get_bert_embeddings([text])
        
        # 转换为tensor
        batch_x = torch.tensor(embeddings, dtype=torch.float32).to(device)
        batch_x = torch.mean(batch_x, dim=1)
        
        with torch.no_grad():
            # 使用注意力机制
            attention_output = attention_model(batch_x, batch_x, batch_x)
            # 获取分类结果
            outputs = classifier_model(attention_output)
            outputs = torch.mean(outputs, dim=1)
            # 获取预测标签和概率
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item(), probabilities[0][predicted.item()].item()
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
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
    username = session.get('username')
    TopicList = getAllTopicData()
    defaultTopic = TopicList[0][0]
    if request.args.get('Topic'):
        defaultTopic = request.args.get('Topic')
    TopicLen = getTopicLen(defaultTopic)
    X, Y = getTopicCreatedAtandpredictData(defaultTopic)
    
    # 获取模型选择参数
    model_type = request.args.get('model', 'pro')  # 默认使用改进模型
    
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
            sentences = f"{sentences} (置信度: {confidence:.2f})"
        else:
            sentences = '预测失败'
    
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
