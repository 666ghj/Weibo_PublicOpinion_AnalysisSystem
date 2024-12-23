from flask import Flask, session, render_template, redirect, Blueprint, request
from utils.mynlp import SnowNLP
from utils.getHomePageData import *
from utils.getHotWordPageData import *
from utils.getTableData import *
from utils.getPublicData import getAllHotWords, getAllTopics
from utils.getEchartsData import *
from utils.getTopicPageData import *
from utils.yuqingpredict import *
from utils.getPublicData import getAllHotWords

pb = Blueprint('page',
               __name__,
               url_prefix='/page',
               template_folder='templates')


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
    X, Y, biedata = getYuQingCharDataOne()
    biedata1, biedata2 = getYuQingCharDataTwo()
    x1Data, y1Data = getYuQingCharDataThree()
    return render_template('yuqingChar.html',
                           username=username,
                           xData=X,
                           yData=Y,
                           biedata=biedata,
                           biedata1=biedata1,
                           biedata2=biedata2,
                           x1Data=x1Data,
                           y1Data=y1Data)

@pb.route('/yuqingpredict')
def yuqingpredict():
    username = session.get('username')
    TopicList = getAllTopicData()
    defaultTopic = TopicList[0][0]
    if request.args.get('Topic'):
        defaultTopic = request.args.get('Topic')
    TopicLen = getTopicLen(defaultTopic)
    X, Y = getTopicCreatedAtandpredictData(defaultTopic)
    sentences = ''
    value = SnowNLP(defaultTopic).sentiments
    if value == 0.5:
        sentences = '中性'
    elif value > 0.5:
        sentences = '正面'
    elif value < 0.5:
        sentences = '负面'
    comments = getCommentFilterDataTopic(defaultTopic)
    return render_template('yuqingpredict.html',
                           username=username,
                           hotWordList=TopicList,
                           defaultHotWord=defaultTopic,
                           hotWordLen=TopicLen,
                           sentences=sentences,
                           xData=X,
                           yData=Y,
                           comments=comments)


@pb.route('/articleCloud')
def articleCloud():
    username = session.get('username')
    return render_template('articleContentCloud.html', username=username)
