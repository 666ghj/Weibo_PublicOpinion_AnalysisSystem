from flask import Flask, session, render_template, redirect, Blueprint, request
from utils.mynlp import SnowNLP
from utils.getHomePageData import *
from utils.getHotWordPageData import *
from utils.getTableData import *
from utils.getPublicData import getAllHotWords, getAllTopics
from utils.getEchartsData import *
from utils.getTopicPageData import *

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
                           X=X,
                           Y=Y,
                           typeChart=typeChart,
                           createAtChart=createAtChart)


@pb.route('/hotWord')
def hotWord():
    username = session.get('username')
    hotWordList = getAllHotWords()
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
                           X=X,
                           Y=Y,
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
                           X=X,
                           Y=Y,
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
                           X=X,
                           Y=Y,
                           x1Data=x1Data,
                           y1Data=y1Data,
                           x2Data=x2Data,
                           y2Data=y2Data)


@pb.route('/ipChar')
def ipChar():
    username = session.get('username')
    articleRegionData = getIPCharByArticleRegion()
    commentRegionData = getIPCharByCommentsRegion()
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
                           X=X,
                           Y=Y,
                           genderPieData=genderPieData)


@pb.route('/yuqingChar')
def yuqingChar():
    username = session.get('username')
    X, Y, finaldata = getYuQingCharDataOne()
    finaldata1, finaldata2 = getYuQingCharDataTwo()
    x1Data, y1Data = getYuQingCharDataThree()
    return render_template('yuqingChar.html',
                           username=username,
                           X=X,
                           Y=Y,
                           finaldata=finaldata,
                           finaldata1=finaldata1,
                           finaldata2=finaldata2,
                           x1Data=x1Data,
                           y1Data=y1Data)


@pb.route('/articleCloud')
def articleCloud():
    username = session.get('username')
    return render_template('articleContentCloud.html', username=username)
