from utils.getPublicData import *
from utils.mynlp import SnowNLP
articleList = getAllArticleData()
commentList = getAllCommentsData()

def getTypeList():
    return list(set([x[8] for x in getAllArticleData()]))

def getArticleByType(type):
    articles = []
    for i in articleList:
        if i[8] == type:
            articles.append(i)
    return articles

def getArticleCharLikeCount(type):
    articles = getArticleByType(type)
    xData = ['0-100','100-1000','1000-5000','5000-15000','15000-30000','30000-50000','50000-~']
    yData = [0 for x in range(len(xData))]
    for article in articles:
        likeCount = int(article[1])
        if likeCount < 100:
            yData[0] += 1
        elif likeCount < 1000:
            yData[1] += 1
        elif likeCount < 5000:
            yData[2] += 1
        elif likeCount < 15000:
            yData[3] += 1
        elif likeCount < 30000:
            yData[4] += 1
        elif likeCount < 50000:
            yData[5] += 1
        elif likeCount >= 50000:
            yData[6] += 1
    return xData,yData

def getArticleCharCommentsLen(type):
    articles = getArticleByType(type)
    xData = ['0-100','100-500','500-1000','1000-1500','1500-3000','3000-5000','5000-10000','10000-15000','15000-~']
    yData = [0 for x in range(len(xData))]
    for article in articles:
        commentLen = int(article[2])
        if commentLen < 100:
            yData[0] += 1
        elif commentLen < 500:
            yData[1] += 1
        elif commentLen < 5000:
            yData[2] += 1
        elif commentLen < 1000:
            yData[3] += 1
        elif commentLen < 1500:
            yData[4] += 1
        elif commentLen < 3000:
            yData[5] += 1
        elif commentLen < 5000:
            yData[6] += 1
        elif commentLen < 10000:
            yData[7] += 1
        elif commentLen >= 15000:
            yData[8] += 1
    return xData,yData

def getArticleCharRepotsLen(type):
    articles = getArticleByType(type)
    xData = ['0-100','100-300','300-500','500-1000','1000-2000','2000-3000','3000-4000','4000-5000','5000-10000','10000-15000','15000-30000','30000-70000','70000-~']
    yData = [0 for x in range(len(xData))]
    for article in articles:
        repostsCount = int(article[3])
        if repostsCount < 100:
            yData[0] += 1
        elif repostsCount < 300:
            yData[1] += 1
        elif repostsCount < 500:
            yData[2] += 1
        elif repostsCount < 1000:
            yData[3] += 1
        elif repostsCount < 3000:
            yData[4] += 1
        elif repostsCount < 4000:
            yData[5] += 1
        elif repostsCount < 5000:
            yData[6] += 1
        elif repostsCount < 10000:
            yData[7] += 1
        elif repostsCount < 15000:
            yData[8] += 1
        elif repostsCount < 30000:
            yData[9] += 1
        elif repostsCount < 70000:
            yData[10] += 1
        elif repostsCount >= 70000:
            yData[11] += 1
    return xData,yData

def getIPCharByArticleRegion():
    articleRegionDic = {}
    for i in articleList:
        if i[4] != '无':
            if articleRegionDic.get(i[4],-1) == -1:
                articleRegionDic[i[4]] = 1
            else:
                articleRegionDic[i[4]] += 1
    resultData = []
    for key,value in articleRegionDic.items():
        resultData.append({
            'name':key,
            'value':value
        })
    return resultData

def getIPCharByCommentsRegion():
    commentRegionDic = {}
    for i in commentList:
        if i[3] != '无':
            if commentRegionDic.get(i[3],-1) == -1:
                commentRegionDic[i[3]] = 1
            else:
                commentRegionDic[i[3]] += 1
    resultData = []
    for key,value in commentRegionDic.items():
        resultData.append({
            'name':key,
            'value':value
        })
    return resultData

def getCommentCharDataOne():
    xData = []
    rangeNum = 20
    for item in range(1,100):
        xData.append(str(rangeNum * item) + '-' + str(rangeNum * (item + 1)))
    yData = [0 for x in range(len(xData))]
    for comment in commentList:
        for item in range(99):
            if int(comment[2]) < rangeNum * (item + 2):
                yData[item] += 1
                break
    return xData,yData

def getCommentCharDataTwo():
    genderDic = {}
    for i in commentList:
        if genderDic.get(i[6],-1) == -1:
            genderDic[i[6]] = 1
        else:
            genderDic[i[6]] += 1
    resultData = [{
        'name':x[0],
        'value':x[1]
    } for x in genderDic.items()]
    return resultData

def getYuQingCharDataOne():
    hotWordList = getAllHotWords()
    xData = ['正面','中性','负面']
    yData = [0,0,0]
    for word in hotWordList:
        emotionValue = SnowNLP(word[0]).sentiments
        if emotionValue > 0.4:
            yData[0] += 1
        elif emotionValue < 0.2:
            yData[2] += 1
        else:
            yData[1] += 1
    bieData = [{
        'name':x,
        'value':yData[index]
    } for index,x in enumerate(xData)]
    return xData,yData,bieData

def getYuQingCharDataTwo():
    xData = ['正面', '中性', '负面']
    bieData1 = [{
        'name':x,
        'value':0
    } for x in xData]
    bieData2 = [{
        'name': x,
        'value': 0
    } for x in xData]

    for comment in commentList:
        emotionValue = SnowNLP(comment[4]).sentiments
        if emotionValue > 0.4:
            bieData1[0]['value'] += 1
        elif emotionValue < 0.2:
            bieData1[2]['value'] += 1
        else:
            bieData1[1]['value'] += 1
    for artile in articleList:
        emotionValue = SnowNLP(artile[5]).sentiments
        if emotionValue > 0.4:
            bieData2[0]['value'] += 1
        elif emotionValue < 0.2:
            bieData2[2]['value'] += 1
        else:
            bieData2[1]['value'] += 1
    return bieData1,bieData2

def getYuQingCharDataThree():
    hotWordList = getAllHotWords()
    x1Data = []
    y1Data = []
    for i in hotWordList[:10]:
        x1Data.append(i[0])
        y1Data.append(int(i[1]))
    return x1Data,y1Data

