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

def getArticleLikeCount(type):
    articles = getArticleByType(type)
    X = ['0-100','100-1000','1000-5000','5000-15000','15000-30000','30000-50000','50000-~']
    Y = [0 for x in range(len(X))]
    for article in articles:
        likeCount = int(article[1])
        if likeCount < 100:
            Y[0] += 1
        elif likeCount < 1000:
            Y[1] += 1
        elif likeCount < 5000:
            Y[2] += 1
        elif likeCount < 15000:
            Y[3] += 1
        elif likeCount < 30000:
            Y[4] += 1
        elif likeCount < 50000:
            Y[5] += 1
        elif likeCount >= 50000:
            Y[6] += 1
    return X,Y

def getArticleCommentsLen(type):
    articles = getArticleByType(type)
    X = ['0-100','100-500','500-1000','1000-1500','1500-3000','3000-5000','5000-10000','10000-15000','15000-~']
    Y = [0 for x in range(len(X))]
    for article in articles:
        commentLen = int(article[2])
        if commentLen < 100:
            Y[0] += 1
        elif commentLen < 500:
            Y[1] += 1
        elif commentLen < 5000:
            Y[2] += 1
        elif commentLen < 1000:
            Y[3] += 1
        elif commentLen < 1500:
            Y[4] += 1
        elif commentLen < 3000:
            Y[5] += 1
        elif commentLen < 5000:
            Y[6] += 1
        elif commentLen < 10000:
            Y[7] += 1
        elif commentLen >= 15000:
            Y[8] += 1
    return X,Y

def getArticleRepotsLen(type):
    articles = getArticleByType(type)
    X = ['0-100','100-300','300-500','500-1000','1000-2000','2000-3000','3000-4000','4000-5000','5000-10000','10000-15000','15000-30000','30000-70000','70000-~']
    Y = [0 for x in range(len(X))]
    for article in articles:
        repostsCount = int(article[3])
        if repostsCount < 100:
            Y[0] += 1
        elif repostsCount < 300:
            Y[1] += 1
        elif repostsCount < 500:
            Y[2] += 1
        elif repostsCount < 1000:
            Y[3] += 1
        elif repostsCount < 3000:
            Y[4] += 1
        elif repostsCount < 4000:
            Y[5] += 1
        elif repostsCount < 5000:
            Y[6] += 1
        elif repostsCount < 10000:
            Y[7] += 1
        elif repostsCount < 15000:
            Y[8] += 1
        elif repostsCount < 30000:
            Y[9] += 1
        elif repostsCount < 70000:
            Y[10] += 1
        elif repostsCount >= 70000:
            Y[11] += 1
    return X,Y

def getIPByArticleRegion():
    articleRegionDic = {}
    for i in articleList:
        if i[4] != '无':
            if i[4] in articleRegionDic.keys():
                articleRegionDic[i[4]] += 1
            else:
                articleRegionDic[i[4]] = 1
    resultData = []
    for key,value in articleRegionDic.items():
        resultData.append({
            'name':key,
            'value':value
        })
    return resultData

def getIPByCommentsRegion():
    commentRegionDic = {}
    for i in commentList:
        if i[3] != '无':
            if i[3] in commentRegionDic.keys():
                commentRegionDic[i[3]] += 1
            else:
                commentRegionDic[i[3]] = 1
    resultData = []
    for key,value in commentRegionDic.items():
        resultData.append({
            'name':key,
            'value':value
        })
    return resultData

def getCommentDataOne():
    X = []
    rangeNum = 20
    for item in range(100):
        X.append(str(rangeNum * item) + '-' + str(rangeNum * (item + 1)))
    Y = [0 for x in range(len(X))]
    for comment in commentList:
        for item in range(100):
            if int(comment[2]) < rangeNum * (item + 1):
                Y[item] += 1
                break
    return X,Y

def getCommentDataTwo():
    genderDic = {}
    for i in commentList:
        if i[6] in genderDic.keys():
            genderDic[i[6]] += 1
        else:
            genderDic[i[6]] = 1
    resultData = [{
        'name':x[0],
        'value':x[1]
    } for x in genderDic.items()]
    return resultData

def getYuQingCharDataOne():
    hotWordList = getAllHotWords()
    X = ['正面','中性','负面']
    Y = [0,0,0]
    for word in hotWordList:
        emotionValue = SnowNLP(word[0]).sentiments
        if emotionValue > 0.4:
            Y[0] += 1
        elif emotionValue < 0.2:
            Y[2] += 1
        else:
            Y[1] += 1
    biedata = [{
        'name':x,
        'value':Y[index]
    } for index,x in enumerate(X)]
    return X,Y,biedata

def getYuQingCharDataTwo():
    X = ['正面', '中性', '负面']
    biedata1 = [{
        'name':x,
        'value':0
    } for x in X]
    biedata2 = [{
        'name': x,
        'value': 0
    } for x in X]

    for comment in commentList:
        emotionValue = SnowNLP(comment[4]).sentiments
        if emotionValue > 0.4:
            biedata1[0]['value'] += 1
        elif emotionValue < 0.2:
            biedata1[2]['value'] += 1
        else:
            biedata1[1]['value'] += 1
    for artile in articleList:
        emotionValue = SnowNLP(artile[5]).sentiments
        if emotionValue > 0.4:
            biedata2[0]['value'] += 1
        elif emotionValue < 0.2:
            biedata2[2]['value'] += 1
        else:
            biedata2[1]['value'] += 1
    return biedata1,biedata2

def getYuQingCharDataThree():
    hotWordList = getAllHotWords()
    x1Data = []
    y1Data = []
    for i in hotWordList[:10]:
        x1Data.append(i[0])
        y1Data.append(int(i[1]))
    return x1Data,y1Data

