from utils.getPublicData import getAllCommentsData,getAllArticleData
from datetime import datetime
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
commentsList = getAllCommentsData()
articleList = getAllArticleData()

def getHomeTagsData():# 统计数据库中文章个数，最高点赞作者，发布文章最多的城市
    articleLenMax = len(articleList)
    likeCountMax = 0
    likeCountMaxAuthorName = ''
    cityDic = {}
    for article in articleList:
        if likeCountMax < int(article[1]):
            likeCountMax = int(article[1])
            likeCountMaxAuthorName = article[11]
        if article[4] != '无':
            if article[4] in cityDic.keys():
                cityDic[article[4]] += 1
            else:
                cityDic[article[4]] = 1
    cityDicSorted = list(sorted(cityDic.items(),key=lambda x:x[1],reverse=True))
    return articleLenMax,likeCountMaxAuthorName,cityDicSorted[0][0]

def getHomeCommentsLikeCountTopFore():# 获取评论中点赞最高的前四条评论
    return list(sorted(commentsList,key=lambda x:int(x[2]),reverse=True))[:4]

def getHomeArticleCreatedAtChart():# 根据日期分别计算该日期的文章数
    X = list(set([x[7] for x in articleList]))
    X = list(sorted(X,key=lambda x:datetime.strptime(x,'%Y-%m-%d').timestamp(),reverse=True))
    Y = [0 for x in range(len(X))]
    for article in articleList:
        for index,j in enumerate(X):# 返回索引和值
            if article[7] == j:
                Y[index] += 1
    return X,Y

def getHomeTypeChart():# 统计每种类型的文章数量
    typeDic = {}
    for article in articleList:
        if article[8] in typeDic.keys():
            typeDic[article[8]] += 1
        else:
            typeDic[article[8]] = 1
    resultData = []
    for key,value in typeDic.items():
        resultData.append({
            'name':key,
            'value':value
        })
    return resultData

def getHomeCommentCreatedChart():# 统计每天用户评论数量
    createAtDic = {}
    for comment in commentsList:
        if comment[1] in createAtDic.keys():
            createAtDic[comment[1]] += 1
        else:
            createAtDic[comment[1]] = 1
    resultData = []
    for key, value in createAtDic.items():
        resultData.append({
            'name': key,
            'value': value
        })
    return resultData

def stopWordList():
    return [line.strip() for line in open('./stopWords.txt',encoding='utf8').readlines()]

def getUserNameWordCloud():# 生成用户名字词云
    text = ''
    stopWords = stopWordList()
    for comment in commentsList:
        text += comment[5]
    cut = jieba.cut(text)
    newCut = []
    for word in cut:
        if word not in stopWords:newCut.append(word)
    string = ' '.join(newCut)
    wc = WordCloud(
        width=1000,
        height=600,
        background_color='#fff',
        colormap='Blues',
        font_path='STHUPO.TTF'
    )
    wc.generate_from_text(string)
    fig = plt.figure(1)
    plt.imshow(wc)
    plt.axis('off')
    plt.savefig('./static/authorNameCloud.jpg',dpi=500)

