from utils.getPublicData import *
articleList = getAllArticleData()
commentList = getAllCommentsData()

def getTypeList():# 返回爬取到的所有文章的类型（已去重）
    return list(set([x[8] for x in getAllArticleData()]))

def getArticleByType(type):# 根据特定文章类型筛选文章
    articles = []
    for i in articleList:
        if i[8] == type:
            articles.append(i)
    return articles

def getArticleCharLikeCount(type):# 统计特定类型文章的点赞数分布
    articles = getArticleByType(type)
    xData = ['0-100','100-1000','1000-5000','5000-15000','15000-30000','30000-50000','50000-~']
    yData = [0 for x in range(len(xData))]# 初始化为长度和xData相同但是每一个元素都是零的列表
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

def getArticleCharCommentsLen(type):# 统计特定类型文章的评论数分布
    articles = getArticleByType(type)
    xData = ['0-100','100-500','500-1000','1000-1500','1500-3000','3000-5000','5000-10000','10000-15000','15000-~']
    yData = [0 for x in range(len(xData))]# 初始化为长度和xData相同但是每一个元素都是零的列表
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

def getArticleCharRepotsLen(type):# 统计特定类型文章的转发数分布
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

def getIPCharByArticleRegion():#统计文章发布地域的分布情况
    articleRegionDic = {}
    for i in articleList:
        if i[4] != '无':# 如果ip为确定值的话就进行下一步
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

def getIPCharByCommentsRegion():#统计评论发布地域的分布情况
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

def getCommentCharDataOne():# 统计评论点赞数的分布情况
    xData = []
    rangeNum = 20
    for item in range(100):
        xData.append(str(rangeNum * item) + '-' + str(rangeNum * (item + 1)))
    yData = [0 for x in range(len(xData))]
    for comment in commentList:
        for item in range(100):
            if int(comment[2]) < rangeNum * (item + 1):
                yData[item] += 1
                break
    return xData,yData

def getCommentCharDataTwo():# 统计评论数据中不同性别的数量
    genderDic = {}
    for i in commentList:
        if i[6] in genderDic.keys():
            genderDic[i[6]] += 1
        else:
            genderDic[i[6]] = 1
    resultData = []
    for key,value in genderDic.items():
        resultData.append({
            'name':key,
            'value':value
        })
    return resultData

