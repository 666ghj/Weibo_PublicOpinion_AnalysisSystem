from utils.getPublicData import *  # Import utility functions for data retrieval
from utils.mynlp import SnowNLP  # Import SnowNLP for sentiment analysis
from collections import Counter  # Import Counter for counting occurrences

articleList = getAllArticleData()  # Retrieve all article data
commentList = getAllCommentsData()  # Retrieve all comment data

def getTypeList():
    # Return a list of unique article types
    return list(set([x[8] for x in articleList]))

def getArticleByType(type):
    # Return a list of articles that match the specified type
    return [article for article in articleList if article[8] == type]

def getArticleLikeCount(type):
    # Categorize articles by the number of likes they have
    articles = getArticleByType(type)
    intervals = [(0, 100), (100, 1000), (1000, 5000), (5000, 15000),
                 (15000, 30000), (30000, 50000), (50000, float('inf'))]
    X = ['0-100','100-1000','1000-5000','5000-15000','15000-30000',
         '30000-50000','50000-~']
    Y = [0] * len(intervals)
    for article in articles:
        likeCount = int(article[1])
        for i, (lower, upper) in enumerate(intervals):
            if lower <= likeCount < upper:
                Y[i] += 1
                break
    return X, Y

def getArticleCommentsLen(type):
    # Categorize articles by the length of comments they have
    articles = getArticleByType(type)
    intervals = [(0, 100), (100, 500), (500, 1000), (1000, 1500),
                 (1500, 3000), (3000, 5000), (5000, 10000),
                 (10000, 15000), (15000, float('inf'))]
    X = ['0-100','100-500','500-1000','1000-1500','1500-3000',
         '3000-5000','5000-10000','10000-15000','15000-~']
    Y = [0] * len(intervals)
    for article in articles:
        commentLen = int(article[2])
        for i, (lower, upper) in enumerate(intervals):
            if lower <= commentLen < upper:
                Y[i] += 1
                break
    return X, Y

def getArticleRepotsLen(type):
    # Categorize articles by the number of reposts
    articles = getArticleByType(type)
    intervals = [(0, 100), (100, 300), (300, 500), (500, 1000),
                 (1000, 2000), (2000, 3000), (3000, 4000),
                 (4000, 5000), (5000, 10000), (10000, 15000),
                 (15000, 30000), (30000, 70000), (70000, float('inf'))]
    X = ['0-100','100-300','300-500','500-1000','1000-2000','2000-3000',
         '3000-4000','4000-5000','5000-10000','10000-15000','15000-30000',
         '30000-70000','70000-~']
    Y = [0] * len(intervals)
    for article in articles:
        repostsCount = int(article[3])
        for i, (lower, upper) in enumerate(intervals):
            if lower <= repostsCount < upper:
                Y[i] += 1
                break
    return X, Y

def getIPByArticleRegion():
    # Count articles by their regions, excluding '无'
    regions = [article[4] for article in articleList if article[4] != '无']
    region_counts = Counter(regions)
    resultData = [{'name': key, 'value': value} for key, value in region_counts.items()]
    return resultData

def getIPByCommentsRegion():
    # Count comments by their regions, excluding '无'
    regions = [comment[3] for comment in commentList if comment[3] != '无']
    region_counts = Counter(regions)
    resultData = [{'name': key, 'value': value} for key, value in region_counts.items()]
    return resultData

def getCommentDataOne():
    # Categorize comments based on some numerical value, possibly length or count
    rangeNum = 20
    intervals = [(rangeNum * i, rangeNum * (i + 1)) for i in range(100)]
    X = [f"{lower}-{upper}" for lower, upper in intervals]
    Y = [0] * len(intervals)
    for comment in commentList:
        comment_value = int(comment[2])
        for i, (lower, upper) in enumerate(intervals):
            if lower <= comment_value < upper:
                Y[i] += 1
                break
    return X, Y

def getCommentDataTwo():
    # Count comments by gender
    genders = [comment[6] for comment in commentList]
    gender_counts = Counter(genders)
    resultData = [{'name': key, 'value': value} for key, value in gender_counts.items()]
    return resultData

def getYuQingCharDataOne():
    # Analyze sentiment of hot words
    hotWordList = getAllHotWords()
    sentiments = []
    for word in hotWordList:
        emotionValue = SnowNLP(word[0]).sentiments
        if emotionValue > 0.4:
            sentiments.append('正面')
        elif emotionValue < 0.2:
            sentiments.append('负面')
        else:
            sentiments.append('中性')
    counts = Counter(sentiments)
    X = ['正面','中性','负面']
    Y = [counts.get(sentiment, 0) for sentiment in X]
    biedata = [{'name': x, 'value': y} for x, y in zip(X, Y)]
    return X, Y, biedata

def getYuQingCharDataTwo():
    # Analyze sentiment of comments and articles
    comment_sentiments = []
    for comment in commentList:
        emotionValue = SnowNLP(comment[4]).sentiments
        if emotionValue > 0.4:
            comment_sentiments.append('正面')
        elif emotionValue < 0.2:
            comment_sentiments.append('负面')
        else:
            comment_sentiments.append('中性')
    comment_counts = Counter(comment_sentiments)
    
    article_sentiments = []
    for article in articleList:
        emotionValue = SnowNLP(article[5]).sentiments
        if emotionValue > 0.4:
            article_sentiments.append('正面')
        elif emotionValue < 0.2:
            article_sentiments.append('负面')
        else:
            article_sentiments.append('中性')
    article_counts = Counter(article_sentiments)
    
    X = ['正面', '中性', '负面']
    biedata1 = [{'name': x, 'value': comment_counts.get(x, 0)} for x in X]
    biedata2 = [{'name': x, 'value': article_counts.get(x, 0)} for x in X]
    return biedata1, biedata2

def getYuQingCharDataThree():
    # Retrieve top 10 hot words and their counts
    hotWordList = getAllHotWords()
    x1Data = [word[0] for word in hotWordList[:10]]
    y1Data = [int(word[1]) for word in hotWordList[:10]]
    return x1Data, y1Data