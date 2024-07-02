from utils.getPublicData import *

def getHotWordLen(hotWord):# 统计包含特定热词评论数量
    commentsList = getAllCommentsData()
    hotWordLen = 0
    for i in commentsList:
        if i[4].find(hotWord) != -1:
            hotWordLen+=1
    return hotWordLen

def getHotWordPageCreatedAtCharData(hotWord):# 统计包含特定热词的评论在每个日期的数量，并返回日期和对应的评论数量
    commentsList = getAllCommentsData()
    createdAt = {}
    for i in commentsList:
        if i[4].find(hotWord) != -1:
            if i[1] in createdAt.keys():
                createdAt[i[1]] += 1
            else:
                createdAt[i[1]] = 1
    return list(createdAt.keys()),list(createdAt.values())

def getCommentFilterData(hotWord):# 筛选包含特定热词的评论并返回这些评论的数据
    commentsList = getAllCommentsData()
    commentData = []
    for i in commentsList:
        if i[4].find(hotWord) != -1:
            commentData.append(i)
    return commentData