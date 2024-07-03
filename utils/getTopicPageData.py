from utils.getPublicData import *

def getTopicLen(topic):# 统计特定话题下的评论数目
    commentsList = getAllCommentsData()
    topic_len = 0
    for i in commentsList:
        if i[9] == topic:
            topicLen+=1
    return topic_len

def getTopicPageCreatedAtCharData(topic):# 统计包含特定热词的评论在每个日期的数量，并返回日期和对应的评论数量
    commentsList = getAllCommentsData()
    createdAt = {}
    for i in commentsList:
        if i[9]==topic:
            if i[1] in createdAt.keys():
                createdAt[i[1]] += 1
            else:
                createdAt[i[1]] = 1
    return list(createdAt.keys()),list(createdAt.values())

def getCommentFilterDataTopic(topic):# 筛选包含特定热词的评论并返回这些评论的数据
    commentsList = getAllCommentsData()
    commentData = []
    for i in commentsList:
        if i[9] == topic:
            commentData.append(i)
    return commentData