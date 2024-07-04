from utils.getPublicData import *
from utils.predict import *
articleList = getAllArticleData()
commentList = getAllCommentsData()
import csv
import os
import datetime
def getTopicByArticle():# 返回文章内容的话题字典
    articleTopicDic = {}
    for i in articleList:
        if i[14] != None:
            if i[14] in articleTopicDic.keys():
                articleTopicDic[i[14]] += 1
            else:
                articleTopicDic[i[14]] = 1
    resultData = []
    for key,value in articleTopicDic.items():
        resultData.append({
            'name':key,
            'value':value
        })
    return resultData

def getTopicByComments():# 返回评论内容的话题字典
    commentsTopicDic = {}
    for i in commentList:
        if i[9] != None:
            if i[9] in commentsTopicDic:
                commentsTopicDic[i[9]] += 1
            else:
                commentsTopicDic[i[9]] = 1
    resultData = []
    for key,value in commentsTopicDic.items():
        resultData.append({
            'name':key,
            'value':value
        })
    return resultData

def mergeTopics(article_topics, comment_topics):# 合并话题
    merged_dict = {}
    for topic in article_topics + comment_topics:
        if topic['name'] in merged_dict:
            merged_dict[topic['name']] += topic['value']
        else:
            merged_dict[topic['name']] = topic['value']
    merged_dict = sorted(merged_dict.items(), key=lambda item: item[1], reverse=True)
    merged_list = [[key, str(value)] for key, value in merged_dict]
    return merged_list
def getAllTopicData():
    # 读取合并文件 merge.csv
    # data = []
    # df = pd.read_csv('./merged_topics.csv',encoding='utf8')
    # for i in df.values:
    #     try:
    #         data.append([
    #             re.search('[\u4e00-\u9fa5]+',str(i)).group(),
    #             re.search('\d+',str(i)).group()
    #         ])
    #     except:
    #         continue
    return mergeTopics(getTopicByArticle(), getTopicByComments())

def getTopicCreatedAtandpredictData(topic):# 统计特定话题的评论在每个日期的数量，并返回日期和对应的评论数量
    createdAt = {}
    for i in articleList:
        if i[14]==topic:
            if i[7] in createdAt.keys():
                createdAt[i[7]] += 1
            else:
                createdAt[i[7]] = 1
    for i in commentList:
        if i[9]==topic:
            if i[1] in createdAt.keys():
                createdAt[i[1]] += 1
            else:
                createdAt[i[1]] = 1
    createdAt = {k: createdAt[k] for k in sorted(createdAt, key=lambda date: datetime.datetime.strptime(date, "%Y-%m-%d"))}
    createdAt.update(predict_future_values(createdAt))
    sorted_data = {k: createdAt[k] for k in sorted(createdAt, key=lambda date: datetime.datetime.strptime(date, "%Y-%m-%d"))}
    # result_list = [0] * (len(sorted_data) - 5) + [1] * 5
    print(list(createdAt.keys()),list(createdAt.values()))
    return list(createdAt.keys()),list(createdAt.values())

def writeTopicsToCSV(topics, file_name):
    # 检查文件是否存在，如果存在则附加写入，否则新建一个
    file_exists = os.path.isfile(file_name)
    # 按值的降序排序
    sorted_topics = sorted(topics, key=lambda x: x['value'], reverse=True)
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 如果文件不存在，则写入表头
        if not file_exists:
            writer.writeheader()
        # 写入数据
        for topic in sorted_topics:
            writer.writerow(topic)
if __name__ == '__main__':
    # 将话题数据写入 CSV 文件
    # print(mergeTopics(getTopicByArticle(), getTopicByComments()))
    # writeTopicsToCSV(merged_topics, 'merged_topics.csv')
    print(getAllTopicData())

