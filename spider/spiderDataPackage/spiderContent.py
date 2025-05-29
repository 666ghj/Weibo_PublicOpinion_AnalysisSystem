import time
import requests
import csv
import os
import random
from datetime import datetime
from .settings import navAddr, articleAddr
from requests.exceptions import RequestException

# 初始化文章数据文件
def init():
    if not os.path.exists(articleAddr):
        with open(articleAddr, 'w', encoding='utf-8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([
                'id', 'likeNum', 'commentsLen', 'reposts_count', 'region', 'content', 'contentLen',
                'created_at', 'type', 'detailUrl', 'authorAvatar', 'authorName', 'authorDetail', 'isVip'
            ])

# 写入数据到CSV
def write(row):
    with open(articleAddr, 'a', encoding='utf-8', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)

# 获取数据，支持多账号
def fetchData(url, params, headers_list):
    headers = random.choice(headers_list)
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()['statuses']
        else:
            return None
    except RequestException as e:
        print(f"请求失败：{e}")
        return None

# 获取类型列表
def getTypeList():
    typeList = []
    with open(navAddr, 'r', encoding='utf-8') as reader:
        readerCsv = csv.reader(reader)
        next(reader)
        for nav in readerCsv:
            typeList.append(nav)
    return typeList

# 解析文章数据
def readJson(response, type):
    for article in response:
        id = article['id']
        likeNum = article['attitudes_count']
        commentsLen = article['comments_count']
        reposts_count = article['reposts_count']
        region = article.get('region_name', '无').replace('发布于 ', '')
        content = article['text_raw']
        contentLen = article['textLength']
        created_at = datetime.strptime(article['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime('%Y-%m-%d')
        detailUrl = f"https://weibo.com/{article['id']}/{article['mblogid']}" if 'mblogid' in article else '无'
        authorAvatar = article['user']['avatar_large']
        authorName = article['user']['screen_name']
        authorDetail = f"https://weibo.com/u/{article['user']['id']}"
        isVip = article['user']['v_plus']
        write([id, likeNum, commentsLen, reposts_count, region, content, contentLen, created_at, type, detailUrl, authorAvatar, authorName, authorDetail, isVip])

# 启动爬虫
def start(headers_list, typeNum=14, pageNum=3, delay=2):
    articleUrl = 'https://weibo.com/ajax/feed/hottimeline'
    init()
    typeList = getTypeList()
    for type in typeList[:typeNum]:
        for page in range(pageNum):
            print(f'正在爬取的类型：{type[0]} 中的第{page + 1}页文章数据')
            time.sleep(random.uniform(1, delay))  # 随机延时
            params = {
                'group_id': type[1],
                'containerid': type[2],
                'max_id': page,
                'count': 10,
                'extparam': 'discover|new_feed'
            }
            response = fetchData(articleUrl, params, headers_list)
            if response:
                readJson(response, type[0])

if __name__ == '__main__':
    headers_list = [
        {
            'Cookie': 'your_cookie_here',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0'
        },
        {
            'Cookie': 'another_cookie_here',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0'
        }
    ]
    start(headers_list)
