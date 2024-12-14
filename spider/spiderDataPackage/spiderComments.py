import time
import requests
import csv
import os
import random
from datetime import datetime
from .settings import articleAddr, commentsAddr
from requests.exceptions import RequestException

# 初始化，创建评论数据文件
def init():
    if not os.path.exists(commentsAddr):
        with open(commentsAddr, 'w', encoding='utf-8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([
                'articleId', 'created_at', 'likes_counts', 'region', 'content',
                'authorName', 'authorGender', 'authorAddress', 'authorAvatar'
            ])

# 写入评论数据到CSV
def write(row):
    with open(commentsAddr, 'a', encoding='utf-8', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)

# 获取数据，支持多账号随机切换
def fetchData(url, params, headers_list):
    headers = random.choice(headers_list)
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()['data']
        else:
            return None
    except RequestException as e:
        print(f"请求失败：{e}")
        return None

# 获取文章列表
def getArticleList():
    articleList = []
    with open(articleAddr, 'r', encoding='utf-8') as reader:
        readerCsv = csv.reader(reader)
        next(reader)
        for nav in readerCsv:
            articleList.append(nav)
    return articleList

# 解析评论数据
def readJson(response, articleId):
    for comment in response:
        created_at = datetime.strptime(comment['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime('%Y-%m-%d')
        likes_counts = comment['like_counts']
        region = comment.get('source', '无').replace('来自', '') 
        content = comment['text_raw']
        authorName = comment['user']['screen_name']
        authorGender = comment['user']['gender']
        authorAddress = comment['user']['location']
        authorAvatar = comment['user']['avatar_large']
        write([articleId, created_at, likes_counts, region, content, authorName, authorGender, authorAddress, authorAvatar])

# 启动爬虫
def start(headers_list, delay=2):
    commentUrl = 'https://weibo.com/ajax/statuses/buildComments'
    init()
    articleList = getArticleList()
    for article in articleList:
        articleId = article[0]
        print(f'正在爬取id值为{articleId}的文章评论')
        time.sleep(random.uniform(1, delay))  # 随机延时，避免频繁访问
        params = {'id': int(articleId), 'is_show_bulletin': 2}
        response = fetchData(commentUrl, params, headers_list)
        if response:
            readJson(response, articleId)

if __name__ == '__main__':
    # 这里的headers_list应该包含多个账号的cookie
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
