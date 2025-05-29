import requests
import pandas as pd
import time
import os
import random
from datetime import datetime
from .settings import articleAddr, commentsAddr, commentsUrl
from utils.logger import spider_logger as logging
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

def getComments(articleId):
    """
    获取指定文章的评论数据
    """
    try:
        # 构建请求URL和头部
        url = f"{commentsUrl}{articleId}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # 解析响应数据
        data = response.json()
        if data['code'] == 200:
            return data['data']
        else:
            logging.error(f"获取评论失败，状态码：{data['code']}")
            return None
            
    except requests.RequestException as e:
        logging.error(f"请求失败：{e}")
        return None

def start():
    """
    开始爬取评论数据
    """
    try:
        # 读取文章数据
        article_df = pd.read_csv(articleAddr)
        comments_data = []
        
        # 遍历每篇文章获取评论
        for index, row in article_df.iterrows():
            article_id = row['id']
            logging.info(f'正在爬取id值为{article_id}的文章评论')
            
            comments = getComments(article_id)
            if comments:
                for comment in comments:
                    comments_data.append({
                        'article_id': article_id,
                        'content': comment.get('content', ''),
                        'created_at': comment.get('created_at', ''),
                        'like_count': comment.get('like_count', 0)
                    })
            
            # 避免请求过于频繁
            time.sleep(1)
        
        # 保存评论数据
        if comments_data:
            comments_df = pd.DataFrame(comments_data)
            comments_df.to_csv(commentsAddr, index=False, encoding='utf-8')
            logging.info(f"成功保存{len(comments_data)}条评论数据")
        else:
            logging.warning("未获取到任何评论数据")
            
    except Exception as e:
        logging.error(f"爬取评论数据时发生错误：{e}")

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
    start()
