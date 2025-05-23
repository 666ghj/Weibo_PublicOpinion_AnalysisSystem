import requests
import pandas as pd
import csv
import time
import os
import random
from datetime import datetime
from spider.spiderDataPackage.settings import articleAddr, commentsAddr, commentsUrl
from utils.logger import spider_logger as logging
from requests.exceptions import RequestException
import json

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

def getComments(articleId, headers_list=None):
    """
    获取指定文章的评论数据
    """
    try:
        # 使用传入的headers_list或默认headers
        if headers_list is None:
            headers = {
                'Cookie': 'Your_Cookie',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
                'Referer': 'https://weibo.com/',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            }
        else:
            headers = random.choice(headers_list)
            # 确保headers包含必要的字段
            if 'Referer' not in headers:
                headers['Referer'] = 'https://weibo.com/'
            if 'Accept' not in headers:
                headers['Accept'] = 'application/json, text/plain, */*'

        url = f"https://weibo.com/ajax/statuses/buildComments?flow=0&is_reload=1&id={articleId}&is_show_bulletin=2&is_mix=0"
        logging.info(f"请求评论URL: {url}")
        
        response = requests.get(url, headers=headers, timeout=15)
        
        # 输出状态码和响应内容前100个字符用于调试
        logging.info(f"状态码: {response.status_code}")
        content_preview = response.text[:100].replace('\n', ' ') if response.text else "空响应"
        logging.info(f"响应内容预览: {content_preview}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # 直接处理数据，不保存到本地
                if 'ok' in data and data['ok'] == 1:
                    if 'data' in data and isinstance(data['data'], list):
                        comments_data = data['data']
                        logging.info(f"成功获取到{len(comments_data)}条评论")
                        return comments_data

                    if 'comments' in data and isinstance(data['comments'], list):
                        logging.info(f"成功获取到{len(data['comments'])}条评论")
                        return data['comments']
                    
                    logging.warning("API返回成功但找不到评论数据，返回结构：" + json.dumps(list(data.keys())))
                else:
                    error_msg = data.get('message', '未知错误')
                    logging.error(f"API返回错误: {data.get('ok')}, {error_msg}")
                
                return None
                    
            except ValueError as e:
                logging.error(f"JSON解析失败: {e}")
                return None
        else:
            logging.warning(f"HTTP错误: {response.status_code}")
            return None
            
    except Exception as e:
        logging.error(f"获取评论时发生意外错误: {e}")
        import traceback
        logging.error(traceback.format_exc())  # 打印详细错误堆栈
        return None

def create_mock_comments(articleId):
    """
    当无法从API获取真实评论时，生成模拟评论数据
    """
    import faker
    import random
    
    fake = faker.Faker('zh_CN')  # 使用中文本地化
    
    # 为测试生成10条模拟评论
    mock_comments = []
    for i in range(10):
        created_at = fake.date_time_between(start_date='-30d', end_date='now').strftime('%Y-%m-%d')
        likes_count = random.randint(0, 1000)
        region = fake.city()
        content = fake.sentence(nb_words=15)
        author_name = fake.name()
        author_gender = random.choice(['m', 'f'])
        author_address = fake.province()
        author_avatar = f"https://tvax1.sinaimg.cn/crop.0.0.{random.randint(180,1000)}.{random.randint(180,1000)}/50/{random.randint(10000,99999)}102.jpg"
        
        mock_comment = {
            'articleId': articleId,
            'created_at': created_at,
            'likes_counts': likes_count,
            'region': region,
            'content': content,
            'authorName': author_name,
            'authorGender': author_gender,
            'authorAddress': author_address,
            'authorAvatar': author_avatar
        }
        mock_comments.append(mock_comment)
    
    logging.info(f"已为文章 {articleId} 生成 {len(mock_comments)} 条模拟评论")
    return mock_comments

def start(headers_list=None):
    """
    开始爬取评论数据
    """
    try:
        # 如果没有提供headers_list，使用默认值
        if headers_list is None:
            headers_list = [
                {
                    'Cookie': 'Your_Cookie',
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'
                }
            ]
        
        # 确保commentsAddr目录存在
        comments_dir = os.path.dirname(commentsAddr)
        if not os.path.exists(comments_dir):
            os.makedirs(comments_dir)
            
        # 初始化评论文件
        init()
            
        # 设置USE_MOCK_DATA环境变量来使用模拟数据
        use_mock = os.environ.get('USE_MOCK_DATA', 'false').lower() == 'true'
        if use_mock:
            logging.info("将使用模拟数据生成评论")
        
        # 在评论获取失败时使用模拟数据
        try:
            import faker
            has_faker = True
        except ImportError:
            has_faker = False
            logging.warning("未安装faker库，无法生成模拟数据。请运行 pip install faker 来安装")
            
        # 读取文章数据
        try:
            article_df = pd.read_csv(articleAddr)
            logging.info(f"成功读取{len(article_df)}篇文章数据")
        except Exception as e:
            logging.error(f"读取文章数据失败: {e}")
            return
            
        # 直接向CSV文件写入数据，不缓存在内存中
        with open(commentsAddr, 'w', encoding='utf-8', newline='') as f:
            # 定义CSV字段
            fieldnames = [
                'articleId', 'created_at', 'likes_counts', 'region', 'content',
                'authorName', 'authorGender', 'authorAddress', 'authorAvatar'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 限制处理文章数量，避免请求过多
            article_count = min(len(article_df), 100)
            comments_count = 0
            
            # 遍历每篇文章获取评论
            for index, row in article_df.iterrows():
                if index >= article_count:
                    break
                    
                article_id = str(row['id'])
                logging.info(f'正在爬取id值为{article_id}的文章评论 ({index+1}/{article_count})')
                
                # 如果设置了USE_MOCK_DATA并安装了faker，则直接生成模拟数据
                if use_mock and has_faker:
                    logging.info("根据设置使用模拟数据")
                    mock_data = create_mock_comments(article_id)
                    for comment in mock_data:
                        writer.writerow(comment)
                        comments_count += 1
                else:
                    # 尝试从API获取真实评论
                    comments = getComments(article_id, headers_list)
                    
                    if comments:
                        logging.info(f"成功获取到{len(comments)}条评论数据")
                        article_comments_count = 0
                        
                        for comment in comments:
                            try:
                                # 提取评论数据
                                comment_data = {
                                    'articleId': article_id,
                                    'created_at': comment.get('created_at', ''),
                                    'likes_counts': comment.get('like_counts', 0),
                                    'region': comment.get('source', '未知').replace('来自', '') if comment.get('source') else '未知',
                                    'content': comment.get('text', comment.get('text_raw', '')),
                                    'authorName': comment.get('user', {}).get('screen_name', '匿名用户') if isinstance(comment.get('user'), dict) else '匿名用户',
                                    'authorGender': comment.get('user', {}).get('gender', '未知') if isinstance(comment.get('user'), dict) else '未知',
                                    'authorAddress': comment.get('user', {}).get('location', '未知') if isinstance(comment.get('user'), dict) else '未知',
                                    'authorAvatar': comment.get('user', {}).get('avatar_large', '') if isinstance(comment.get('user'), dict) else ''
                                }
                                
                                # 直接写入CSV
                                writer.writerow(comment_data)
                                comments_count += 1
                                article_comments_count += 1
                                
                                # 显示第一条评论的样例数据
                                if article_comments_count == 1:
                                    logging.info(f"评论样例: {comment_data['content'][:50]}...")
                                    
                            except Exception as e:
                                logging.error(f"处理评论数据时出错: {e}")
                                continue
                    elif has_faker and use_mock:
                        # API获取失败时使用模拟数据
                        logging.info("API获取失败，使用模拟数据替代")
                        mock_data = create_mock_comments(article_id)
                        for comment in mock_data:
                            writer.writerow(comment)
                            comments_count += 1
                
                # 休眠时间
                time.sleep(random.uniform(1.5, 3.0))  # 随机延迟
            
            logging.info(f"已完成爬取，共写入{comments_count}条评论数据")
            
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
