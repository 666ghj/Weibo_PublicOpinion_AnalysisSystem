import requests
import csv
import numpy as np
import os
import random
from .settings import navAddr
from requests.exceptions import RequestException

# 初始化导航数据文件
def init():
    if not os.path.exists(navAddr):
        with open(navAddr, 'w', encoding='utf-8', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['typeName', 'gid', 'containerid'])

# 写入导航数据
def write(row):
    with open(navAddr, 'a', encoding='utf-8', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)

# 获取数据，支持多账号
def fetchData(url, headers_list):
    headers = random.choice(headers_list)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()['data']['modules']
        else:
            return None
    except RequestException as e:
        print(f"请求失败：{e}")
        return None

# 解析导航数据
def readJson(response):
    for module in response:
        if 'type' in module and 'typeName' in module:
            typeName = module['typeName']
            for submodule in module['modules']:
                if 'id' in submodule and 'containerid' in submodule:
                    gid = submodule['id']
                    containerid = submodule['containerid']
                    write([typeName, gid, containerid])

# 启动爬虫
def start(headers_list):
    navUrl = 'https://weibo.com/ajax/side/hot'
    init()
    response = fetchData(navUrl, headers_list)
    if response:
        readJson(response)

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
