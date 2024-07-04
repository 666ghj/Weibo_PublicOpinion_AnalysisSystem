import time
import requests
import csv
import os
from datetime import datetime
from .settings import navAddr,articleAddr

def init():
    if not os.path.exists(articleAddr):
        with open(articleAddr,'w',encoding='utf-8',newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([
                'id',
                'likeNum',
                'commentsLen',
                'reposts_count',
                'region',
                'content',
                'contentLen',
                'created_at',
                'type',
                'detailUrl',# followBtnCode>uid + mblogid
                'authorAvatar',
                'authorName',
                'authorDetail',
                'isVip' # v_plus
            ])

def write(row):
    with open(articleAddr, 'a', encoding='utf-8', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)

def fetchData(url,params):
    headers = {
        'Cookie':'SINAGLOBAL=2555941826014.1074.1676801766625; ULV=1719829459275:6:1:2:4660996305989.918.1719827559898:1719743122299; UOR=,,www.baidu.com; XSRF-TOKEN=VtLXviYSIs8lor7sz4iGyigL; SUB=_2A25LhvU9DeRhGeFH6FIX-S3MyD2IHXVo-gj1rDV8PUJbkNAGLRXMkW1Ne2nhI3Gle25QJK0Z99J3trq_NZn6YKJ-; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WW3Mv8V5EupQbbKh.vaZIwU5JpX5KzhUgL.FoM4e05c1Ke7e022dJLoIp7LxKML1KBLBKnLxKqL1hnLBoM41hz41hqReKqN; WBPSESS=Dt2hbAUaXfkVprjyrAZT_LRaDLsnxG-kIbeYwnBb5OUKZiwfVr_UrcYfWuqG-4ZVDM5HeU3HXkDNK_thfRfdS9Ao6ezT30jDksv-CpaVmlTAqGUHjJ7PYkH5aCK4HLxmRq14ZalmQNwzfWMPa4y0VNRLuYdg7L1s49ymNq_5v5vusoz0r4ki6u-MHGraF0fbUTgX14x0kHayEwOoxfLI-w==; SCF=AqmJWo31oFV5itnRgWNU1-wHQTL6PmkBLf3gDuqpdqAIfaWguDTMre6Oxjf5Uzs74JAh2r0DdV1sJ1g6m-wJ5NQ.; _s_tentry=-; Apache=4660996305989.918.1719827559898; PC_TOKEN=7955a7ab1f; appkey=; geetest_token=602cd4e3a7ed1898808f8adfe1a2048b; ALF=1722421868',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0'
    }
    response = requests.get(url,headers=headers,params=params)
    if response.status_code == 200:
        return response.json()['statuses']
    else:
        return None

def getTypeList():
    typeList = []
    with open(navAddr,'r',encoding='utf-8') as reader:
        readerCsv = csv.reader(reader)
        next(reader)
        for nav in readerCsv:
            typeList.append(nav)
    return typeList

def readJson(response,type):
    for artice in response:
        id = artice['id']
        likeNum = artice['attitudes_count']
        commentsLen = artice['comments_count']
        reposts_count = artice['reposts_count']
        try:
            region = artice['region_name'].replace('发布于 ', '')
        except:
            region = '无'
        content = artice['text_raw']
        contentLen = artice['textLength']
        created_at = datetime.strptime(artice['created_at'],'%a %b %d %H:%M:%S %z %Y').strftime('%Y-%m-%d')
        type = type
        try:
            detailUrl = 'https://weibo.com/' + str(artice['id']) + '/' + str(artice['mblogid'])
        except:
            detailUrl = '无'
        authorAvatar = artice['user']['avatar_large']
        authorName = artice['user']['screen_name']
        authorDetail = 'https://weibo.com/u/' + str(artice['user']['id'])
        isVip = artice['user']['v_plus']
        write([
            id,
            likeNum,
            commentsLen,
            reposts_count,
            region,
            content,
            contentLen,
            created_at,
            type,
            detailUrl,
            authorAvatar,
            authorName,
            authorDetail,
            isVip
        ])

def start(typeNum=14,pageNum=3):
    articleUrl = 'https://weibo.com/ajax/feed/hottimeline'
    init()
    typeList = getTypeList()
    typeNumCount = 0
    for type in typeList:
        if typeNumCount > typeNum:return
        time.sleep(2)
        for page in range(0,pageNum):
            print('正在爬取的类型：%s 中的第%s页文章数据' % (type[0],page + 1))
            time.sleep(1)
            parmas = {
                'group_id':type[1],
                'containerid':type[2],
                'max_id':page,
                'count':10,
                'extparam':'discover|new_feed'
            }
            response = fetchData(articleUrl,parmas)
            readJson(response,type[0])
        typeNumCount += 1

if __name__ == '__main__':
    start()








