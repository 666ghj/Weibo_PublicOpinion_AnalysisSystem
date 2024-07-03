import requests
import csv
import numpy as np
import os

def init():
    if not os.path.exists('./nav.csv'):
        with open('./nav.csv','w',encoding='utf-8',newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([
                'typeName',
                'gid',
                'containerid'
            ])

def writerRow(row):
    with open('./nav.csv', 'a', encoding='utf-8', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)

def get_data(url):
    headers = {
        'Cookie':'SINAGLOBAL=2555941826014.1074.1676801766625; ULV=1719829459275:6:1:2:4660996305989.918.1719827559898:1719743122299; UOR=,,www.baidu.com; XSRF-TOKEN=VtLXviYSIs8lor7sz4iGyigL; SUB=_2A25LhvU9DeRhGeFH6FIX-S3MyD2IHXVo-gj1rDV8PUJbkNAGLRXMkW1Ne2nhI3Gle25QJK0Z99J3trq_NZn6YKJ-; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WW3Mv8V5EupQbbKh.vaZIwU5JpX5KzhUgL.FoM4e05c1Ke7e022dJLoIp7LxKML1KBLBKnLxKqL1hnLBoM41hz41hqReKqN; WBPSESS=Dt2hbAUaXfkVprjyrAZT_LRaDLsnxG-kIbeYwnBb5OUKZiwfVr_UrcYfWuqG-4ZVDM5HeU3HXkDNK_thfRfdS9Ao6ezT30jDksv-CpaVmlTAqGUHjJ7PYkH5aCK4HLxmRq14ZalmQNwzfWMPa4y0VNRLuYdg7L1s49ymNq_5v5vusoz0r4ki6u-MHGraF0fbUTgX14x0kHayEwOoxfLI-w==; SCF=AqmJWo31oFV5itnRgWNU1-wHQTL6PmkBLf3gDuqpdqAIfaWguDTMre6Oxjf5Uzs74JAh2r0DdV1sJ1g6m-wJ5NQ.; _s_tentry=-; Apache=4660996305989.918.1719827559898; PC_TOKEN=7955a7ab1f; appkey=; geetest_token=602cd4e3a7ed1898808f8adfe1a2048b; ALF=1722421868',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0'
    }
    params = {
        'is_new_segment':1,
        'fetch_hot':1
    }
    response = requests.get(url,headers=headers,params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def parse_json(response):
    navList = np.append(response['groups'][3]['group'],response['groups'][4]['group'])
    for nav in navList:
        navName = nav['title']
        gid = nav['gid']
        containerid = nav['containerid']
        writerRow([
            navName,
            gid,
            containerid
        ])


if __name__ == '__main__':
    init()
    url = 'https://weibo.com/ajax/feed/allGroups'
    response = get_data(url)
    parse_json(response)