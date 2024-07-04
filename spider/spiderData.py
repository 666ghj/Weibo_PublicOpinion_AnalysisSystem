from spiderDataPackage.spiderNav import start as spiderNav
from spiderDataPackage.spiderContent import start as spiderContent
from spiderDataPackage.spiderComments import start as spiderComments
import os

def spiderData():
    if not os.path.exists('./nav.csv'):
        print('正在爬取导航栏数据')
        spiderNav()
    print('正在爬取文章数据')
    spiderContent(1,1)
    print('正在爬取文章评论数据')
    spiderComments()

if __name__ == '__main__':
    spiderData()