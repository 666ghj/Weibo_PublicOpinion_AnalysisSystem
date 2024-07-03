from spiderContent import start as spiderContentStart
from spiderComments import start as spiderCommentsStart
from saveData import save_to_sql as saveData

def main():
    print('正在爬取文章数据')
    spiderContentStart(1,1)
    print('正在爬取文章评论数据')
    spiderCommentsStart()
    print('正在存储数据')
    saveData()
    print("爬取数据更新")

if __name__ == '__main__':
    main()