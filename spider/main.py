from spiderData import spiderData 
from saveData import saveData

def main():
    print('正在爬取数据')
    spiderData()
    print('正在存储数据')
    saveData()
    print("爬取数据更新")

if __name__ == '__main__':
    main()