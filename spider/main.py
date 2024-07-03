from spiderData import spiderData 
from saveData import save_to_sql as saveData

def main():
    try:
        spiderData()
        saveData()
        print("爬取数据更新")
    except:
        print("爬取数据失败")

if __name__ == '__main__':
    main()