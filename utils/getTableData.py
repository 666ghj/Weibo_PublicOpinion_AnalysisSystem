from utils.getPublicData import getAllArticleData
from utils.mynlp import SnowNLP

def getTableDataList(flag):
    if flag:
        tableList = []
        articeList = getAllArticleData()
        for article in articeList:
            item = list(article)
            value = ''
            if SnowNLP(item[5]).sentiments > 0.6:
                value = '正面'
            elif SnowNLP(item[5]).sentiments < 0.4:
                value = '负面'
            else:
                value = '中性'
            item.append(value)
            tableList.append(item)
        return tableList
    else:
        return getAllArticleData()