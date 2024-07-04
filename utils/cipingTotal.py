import jieba
import re

def main():
    reader = open('./cutComments.txt','r',encoding='utf8')
    strs = reader.read()
    result = open('cipingTotal.csv', 'w', encoding='utf8')

    # 分词，去重，列表
    word_list = jieba.cut(strs,cut_all=True)

    new_words = []
    for i in word_list:
        m = re.search("\d+",i)
        n = re.search("\W+",i)
        if not m and not n and len(i) > 1:
            new_words.append(i)

    # 统计词频
    word_count = {}
    for i in set(new_words):
        word_count[i] = new_words.count(i)

    # 格式整理
    list_count = sorted(word_count.items(),key=lambda x:x[1],reverse=True)
    # list_count = [[key, str(value)] for key, value in list_count]
    # return list_count
    for i in range(100):
        print(list_count[i],file=result)

if __name__ == '__main__':
    print(main())