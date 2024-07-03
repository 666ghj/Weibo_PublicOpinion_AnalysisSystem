from utils.getPublicData import getAllCommentsData
import jieba
targetTxt = 'cutComments.txt'

def stopWordList():
    stopWords = [line.strip() for line in open('./stopWords.txt',encoding='utf8').readlines()]
    return stopWords

def seg_depart(sentence):
    sentence_depart = jieba.cut(" ".join([x[4] for x in sentence]).strip())
    stopWords = stopWordList()
    outStr = ''
    for word in sentence_depart:
        if word not in stopWords:
            if word != '\t':
                outStr += word
    return outStr

def writer_comments_cuts():
    with open(targetTxt,'a+',encoding='utf-8') as targetFile:
        seg = jieba.cut(seg_depart(getAllCommentsData()),cut_all=True)
        output = ' '.join(seg)
        targetFile.write(output)
        targetFile.write('\n')
        print('写入成功')


if __name__ == '__main__':
    writer_comments_cuts()