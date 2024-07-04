from utils.getPublicData import getAllCommentsData
import jieba
import re
targetTxt = 'cutComments.txt'

def stopWordList():
    stopWords = [line.strip() for line in open('./stopWords.txt',encoding='utf8').readlines()]
    return stopWords

def seg_depart(sentence):
    sentence_depart = jieba.cut(" ".join([clean(x[4]) for x in sentence]).strip())
    stopWords = stopWordList()
    outStr = ''
    for word in sentence_depart:
        if word not in stopWords:
            if word != '\t':
                outStr += word
    return outStr

def writer_comments_cuts():
    with open(targetTxt,'w+',encoding='utf-8') as targetFile:
        seg = jieba.cut(seg_depart(getAllCommentsData()))
        output = ' '.join(seg)
        targetFile.write(output)
        targetFile.write('\n')
        print('写入成功')

def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)  # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    # 去除emoji表情的正则表达式
    text = re.compile(u'[\U00010000-\U0010ffff]').sub('',text)
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)  # 去除网址
    text = text.replace("转发微博", "")  # 去除无意义的词语
    text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
    return text.strip()

if __name__ == '__main__':
    writer_comments_cuts()