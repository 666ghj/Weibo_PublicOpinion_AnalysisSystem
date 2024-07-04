import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from pymysql import *
import json
import numpy as np
def stopWordList():
    return [line.strip() for line in open('./model/stopWords.txt',encoding='utf8').readlines()]

def get_img(field,tableName,targetImgSrc,resImgSrc):
    con = connect(host='47.92.235.6',user='XiaoXueQi',password='XiaoXueQi',database='Weibo_PublicOpinion_AnalysisSystem',port=3306,charset='utf8mb4')
    cuser = con.cursor()
    sql = f'select {field} from {tableName}'
    cuser.execute(sql)
    data = cuser.fetchall()
    text = ''
    for item in data:
        text += item[0]
    cuser.close()
    con.close()

    cut = jieba.cut(text)
    newCut = []
    for word in cut:
        if word not in stopWordList():newCut.append(word)
    string = ' '.join(newCut)

    img = Image.open(targetImgSrc)
    img_arr = np.array(img)
    wc = WordCloud(
        background_color="#fff",
        mask=img_arr,
        font_path='STHUPO.TTF'
    )
    wc.generate_from_text(string)

    fig = plt.figure(1)
    plt.imshow(wc)

    plt.axis('off')

    plt.savefig(resImgSrc,dpi=500)


# get_img('content','comments','./static/comment.jpg','./static/commentCloud.jpg')
get_img('content','article','./static/content.jpg','./static/contentCloud.jpg')
