import os
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('mysql+pymysql://XiaoXueQi:XiaoXueQi@47.92.235.6/Weibo_PublicOpinion_AnalysisSystem?charset=utf8mb4')

def saveData():
    try:
        oldArticle = pd.read_sql('select * from article',engine)
        newArticle = pd.read_csv('article.csv')
        oldComment = pd.read_sql('select * from comments',engine)
        newComment = pd.read_csv('comments.csv')

        mergeArticle = pd.concat([newArticle,oldArticle],join='inner')
        mergeComment = pd.concat([newComment,oldComment],join='inner')

        mergeArticle.drop_duplicates(subset='id',keep='last',inplace=True)
        mergeComment.drop_duplicates(subset='content',keep='last',inplace=True)

        mergeArticle.to_sql('article', con=engine, if_exists='replace', index=False)
        mergeComment.to_sql('comments', con=engine, if_exists='replace', index=False)
    except:
        newArticle = pd.read_csv('article.csv')
        newComment = pd.read_csv('comments.csv')
        newArticle.to_sql('article',con=engine,if_exists='replace',index=False)
        newComment.to_sql('comments',con=engine,if_exists='replace',index=False)

    os.remove('./article.csv')
    os.remove('./comments.csv')

if __name__ == '__main__':
    saveData()