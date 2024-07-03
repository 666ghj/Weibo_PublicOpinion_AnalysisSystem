import os
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('mysql+pymysql://XiaoXueQi:XiaoXueQi@10.92.35.13/Weibo_PublicOpinion_AnalysisSystem?charset=utf8mb4')

def save_to_sql():
    try:
        artileOldPd = pd.read_sql('select * from article',engine)
        articleNewPd = pd.read_csv('article.csv')
        commentOldPd = pd.read_sql('select * from comments',engine)
        commentNewPd = pd.read_csv('comments.csv')

        concatArticlePd = pd.concat([articleNewPd,artileOldPd],join='inner')
        concatCommentsPd = pd.concat([commentNewPd,commentOldPd],join='inner')

        concatArticlePd.drop_duplicates(subset='id',keep='last',inplace=True)
        concatCommentsPd.drop_duplicates(subset='content',keep='last',inplace=True)

        concatArticlePd.to_sql('article', con=engine, if_exists='replace', index=False)
        concatCommentsPd.to_sql('comments', con=engine, if_exists='replace', index=False)
    except:
        articleNewPd = pd.read_csv('article.csv')
        commentNewPd = pd.read_csv('comments.csv')
        articleNewPd.to_sql('article',con=engine,if_exists='replace',index=False)
        commentNewPd.to_sql('comments',con=engine,if_exists='replace',index=False)

    os.remove('./article.csv')
    os.remove('./comments.csv')

if __name__ == '__main__':
    save_to_sql()