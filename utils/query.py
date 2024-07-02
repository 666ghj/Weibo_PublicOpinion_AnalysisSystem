from pymysql import *
conn = connect(host='10.92.35.13',port=3306,user='XiaoXueQi',password='XiaoXueQi',database='Weibo_PublicOpinion_AnalysisSystem')
cursor = conn.cursor()
def query(sql,params,type="no_select"):
    params = tuple(params)
    cursor.execute(sql,params)
    conn.ping(reconnect=True)
    if type != 'no_select':
        data_list = cursor.fetchall()
        conn.commit()
        return data_list
    else:
        conn.commit()
