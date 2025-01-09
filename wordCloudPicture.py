import os
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pymysql

def stopWordList():
    """
    如果 stopWords.txt 文件内容较大，或被频繁读取，
    可以考虑将其缓存起来，避免重复读文件。
    """
    with open('./model/stopWords.txt', encoding='utf8') as f:
        return [line.strip() for line in f.readlines()]

def generate_word_cloud(text, mask_path, font_path, output_path):
    """生成词云并保存到 output_path"""
    img = Image.open(mask_path)
    img_arr = np.array(img)

    wc = WordCloud(
        background_color="#fff",
        mask=img_arr,
        font_path=font_path
    )
    wc.generate_from_text(text)

    plt.figure(figsize=(8, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # 保存后关闭

def get_db_connection_interactive():
    """
    通过终端交互获取数据库连接参数，若按回车则使用默认值。
    """
    print("请依次输入数据库连接信息（直接按回车使用默认值）：")

    host = input(" 1. 主机 (默认: localhost): ") or "localhost"
    port_str = input(" 2. 端口 (默认: 3306): ") or "3306"
    port = int(port_str)

    user = input(" 3. 用户名 (默认: root): ") or "root"
    password = input(" 4. 密码 (默认: 312517): ") or "12345678"
    db_name = input(" 5. 数据库名 (默认: Weibo_PublicOpinion_AnalysisSystem): ") or "Weibo_PublicOpinion_AnalysisSystem"

    print(f"\n即将连接到数据库: {user}@{host}:{port}/{db_name}\n")
    
    return pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=db_name,
        port=port,
        charset='utf8mb4'
    )

def get_img(field, table_name, target_img_src, res_img_src, connection, font_path='STHUPO.TTF'):
    """ 
    从数据库拉取指定字段的文本数据，分词处理后生成词云。
    :param field: 数据库字段名
    :param table_name: 数据表名
    :param target_img_src: 词云形状图
    :param res_img_src: 输出词云文件路径
    :param connection: 已建立的数据库连接
    :param font_path: 字体文件路径
    """
    cursor = connection.cursor()
    sql = f'SELECT {field} FROM {table_name}'
    cursor.execute(sql)
    data = cursor.fetchall()

    text = ''
    for item in data:
        text += item[0]  # item 是元组 (内容,)，取第一个元素即可

    cursor.close()

    # 分词 & 去停用词
    cut_words = jieba.cut(text)
    stop_words = set(stopWordList())
    filtered_words = [word for word in cut_words if word not in stop_words]
    final_text = ' '.join(filtered_words)

    # 生成词云
    generate_word_cloud(final_text, target_img_src, font_path, res_img_src)

def main():
    # 1. 获取数据库连接（交互式输入）
    connection = get_db_connection_interactive()

    # 2. 根据需求生成词云
    # 例如：从 article 表的 content 字段生成词云
    try:
        get_img(
            field='content', 
            table_name='article', 
            target_img_src='./static/content.jpg', 
            res_img_src='./static/contentCloud.jpg', 
            connection=connection
        )
        print("词云生成完毕！")
    finally:
        # 关闭数据库连接
        connection.close()

if __name__ == '__main__':
    main()
