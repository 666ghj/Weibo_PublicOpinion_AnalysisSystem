
import csv  # 用于处理CSV文件的读写操作
import os  # 用于操作系统相关功能
import sys
import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的父目录路径
parent_dir = os.path.dirname(current_file_path)

# 获取父目录的父目录路径，也就是项目根目录
project_root_dir = os.path.dirname(parent_dir)

# 将项目根目录添加到 Python 路径中
sys.path.append(project_root_dir)

# 现在可以导入 utils 目录中的模块了
from utils.getPublicData import getAllCommentsData  # 自定义函数，用于获取评论数据
from utils.mynlp import SnowNLP  # 引入SnowNLP库，用于中文情感分析
def targetFile():
    targetFile = 'target.csv'  # 定义目标文件名称
    commentsList = getAllCommentsData()  # 获取所有评论数据

    rateData = []  # 用于存储处理后的评论数据
    good = 0  # 记录正面评论数量
    bad = 0  # 记录负面评论数量
    middle = 0  # 记录中性评论数量

    # 遍历所有评论，进行情感分析
    for index, i in enumerate(commentsList): # enumerate 是 Python 中的一个内置函数，它允许我们在遍历可迭代对象（如列表、元组或字符串）时同时获取元素的索引和值。
        # |articleId|created_at | likes_counts | region | content| authorName | authorGender | authorAddress | authorAvatar
        value = SnowNLP(i[4]).sentiments  # 对评论内容进行情感分析
        if value > 0.5:  # 如果情感值大于0.5，判定为正面评论
            good += 1
            rateData.append([i[4], '正面'])
        elif value == 0.5:  # 如果情感值等于0.5，判定为中性评论
            middle += 1
            rateData.append([i[4], '中性'])
        elif value < 0.5:  # 如果情感值小于0.5，判定为负面评论
            bad += 1
            rateData.append([i[4], '负面'])

    # 将处理后的评论数据写入目标文件
    for i in rateData:
        with open(targetFile, 'a+', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(i)  # 将每条数据写入CSV文件

def main():
    targetFile()  # 调用targetFile函数进行数据处理

if __name__ == '__main__':
    main()  # 运行主函数
