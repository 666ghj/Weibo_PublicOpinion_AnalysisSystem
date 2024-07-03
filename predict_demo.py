'''
用于测试预测逻辑
@Author: QST520
'''
import numpy as np

import datetime
import matplotlib.pyplot as plt


def datetime_to_number(date: str):  # 格式化日期转换为 integer
    date_number = datetime.datetime.strptime(date, "%Y-%m-%d")
    base_number = datetime.datetime.strptime("2024-1-1", "%Y-%m-%d")
    return date_number.__sub__(base_number).days


if __name__ == '__main__':  # 预测 demo
    xs = [
        '2024-6-4', '2024-6-5', '2024-6-6', '2024-6-7', '2024-6-8', '2024-6-9',
        '2024-6-10', '2024-6-11', '2024-6-12', '2024-6-13'
    ]
    ys = [15, 14, 16, 15, 16, 13, 12, 11, 9, 8]
    xs = np.array(list(map(datetime_to_number, xs)))
    ys = np.array(ys)
    fit = np.polyfit(xs, ys, 1)
    fn = np.poly1d(fit)
    print('2024-6-14 PREDICTION: ' +
          str(int(fn(datetime_to_number('2024-6-14')))))
    print('2024-6-15 PREDICTION: ' +
          str(int(fn(datetime_to_number('2024-6-15')))))
    print('2024-6-16 PREDICTION: ' +
          str(int(fn(datetime_to_number('2024-6-16')))))
    print('2024-6-17 PREDICTION: ' +
          str(int(fn(datetime_to_number('2024-6-17')))))
    print('2024-6-18 PREDICTION: ' +
          str(int(fn(datetime_to_number('2024-6-18')))))
