import numpy as np
import datetime
import matplotlib.pyplot as plt


def datetime_to_number(date: str):  # 格式化日期转换为 integer
    date_number = datetime.datetime.strptime(date, "%Y-%m-%d")
    base_number = datetime.datetime.strptime("2024-1-1", "%Y-%m-%d")
    return (date_number - base_number).days


def predict_future_values(data):
    # 提取并排序日期
    sorted_dates = sorted(data.keys(), key=lambda date: datetime.datetime.strptime(date, "%Y-%m-%d"))
    sorted_data = {k: data[k] for k in sorted_dates}

    # 将日期转换为整数并提取相应的值
    xs = np.array([datetime_to_number(date) for date in sorted_data.keys()])
    ys = np.array([data[date] for date in sorted_data.keys()])

    # 拟合线性回归模型
    fit = np.polyfit(xs, ys, 1)
    fn = np.poly1d(fit)

    # 获取最新日期，并生成未来三天的日期
    latest_date = sorted_dates[-1]
    latest_date_obj = datetime.datetime.strptime(latest_date, "%Y-%m-%d")
    future_dates = [(latest_date_obj + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 6)]

    # 预测未来日期的值
    predictions = {}
    for date in future_dates:
        date_num = datetime_to_number(date)
        if int(fn(date_num))<=0:
            predictions[date] = 0
        else:
            predictions[date] = int(fn(date_num))

    return predictions


if __name__ == '__main__':
    data = {'2024-06-15': 1, '2024-06-18': 1, '2024-06-22': 1, '2024-06-23': 1, '2024-07-01': 3, '2024-07-02': 4, '2024-07-03': 4, '2024-07-04': 14}
    predictions = predict_future_values(data)
    print(predictions)
    # for date, value in predictions.items():
    #     print(f'{date} PREDICTION: {value}')
