from utils.getPublicData import *
from utils.predict import predict_future_values  # Use the new function
import csv
import os
import datetime
import pandas as pd

def getTopicCreatedAtandpredictData(topic):
    createdAt = {}
    for i in articleList:
        if i[14]==topic:
            if i[7] in createdAt.keys():
                createdAt[i[7]] += 1
            else:
                createdAt[i[7]] = 1
    for i in commentList:
        if i[9]==topic:
            if i[1] in createdAt.keys():
                createdAt[i[1]] += 1
            else:
                createdAt[i[1]] = 1

    # Use the improved time series prediction approach
    predictions = predict_future_values(createdAt, forecast_days=5)

    # Merge historical data and predictions
    combined_data = {**createdAt, **predictions}
    combined_data = {k: combined_data[k] for k in sorted(combined_data, key=lambda date: datetime.datetime.strptime(date, "%Y-%m-%d"))}

    print(list(combined_data.keys()), list(combined_data.values()))
    return list(combined_data.keys()), list(combined_data.values())
