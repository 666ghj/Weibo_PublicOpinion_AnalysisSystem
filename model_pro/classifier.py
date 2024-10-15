import torch
import torch.nn as nn

class FinalClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout_rate=0.3):
        super(FinalClassifier, self).__init__()
        # 增加一个隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层全连接层
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # 第二层全连接层
        self.dropout = nn.Dropout(dropout_rate)  # Dropout 防止过拟合
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层全连接 + ReLU 激活
        x = self.dropout(x)  # Dropout
        out = self.fc2(x)  # 最终输出层（未应用 softmax）
        return out
