import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    def __init__(self, input_size, adapter_size):
        super(AdapterLayer, self).__init__()
        # 第一个全连接层降维
        self.down_project = nn.Linear(input_size, adapter_size)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 第二个全连接层升维
        self.up_project = nn.Linear(adapter_size, input_size)

    def forward(self, x):
        # 通过Adapter层的前向传播
        down_projected = self.down_project(x)
        relu = self.relu(down_projected)
        up_projected = self.up_project(x)
        # 将Adapter的输出与输入相加（残差连接）
        return x + up_projected
