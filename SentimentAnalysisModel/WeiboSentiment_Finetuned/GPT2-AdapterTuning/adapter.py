import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    """
    Adapter层实现
    将其添加到Transformer层中可以实现参数高效微调
    """
    def __init__(self, input_size, adapter_size):
        super(AdapterLayer, self).__init__()
        # 降维全连接层
        self.down_project = nn.Linear(input_size, adapter_size)
        # 激活函数
        self.activation = nn.ReLU()
        # 升维全连接层
        self.up_project = nn.Linear(adapter_size, input_size)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        # 初始化down_project用较小的值
        nn.init.normal_(self.down_project.weight, std=1e-2)
        nn.init.zeros_(self.down_project.bias)
        
        # 初始化up_project为接近零的值，确保训练初期对原始模型影响较小
        nn.init.normal_(self.up_project.weight, std=1e-2)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x):
        # 保存原始输入用于残差连接
        residual = x
        
        # 通过降维层
        x = self.down_project(x)
        # 激活
        x = self.activation(x)
        # 通过升维层
        x = self.up_project(x)
        
        # 残差连接
        return residual + x 