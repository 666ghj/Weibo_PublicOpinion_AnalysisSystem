import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from adapter import AdapterLayer

class GPT2BlockWithAdapter(nn.Module):
    """
    带Adapter的GPT2Block层
    在原始GPT2Block的基础上添加Adapter层实现参数高效微调
    """
    def __init__(self, config):
        super(GPT2BlockWithAdapter, self).__init__()
        # 创建标准的GPT2Block
        self.original_block = GPT2Block(config)
        
        # 添加Adapter层
        adapter_size = 64  # Adapter的隐藏层大小
        self.adapter = AdapterLayer(config.hidden_size, adapter_size)
    
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs  # 使用**kwargs接收所有其他参数
    ):
        # 首先通过原始的GPT2Block，只传递它支持的参数
        outputs = self.original_block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # 原始输出中的第一个元素是隐藏状态
        hidden_states = outputs[0]
        
        # 将隐藏状态通过Adapter层
        hidden_states = self.adapter(hidden_states)
        
        # 更新输出的隐藏状态
        outputs = (hidden_states,) + outputs[1:]
        
        return outputs
    
    def load_state_dict(self, state_dict, strict=True):
        """
        自定义加载参数方法，用于从原始GPT2Block加载参数
        """
        # 将所有参数传递给原始Block
        return self.original_block.load_state_dict(state_dict, strict=strict) 