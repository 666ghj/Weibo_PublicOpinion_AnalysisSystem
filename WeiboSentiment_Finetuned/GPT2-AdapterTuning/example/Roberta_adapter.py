from transformers.models.roberta.modeling_roberta import RobertaLayer

class RobertaLayerWithAdapter(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        # 假设Adapter的大小为64
        adapter_size = 64
        self.adapter = AdapterLayer(config.hidden_size, adapter_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        # 调用原始的前向传播方法
        self_outputs = super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        # 得到Transformer层的输出
        sequence_output = self_outputs[0]
        # 将输出通过Adapter层
        sequence_output = self.adapter(sequence_output)
        # 返回修改后的输出（其他输出保持不变）
        return (sequence_output,) + self_outputs[1:]

"""
RoBERTa的每个RobertaLayer包含一个自注意力（self-attention）机制和一个前馈网络，这些层共同构成了RoBERTa的基础架构。
"""
