from transformers.models.gpt2.modeling_gpt2 import GPT2Block

class GPT2BlockWithAdapter(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        # 假设Adapter的大小为64
        adapter_size = 64
        self.adapter = AdapterLayer(config.n_embd, adapter_size)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 调用原始的前向传播方法
        attn_outputs = super().forward(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 得到Transformer层的输出
        a = attn_outputs[0]  # 输出的第一部分是attention的结果
        # 将输出通过Adapter层
        a = self.adapter(a)
        # 返回修改后的输出（其他输出保持不变）
        outputs = (a,) + attn_outputs[1:]
        return outputs
"""
每个GPT2Block包含了一系列的自注意力（Self-Attention）和前馈网络（Feed-Forward）层，这些层共同构成了模型的基础架构。

"""


