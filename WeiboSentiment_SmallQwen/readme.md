# 微调Qwen3小参数模型来完成情感分析任务

<img src="https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/blob/main/static/image/logo_Qweb3.jpg" alt="微博情感分析示例" width="25%" />

## 项目背景

本文件夹专门用于基于阿里Qwen3系列模型的微博情感分析任务。根据最新的模型评测结果，Qwen3的小参数模型（如0.6B、4B、8B、14B）在话题识别、情感分析等相对简单的自然语言处理任务上表现优异，超越了传统的BERT等基础模型。

qwen 0.6B模型加线性分类器，做特定领域的文本分类和序列标注，优于bert，也优于235B的qwen3 few shot learning。在算力有限的情况下，性价比很高...

在经过了一些相关的调研之后，我觉的将Qwen3的一些小参数模型用在本系统中是一个不错的选择。

虽然这个参数在LLM时代算小，但作为个人开发者计算资源有限，微调他们还是实属不易。