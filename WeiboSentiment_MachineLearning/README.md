# WeiboSentiment
用各种机器学习对中文微博进行情感分析    
语料来源： https://github.com/dengxiuqi/weibo2018
---
##### "微博情感分析"是我本科的毕业设计, 也是我入门NLP的项目, 就把它发出来供大家交流。
##### 2021.06.07更新: 之前的版本写得比较随意, 没想到star破百了, 私下也有一些刚入门NLP的同学因为这个项目联系我, 就更新一下这个项目吧
* 重构项目架构和代码, 提高可读性
* 每个文件中的特征、数据处理方法与模型细节都尽可能避免重复, 以给各位同学提供更多的参考
* 神经网络结构换成了pytorch, 需要`tensorflow 1.0`代码的同学请回退至`445998`版本。    
* 新增了`Bert`模型
* 由于gensim新老版本很多语法不兼容, 将gensim更新为4.0版本
----
#### 项目说明
* 训练集10000条语料, 测试集500条语料
* 使用朴素贝叶斯、SVM、XGBoost、LSTM和Bert, 等多种模型搭建并训练二分类模型
* 前3个模型都采用端到端的训练方法
* LSTM先预训练得到Word2Vec词向量, 在训练神经网络
* `Bert`使用的是哈工大的预训练模型, 用Bert的`[CLS]`位输出在一个下游网络上进行finetune。预训练模型需要自行下载:    
    * github下载地址: https://github.com/ymcui/Chinese-BERT-wwm
    * baidu网盘: https://pan.baidu.com/s/16z-ybrqT6wLdy_mLHtywSw  密码: djkj
    * 下载后将文件夹放在`./model`文件夹下, 并将`bert_config.json`改名为`config.json`
---
#### 实验结果
各种分类器在测试集上的测试结果  

|模型|准确率|AUC|
| :---: | :---: | :---: |
|1.bayes|0.856| - |
|2.svm|0.856| - |
|3.xgboost|0.86| 0.904 |
|4.lstm|0.87| 0.931 |
|5.bert|0.87| 0.929 |
