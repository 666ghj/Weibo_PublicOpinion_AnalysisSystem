<div align="center">

  <!-- # 📊 Weibo Public Opinion Analysis System  -->

  <img src="https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/blob/main/images/logo.png" alt="Weibo Public Opinion Analysis System Logo" width="500">

  [![GitHub Stars](https://img.shields.io/github/stars/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/stargazers)
  [![GitHub Forks](https://img.shields.io/github/forks/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/network)
  [![GitHub Issues](https://img.shields.io/github/issues/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/issues)
  [![GitHub Contributors](https://img.shields.io/github/contributors/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/graphs/contributors)
  [![GitHub License](https://img.shields.io/github/license/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/blob/main/LICENSE)


  [English](./README.md) | [中文文档](./README-CN.md)
</div>

---

**微博舆情分析预测系统** 是一个用于监控、分析和预测社交媒体平台（如微博）上的公众舆情趋势的**社交网络舆情分析系统**。该系统利用深度学习、自然语言处理（NLP）和机器学习技术，从大量社交媒体数据中提取有价值的舆情信息，帮助政府、企业及其他组织及时了解公众态度、应对突发事件并优化决策。📈

通过强大的数据采集与处理能力，微博舆情分析预测系统实现了实时数据收集、情感分析、话题分类和舆情预测等功能，确保用户能够在复杂多变的社交网络环境中获得准确、全面的舆情洞察。系统采用模块化设计，易于维护和扩展，旨在为用户提供一个高效、可靠的舆情分析工具，助力各类组织在信息化时代做出明智决策。

## ✨ 功能

- **实时数据采集**：通过网络爬虫技术，从微博等社交平台实时获取用户生成内容。
- **数据清洗与处理**：对采集到的数据进行预处理，包括分词、去停用词、表情符号和网址的去除等。
- **话题分类**：利用机器学习和自然语言处理技术，对帖子和评论进行自动话题分类。
- **情感分析**：分析文本中的情感倾向（正面、中性、负面），帮助理解公众情绪。
- **舆情监控与预测**：实时监控舆情变化，并基于历史数据预测未来的舆情趋势。
- **数据可视化**：通过图表和图形直观展示分析结果，便于用户理解和决策。
- **用户管理**：提供用户注册、登录和会话管理功能，确保系统的安全性和个性化服务。

## 🚀 开始使用

按照以下步骤在您的系统上运行该项目。

### 前提条件

- [Python](https://www.python.org/) 3.7 或更高版本
- [MySQL](https://www.mysql.com/) 数据库
- [Conda](https://docs.conda.io/en/latest/)（可选，用于环境管理）
- 合法的微博账号（用于数据采集）

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/YourUsername/Weibo-Public-Opinion-Analysis-System.git
   cd Weibo-Public-Opinion-Analysis-System

1. 创建并激活虚拟环境（可选）：

   ```bash
   conda create -n weibo_opinion_analysis python=3.8
   conda activate weibo_opinion_analysis
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 配置MySQL数据库：

   - 运行 `createTables.sql` 创建所需的数据库表。
   - 修改 `config.py` 中的数据库连接配置，确保与您的MySQL设置匹配。

4. 启动Flask应用：

   ```bash
   python app.py
   ```

5. 访问应用： 打开浏览器，访问 `http://localhost:5000` 以使用系统。

## 🛠️ 技术栈

微博舆情分析预测系统采用了一系列现代技术，以确保其高效性和可扩展性：

- **Flask** - 轻量级的Web应用框架。
- **[MySQL](https://www.mysql.com/)** - 关系型数据库，用于存储采集和处理的数据。
- **[Scrapy](https://scrapy.org/)** - 强大的网络爬虫框架，用于数据采集。
- **[Jieba](https://github.com/fxsjy/jieba)** - 中文分词工具，用于文本预处理。
- **[SnowNLP](https://github.com/isnowfy/snownlp)** - 中文自然语言处理库，用于情感分析。
- **[BERT](https://github.com/google-research/bert)** - 预训练的语言模型，用于话题分类。
- **Pandas** - 数据分析和处理库。
- **[Matplotlib](https://matplotlib.org/)** - 数据可视化库。
- **[Scikit-learn](https://scikit-learn.org/)** - 机器学习库，用于模型训练和评估。
- **[TensorFlow](https://www.tensorflow.org/)** 或 **[PyTorch](https://pytorch.org/)** - 深度学习框架，用于高级模型开发。

## 🤝 贡献

我们欢迎您的贡献！以下是参与项目的步骤：

1. Fork 本仓库。
2. 创建您的功能分支 (`git checkout -b feature/新功能`)。
3. 提交您的更改 (`git commit -m '添加新功能'`)。
4. 推送到分支 (`git push origin feature/新功能`)。
5. 打开一个 Pull Request。

请确保在提交之前运行所有测试，并遵循项目的编码规范。

## 📜 许可证

本项目采用 [MIT License](https://github.com/YourUsername/Weibo-Public-Opinion-Analysis-System/blob/main/LICENSE) 许可证 - 详情请参阅 [LICENSE](https://github.com/YourUsername/Weibo-Public-Opinion-Analysis-System/blob/main/LICENSE) 文件。

## 🌟 支持一下

如果您喜欢这个项目，请在 [GitHub](https://github.com/YourUsername/Weibo-Public-Opinion-Analysis-System) 上给它一个星 ⭐！

## 📫 联系我们

有任何问题或建议，欢迎通过以下方式联系我们：

- GitHub Issues: [创建新问题](https://github.com/YourUsername/Weibo-Public-Opinion-Analysis-System/issues)
- 邮箱: your-email@example.com

## ✨ 贡献者

感谢以下这些优秀的贡献者：

[![Contributors](https://contrib.rocks/image?repo=666ghj/Weibo_PublicOpinion_AnalysisSystem)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/graphs/contributors)
