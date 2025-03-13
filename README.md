<div align="center">

  <!-- # 📊 Weibo Public Opinion Analysis System  -->

  <img src="https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/blob/main/static/image/logo_compressed.png" alt="Weibo Public Opinion Analysis System Logo" width="800">

  [![GitHub Stars](https://img.shields.io/github/stars/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/stargazers)
  [![GitHub Forks](https://img.shields.io/github/forks/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/network)
  [![GitHub Issues](https://img.shields.io/github/issues/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/issues)
  [![GitHub Contributors](https://img.shields.io/github/contributors/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/graphs/contributors)
  [![GitHub License](https://img.shields.io/github/license/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/blob/main/LICENSE)


  [English](./README.md) | [中文文档](./README-CN.md)
</div>

🚀The latest 2.0 version has fully upgraded AI modes—welcome to experience it!⬇️

<div style="display: flex; justify-content: center; align-items: center; gap: 10px; width: 90%; margin: auto;">
    <img src="https://github.com/user-attachments/assets/7ea3f84c-4e08-448e-b2f5-35e1a1e4fbba" style="width: 45%; height: auto;">
    <img src="https://github.com/user-attachments/assets/37c69bb2-c579-4a81-86d4-a38522cebd73" style="width: 45%; height: auto;">
</div>

**Weibo Public Opinion Analysis and Prediction System** is a **social network public opinion analysis system** designed to monitor, analyze, and predict public opinion trends on social media platforms such as Weibo. This system leverages deep learning, natural language processing (NLP), and machine learning technologies to extract valuable public opinion information from vast amounts of social media data, helping governments, enterprises, and other organizations promptly understand public attitudes, respond to emergencies, and optimize decision-making. 📈

Through powerful data collection and processing capabilities, the Weibo Public Opinion Analysis and Prediction System achieves real-time data collection, sentiment analysis, topic classification, and public opinion prediction, ensuring that users can obtain accurate and comprehensive insights into public opinion in the complex and changing social network environment. The system adopts a modular design, making it easy to maintain and expand, aiming to provide users with an efficient and reliable public opinion analysis tool, assisting various organizations in making informed decisions in the information age.

## ✨ Features

- **Real-time Data Collection**: Utilize web scraping technologies to obtain user-generated content from social platforms like Weibo in real-time.
- **Data Cleaning and Processing**: Preprocess collected data, including tokenization, removal of stop words, emojis, and URLs.
- **Topic Classification**: Automatically classify posts and comments into topics using machine learning and natural language processing techniques.
- **Sentiment Analysis**: Analyze the sentiment orientation (positive, neutral, negative) within texts to understand public emotions.
- **Public Opinion Monitoring and Prediction**: Monitor changes in public opinion in real-time and predict future trends based on historical data.
- **Data Visualization**: Display analysis results through charts and graphics for easy understanding and decision-making.
- **User Management**: Provide user registration, login, and session management features to ensure system security and personalized services.

## 🚀 Getting Started

Follow the steps below to run the project on your system.

### Prerequisites

- [Python](https://www.python.org/) 3.7 or higher
- [MySQL](https://www.mysql.com/) Database
- [Conda](https://docs.conda.io/en/latest/) (optional, for environment management)
- A valid Weibo account (for data collection)
- At least one of the following API keys for AI analysis features:
  - OpenAI API key
  - Anthropic (Claude) API key
  - DeepSeek API key

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem.git
   cd Weibo-Public-Opinion-Analysis-System

2. Create and activate a virtual environment (optional):

   ```bash
   conda create -n weibo_opinion_analysis python=3.8
   conda activate weibo_opinion_analysis
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the MySQL database:

   - Run `createTables.sql` to create the necessary database tables.
   - Modify the database connection settings in `config.py` to match your MySQL configuration.

5. Configure AI Analysis (Optional):
   
   Set up environment variables for AI analysis features:
   ```bash
   # For OpenAI API (Required for GPT models)
   export OPENAI_API_KEY="your-openai-key"
   
   # For Anthropic API (Required for Claude models)
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # For DeepSeek API (Required for DeepSeek models)
   export DEEPSEEK_API_KEY="your-deepseek-key"
   ```
   
   Note: At least one API key must be configured to use AI analysis features.
   
   Supported AI Models:
   - OpenAI: GPT-3.5-Turbo, GPT-4
   - Anthropic: Claude-3 (Opus, Sonnet, Haiku)
   - DeepSeek: DeepSeek-V3 (deepseek-chat), DeepSeek-R1 (deepseek-reasoner)

6. Start the Flask application:

   ```bash
   python app.py
   ```

7. Access the application: Open your browser and navigate to http://localhost:5000 to use the system.

## 🛠️ Technology Stack

The Weibo Public Opinion Analysis and Prediction System employs a range of modern technologies to ensure efficiency and scalability:

- **[Flask](https://flask.palletsprojects.com/en/stable/)** - A lightweight web application framework.
- **[MySQL](https://www.mysql.com/)** - A relational database used to store collected and processed data.
- **[Scrapy](https://scrapy.org/)** - A powerful web scraping framework used for data collection.
- **[Jieba](https://github.com/fxsjy/jieba)** - A Chinese text segmentation tool used for text preprocessing.
- **[SnowNLP](https://github.com/isnowfy/snownlp)** - A Chinese natural language processing library used for sentiment analysis.
- **[BERT](https://github.com/google-research/bert)** - A pre-trained language model used for topic classification.
- **[Pandas](https://pandas.pydata.org/)** - A data analysis and manipulation library.
- **[Matplotlib](https://matplotlib.org/)** - A data visualization library.
- **[Scikit-learn](https://scikit-learn.org/)** - A machine learning library used for model training and evaluation.
- **[TensorFlow](https://www.tensorflow.org/)** 或 **[PyTorch](https://pytorch.org/)** - Deep learning frameworks used for advanced model development.
- **[OpenAI GPT](https://openai.com/)** - Advanced language models for text analysis.
- **[Anthropic Claude](https://www.anthropic.com/)** - AI models for sophisticated text analysis.
- **[DeepSeek](https://deepseek.com/)** - Advanced Chinese-English bilingual AI models.

## 🤝 Contribution

We welcome your contributions! Follow the steps below to participate in the project:

1. Fork this repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure that all tests pass before submitting and follow the project's coding standards.

## 📜 License

This project is licensed under the [GPL-2.0 License](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/blob/main/LICENSE) - see the [LICENSE](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/blob/main/LICENSE) file for details.

## 🌟 Show Your Support

If you like this project, please give it a star ⭐ on [GitHub](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem)!

## 📫 Contact Us

If you have any questions or suggestions, feel free to contact us through the following methods:

- GitHub Issues: [Create a new issue](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/issues)
- Email: 670939375@qq.com

## ✨ Contributors

Thanks to the following contributors:

[![Contributors](https://contrib.rocks/image?repo=666ghj/Weibo_PublicOpinion_AnalysisSystem)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/graphs/contributors)
