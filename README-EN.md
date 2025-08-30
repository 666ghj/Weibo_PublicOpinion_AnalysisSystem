<div align="center">

<img src="static/image/logo_compressed.png" alt="Weibo Public Opinion Analysis System Logo" width="600">

[![GitHub Stars](https://img.shields.io/github/stars/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/stargazers)
[![GitHub Watchers](https://img.shields.io/github/watchers/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/watchers)
[![GitHub Forks](https://img.shields.io/github/forks/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/network)
[![GitHub Issues](https://img.shields.io/github/issues/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/issues)
[![GitHub License](https://img.shields.io/github/license/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/blob/main/LICENSE)

[English](./README-EN.md) | [中文文档](./README.md)

</div>

## 📝 Project Overview

**"WeiYu"** is an innovative multi-agent public opinion analysis system built from scratch, featuring universal simplicity across all platforms.

See the system-generated research report on "Wuhan University Public Opinion":[In-depth Analysis Report on Wuhan University's Brand Reputation](./final_reports/final_report__20250827_131630.html)

Beyond just report quality, compared to similar products, we have 🚀 six major advantages:

1. **AI-Driven Comprehensive Monitoring**: AI crawler clusters operate 24/7 non-stop, comprehensively covering 10+ key domestic and international social media platforms including Weibo, Xiaohongshu, TikTok, Kuaishou, etc. Not only capturing trending content in real-time, but also drilling down to massive user comments, letting you hear the most authentic and widespread public voice.

2. **Composite Analysis Engine Beyond LLM**: We not only rely on 5 types of professionally designed Agents, but also integrate middleware such as fine-tuned models and statistical models. Through multi-model collaborative work, we ensure the depth, accuracy, and multi-dimensional perspective of analysis results.

3. **Powerful Multimodal Capabilities**: Breaking through text and image limitations, capable of deep analysis of short video content from TikTok, Kuaishou, etc., and precisely extracting structured multimodal information cards such as weather, calendar, stocks from modern search engines, giving you comprehensive control over public opinion dynamics.

4. **Agent "Forum" Collaboration Mechanism**: Endowing different Agents with unique toolsets and thinking patterns, conducting chain-of-thought collision and debate through the "forum" mechanism. This not only avoids the thinking limitations of single models and homogenization caused by communication, but also catalyzes higher-quality collective intelligence and decision support.

5. **Seamless Integration of Public and Private Domain Data**: The platform not only analyzes public opinion, but also provides high-security interfaces supporting seamless integration of your internal business databases with public opinion data. Breaking through data barriers, providing powerful analysis capabilities of "external trends + internal insights" for vertical businesses.

6. **Lightweight and Highly Extensible Framework**: Based on pure Python modular design, achieving lightweight, one-click deployment. Clear code structure allows developers to easily integrate custom models and business logic, enabling rapid platform expansion and deep customization.

**Starting with public opinion, but not limited to public opinion**. The goal of "WeiYu" is to become a simple and universal data analysis engine that drives all business scenarios.

<div align="center">
<img src="static/image/system_schematic.png" alt="banner" width="800">

Say goodbye to traditional data dashboards. In "WeiYu", everything starts with a simple question - you just need to ask your analysis needs like a conversation
</div>

## 🏗️ System Architecture

### Overall Architecture Diagram

Still drawing...

### Project Code Structure Tree

```
Weibo_PublicOpinion_AnalysisSystem/
├── QueryEngine/                   # Domestic and international news breadth search Agent
│   ├── agent.py                   # Agent main logic
│   ├── llms/                      # LLM interface wrapper
│   ├── nodes/                     # Processing nodes
│   ├── tools/                     # Search tools
│   ├── utils/                     # Utility functions
│   └── ...                        # Other modules
├── MediaEngine/                   # Powerful multimodal understanding Agent
│   ├── agent.py                   # Agent main logic
│   ├── llms/                      # LLM interfaces
│   ├── tools/                     # Search tools
│   └── ...                        # Other modules
├── InsightEngine/                 # Private database mining Agent
│   ├── agent.py                   # Agent main logic
│   ├── llms/                      # LLM interface wrapper
│   │   ├── deepseek.py            # DeepSeek API
│   │   ├── kimi.py                # Kimi API
│   │   ├── openai_llm.py          # OpenAI format API
│   │   └── base.py                # LLM base class
│   ├── nodes/                     # Processing nodes
│   │   ├── first_search_node.py   # First search node
│   │   ├── reflection_node.py     # Reflection node
│   │   ├── summary_nodes.py       # Summary node
│   │   ├── search_node.py         # Search node
│   │   ├── sentiment_node.py      # Sentiment analysis node
│   │   └── insight_node.py        # Insight generation node
│   ├── tools/                     # Database query and analysis tools
│   │   ├── media_crawler_db.py    # Database query tool
│   │   └── sentiment_analyzer.py  # Sentiment analysis integration tool
│   ├── state/                     # State management
│   │   ├── __init__.py
│   │   └── state.py               # Agent state definition
│   ├── prompts/                   # Prompt templates
│   │   ├── __init__.py
│   │   └── prompts.py             # Various prompts
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       └── helpers.py             # Helper functions
├── ReportEngine/                  # Multi-round report generation Agent
│   ├── agent.py                   # Agent main logic
│   ├── llms/                      # LLM interfaces
│   │   └── gemini.py              # Gemini API dedicated
│   ├── nodes/                     # Report generation nodes
│   │   ├── template_selection.py  # Template selection node
│   │   └── html_generation.py     # HTML generation node
│   ├── report_template/           # Report template library
│   │   ├── 社会公共热点事件分析.md
│   │   ├── 商业品牌舆情监测.md
│   │   └── ...                    # More templates
│   └── flask_interface.py         # Flask API interface
├── ForumEngine/                   # Forum engine simple implementation
│   └── monitor.py                 # Log monitoring and forum management
├── MindSpider/                    # Weibo crawler system
│   ├── main.py                    # Crawler main program
│   ├── BroadTopicExtraction/      # Topic extraction module
│   │   ├── get_today_news.py      # Today's news fetching
│   │   └── topic_extractor.py     # Topic extractor
│   ├── DeepSentimentCrawling/     # Deep sentiment crawling
│   │   ├── MediaCrawler/          # Media crawler core
│   │   └── platform_crawler.py    # Platform crawler management
│   └── schema/                    # Database schema
│       └── init_database.py       # Database initialization
├── SentimentAnalysisModel/        # Sentiment analysis model collection
│   ├── WeiboSentiment_Finetuned/  # Fine-tuned BERT/GPT-2 models
│   ├── WeiboMultilingualSentiment/# Multilingual sentiment analysis (recommended)
│   ├── WeiboSentiment_SmallQwen/  # Small parameter Qwen3 fine-tuning
│   └── WeiboSentiment_MachineLearning/ # Traditional machine learning methods
├── SingleEngineApp/               # Individual Agent Streamlit applications
│   ├── query_engine_streamlit_app.py
│   ├── media_engine_streamlit_app.py
│   └── insight_engine_streamlit_app.py
├── templates/                     # Flask templates
│   └── index.html                 # Main interface frontend
├── static/                        # Static resources
├── logs/                          # Runtime log directory
├── final_reports/                 # Final generated HTML report files
├── utils/                         # Common utility functions
├── app.py                         # Flask main application entry
├── config.py                      # Global configuration file
└── requirements.txt               # Python dependency list
```

## 🚀 Quick Start

### System Requirements

- **Operating System**: Windows, Linux, MacOS
- **Python Version**: 3.9+
- **Conda**: Anaconda or Miniconda
- **Database**: MySQL (optional, you can choose our cloud database service)
- **Memory**: 2GB+ recommended

### 1. Create Conda Environment

```bash
# Create conda environment
conda create -n your_conda_name python=3.11
conda activate your_conda_name
```

### 2. Install Dependencies

```bash
# Basic dependency installation
pip install -r requirements.txt

#========Below are optional========
# If you need local sentiment analysis functionality, install PyTorch
# CPU version
pip install torch torchvision torchaudio

# CUDA 11.8 version (if you have GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and other AI-related dependencies
pip install transformers scikit-learn xgboost
```

### 3. Install Playwright Browser Drivers

```bash
# Install browser drivers (for crawler functionality)
playwright install chromium
```

### 4. System Configuration

#### 4.1 Configure API Keys

Edit the `config.py` file and fill in your API keys (you can also choose your own models and search proxies):

```python
# MySQL Database Configuration
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "your_username"
DB_PASSWORD = "your_password"
DB_NAME = "weibo_analysis"
DB_CHARSET = "utf8mb4"

# DeepSeek API (Apply at: https://www.deepseek.com/)
DEEPSEEK_API_KEY = "your_deepseek_api_key"

# Tavily Search API (Apply at: https://www.tavily.com/)
TAVILY_API_KEY = "your_tavily_api_key"

# Kimi API (Apply at: https://www.kimi.com/)
KIMI_API_KEY = "your_kimi_api_key"

# Gemini API (Apply at: https://api.chataiapi.com/)
GEMINI_API_KEY = "your_gemini_api_key"

# Bocha Search API (Apply at: https://open.bochaai.com/)
BOCHA_Web_Search_API_KEY = "your_bocha_api_key"

# Silicon Flow API (Apply at: https://siliconflow.cn/)
GUIJI_QWEN3_API_KEY = "your_guiji_api_key"
```

#### 4.2 Database Initialization

**Option 1: Use Local Database**
```bash
# Local MySQL database initialization
cd MindSpider
python schema/init_database.py
```

**Option 2: Use Cloud Database Service (Recommended)**

We provide convenient cloud database service with 100,000+ daily real public opinion data, currently **free application** during the promotion period!

- Real public opinion data, updated in real-time
- Multi-dimensional tag classification
- High-availability cloud service
- Professional technical support

**Contact us to apply for free cloud database access: 📧 670939375@qq.com**

### 5. Launch System

#### 5.1 Complete System Launch (Recommended)

```bash
# In project root directory, activate conda environment
conda activate your_conda_name

# Start main application
python app.py
```

> Note: Data crawling requires separate operation, see section 5.3 for guidance

Visit http://localhost:5000 to use the complete system

#### 5.2 Launch Individual Agents

```bash
# Start QueryEngine
streamlit run SingleEngineApp/query_engine_streamlit_app.py --server.port 8503

# Start MediaEngine  
streamlit run SingleEngineApp/media_engine_streamlit_app.py --server.port 8502

# Start InsightEngine
streamlit run SingleEngineApp/insight_engine_streamlit_app.py --server.port 8501
```

#### 5.3 Crawler System Standalone Use

This section has detailed configuration documentation: [MindSpider Usage Guide](./MindSpider/README.md)

```bash
# Enter crawler directory
cd MindSpider

# Project initialization
python main.py --setup

# Run complete crawler workflow
python main.py --complete --date 2024-01-20

# Run topic extraction only
python main.py --broad-topic --date 2024-01-20

# Run deep crawling only
python main.py --deep-sentiment --platforms xhs dy wb
```

## ⚙️ Advanced Configuration

### Modify Key Parameters

#### Agent Configuration Parameters

Each agent has dedicated configuration files that can be adjusted according to needs:

```python
# QueryEngine/utils/config.py
class Config:
    max_reflections = 2           # Reflection rounds
    max_search_results = 15       # Maximum search results
    max_content_length = 8000     # Maximum content length
    
# MediaEngine/utils/config.py  
class Config:
    comprehensive_search_limit = 10  # Comprehensive search limit
    web_search_limit = 15           # Web search limit
    
# InsightEngine/utils/config.py
class Config:
    default_search_topic_globally_limit = 200    # Global search limit
    default_get_comments_limit = 500             # Comment retrieval limit
    max_search_results_for_llm = 50              # Max results for LLM
```

#### Sentiment Analysis Model Configuration

```python
# InsightEngine/tools/sentiment_analyzer.py
SENTIMENT_CONFIG = {
    'model_type': 'multilingual',     # Options: 'bert', 'multilingual', 'qwen'
    'confidence_threshold': 0.8,      # Confidence threshold
    'batch_size': 32,                 # Batch size
    'max_sequence_length': 512,       # Max sequence length
}
```

### Integrate Different LLM Models

The system supports multiple LLM providers, switchable in each agent's configuration:

```python
# Configure in each Engine's utils/config.py
class Config:
    default_llm_provider = "deepseek"  # Options: "deepseek", "openai", "kimi", "gemini", "qwen"
    
    # DeepSeek configuration
    deepseek_api_key = "your_api_key"
    deepseek_model = "deepseek-chat"
    
    # OpenAI compatible configuration
    openai_api_key = "your_api_key"
    openai_model = "gpt-3.5-turbo"
    openai_base_url = "https://api.openai.com/v1"
    
    # Kimi configuration
    kimi_api_key = "your_api_key"  
    kimi_model = "moonshot-v1-8k"
    
    # Gemini configuration
    gemini_api_key = "your_api_key"
    gemini_model = "gemini-pro"
```

### Change Sentiment Analysis Models

The system integrates multiple sentiment analysis methods, selectable based on needs:

#### 1. Multilingual Sentiment Analysis

```bash
cd SentimentAnalysisModel/WeiboMultilingualSentiment
python predict.py --text "This product is amazing!" --lang "en"
```

#### 2. Small Parameter Qwen3 Fine-tuning

```bash
cd SentimentAnalysisModel/WeiboSentiment_SmallQwen
python predict_universal.py --text "This event was very successful"
```

#### 3. BERT-based Fine-tuned Model

```bash
# Use BERT Chinese model
cd SentimentAnalysisModel/WeiboSentiment_Finetuned/BertChinese-Lora
python predict.py --text "This product is really great"
```

#### 4. GPT-2 LoRA Fine-tuned Model

```bash
cd SentimentAnalysisModel/WeiboSentiment_Finetuned/GPT2-Lora
python predict.py --text "I'm not feeling great today"
```

#### 5. Traditional Machine Learning Methods

```bash
cd SentimentAnalysisModel/WeiboSentiment_MachineLearning
python predict.py --model_type "svm" --text "Service attitude needs improvement"
```

### Integrate Custom Business Database

#### 1. Modify Database Connection Configuration

```python
# Add your business database configuration in config.py
BUSINESS_DB_HOST = "your_business_db_host"
BUSINESS_DB_PORT = 3306
BUSINESS_DB_USER = "your_business_user"
BUSINESS_DB_PASSWORD = "your_business_password"
BUSINESS_DB_NAME = "your_business_database"
```

#### 2. Create Custom Data Access Tools

```python
# InsightEngine/tools/custom_db_tool.py
class CustomBusinessDBTool:
    """Custom business database query tool"""
    
    def __init__(self):
        self.connection_config = {
            'host': config.BUSINESS_DB_HOST,
            'port': config.BUSINESS_DB_PORT,
            'user': config.BUSINESS_DB_USER,
            'password': config.BUSINESS_DB_PASSWORD,
            'database': config.BUSINESS_DB_NAME,
        }
    
    def search_business_data(self, query: str, table: str):
        """Query business data"""
        # Implement your business logic
        pass
    
    def get_customer_feedback(self, product_id: str):
        """Get customer feedback data"""
        # Implement customer feedback query logic
        pass
```

#### 3. Integrate into InsightEngine

```python
# Integrate custom tools in InsightEngine/agent.py
from .tools.custom_db_tool import CustomBusinessDBTool

class DeepSearchAgent:
    def __init__(self, config=None):
        # ... other initialization code
        self.custom_db_tool = CustomBusinessDBTool()
    
    def execute_custom_search(self, query: str):
        """Execute custom business data search"""
        return self.custom_db_tool.search_business_data(query, "your_table")
```

### Custom Report Templates

#### 1. Upload in Web Interface

The system supports uploading custom template files (.md or .txt format), selectable when generating reports.

#### 2. Create Template Files

Create new templates in the `ReportEngine/report_template/` directory, and our Agent will automatically select the most appropriate template.

## 🤝 Contributing Guide

We welcome all forms of contributions!

### How to Contribute

1. **Fork the project** to your GitHub account
2. **Create Feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to branch**: `git push origin feature/AmazingFeature`
5. **Open Pull Request**

### Development Standards

- Code follows PEP8 standards
- Commit messages use clear Chinese/English descriptions
- New features need corresponding test cases
- Update related documentation

## 📄 License

This project is licensed under the [GPL-2.0 License](LICENSE). Please see the LICENSE file for details.

## 🎉 Support & Contact

### Get Help

- **Project Homepage**: [GitHub Repository](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem)
- **Issue Reporting**: [Issues Page](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/issues)
- **Feature Requests**: [Discussions Page](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/discussions)

### Contact Information

- 📧 **Email**: 670939375@qq.com

### Business Cooperation

- 🏢 **Enterprise Custom Development**
- 📊 **Big Data Services**
- 🎓 **Academic Collaboration**
- 💼 **Technical Training**

### Cloud Service Application

**Free Cloud Database Service Application**:
📧 Send email to: 670939375@qq.com  
📝 Subject: WeiYu Cloud Database Application  
📝 Description: Your use case and requirements  

## 👥 Contributors

Thanks to these excellent contributors:

[![Contributors](https://contrib.rocks/image?repo=666ghj/Weibo_PublicOpinion_AnalysisSystem)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/graphs/contributors)

---

<div align="center">

**⭐ If this project helps you, please give us a star!**

</div>