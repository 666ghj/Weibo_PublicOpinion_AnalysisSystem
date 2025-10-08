# -*- coding: utf-8 -*-
"""
Intelligence Public Opinion Analysis Platform Configuration File
Stores database connection information and API keys
"""

# ============================== 数据库配置 ==============================
# MySQL Database Configuration
DB_HOST = "your_database_host"  # e.g., "localhost" or "127.0.0.1"
DB_PORT = 3306  # e.g., 3306
DB_USER = "your_database_user"
DB_PASSWORD = "your_database_password"
DB_NAME = "your_database_name"
DB_CHARSET = "utf8mb4"
# 我们也提供云数据库资源便捷配置，日均10w+数据，学术研究可免费申请，联系我们：670939375@qq.com


# ============================== LLM配置 ==============================
# 重要提醒：推荐第一次先按照默认模型安排配置，成功跑通后再更改自己的模型！

# DeepSeek API Key (openai调用格式)
# 用于Query Agent
# 申请地址https://www.deepseek.com/
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxx"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Kimi API Key (openai调用格式)
# 用于Insight Agent
# 申请地址https://platform.moonshot.cn/
KIMI_API_KEY = "sk-xxxxxxxxxxxxxxxxx"
KIMI_BASE_URL = "https://api.moonshot.cn/v1"

# Gemini API Key (openai调用格式)
# 用于Media Agent与Report Agent
# 这里我用了一个中转api来接入Gemini，申请地址https://api.chataiapi.com/，你也可以使用其他
GEMINI_API_KEY = "sk-xxxxxxxxxxxxxxxxx"
GEMINI_BASE_URL = "https://www.chataiapi.com/v1"

# Siliconflow API Key (openai调用格式)
# 用于Forum Host与keyword Optimizer
# 申请地址https://siliconflow.cn/
GUIJI_QWEN3_API_KEY = "sk-xxxxxxxxxxxxxxxxx"
GUIJI_QWEN3_BASE_URL = "https://api.siliconflow.cn/v1"

# 调试阶段出于成本考虑，没有使用ChatGPT与Claude，您也可以接入自己的模型，只要符合openai调用格式即可


# ============================== Web工具配置 ==============================
# Tavily Search API Key
# 申请地址https://www.tavily.com/
TAVILY_API_KEY = "tvly-xxxxxxxxxxxxxxxxx"

# Bocha Search API Key
# 申请地址https://open.bochaai.com/
BOCHA_Web_Search_API_KEY = "sk-xxxxxxxxxxxxxxxxx"
