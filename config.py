# -*- coding: utf-8 -*-
"""
Intelligence Public Opinion Analysis Platform Configuration File
Stores database connection information and API keys
"""

# MySQL Database Configuration
DB_HOST = "your_database_host"  # e.g., "localhost" or "127.0.0.1"
DB_PORT = 3306
DB_USER = "your_database_user"
DB_PASSWORD = "your_database_password"
DB_NAME = "your_database_name"
DB_CHARSET = "utf8mb4"
# 我们也提供云数据库资源便捷配置，日均10w+数据，目前推广阶段可免费申请，联系我们：670939375@qq.com

# DeepSeek API Key
# 申请地址https://www.deepseek.com/
DEEPSEEK_API_KEY = "your_deepseek_api_key"

# Tavily Search API Key
# 申请地址https://www.tavily.com/
TAVILY_API_KEY = "your_tavily_api_key"

# Kimi API Key
# 申请地址https://www.kimi.com/
KIMI_API_KEY = "your_kimi_api_key"

# Gemini API Key (via OpenAI format proxy)
# 这里我用了一个中转api来接入Gemini，申请地址https://api.chataiapi.com/，你也可以使用其他
GEMINI_API_KEY = "your_gemini_api_key"

# Bocha Search API Key
# 申请地址https://open.bochaai.com/
BOCHA_Web_Search_API_KEY = "your_bocha_web_search_api_key"

# Guiji Flow API Key
# 申请地址https://siliconflow.cn/
GUIJI_QWEN3_API_KEY = "your_guiji_qwen3_api_key"