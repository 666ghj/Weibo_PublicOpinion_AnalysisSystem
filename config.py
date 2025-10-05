# -*- coding: utf-8 -*-
"""Central configuration for the public opinion analysis platform."""

# =========================
# Database configuration
# =========================
DB_HOST = "your_database_host"
DB_PORT = 3306
DB_USER = "your_database_user"
DB_PASSWORD = "your_database_password"
DB_NAME = "your_database_name"
DB_CHARSET = "utf8mb4"

# =========================
# LLM provider defaults
# =========================
DEFAULT_LLM_PROVIDER = "deepseek"

# DeepSeek (OpenAI-compatible)
DEEPSEEK_API_KEY = "your_deepseek_api_key"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_BASE = "https://api.deepseek.com"

# OpenAI
OPENAI_API_KEY = "your_openai_api_key"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_BASE = "https://api.openai.com/v1"

# Moonshot Kimi
KIMI_API_KEY = "your_kimi_api_key"
KIMI_MODEL = "kimi-k2-0711-preview"
KIMI_API_BASE = "https://api.moonshot.cn/v1"

# Gemini (via OpenAI-compatible proxy)
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_API_BASE = "https://www.chataiapi.com/v1"

# SiliconFlow Qwen (used for keyword optimisation and forum host)
GUIJI_QWEN3_API_KEY = "your_guiji_qwen3_api_key"
GUIJI_QWEN3_API_BASE = "https://api.siliconflow.cn/v1"
GUIJI_QWEN3_KEYWORD_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
GUIJI_QWEN3_FORUM_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# =========================
# External search providers
# =========================
TAVILY_API_KEY = "your_tavily_api_key"

# Bocha search (kept backward compatible alias)
BOCHA_API_KEY = "your_bocha_web_search_api_key"
BOCHA_Web_Search_API_KEY = BOCHA_API_KEY

# =========================
# Miscellaneous options
# =========================
# 我们也提供云数据库资源便捷配置，日均10w+数据，目前推广阶段可免费申请，联系我们：670939375@qq.com
