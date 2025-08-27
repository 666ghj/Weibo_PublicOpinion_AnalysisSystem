# -*- coding: utf-8 -*-
"""
智能舆情分析平台配置文件
存储数据库连接信息和API密钥
"""

# MySQL数据库配置
DB_HOST = "rm-2zeib6b13f6tt9kncoo.mysql.rds.aliyuncs.com"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = "mneDccc7sHHANtFk"
DB_NAME = "media_crawler"
DB_CHARSET = "utf8mb4"

# DeepSeek API密钥
DEEPSEEK_API_KEY = "sk-db84c08a6f9a439b8eb798ad9ef22225"

# Tavily搜索API密钥
TAVILY_API_KEY = "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"

# Kimi API密钥
KIMI_API_KEY = "sk-H3vxh28PQMJajvAon6nrqVFcf9Igs5cVKVn2v7UUthRrmje3"

# Gemini API密钥（中转，OpenAI调用格式）
GEMINI_API_KEY = "sk-JjKFgVz5NsXAWjflIFM82Z3eGwpunP7kq0HBiLh0suRJDLtp"

# 博查搜索API密钥
BOCHA_Web_Search_API_KEY = "sk-8dfcc8b40d81448ca41f1d8d50aba2e9"

# 硅基流动API密钥
GUIJI_QWEN3_API_KEY = "sk-qrkvwyhqodwwdldpzsuaipoxiepqeqelhguwkoklbdnemybt"