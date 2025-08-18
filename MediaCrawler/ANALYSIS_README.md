# MediaCrawler 舆情分析功能使用指南

## 🎯 功能概述

MediaCrawler现已集成完整的微博舆情分析功能，支持：
- **情感分析**: 自动识别帖子和评论的情感倾向
- **话题检测**: 自动发现和聚类讨论话题
- **实时分析**: 数据爬取时自动触发分析
- **Web API**: 提供完整的RESTful API接口

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install transformers torch jieba flask aiofiles aiosqlite
```

### 2. 启动服务
```bash
cd MediaCrawler
python flask_app.py
```
服务将在 http://localhost:5001 启动

### 3. 检查系统状态
```bash
curl http://localhost:5001/system/status
```

## 📊 API接口说明

### 爬虫控制
```bash
# 启动爬虫任务
curl -X POST http://localhost:5001/crawler/start \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "wb",
    "keywords": "人工智能,AI",
    "type": "search",
    "save_data_option": "sqlite",
    "get_comment": true
  }'

# 查看爬虫状态
curl http://localhost:5001/crawler/status
```

### 数据查询
```bash
# 获取帖子数据
curl "http://localhost:5001/data/posts?limit=10&platform=wb"

# 获取评论数据
curl "http://localhost:5001/data/comments?limit=20&post_id=帖子ID"
```

### 情感分析
```bash
# 分析帖子情感
curl -X POST http://localhost:5001/analysis/sentiment/posts \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "multilingual",
    "limit": 50
  }'

# 分析评论情感
curl -X POST http://localhost:5001/analysis/sentiment/comments \
  -H "Content-Type: application/json" \
  -d '{
    "post_id": "帖子ID",
    "model_type": "multilingual",
    "limit": 100
  }'
```

### 话题检测
```bash
# 检测帖子话题
curl -X POST http://localhost:5001/analysis/topics/posts \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "bertopic",
    "num_topics": 10,
    "limit": 100
  }'

# 检测评论话题
curl -X POST http://localhost:5001/analysis/topics/comments \
  -H "Content-Type: application/json" \
  -d '{
    "post_id": "帖子ID",
    "model_type": "keyword_extraction",
    "num_topics": 5
  }'
```

### 综合分析
```bash
# 一键综合分析
curl -X POST http://localhost:5001/analysis/comprehensive \
  -H "Content-Type: application/json" \
  -d '{
    "include_comments": true,
    "sentiment_model": "multilingual",
    "topic_model": "bertopic",
    "num_topics": 10,
    "limit": 50
  }'
```

## ⚙️ 自动分析配置

### 查看钩子状态
```bash
curl http://localhost:5001/analysis/hooks/status
```

### 配置自动分析
```bash
# 启用自动情感分析，禁用话题检测
curl -X POST http://localhost:5001/analysis/hooks/config \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "auto_sentiment": true,
    "auto_topic": false,
    "batch_size": 10
  }'

# 强制处理待分析数据
curl -X POST http://localhost:5001/analysis/hooks/flush
```

## 📁 数据存储结构

### 原始数据
- **SQLite数据库**: `schema/sqlite_tables.db`
  - `weibo_note` 表: 帖子数据
  - `weibo_comment` 表: 评论数据

### 分析结果
- **结果目录**: `data/analysis_results/`
  - `sentiment_post_results.jsonl`: 帖子情感分析结果
  - `sentiment_comment_results.jsonl`: 评论情感分析结果
  - `topic_posts_results.jsonl`: 帖子话题检测结果
  - `topic_comments_results.jsonl`: 评论话题检测结果
  - `batch_posts_results.jsonl`: 批处理结果记录

## 🎛️ 配置选项

### 情感分析模型
- `multilingual`: 多语言情感分析模型 (推荐)
- `machine_learning`: 传统机器学习模型
- `qwen`: Qwen大语言模型

### 话题检测模型
- `bertopic`: BERTopic话题建模 (推荐)
- `keyword_extraction`: 关键词提取聚类

### 自动分析配置
- `auto_sentiment`: 是否自动进行情感分析
- `auto_topic`: 是否自动进行话题检测
- `batch_size`: 批处理大小 (1-100)

## 📊 分析结果格式

### 情感分析结果
```json
{
  "post_id": "帖子ID",
  "sentiment_label": 3,
  "sentiment_text": "正面",
  "confidence": 0.85,
  "sentiment_scores": {
    "very_negative": 0.02,
    "negative": 0.05,
    "neutral": 0.08,
    "positive": 0.35,
    "very_positive": 0.50
  },
  "model_name": "multilingual-sentiment-analysis",
  "text_content": "分析的文本内容..."
}
```

### 话题检测结果
```json
{
  "topics": [
    {
      "topic_id": "topic_0",
      "topic_name": "话题1: 人工智能",
      "keywords": ["人工智能", "AI", "机器学习", "深度学习", "算法"],
      "document_count": 25,
      "related_posts": ["post_id_1", "post_id_2"],
      "description": "关于人工智能技术的讨论"
    }
  ],
  "topic_assignments": [0, 1, 0, 2, -1],
  "model_name": "bertopic"
}
```

## 🔧 故障排除

### 常见问题

1. **分析模块不可用**
   - 检查是否安装了必要依赖: `transformers`, `torch`, `jieba`
   - 查看服务器日志获取详细错误信息

2. **数据库文件不存在**
   - 确保先运行爬虫任务生成数据
   - 检查 `schema/sqlite_tables.db` 文件是否存在

3. **分析速度慢**
   - 首次使用会下载模型文件，需要网络连接
   - 可以调整批处理大小优化性能
   - 考虑禁用话题检测以提高速度

4. **内存不足**
   - 减少批处理大小
   - 选择更轻量的模型
   - 分批处理大量数据

### 日志查看
- 服务器日志会显示详细的错误信息
- 分析失败不会影响数据存储
- 可以通过 `/system/status` 接口查看系统状态

## 🎯 最佳实践

1. **首次使用**
   - 先运行小规模测试确保功能正常
   - 检查网络连接以下载模型文件
   - 配置合适的批处理大小

2. **生产环境**
   - 启用自动情感分析，按需启用话题检测
   - 定期清理分析结果文件
   - 监控系统资源使用情况

3. **性能优化**
   - 使用SQLite存储以获得最佳性能
   - 根据硬件配置调整批处理大小
   - 考虑使用更快的情感分析模型

## 📞 技术支持

如遇到问题，请：
1. 查看 `INTEGRATION_CHANGELOG.md` 了解详细修改内容
2. 检查系统状态接口获取诊断信息
3. 查看服务器日志获取错误详情

---

**享受强大的微博舆情分析功能！** 🎉
