# 声明：本代码仅供学习和研究目的使用。请遵守平台条款与法律法规。
#使用前使用cd MediaCrawler命令切换目录，代码中相对路径是相对于MediaCrawler目录的

from __future__ import annotations

import threading
import traceback
import asyncio
import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional, List
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string

# 添加项目根目录到Python路径，以便导入分析模块
project_root = Path(__file__).parent.parent
current_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# 复用 MediaCrawler 的内部模块
import config
import db as mc_db
from main import CrawlerFactory

# 导入分析模块
try:
    from analysis.sentiment_analyzer import sentiment_service
    from analysis.topic_detector import topic_service
    from analysis.analysis_hooks import analysis_hooks
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Analysis modules not available: {e}")
    ANALYSIS_AVAILABLE = False

app = Flask(__name__)

# 添加首页路由
@app.route("/")
def home():
    """系统首页"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>微博舆情分析系统</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .api-list { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .api-item { margin: 10px 0; padding: 10px; background: white; border-left: 4px solid #007bff; }
            .method { font-weight: bold; color: #007bff; }
            .url { font-family: monospace; background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 微博舆情分析系统</h1>
            <p>欢迎使用微博舆情分析系统！系统已成功启动并运行。</p>

            <h2>📋 可用的API接口</h2>
            <div class="api-list">
                <div class="api-item">
                    <span class="method">GET</span> <span class="url">/system/status</span> - 查看系统状态
                </div>
                <div class="api-item">
                    <span class="method">POST</span> <span class="url">/crawler/start</span> - 启动爬虫
                </div>
                <div class="api-item">
                    <span class="method">GET</span> <span class="url">/crawler/status</span> - 查看爬虫状态
                </div>
                <div class="api-item">
                    <span class="method">GET</span> <span class="url">/data/posts</span> - 获取帖子数据
                </div>
                <div class="api-item">
                    <span class="method">GET</span> <span class="url">/data/comments</span> - 获取评论数据
                </div>
                <div class="api-item">
                    <span class="method">POST</span> <span class="url">/analysis/sentiment/posts</span> - 分析帖子情感
                </div>
                <div class="api-item">
                    <span class="method">POST</span> <span class="url">/analysis/topics/posts</span> - 检测帖子话题
                </div>
                <div class="api-item">
                    <span class="method">POST</span> <span class="url">/analysis/comprehensive</span> - 综合分析
                </div>
            </div>

            <h2>🔗 快速链接</h2>
            <p>
                <a href="/system/status" target="_blank">📊 查看系统状态</a> |
                <a href="/data/posts?limit=10" target="_blank">📝 查看帖子数据</a> |
                <a href="/data/comments?limit=10" target="_blank">💬 查看评论数据</a> |
                <a href="/analysis" target="_blank">🔍 数据分析页面</a>
            </p>

            <h2>📖 使用说明</h2>
            <ol>
                <li>首先配置爬虫参数并启动爬虫收集数据</li>
                <li>使用数据查询接口查看收集到的数据</li>
                <li>使用分析接口进行情感分析和话题检测</li>
                <li>查看分析结果和统计信息</li>
            </ol>
        </div>
    </body>
    </html>
    """

@app.route("/analysis")
def analysis_page():
    """数据分析页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>数据分析 - 微博舆情分析系统</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .analysis-section { margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #0056b3; }
            .result-area { margin-top: 20px; padding: 15px; background: white; border: 1px solid #ddd; border-radius: 5px; min-height: 200px; }
            .loading { color: #666; font-style: italic; }
            .error { color: #dc3545; }
            .success { color: #28a745; }
            input, select { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 数据分析工具</h1>

            <div class="analysis-section">
                <h2>📊 情感分析</h2>
                <p>分析帖子的情感倾向（正面、负面、中性）</p>
                <label>分析数量: <input type="number" id="sentiment-limit" value="20" min="1" max="100"></label>
                <label>模型类型:
                    <select id="sentiment-model">
                        <option value="multilingual">多语言模型</option>
                        <option value="machine_learning">机器学习模型</option>
                        <option value="qwen">Qwen模型</option>
                    </select>
                </label>
                <br>
                <button class="btn" onclick="runSentimentAnalysis()">开始情感分析</button>
                <div id="sentiment-result" class="result-area">点击按钮开始分析...</div>
            </div>

            <div class="analysis-section">
                <h2>🏷️ 话题检测</h2>
                <p>检测帖子中的主要话题和关键词</p>
                <label>分析数量: <input type="number" id="topic-limit" value="50" min="1" max="200"></label>
                <label>话题数量: <input type="number" id="topic-num" value="5" min="1" max="20"></label>
                <label>模型类型:
                    <select id="topic-model">
                        <option value="bertopic">BERTopic模型</option>
                        <option value="keyword_extraction">关键词提取</option>
                    </select>
                </label>
                <br>
                <button class="btn" onclick="runTopicAnalysis()">开始话题检测</button>
                <div id="topic-result" class="result-area">点击按钮开始分析...</div>
            </div>

            <div class="analysis-section">
                <h2>🎯 综合分析</h2>
                <p>同时进行情感分析和话题检测，生成综合报告</p>
                <label>分析数量: <input type="number" id="comprehensive-limit" value="30" min="1" max="100"></label>
                <label>话题数量: <input type="number" id="comprehensive-topics" value="8" min="1" max="15"></label>
                <br>
                <button class="btn" onclick="runComprehensiveAnalysis()">开始综合分析</button>
                <div id="comprehensive-result" class="result-area">点击按钮开始分析...</div>
            </div>
        </div>

        <script>
            async function runSentimentAnalysis() {
                const resultDiv = document.getElementById('sentiment-result');
                const limit = document.getElementById('sentiment-limit').value;
                const model = document.getElementById('sentiment-model').value;

                resultDiv.innerHTML = '<div class="loading">🔄 正在进行情感分析，请稍候...</div>';

                try {
                    const response = await fetch('/analysis/sentiment/posts', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ model_type: model, limit: parseInt(limit) })
                    });

                    const data = await response.json();

                    if (data.ok) {
                        let html = '<div class="success">✅ 情感分析完成！</div>';
                        html += `<p><strong>分析结果数量:</strong> ${data.count}</p>`;

                        // 统计情感分布
                        const sentimentCounts = { positive: 0, negative: 0, neutral: 0 };
                        data.results.forEach(result => {
                            if (result.sentiment_label !== undefined) {
                                if (result.sentiment_label >= 3) sentimentCounts.positive++;
                                else if (result.sentiment_label <= 1) sentimentCounts.negative++;
                                else sentimentCounts.neutral++;
                            }
                        });

                        html += '<h4>📈 情感分布统计:</h4>';
                        html += `<p>😊 正面: ${sentimentCounts.positive} (${(sentimentCounts.positive/data.count*100).toFixed(1)}%)</p>`;
                        html += `<p>😐 中性: ${sentimentCounts.neutral} (${(sentimentCounts.neutral/data.count*100).toFixed(1)}%)</p>`;
                        html += `<p>😞 负面: ${sentimentCounts.negative} (${(sentimentCounts.negative/data.count*100).toFixed(1)}%)</p>`;

                        html += '<h4>📝 详细结果 (前5条):</h4>';
                        data.results.slice(0, 5).forEach((result, index) => {
                            const sentiment = result.sentiment_text || '未知';
                            const confidence = result.confidence ? (result.confidence * 100).toFixed(1) + '%' : '未知';
                            const text = result.text_content || '无内容';
                            html += `<div style="margin: 10px 0; padding: 10px; border-left: 3px solid #007bff;">`;
                            html += `<strong>帖子 ${index + 1}:</strong> ${sentiment} (置信度: ${confidence})<br>`;
                            html += `<small>${text}</small></div>`;
                        });

                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = `<div class="error">❌ 分析失败: ${data.msg}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="error">❌ 请求失败: ${error.message}</div>`;
                }
            }

            async function runTopicAnalysis() {
                const resultDiv = document.getElementById('topic-result');
                const limit = document.getElementById('topic-limit').value;
                const numTopics = document.getElementById('topic-num').value;
                const model = document.getElementById('topic-model').value;

                resultDiv.innerHTML = '<div class="loading">🔄 正在进行话题检测，请稍候...</div>';

                try {
                    const response = await fetch('/analysis/topics/posts', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_type: model,
                            num_topics: parseInt(numTopics),
                            limit: parseInt(limit)
                        })
                    });

                    const data = await response.json();

                    if (data.ok) {
                        let html = '<div class="success">✅ 话题检测完成！</div>';
                        html += `<p><strong>检测到的话题数量:</strong> ${data.topics ? data.topics.length : 0}</p>`;

                        if (data.topics && data.topics.length > 0) {
                            html += '<h4>🏷️ 主要话题:</h4>';
                            data.topics.forEach((topic, index) => {
                                html += `<div style="margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">`;
                                html += `<h5>${topic.topic_name || '话题 ' + (index + 1)}</h5>`;
                                html += `<p><strong>关键词:</strong> ${topic.keywords ? topic.keywords.join(', ') : '无'}</p>`;
                                html += `<p><strong>相关帖子数:</strong> ${topic.document_count || 0}</p>`;
                                if (topic.description) {
                                    html += `<p><strong>描述:</strong> ${topic.description}</p>`;
                                }
                                html += `</div>`;
                            });
                        }

                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = `<div class="error">❌ 分析失败: ${data.msg}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="error">❌ 请求失败: ${error.message}</div>`;
                }
            }

            async function runComprehensiveAnalysis() {
                const resultDiv = document.getElementById('comprehensive-result');
                const limit = document.getElementById('comprehensive-limit').value;
                const numTopics = document.getElementById('comprehensive-topics').value;

                resultDiv.innerHTML = '<div class="loading">🔄 正在进行综合分析，请稍候...</div>';

                try {
                    const response = await fetch('/analysis/comprehensive', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            include_comments: false,
                            sentiment_model: 'multilingual',
                            topic_model: 'bertopic',
                            num_topics: parseInt(numTopics),
                            limit: parseInt(limit)
                        })
                    });

                    const data = await response.json();

                    if (data.ok) {
                        let html = '<div class="success">✅ 综合分析完成！</div>';

                        // 显示摘要
                        if (data.summary) {
                            html += '<h4>📊 分析摘要:</h4>';
                            html += `<p><strong>总帖子数:</strong> ${data.summary.total_posts}</p>`;

                            if (data.summary.sentiment_distribution) {
                                const dist = data.summary.sentiment_distribution;
                                html += '<h5>情感分布:</h5>';
                                html += `<p>😊 正面: ${dist.positive} | 😐 中性: ${dist.neutral} | 😞 负面: ${dist.negative}</p>`;
                            }

                            if (data.summary.top_topics && data.summary.top_topics.length > 0) {
                                html += '<h5>热门话题:</h5>';
                                data.summary.top_topics.forEach((topic, index) => {
                                    html += `<p><strong>${index + 1}. ${topic.name}:</strong> ${topic.keywords.join(', ')} (${topic.document_count}条帖子)</p>`;
                                });
                            }
                        }

                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = `<div class="error">❌ 分析失败: ${data.msg}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="error">❌ 请求失败: ${error.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """

# 简单的全局运行状态（单任务）
_state: Dict[str, Any] = {
    "status": "idle",            # idle | running | done | error
    "params": None,               # 最近一次任务的参数
    "error": None,                # 错误信息
    "thread": None,               # 运行线程
}

_lock = threading.Lock()


def _run_crawl(params: Dict[str, Any]):
    async def _job():
        # 设置运行参数到 config
        config.PLATFORM = params.get("platform", config.PLATFORM)
        config.LOGIN_TYPE = params.get("lt", config.LOGIN_TYPE)
        config.CRAWLER_TYPE = params.get("type", config.CRAWLER_TYPE)
        config.KEYWORDS = params.get("keywords", config.KEYWORDS)
        # 布尔值转化
        if "get_comment" in params:
            config.ENABLE_GET_COMMENTS = bool(params["get_comment"])  
        if "get_sub_comment" in params:
            config.ENABLE_GET_SUB_COMMENTS = bool(params["get_sub_comment"])  
        if "save_data_option" in params:
            config.SAVE_DATA_OPTION = params["save_data_option"]
        if "cookies" in params:
            config.COOKIES = params["cookies"]

        # 初始化数据库（如需要）
        if config.SAVE_DATA_OPTION in ["db", "sqlite"]:
            await mc_db.init_db()

        # 创建并启动爬虫
        crawler = CrawlerFactory.create_crawler(platform=config.PLATFORM)
        await crawler.start()

    try:
        asyncio.run(_job())
        with _lock:
            _state["status"] = "done"
            _state["thread"] = None
    except Exception:
        err = traceback.format_exc()
        with _lock:
            _state["status"] = "error"
            _state["error"] = err
            _state["thread"] = None


@app.route("/crawler/start", methods=["POST"])
def start_crawler():
    """
    启动一次 MediaCrawler 任务。
    请求 JSON 字段（部分可选）：
      - platform: xhs|dy|ks|bili|wb|tieba|zhihu
      - lt: 登录方式 qrcode|phone|cookie
      - type: 爬取类型 search|detail|creator
      - keywords: 关键词（逗号分隔）
      - save_data_option: csv|db|json|sqlite
      - get_comment: bool
      - get_sub_comment: bool
      - cookies: str（cookie 登录时）
    """
    params = request.get_json(silent=True) or {}

    with _lock:
        if _state["status"] == "running":
            return jsonify({"ok": False, "msg": "已有任务在运行，请先等待完成或停止"}), 409

        # 清理上一次状态
        _state.update({
            "status": "running",
            "params": params,
            "error": None,
        })

        t = threading.Thread(target=_run_crawl, args=(params,), daemon=True)
        _state["thread"] = t
        t.start()

    return jsonify({"ok": True, "status": _state["status"], "params": _state["params"]})


@app.route("/crawler/status", methods=["GET"])
def crawler_status():
    with _lock:
        resp = {k: v for k, v in _state.items() if k != "thread"}
    return jsonify(resp)


@app.route("/crawler/stop", methods=["POST"])  # 软停止占位：当前仅返回说明
def crawler_stop():
    """
    说明：当前 MediaCrawler 未提供统一的停止钩子，强行停止可能导致资源未正确释放。
    如需软停止，请明确需求，我可以后续在各平台 crawler 中增加可中断检查点。
    """
    with _lock:
        running = _state.get("status") == "running"
    if not running:
        return jsonify({"ok": True, "msg": "当前无运行任务"})
    return jsonify({"ok": False, "msg": "暂未实现软停止。如需请告知，我会增加安全的停止机制。"}), 501


# ==================== 数据查询接口 ====================

@app.route("/data/posts", methods=["GET"])
def get_posts():
    """
    获取爬取的帖子数据
    参数:
        - platform: 平台名称 (可选)
        - limit: 返回数量限制 (默认100)
        - offset: 偏移量 (默认0)
    """
    platform = request.args.get("platform")
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    try:
        # 从数据库查询数据
        if config.SAVE_DATA_OPTION == "sqlite":
            db_path = "schema/sqlite_tables.db"
            if not os.path.exists(db_path):
                return jsonify({"ok": False, "msg": "数据库文件不存在"}), 404

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 构建查询
            query = "SELECT * FROM weibo_note"
            params = []

            if platform:
                query += " WHERE platform = ?"
                params.append(platform)

            query += f" ORDER BY add_ts DESC LIMIT {limit} OFFSET {offset}"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            posts = [dict(row) for row in rows]
            conn.close()

            return jsonify({"ok": True, "data": posts, "count": len(posts)})

        elif config.SAVE_DATA_OPTION == "db":
            # MySQL数据库查询
            import pymysql
            from config.db_config import MYSQL_DB_HOST, MYSQL_DB_PORT, MYSQL_DB_USER, MYSQL_DB_PWD, MYSQL_DB_NAME

            conn = pymysql.connect(
                host=MYSQL_DB_HOST,
                port=int(MYSQL_DB_PORT),
                user=MYSQL_DB_USER,
                password=MYSQL_DB_PWD,
                database=MYSQL_DB_NAME,
                charset='utf8mb4'
            )

            cursor = conn.cursor(pymysql.cursors.DictCursor)

            # 构建查询
            query = "SELECT * FROM weibo_note"
            params = []

            if platform:
                query += " WHERE platform = %s"
                params.append(platform)

            query += f" ORDER BY add_ts DESC LIMIT {limit} OFFSET {offset}"

            cursor.execute(query, params)
            posts = cursor.fetchall()

            conn.close()

            return jsonify({"ok": True, "data": posts, "count": len(posts)})
        else:
            return jsonify({"ok": False, "msg": "不支持的数据库类型"}), 400

    except Exception as e:
        return jsonify({"ok": False, "msg": f"查询失败: {str(e)}"}), 500

@app.route("/data/comments", methods=["GET"])
def get_comments():
    """
    获取爬取的评论数据
    参数:
        - post_id: 帖子ID (可选)
        - limit: 返回数量限制 (默认100)
        - offset: 偏移量 (默认0)
    """
    post_id = request.args.get("post_id")
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    try:
        if config.SAVE_DATA_OPTION == "sqlite":
            db_path = "schema/sqlite_tables.db"
            if not os.path.exists(db_path):
                return jsonify({"ok": False, "msg": "数据库文件不存在"}), 404

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 构建查询
            query = "SELECT * FROM weibo_comment"
            params = []

            if post_id:
                query += " WHERE note_id = ?"
                params.append(post_id)

            query += f" ORDER BY add_ts DESC LIMIT {limit} OFFSET {offset}"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            comments = [dict(row) for row in rows]
            conn.close()

            return jsonify({"ok": True, "data": comments, "count": len(comments)})

        elif config.SAVE_DATA_OPTION == "db":
            # MySQL数据库查询
            import pymysql
            from config.db_config import MYSQL_DB_HOST, MYSQL_DB_PORT, MYSQL_DB_USER, MYSQL_DB_PWD, MYSQL_DB_NAME

            conn = pymysql.connect(
                host=MYSQL_DB_HOST,
                port=int(MYSQL_DB_PORT),
                user=MYSQL_DB_USER,
                password=MYSQL_DB_PWD,
                database=MYSQL_DB_NAME,
                charset='utf8mb4'
            )

            cursor = conn.cursor(pymysql.cursors.DictCursor)

            # 构建查询
            query = "SELECT * FROM weibo_comment"
            params = []

            if post_id:
                query += " WHERE note_id = %s"
                params.append(post_id)

            query += f" ORDER BY add_ts DESC LIMIT {limit} OFFSET {offset}"

            cursor.execute(query, params)
            comments = cursor.fetchall()

            conn.close()

            return jsonify({"ok": True, "data": comments, "count": len(comments)})
        else:
            return jsonify({"ok": False, "msg": "不支持的数据库类型"}), 400

    except Exception as e:
        return jsonify({"ok": False, "msg": f"查询失败: {str(e)}"}), 500

# ==================== 情感分析接口 ====================

@app.route("/analysis/sentiment/posts", methods=["POST"])
def analyze_posts_sentiment():
    """
    分析帖子情感
    请求JSON字段:
        - post_ids: 帖子ID列表 (可选，不提供则分析最新的帖子)
        - model_type: 模型类型 (可选: multilingual, machine_learning, qwen)
        - limit: 分析数量限制 (默认50)
    """
    if not ANALYSIS_AVAILABLE:
        return jsonify({"ok": False, "msg": "分析模块不可用"}), 503

    params = request.get_json(silent=True) or {}
    post_ids = params.get("post_ids", [])
    model_type = params.get("model_type", "multilingual")
    limit = params.get("limit", 50)

    async def _analyze():
        try:
            # 获取帖子数据
            posts_data = []

            if post_ids:
                # 根据ID获取特定帖子
                if config.SAVE_DATA_OPTION == "sqlite":
                    db_path = "schema/sqlite_tables.db"
                    if os.path.exists(db_path):
                        conn = sqlite3.connect(db_path)
                        conn.row_factory = sqlite3.Row
                        cursor = conn.cursor()

                        placeholders = ",".join(["?" for _ in post_ids])
                        cursor.execute(f"SELECT * FROM weibo_note WHERE note_id IN ({placeholders})", post_ids)
                        rows = cursor.fetchall()
                        posts_data = [dict(row) for row in rows]
                        conn.close()
                elif config.SAVE_DATA_OPTION == "db":
                    # MySQL数据库查询
                    import pymysql
                    from config.db_config import MYSQL_DB_HOST, MYSQL_DB_PORT, MYSQL_DB_USER, MYSQL_DB_PWD, MYSQL_DB_NAME

                    conn = pymysql.connect(
                        host=MYSQL_DB_HOST,
                        port=int(MYSQL_DB_PORT),
                        user=MYSQL_DB_USER,
                        password=MYSQL_DB_PWD,
                        database=MYSQL_DB_NAME,
                        charset='utf8mb4'
                    )

                    cursor = conn.cursor(pymysql.cursors.DictCursor)
                    placeholders = ",".join(["%s" for _ in post_ids])
                    cursor.execute(f"SELECT * FROM weibo_note WHERE note_id IN ({placeholders})", post_ids)
                    posts_data = cursor.fetchall()
                    conn.close()
            else:
                # 获取最新的帖子
                if config.SAVE_DATA_OPTION == "sqlite":
                    db_path = "schema/sqlite_tables.db"
                    if os.path.exists(db_path):
                        conn = sqlite3.connect(db_path)
                        conn.row_factory = sqlite3.Row
                        cursor = conn.cursor()

                        cursor.execute(f"SELECT * FROM weibo_note ORDER BY add_ts DESC LIMIT {limit}")
                        rows = cursor.fetchall()
                        posts_data = [dict(row) for row in rows]
                        conn.close()
                    else:
                        return {"ok": False, "msg": "数据库文件不存在"}
                elif config.SAVE_DATA_OPTION == "db":
                    # MySQL数据库查询
                    import pymysql
                    from config.db_config import MYSQL_DB_HOST, MYSQL_DB_PORT, MYSQL_DB_USER, MYSQL_DB_PWD, MYSQL_DB_NAME

                    conn = pymysql.connect(
                        host=MYSQL_DB_HOST,
                        port=int(MYSQL_DB_PORT),
                        user=MYSQL_DB_USER,
                        password=MYSQL_DB_PWD,
                        database=MYSQL_DB_NAME,
                        charset='utf8mb4'
                    )

                    cursor = conn.cursor(pymysql.cursors.DictCursor)
                    cursor.execute(f"SELECT * FROM weibo_note ORDER BY add_ts DESC LIMIT {limit}")
                    posts_data = cursor.fetchall()
                    conn.close()

            if not posts_data:
                return {"ok": False, "msg": "未找到帖子数据"}

            # 分析情感
            results = []
            for post in posts_data:
                try:
                    result = await sentiment_service.analyze_post_sentiment(post, model_type)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "post_id": post.get("note_id", ""),
                        "error": str(e)
                    })

            return {"ok": True, "results": results, "count": len(results)}

        except Exception as e:
            return {"ok": False, "msg": f"分析失败: {str(e)}"}

    # 运行异步分析
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_analyze())
        return jsonify(result)
    finally:
        loop.close()

@app.route("/analysis/sentiment/comments", methods=["POST"])
def analyze_comments_sentiment():
    """
    分析评论情感
    请求JSON字段:
        - comment_ids: 评论ID列表 (可选)
        - post_id: 帖子ID (可选，分析该帖子下的评论)
        - model_type: 模型类型 (可选)
        - limit: 分析数量限制 (默认100)
    """
    if not ANALYSIS_AVAILABLE:
        return jsonify({"ok": False, "msg": "分析模块不可用"}), 503

    params = request.get_json(silent=True) or {}
    comment_ids = params.get("comment_ids", [])
    post_id = params.get("post_id")
    model_type = params.get("model_type", "multilingual")
    limit = params.get("limit", 100)

    async def _analyze():
        try:
            # 获取评论数据
            comments_data = []
            if config.SAVE_DATA_OPTION == "sqlite":
                db_path = "schema/sqlite_tables.db"
                if not os.path.exists(db_path):
                    return {"ok": False, "msg": "数据库文件不存在"}

                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if comment_ids:
                    placeholders = ",".join(["?" for _ in comment_ids])
                    cursor.execute(f"SELECT * FROM weibo_comment WHERE comment_id IN ({placeholders})", comment_ids)
                elif post_id:
                    cursor.execute(f"SELECT * FROM weibo_comment WHERE note_id = ? ORDER BY add_ts DESC LIMIT {limit}", (post_id,))
                else:
                    cursor.execute(f"SELECT * FROM weibo_comment ORDER BY add_ts DESC LIMIT {limit}")

                rows = cursor.fetchall()
                comments_data = [dict(row) for row in rows]
                conn.close()

            if not comments_data:
                return {"ok": False, "msg": "未找到评论数据"}

            # 分析情感
            results = []
            for comment in comments_data:
                try:
                    result = await sentiment_service.analyze_comment_sentiment(comment, model_type)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "comment_id": comment.get("comment_id", ""),
                        "error": str(e)
                    })

            return {"ok": True, "results": results, "count": len(results)}

        except Exception as e:
            return {"ok": False, "msg": f"分析失败: {str(e)}"}

    # 运行异步分析
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_analyze())
        return jsonify(result)
    finally:
        loop.close()

# ==================== 话题检测接口 ====================

@app.route("/analysis/topics/posts", methods=["POST"])
def detect_posts_topics():
    """
    检测帖子话题
    请求JSON字段:
        - post_ids: 帖子ID列表 (可选)
        - model_type: 模型类型 (可选: bertopic, keyword_extraction)
        - num_topics: 话题数量 (默认10)
        - limit: 分析数量限制 (默认100)
    """
    if not ANALYSIS_AVAILABLE:
        return jsonify({"ok": False, "msg": "分析模块不可用"}), 503

    params = request.get_json(silent=True) or {}
    post_ids = params.get("post_ids", [])
    model_type = params.get("model_type", "bertopic")
    num_topics = params.get("num_topics", 10)
    limit = params.get("limit", 100)

    async def _analyze():
        try:
            # 获取帖子数据
            posts_data = []

            if config.SAVE_DATA_OPTION == "sqlite":
                db_path = "schema/sqlite_tables.db"
                if not os.path.exists(db_path):
                    return {"ok": False, "msg": "数据库文件不存在"}

                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if post_ids:
                    placeholders = ",".join(["?" for _ in post_ids])
                    cursor.execute(f"SELECT * FROM weibo_note WHERE note_id IN ({placeholders})", post_ids)
                else:
                    cursor.execute(f"SELECT * FROM weibo_note ORDER BY add_ts DESC LIMIT {limit}")

                rows = cursor.fetchall()
                posts_data = [dict(row) for row in rows]
                conn.close()

            elif config.SAVE_DATA_OPTION == "db":
                # MySQL数据库查询
                import pymysql
                from config.db_config import MYSQL_DB_HOST, MYSQL_DB_PORT, MYSQL_DB_USER, MYSQL_DB_PWD, MYSQL_DB_NAME

                conn = pymysql.connect(
                    host=MYSQL_DB_HOST,
                    port=int(MYSQL_DB_PORT),
                    user=MYSQL_DB_USER,
                    password=MYSQL_DB_PWD,
                    database=MYSQL_DB_NAME,
                    charset='utf8mb4'
                )

                cursor = conn.cursor(pymysql.cursors.DictCursor)

                if post_ids:
                    placeholders = ",".join(["%s" for _ in post_ids])
                    cursor.execute(f"SELECT * FROM weibo_note WHERE note_id IN ({placeholders})", post_ids)
                else:
                    cursor.execute(f"SELECT * FROM weibo_note ORDER BY add_ts DESC LIMIT {limit}")

                posts_data = cursor.fetchall()
                conn.close()

            if not posts_data:
                return {"ok": False, "msg": "未找到帖子数据"}

            # 检测话题
            result = await topic_service.detect_topics_from_posts(posts_data, model_type, num_topics)
            result["ok"] = True

            return result

        except Exception as e:
            return {"ok": False, "msg": f"话题检测失败: {str(e)}"}

    # 运行异步分析
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_analyze())
        return jsonify(result)
    finally:
        loop.close()

@app.route("/analysis/topics/comments", methods=["POST"])
def detect_comments_topics():
    """
    检测评论话题
    请求JSON字段:
        - comment_ids: 评论ID列表 (可选)
        - post_id: 帖子ID (可选，分析该帖子下的评论)
        - model_type: 模型类型 (可选)
        - num_topics: 话题数量 (默认10)
        - limit: 分析数量限制 (默认200)
    """
    if not ANALYSIS_AVAILABLE:
        return jsonify({"ok": False, "msg": "分析模块不可用"}), 503

    params = request.get_json(silent=True) or {}
    comment_ids = params.get("comment_ids", [])
    post_id = params.get("post_id")
    model_type = params.get("model_type", "bertopic")
    num_topics = params.get("num_topics", 10)
    limit = params.get("limit", 200)

    async def _analyze():
        try:
            # 获取评论数据
            comments_data = []
            if config.SAVE_DATA_OPTION == "sqlite":
                db_path = "schema/sqlite_tables.db"
                if not os.path.exists(db_path):
                    return {"ok": False, "msg": "数据库文件不存在"}

                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if comment_ids:
                    placeholders = ",".join(["?" for _ in comment_ids])
                    cursor.execute(f"SELECT * FROM weibo_comment WHERE comment_id IN ({placeholders})", comment_ids)
                elif post_id:
                    cursor.execute(f"SELECT * FROM weibo_comment WHERE note_id = ? ORDER BY add_ts DESC LIMIT {limit}", (post_id,))
                else:
                    cursor.execute(f"SELECT * FROM weibo_comment ORDER BY add_ts DESC LIMIT {limit}")

                rows = cursor.fetchall()
                comments_data = [dict(row) for row in rows]
                conn.close()

            if not comments_data:
                return {"ok": False, "msg": "未找到评论数据"}

            # 检测话题
            result = await topic_service.detect_topics_from_comments(comments_data, model_type, num_topics)
            result["ok"] = True

            return result

        except Exception as e:
            return {"ok": False, "msg": f"话题检测失败: {str(e)}"}

    # 运行异步分析
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_analyze())
        return jsonify(result)
    finally:
        loop.close()

# ==================== 综合分析接口 ====================

@app.route("/analysis/comprehensive", methods=["POST"])
def comprehensive_analysis():
    """
    综合分析接口 - 同时进行情感分析和话题检测
    请求JSON字段:
        - post_ids: 帖子ID列表 (可选)
        - include_comments: 是否包含评论分析 (默认true)
        - sentiment_model: 情感分析模型 (可选)
        - topic_model: 话题检测模型 (可选)
        - num_topics: 话题数量 (默认10)
        - limit: 分析数量限制 (默认50)
    """
    if not ANALYSIS_AVAILABLE:
        return jsonify({"ok": False, "msg": "分析模块不可用"}), 503

    params = request.get_json(silent=True) or {}
    post_ids = params.get("post_ids", [])
    include_comments = params.get("include_comments", True)
    sentiment_model = params.get("sentiment_model", "multilingual")
    topic_model = params.get("topic_model", "bertopic")
    num_topics = params.get("num_topics", 10)
    limit = params.get("limit", 50)

    async def _analyze():
        try:
            result = {
                "ok": True,
                "post_sentiment": [],
                "comment_sentiment": [],
                "post_topics": {},
                "comment_topics": {},
                "summary": {}
            }

            # 获取帖子数据
            posts_data = []
            comments_data = []

            if config.SAVE_DATA_OPTION == "sqlite":
                db_path = "schema/sqlite_tables.db"
                if not os.path.exists(db_path):
                    return {"ok": False, "msg": "数据库文件不存在"}

                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if post_ids:
                    placeholders = ",".join(["?" for _ in post_ids])
                    cursor.execute(f"SELECT * FROM weibo_note WHERE note_id IN ({placeholders})", post_ids)
                else:
                    cursor.execute(f"SELECT * FROM weibo_note ORDER BY add_ts DESC LIMIT {limit}")

                rows = cursor.fetchall()
                posts_data = [dict(row) for row in rows]

                # 如果需要分析评论，获取相关评论
                if include_comments and posts_data:
                    post_id_list = [post["note_id"] for post in posts_data]
                    placeholders = ",".join(["?" for _ in post_id_list])
                    cursor.execute(f"SELECT * FROM weibo_comment WHERE note_id IN ({placeholders}) ORDER BY add_ts DESC", post_id_list)
                    comment_rows = cursor.fetchall()
                    comments_data = [dict(row) for row in comment_rows]

                conn.close()

            elif config.SAVE_DATA_OPTION == "db":
                # MySQL数据库查询
                import pymysql
                from config.db_config import MYSQL_DB_HOST, MYSQL_DB_PORT, MYSQL_DB_USER, MYSQL_DB_PWD, MYSQL_DB_NAME

                conn = pymysql.connect(
                    host=MYSQL_DB_HOST,
                    port=int(MYSQL_DB_PORT),
                    user=MYSQL_DB_USER,
                    password=MYSQL_DB_PWD,
                    database=MYSQL_DB_NAME,
                    charset='utf8mb4'
                )

                cursor = conn.cursor(pymysql.cursors.DictCursor)

                if post_ids:
                    placeholders = ",".join(["%s" for _ in post_ids])
                    cursor.execute(f"SELECT * FROM weibo_note WHERE note_id IN ({placeholders})", post_ids)
                else:
                    cursor.execute(f"SELECT * FROM weibo_note ORDER BY add_ts DESC LIMIT {limit}")

                posts_data = cursor.fetchall()

                # 如果需要分析评论，获取相关评论（如果评论表存在）
                if include_comments and posts_data:
                    try:
                        post_id_list = [post["note_id"] for post in posts_data]
                        placeholders = ",".join(["%s" for _ in post_id_list])
                        cursor.execute(f"SELECT * FROM weibo_comment WHERE note_id IN ({placeholders}) ORDER BY add_ts DESC", post_id_list)
                        comments_data = cursor.fetchall()
                    except Exception:
                        # 如果评论表不存在，忽略错误
                        comments_data = []

                conn.close()

            if not posts_data:
                return {"ok": False, "msg": "未找到帖子数据"}

            # 并行执行分析任务
            tasks = []

            # 帖子情感分析
            async def analyze_post_sentiment():
                results = []
                for post in posts_data:
                    try:
                        sentiment_result = await sentiment_service.analyze_post_sentiment(post, sentiment_model)
                        results.append(sentiment_result)
                    except Exception as e:
                        results.append({"post_id": post.get("note_id", ""), "error": str(e)})
                return results

            # 帖子话题检测
            async def analyze_post_topics():
                return await topic_service.detect_topics_from_posts(posts_data, topic_model, num_topics)

            # 评论分析
            async def analyze_comment_sentiment():
                if not include_comments or not comments_data:
                    return []
                results = []
                for comment in comments_data[:200]:  # 限制评论数量
                    try:
                        sentiment_result = await sentiment_service.analyze_comment_sentiment(comment, sentiment_model)
                        results.append(sentiment_result)
                    except Exception as e:
                        results.append({"comment_id": comment.get("comment_id", ""), "error": str(e)})
                return results

            async def analyze_comment_topics():
                if not include_comments or not comments_data:
                    return {}
                return await topic_service.detect_topics_from_comments(comments_data[:200], topic_model, num_topics)

            # 执行所有分析任务
            post_sentiment_results, post_topic_results, comment_sentiment_results, comment_topic_results = await asyncio.gather(
                analyze_post_sentiment(),
                analyze_post_topics(),
                analyze_comment_sentiment(),
                analyze_comment_topics(),
                return_exceptions=True
            )

            # 处理结果
            result["post_sentiment"] = post_sentiment_results if not isinstance(post_sentiment_results, Exception) else []
            result["post_topics"] = post_topic_results if not isinstance(post_topic_results, Exception) else {}
            result["comment_sentiment"] = comment_sentiment_results if not isinstance(comment_sentiment_results, Exception) else []
            result["comment_topics"] = comment_topic_results if not isinstance(comment_topic_results, Exception) else {}

            # 生成摘要统计
            summary = {
                "total_posts": len(posts_data),
                "total_comments": len(comments_data) if include_comments else 0,
                "sentiment_distribution": {},
                "top_topics": []
            }

            # 统计情感分布
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for sentiment_result in result["post_sentiment"]:
                if "sentiment_label" in sentiment_result:
                    label = sentiment_result["sentiment_label"]
                    if label in [3, 4]:  # 正面
                        sentiment_counts["positive"] += 1
                    elif label in [0, 1]:  # 负面
                        sentiment_counts["negative"] += 1
                    else:  # 中性
                        sentiment_counts["neutral"] += 1

            summary["sentiment_distribution"] = sentiment_counts

            # 提取热门话题
            if "topics" in result["post_topics"]:
                summary["top_topics"] = [
                    {
                        "name": topic["topic_name"],
                        "keywords": topic["keywords"][:5],
                        "document_count": topic["document_count"]
                    }
                    for topic in result["post_topics"]["topics"][:5]
                ]

            result["summary"] = summary

            return result

        except Exception as e:
            return {"ok": False, "msg": f"综合分析失败: {str(e)}"}

    # 运行异步分析
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_analyze())
        return jsonify(result)
    finally:
        loop.close()

# ==================== 分析钩子管理接口 ====================

@app.route("/analysis/hooks/status", methods=["GET"])
def get_hooks_status():
    """获取分析钩子状态"""
    if not ANALYSIS_AVAILABLE:
        return jsonify({"ok": False, "msg": "分析模块不可用"}), 503

    stats = analysis_hooks.get_analysis_stats()
    return jsonify({"ok": True, "stats": stats})

@app.route("/analysis/hooks/config", methods=["POST"])
def configure_hooks():
    """配置分析钩子
    请求JSON字段:
        - enabled: 是否启用钩子 (bool)
        - auto_sentiment: 是否自动情感分析 (bool)
        - auto_topic: 是否自动话题检测 (bool)
        - batch_size: 批处理大小 (int)
    """
    if not ANALYSIS_AVAILABLE:
        return jsonify({"ok": False, "msg": "分析模块不可用"}), 503

    params = request.get_json(silent=True) or {}

    try:
        if "enabled" in params:
            if params["enabled"]:
                analysis_hooks.enabled = True
            else:
                analysis_hooks.disable_auto_analysis()

        if "auto_sentiment" in params or "auto_topic" in params:
            auto_sentiment = params.get("auto_sentiment", analysis_hooks.auto_sentiment)
            auto_topic = params.get("auto_topic", analysis_hooks.auto_topic)
            analysis_hooks.enable_auto_analysis(auto_sentiment, auto_topic)

        if "batch_size" in params:
            batch_size = int(params["batch_size"])
            if 1 <= batch_size <= 100:
                analysis_hooks.batch_size = batch_size
            else:
                return jsonify({"ok": False, "msg": "批处理大小必须在1-100之间"}), 400

        return jsonify({"ok": True, "msg": "配置更新成功", "stats": analysis_hooks.get_analysis_stats()})

    except Exception as e:
        return jsonify({"ok": False, "msg": f"配置失败: {str(e)}"}), 500

@app.route("/analysis/hooks/flush", methods=["POST"])
def flush_pending_analysis():
    """强制处理所有待分析的数据"""
    if not ANALYSIS_AVAILABLE:
        return jsonify({"ok": False, "msg": "分析模块不可用"}), 503

    async def _flush():
        try:
            await analysis_hooks.flush_pending()
            return {"ok": True, "msg": "待分析数据处理完成"}
        except Exception as e:
            return {"ok": False, "msg": f"处理失败: {str(e)}"}

    # 运行异步处理
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_flush())
        return jsonify(result)
    finally:
        loop.close()

# ==================== 系统状态接口 ====================

@app.route("/system/status", methods=["GET"])
def system_status():
    """获取系统整体状态"""
    status = {
        "crawler": {
            "status": _state["status"],
            "platform": config.PLATFORM,
            "save_data_option": config.SAVE_DATA_OPTION
        },
        "analysis": {
            "available": ANALYSIS_AVAILABLE,
            "hooks": analysis_hooks.get_analysis_stats() if ANALYSIS_AVAILABLE else None
        },
        "database": {
            "type": config.SAVE_DATA_OPTION,
            "path": "schema/sqlite_tables.db" if config.SAVE_DATA_OPTION == "sqlite" else None
        }
    }

    # 检查数据库连接和统计
    if config.SAVE_DATA_OPTION == "sqlite":
        db_path = "schema/sqlite_tables.db"
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # 获取表统计
                cursor.execute("SELECT COUNT(*) FROM weibo_note")
                posts_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM weibo_comment")
                comments_count = cursor.fetchone()[0]

                status["database"]["posts_count"] = posts_count
                status["database"]["comments_count"] = comments_count
                status["database"]["available"] = True

                conn.close()
            except Exception as e:
                status["database"]["error"] = str(e)
                status["database"]["available"] = False
        else:
            status["database"]["available"] = False

    elif config.SAVE_DATA_OPTION == "db":
        # MySQL数据库检查
        try:
            import pymysql
            from config.db_config import MYSQL_DB_HOST, MYSQL_DB_PORT, MYSQL_DB_USER, MYSQL_DB_PWD, MYSQL_DB_NAME

            conn = pymysql.connect(
                host=MYSQL_DB_HOST,
                port=int(MYSQL_DB_PORT),
                user=MYSQL_DB_USER,
                password=MYSQL_DB_PWD,
                database=MYSQL_DB_NAME,
                charset='utf8mb4'
            )

            cursor = conn.cursor()

            # 获取表统计
            cursor.execute("SELECT COUNT(*) FROM weibo_note")
            posts_count = cursor.fetchone()[0]

            # 检查评论表是否存在
            try:
                cursor.execute("SELECT COUNT(*) FROM weibo_comment")
                comments_count = cursor.fetchone()[0]
            except Exception:
                comments_count = 0  # 评论表不存在

            status["database"]["posts_count"] = posts_count
            status["database"]["comments_count"] = comments_count
            status["database"]["available"] = True
            status["database"]["host"] = MYSQL_DB_HOST
            status["database"]["database"] = MYSQL_DB_NAME

            conn.close()
        except Exception as e:
            status["database"]["error"] = str(e)
            status["database"]["available"] = False

    return jsonify({"ok": True, "status": status})

if __name__ == "__main__":
    # 开发默认端口 5001，可按需修改
    app.run(host="0.0.0.0", port=5001, debug=False)

