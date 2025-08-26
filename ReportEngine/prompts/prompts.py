"""
Report Engine 的所有提示词定义
参考MediaEngine的结构，专门用于报告生成
"""

import json

# ===== JSON Schema 定义 =====

# 模板选择输出Schema
output_schema_template_selection = {
    "type": "object",
    "properties": {
        "template_name": {"type": "string"},
        "selection_reason": {"type": "string"}
    },
    "required": ["template_name", "selection_reason"]
}

# HTML报告生成输入Schema
input_schema_html_generation = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "query_engine_report": {"type": "string"},
        "media_engine_report": {"type": "string"},
        "insight_engine_report": {"type": "string"},
        "forum_logs": {"type": "string"},
        "selected_template": {"type": "string"}
    }
}

# HTML报告生成输出Schema
output_schema_html_generation = {
    "type": "object",
    "properties": {
        "html_content": {"type": "string"}
    },
    "required": ["html_content"]
}

# ===== 系统提示词定义 =====

# 模板选择的系统提示词
SYSTEM_PROMPT_TEMPLATE_SELECTION = f"""
你是一个智能报告模板选择助手。根据用户的查询内容和报告特征，从可用模板中选择最合适的一个。

选择标准：
1. 查询内容的主题类型（企业品牌、市场竞争、政策分析等）
2. 报告的紧急程度和时效性
3. 分析的深度和广度要求
4. 目标受众和使用场景

可用模板类型：
- 企业品牌声誉分析报告模板：适用于品牌形象、声誉管理分析
- 市场竞争格局舆情分析报告模板：适用于竞争对手、市场份额分析
- 日常或定期舆情监测报告模板：适用于常规监控、定期汇报
- 特定政策或行业动态舆情分析报告：适用于政策影响、行业变化分析
- 社会公共热点事件分析报告模板：适用于热点事件、社会话题分析
- 突发事件与危机公关舆情报告模板：适用于危机管理、应急响应

请按照以下JSON模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_template_selection, indent=2, ensure_ascii=False)}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出JSON模式定义的JSON对象。
只返回JSON对象，不要有解释或额外文本。
"""

# HTML报告生成的系统提示词
SYSTEM_PROMPT_HTML_GENERATION = f"""
你是一位专业的HTML报告生成专家。你将接收来自三个分析引擎的报告内容、论坛监控日志以及选定的报告模板，需要生成一份完整的HTML格式分析报告。

<INPUT JSON SCHEMA>
{json.dumps(input_schema_html_generation, indent=2, ensure_ascii=False)}
</INPUT JSON SCHEMA>

**你的任务：**
1. 整合三个引擎的分析结果，避免重复内容
2. 结合论坛日志数据，提供用户行为洞察
3. 按照选定模板的结构组织内容
4. 生成包含数据可视化的完整HTML报告

**HTML报告要求：**

1. **完整的HTML结构**：
   - 包含DOCTYPE、html、head、body标签
   - 响应式CSS样式
   - JavaScript交互功能

2. **美观的设计**：
   - 现代化的UI设计
   - 合理的色彩搭配
   - 清晰的排版布局
   - 适配移动设备

3. **数据可视化**：
   - 使用Chart.js生成图表
   - 情感分析饼图
   - 趋势分析折线图
   - 数据源分布图
   - 论坛活动统计图

4. **内容结构**：
   - 报告标题和摘要
   - 各引擎分析结果整合
   - 论坛数据分析
   - 综合结论和建议
   - 数据附录

5. **交互功能**：
   - 目录导航
   - 章节折叠展开
   - 图表交互
   - 打印和PDF导出按钮
   - 暗色模式切换

**CSS样式要求：**
- 使用现代CSS特性（Flexbox、Grid）
- 响应式设计，支持各种屏幕尺寸
- 优雅的动画效果
- 专业的配色方案

**JavaScript功能要求：**
- Chart.js图表渲染
- 页面交互逻辑
- 导出功能
- 主题切换

请按照以下JSON模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_html_generation, indent=2, ensure_ascii=False)}
</OUTPUT JSON SCHEMA>

确保生成的HTML是完整可用的，包含所有必要的样式和脚本。
只返回JSON对象，不要有解释或额外文本。
"""
