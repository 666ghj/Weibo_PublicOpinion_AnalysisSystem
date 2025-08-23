"""
Streamlit Web界面
为DInsight Agent提供友好的Web界面
"""

import os
import sys
import streamlit as st
from datetime import datetime
import json

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from InsightEngine import DeepSearchAgent, Config
from config import DEEPSEEK_API_KEY, KIMI_API_KEY, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, DB_CHARSET


def main():
    """主函数"""
    st.set_page_config(
        page_title="Insight Agent",
        page_icon="",
        layout="wide"
    )

    st.title("Insight Engine")
    st.markdown("本地舆情数据库深度分析AI代理")

    # ----- 配置被硬编码 -----
    # 强制使用 Kimi
    llm_provider = "kimi"
    model_name = "kimi-k2-0711-preview"
    # 默认高级配置
    max_reflections = 2
    max_content_length = 500000  # Kimi支持长文本

    # 主界面
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("研究查询")
        query = st.text_area(
            "请输入您要研究的问题",
            placeholder="例如：2025年人工智能发展趋势",
            height=100
        )

    with col2:
        st.header("状态信息")
        if 'agent' in st.session_state and hasattr(st.session_state.agent, 'state'):
            progress = st.session_state.agent.get_progress_summary()
            st.metric("总段落数", progress['total_paragraphs'])
            st.metric("已完成", progress['completed_paragraphs'])
            st.progress(progress['progress_percentage'] / 100)
        else:
            st.info("尚未开始研究")

    # 执行按钮
    col1_btn, col2_btn, col3_btn = st.columns([1, 1, 1])
    with col2_btn:
        start_research = st.button("开始研究", type="primary", use_container_width=True)

    # 验证配置
    if start_research:
        if not query.strip():
            st.error("请输入研究查询")
            return

        # 由于强制使用Kimi，只检查KIMI_API_KEY
        if not KIMI_API_KEY:
            st.error("请在您的配置文件(config.py)中设置KIMI_API_KEY")
            return

        # 自动使用配置文件中的API密钥和数据库配置
        db_host = DB_HOST
        db_user = DB_USER
        db_password = DB_PASSWORD
        db_name = DB_NAME
        db_port = DB_PORT
        db_charset = DB_CHARSET

        # 创建配置
        config = Config(
            deepseek_api_key=None,
            openai_api_key=None,
            kimi_api_key=KIMI_API_KEY,  # 强制使用配置文件中的Kimi Key
            db_host=db_host,
            db_user=db_user,
            db_password=db_password,
            db_name=db_name,
            db_port=db_port,
            db_charset=db_charset,
            default_llm_provider=llm_provider,
            deepseek_model="deepseek-chat", # 保留默认值以兼容
            openai_model="gpt-4o-mini", # 保留默认值以兼容
            kimi_model=model_name,
            max_reflections=max_reflections,
            max_content_length=max_content_length,
            output_dir="insight_engine_streamlit_reports"
        )

        # 执行研究
        execute_research(query, config)


def execute_research(query: str, config: Config):
    """执行研究"""
    try:
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 初始化Agent
        status_text.text("正在初始化Agent...")
        agent = DeepSearchAgent(config)
        st.session_state.agent = agent

        progress_bar.progress(10)

        # 生成报告结构
        status_text.text("正在生成报告结构...")
        agent._generate_report_structure(query)
        progress_bar.progress(20)

        # 处理段落
        total_paragraphs = len(agent.state.paragraphs)
        for i in range(total_paragraphs):
            status_text.text(f"正在处理段落 {i + 1}/{total_paragraphs}: {agent.state.paragraphs[i].title}")

            # 初始搜索和总结
            agent._initial_search_and_summary(i)
            progress_value = 20 + (i + 0.5) / total_paragraphs * 60
            progress_bar.progress(int(progress_value))

            # 反思循环
            agent._reflection_loop(i)
            agent.state.paragraphs[i].research.mark_completed()

            progress_value = 20 + (i + 1) / total_paragraphs * 60
            progress_bar.progress(int(progress_value))

        # 生成最终报告
        status_text.text("正在生成最终报告...")
        final_report = agent._generate_final_report()
        progress_bar.progress(90)

        # 保存报告
        status_text.text("正在保存报告...")
        agent._save_report(final_report)
        progress_bar.progress(100)

        status_text.text("研究完成！")

        # 显示结果
        display_results(agent, final_report)

    except Exception as e:
        st.error(f"研究过程中发生错误: {str(e)}")


def display_results(agent: DeepSearchAgent, final_report: str):
    """显示研究结果"""
    st.header("研究结果")

    # 结果标签页（已移除下载选项）
    tab1, tab2 = st.tabs(["最终报告", "详细信息"])

    with tab1:
        st.markdown(final_report)

    with tab2:
        # 段落详情
        st.subheader("段落详情")
        for i, paragraph in enumerate(agent.state.paragraphs):
            with st.expander(f"段落 {i + 1}: {paragraph.title}"):
                st.write("**预期内容:**", paragraph.content)
                st.write("**最终内容:**", paragraph.research.latest_summary[:300] + "..."
                if len(paragraph.research.latest_summary) > 300
                else paragraph.research.latest_summary)
                st.write("**搜索次数:**", paragraph.research.get_search_count())
                st.write("**反思次数:**", paragraph.research.reflection_iteration)

        # 搜索历史
        st.subheader("搜索历史")
        all_searches = []
        for paragraph in agent.state.paragraphs:
            all_searches.extend(paragraph.research.search_history)

        if all_searches:
            for i, search in enumerate(all_searches):
                with st.expander(f"搜索 {i + 1}: {search.query}"):
                    st.write("**URL:**", search.url)
                    st.write("**标题:**", search.title)
                    st.write("**内容预览:**",
                             search.content[:200] + "..." if len(search.content) > 200 else search.content)
                    if search.score:
                        st.write("**相关度评分:**", search.score)


if __name__ == "__main__":
    main()