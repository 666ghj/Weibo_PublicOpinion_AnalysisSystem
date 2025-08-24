"""
Streamlit Web界面
为Media Agent提供友好的Web界面
"""

import os
import sys
import streamlit as st
from datetime import datetime
import json

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from MediaEngine import DeepSearchAgent, Config
from config import DEEPSEEK_API_KEY, BOCHA_Web_Search_API_KEY, GEMINI_API_KEY


def main():
    """主函数"""
    st.set_page_config(
        page_title="Media Agent",
        page_icon="🔍",
        layout="wide"
    )

    st.title("Media Agent")
    st.markdown("具备强大多模态能力的AI代理")

    # 检查URL参数
    try:
        # 尝试使用新版本的query_params
        query_params = st.query_params
        auto_query = query_params.get('query', '')
        auto_search = query_params.get('auto_search', 'false').lower() == 'true'
    except AttributeError:
        # 兼容旧版本
        query_params = st.experimental_get_query_params()
        auto_query = query_params.get('query', [''])[0]
        auto_search = query_params.get('auto_search', ['false'])[0].lower() == 'true'

    # ----- 配置被硬编码 -----
    # 强制使用 Gemini
    llm_provider = "gemini"
    model_name = "gemini-2.5-pro"
    # 默认高级配置
    max_reflections = 2
    max_content_length = 20000

    # 主界面
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("研究查询")
        
        # 如果有自动查询，使用它作为默认值
        default_query = auto_query if auto_query else ""
        
        query = st.text_area(
            "请输入您要研究的问题",
            value=default_query,
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
    
    # 自动搜索逻辑
    if auto_search and auto_query and 'auto_search_executed' not in st.session_state:
        st.session_state.auto_search_executed = True
        start_research = True
        query = auto_query

    # 验证配置
    if start_research:
        if not query.strip():
            st.error("请输入研究查询")
            return

        # 由于强制使用Gemini，检查相关的API密钥
        if not GEMINI_API_KEY:
            st.error("请在您的配置文件(config.py)中设置GEMINI_API_KEY")
            return
        if not BOCHA_Web_Search_API_KEY:
            st.error("请在您的配置文件(config.py)中设置BOCHA_Web_Search_API_KEY")
            return

        # 自动使用配置文件中的API密钥
        gemini_key = GEMINI_API_KEY
        bocha_key = BOCHA_Web_Search_API_KEY

        # 创建配置
        config = Config(
            deepseek_api_key=None,
            openai_api_key=None,
            gemini_api_key=gemini_key,
            bocha_api_key=bocha_key,
            default_llm_provider=llm_provider,
            deepseek_model="deepseek-chat",  # 保留默认值以兼容
            openai_model="gpt-4o-mini",  # 保留默认值以兼容
            gemini_model=model_name,
            max_reflections=max_reflections,
            max_content_length=max_content_length,
            output_dir="media_engine_streamlit_reports"
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