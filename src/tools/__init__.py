"""
工具调用模块
提供外部工具接口，如网络搜索等
"""

from .search import tavily_search, SearchResult

__all__ = ["tavily_search", "SearchResult"]
