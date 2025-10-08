"""
配置管理模块
处理环境变量和配置参数
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """配置类"""
    # API密钥
    deepseek_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    kimi_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    openai_base_url: Optional[str] = None
    kimi_base_url: str = "https://api.moonshot.cn/v1"
    
    # 数据库配置
    db_host: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_name: Optional[str] = None
    db_port: int = 3306
    db_charset: str = "utf8mb4"
    
    # 模型配置
    default_llm_provider: str = "deepseek"  # deepseek、openai 或 kimi
    deepseek_model: str = "deepseek-chat"
    openai_model: str = "gpt-4o-mini"
    kimi_model: str = "kimi-k2-0711-preview"
    
    # 搜索配置
    search_timeout: int = 240
    max_content_length: int = 500000  # 提高5倍以充分利用Kimi的长文本能力
    
    # 数据库查询限制
    default_search_hot_content_limit: int = 100
    default_search_topic_globally_limit_per_table: int = 50
    default_search_topic_by_date_limit_per_table: int = 100
    default_get_comments_for_topic_limit: int = 500
    default_search_topic_on_platform_limit: int = 200
    
    # Agent配置
    max_reflections: int = 3
    max_paragraphs: int = 6
    
    # 结果处理限制
    max_search_results_for_llm: int = 0  # 0表示不限制，传递所有搜索结果给LLM
    max_high_confidence_sentiment_results: int = 0  # 0表示不限制，返回所有高置信度情感分析结果
    
    # 输出配置
    output_dir: str = "reports"
    save_intermediate_states: bool = True
    
    def validate(self) -> bool:
        """验证配置"""
        # 检查必需的API密钥
        if self.default_llm_provider == "deepseek" and not self.deepseek_api_key:
            print("错误: DeepSeek API Key未设置")
            return False
        
        if self.default_llm_provider == "openai" and not self.openai_api_key:
            print("错误: OpenAI API Key未设置")
            return False
        
        if not all([self.db_host, self.db_user, self.db_password, self.db_name]):
            print("错误: 数据库连接信息不完整")
            return False
        
        return True
    
    @classmethod
    def from_file(cls, config_file: str) -> "Config":
        """从配置文件创建配置"""
        if config_file.endswith('.py'):
            # Python配置文件
            import importlib.util
            
            # 动态导入配置文件
            spec = importlib.util.spec_from_file_location("config", config_file)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            return cls(
                deepseek_api_key=getattr(config_module, "DEEPSEEK_API_KEY", None),
                openai_api_key=getattr(config_module, "OPENAI_API_KEY", None),
                kimi_api_key=getattr(config_module, "KIMI_API_KEY", None),
                deepseek_base_url=getattr(config_module, "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                openai_base_url=getattr(config_module, "OPENAI_BASE_URL", None),
                kimi_base_url=getattr(config_module, "KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
                
                db_host=getattr(config_module, "DB_HOST", None),
                db_user=getattr(config_module, "DB_USER", None),
                db_password=getattr(config_module, "DB_PASSWORD", None),
                db_name=getattr(config_module, "DB_NAME", None),
                db_port=getattr(config_module, "DB_PORT", 3306),
                db_charset=getattr(config_module, "DB_CHARSET", "utf8mb4"),
                
                default_llm_provider=getattr(config_module, "DEFAULT_LLM_PROVIDER", "deepseek"),
                deepseek_model=getattr(config_module, "DEEPSEEK_MODEL", "deepseek-chat"),
                openai_model=getattr(config_module, "OPENAI_MODEL", "gpt-4o-mini"),

                search_timeout=getattr(config_module, "SEARCH_TIMEOUT", 240),
                max_content_length=getattr(config_module, "SEARCH_CONTENT_MAX_LENGTH", 200000),
                
                default_search_hot_content_limit=getattr(config_module, "DEFAULT_SEARCH_HOT_CONTENT_LIMIT", 100),
                default_search_topic_globally_limit_per_table=getattr(config_module, "DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE", 50),
                default_search_topic_by_date_limit_per_table=getattr(config_module, "DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE", 100),
                default_get_comments_for_topic_limit=getattr(config_module, "DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT", 500),
                default_search_topic_on_platform_limit=getattr(config_module, "DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT", 200),
                
                max_reflections=getattr(config_module, "MAX_REFLECTIONS", 2),
                max_paragraphs=getattr(config_module, "MAX_PARAGRAPHS", 5),
                
                max_search_results_for_llm=getattr(config_module, "MAX_SEARCH_RESULTS_FOR_LLM", 0),
                max_high_confidence_sentiment_results=getattr(config_module, "MAX_HIGH_CONFIDENCE_SENTIMENT_RESULTS", 0),
                
                output_dir=getattr(config_module, "OUTPUT_DIR", "reports"),
                save_intermediate_states=getattr(config_module, "SAVE_INTERMEDIATE_STATES", True)
            )
        else:
            # .env格式配置文件
            config_dict = {}
            
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            config_dict[key.strip()] = value.strip()
            
            return cls(
                deepseek_api_key=config_dict.get("DEEPSEEK_API_KEY"),
                openai_api_key=config_dict.get("OPENAI_API_KEY"),
                kimi_api_key=config_dict.get("KIMI_API_KEY"),
                deepseek_base_url=config_dict.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                openai_base_url=config_dict.get("OPENAI_BASE_URL"),
                kimi_base_url=config_dict.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
                
                db_host=config_dict.get("DB_HOST"),
                db_user=config_dict.get("DB_USER"),
                db_password=config_dict.get("DB_PASSWORD"),
                db_name=config_dict.get("DB_NAME"),
                db_port=int(config_dict.get("DB_PORT", "3306")),
                db_charset=config_dict.get("DB_CHARSET", "utf8mb4"),
                
                default_llm_provider=config_dict.get("DEFAULT_LLM_PROVIDER", "deepseek"),
                deepseek_model=config_dict.get("DEEPSEEK_MODEL", "deepseek-chat"),
                openai_model=config_dict.get("OPENAI_MODEL", "gpt-4o-mini"),
                kimi_model=config_dict.get("KIMI_MODEL", "kimi-k2-0711-preview"),

                search_timeout=int(config_dict.get("SEARCH_TIMEOUT", "240")),
                max_content_length=int(config_dict.get("SEARCH_CONTENT_MAX_LENGTH", "500000")),
                
                default_search_hot_content_limit=int(config_dict.get("DEFAULT_SEARCH_HOT_CONTENT_LIMIT", "100")),
                default_search_topic_globally_limit_per_table=int(config_dict.get("DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE", "50")),
                default_search_topic_by_date_limit_per_table=int(config_dict.get("DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE", "100")),
                default_get_comments_for_topic_limit=int(config_dict.get("DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT", "500")),
                default_search_topic_on_platform_limit=int(config_dict.get("DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT", "200")),
                
                max_reflections=int(config_dict.get("MAX_REFLECTIONS", "2")),
                max_paragraphs=int(config_dict.get("MAX_PARAGRAPHS", "5")),
                
                max_search_results_for_llm=int(config_dict.get("MAX_SEARCH_RESULTS_FOR_LLM", "0")),
                max_high_confidence_sentiment_results=int(config_dict.get("MAX_HIGH_CONFIDENCE_SENTIMENT_RESULTS", "0")),
                
                output_dir=config_dict.get("OUTPUT_DIR", "reports"),
                save_intermediate_states=config_dict.get("SAVE_INTERMEDIATE_STATES", "true").lower() == "true"
            )


def load_config(config_file: Optional[str] = None) -> Config:
    """
    加载配置
    
    Args:
        config_file: 配置文件路径，如果不指定则使用默认路径
        
    Returns:
        配置对象
    """
    # 确定配置文件路径
    if config_file:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        file_to_load = config_file
    else:
        # 尝试加载常见的配置文件
        for config_path in ["config.py", "config.env", ".env"]:
            if os.path.exists(config_path):
                file_to_load = config_path
                print(f"已找到配置文件: {config_path}")
                break
        else:
            raise FileNotFoundError("未找到配置文件，请创建 config.py 文件")
    
    # 创建配置对象
    config = Config.from_file(file_to_load)
    
    # 验证配置
    if not config.validate():
        raise ValueError("配置验证失败，请检查配置文件中的API密钥")
    
    return config


def print_config(config: Config):
    """打印配置信息（隐藏敏感信息）"""
    print("\n=== 当前配置 ===")
    print(f"LLM提供商: {config.default_llm_provider}")
    print(f"DeepSeek模型: {config.deepseek_model}")
    print(f"OpenAI模型: {config.openai_model}")

    print(f"搜索超时: {config.search_timeout}秒")
    print(f"最大内容长度: {config.max_content_length}")
    print(f"最大反思次数: {config.max_reflections}")
    print(f"最大段落数: {config.max_paragraphs}")
    print(f"输出目录: {config.output_dir}")
    print(f"保存中间状态: {config.save_intermediate_states}")
    
    # 显示API密钥和数据库状态（不显示实际密钥）
    print(f"DeepSeek API Key: {'已设置' if config.deepseek_api_key else '未设置'}")
    print(f"OpenAI API Key: {'已设置' if config.openai_api_key else '未设置'}")
    print(f"数据库连接: {'已配置' if all([config.db_host, config.db_user, config.db_password, config.db_name]) else '未配置'}")
    print(f"数据库主机: {config.db_host}")
    print(f"数据库端口: {config.db_port}")
    print(f"数据库名称: {config.db_name}")
    print("==================\n")
