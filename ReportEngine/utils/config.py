"""
Report Engine配置管理模块
处理环境变量和配置参数
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Report Engine配置类"""
    # API密钥
    gemini_api_key: Optional[str] = None
    gemini_base_url: str = "https://www.chataiapi.com/v1"
    
    # 模型配置
    default_llm_provider: str = "gemini"
    gemini_model: str = "gemini-2.5-pro"
    
    # 报告配置
    max_content_length: int = 200000
    output_dir: str = "final_reports"
    template_dir: str = "ReportEngine/report_template"
    
    # 超时配置 - 专门为长报告生成优化（平均生成时间7分钟）
    api_timeout: float = 900.0      # API调用超时时间（秒），设置为15分钟，适应7分钟平均生成时间
    max_retry_delay: float = 180.0  # 最大重试延迟（秒），设置为3分钟
    max_retries: int = 8            # 最大重试次数，增加到8次
    
    # 日志配置
    log_file: str = "logs/report.log"
    
    # HTML导出配置
    enable_pdf_export: bool = True
    chart_style: str = "modern"  # modern, classic, minimal
    
    def validate(self) -> bool:
        """验证配置"""
        if not self.gemini_api_key:
            print("错误: Gemini API Key未设置")
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
                gemini_api_key=getattr(config_module, "GEMINI_API_KEY", None),
                gemini_base_url=getattr(config_module, "GEMINI_BASE_URL", "https://www.chataiapi.com/v1"),
                default_llm_provider=getattr(config_module, "DEFAULT_LLM_PROVIDER", "gemini"),
                gemini_model=getattr(config_module, "GEMINI_MODEL", "gemini-2.5-pro"),
                max_content_length=getattr(config_module, "MAX_CONTENT_LENGTH", 200000),
                output_dir=getattr(config_module, "REPORT_OUTPUT_DIR", "final_reports"),
                template_dir=getattr(config_module, "TEMPLATE_DIR", "ReportEngine/report_template"),
                api_timeout=getattr(config_module, "REPORT_API_TIMEOUT", 900.0),
                max_retry_delay=getattr(config_module, "REPORT_MAX_RETRY_DELAY", 180.0),
                max_retries=getattr(config_module, "REPORT_MAX_RETRIES", 8),
                log_file=getattr(config_module, "REPORT_LOG_FILE", "logs/report.log"),
                enable_pdf_export=getattr(config_module, "ENABLE_PDF_EXPORT", True),
                chart_style=getattr(config_module, "CHART_STYLE", "modern")
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
                gemini_api_key=config_dict.get("GEMINI_API_KEY"),
                gemini_base_url=config_dict.get("GEMINI_BASE_URL", "https://www.chataiapi.com/v1"),
                default_llm_provider=config_dict.get("DEFAULT_LLM_PROVIDER", "gemini"),
                gemini_model=config_dict.get("GEMINI_MODEL", "gemini-2.5-pro"),
                max_content_length=int(config_dict.get("MAX_CONTENT_LENGTH", "200000")),
                output_dir=config_dict.get("REPORT_OUTPUT_DIR", "final_reports"),
                template_dir=config_dict.get("TEMPLATE_DIR", "ReportEngine/report_template"),
                api_timeout=float(config_dict.get("REPORT_API_TIMEOUT", "900.0")),
                max_retry_delay=float(config_dict.get("REPORT_MAX_RETRY_DELAY", "180.0")),
                max_retries=int(config_dict.get("REPORT_MAX_RETRIES", "8")),
                log_file=config_dict.get("REPORT_LOG_FILE", "logs/report.log"),
                enable_pdf_export=config_dict.get("ENABLE_PDF_EXPORT", "true").lower() == "true",
                chart_style=config_dict.get("CHART_STYLE", "modern")
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
                break
        else:
            raise FileNotFoundError("未找到配置文件，请创建 config.py 文件")
    
    # 创建配置对象
    config = Config.from_file(file_to_load)
    
    # 验证配置
    if not config.validate():
        raise ValueError("Report Engine配置验证失败，请检查配置文件中的API密钥")
    
    return config


def print_config(config: Config):
    """打印配置信息（隐藏敏感信息）"""
    print("\n=== Report Engine配置 ===")
    print(f"LLM提供商: {config.default_llm_provider}")
    print(f"Gemini模型: {config.gemini_model}")
    print(f"最大内容长度: {config.max_content_length}")
    print(f"输出目录: {config.output_dir}")
    print(f"模板目录: {config.template_dir}")
    print(f"API超时时间: {config.api_timeout}秒（{config.api_timeout/60:.1f}分钟）")
    print(f"最大重试延迟: {config.max_retry_delay}秒（{config.max_retry_delay/60:.1f}分钟）")
    print(f"最大重试次数: {config.max_retries}次")
    print(f"日志文件: {config.log_file}")
    print(f"PDF导出: {config.enable_pdf_export}")
    print(f"图表样式: {config.chart_style}")
    print(f"Gemini API Key: {'已设置' if config.gemini_api_key else '未设置'}")
    print("========================\n")
