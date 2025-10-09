"""
Configuration management module for the Report Engine.
"""

import os
from dataclasses import dataclass
from typing import Optional


def _get_value(source, key: str, default=None, *fallback_keys: str):
    candidates = (key,) + fallback_keys
    value = None
    for candidate in candidates:
        if isinstance(source, dict):
            value = source.get(candidate)
        else:
            value = getattr(source, candidate, None)
        if value not in (None, ""):
            break
    if value in (None, ""):
        for candidate in candidates:
            env_val = os.getenv(candidate)
            if env_val not in (None, ""):
                value = env_val
                break
    return value if value not in (None, "") else default


@dataclass
class Config:
    """Report Engine configuration."""

    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model_name: Optional[str] = None
    llm_provider: Optional[str] = None  # compatibility

    max_content_length: int = 200000
    output_dir: str = "final_reports"
    template_dir: str = "ReportEngine/report_template"

    api_timeout: float = 900.0
    max_retry_delay: float = 180.0
    max_retries: int = 8

    log_file: str = "logs/report.log"
    enable_pdf_export: bool = True
    chart_style: str = "modern"

    def __post_init__(self):
        if not self.llm_provider and self.llm_model_name:
            self.llm_provider = self.llm_model_name

    def validate(self) -> bool:
        if not self.llm_api_key:
            print("错误: Report Engine LLM API Key 未设置 (REPORT_ENGINE_API_KEY)。")
            return False
        if not self.llm_model_name:
            print("错误: Report Engine 模型名称未设置 (REPORT_ENGINE_MODEL_NAME)。")
            return False
        return True

    @classmethod
    def from_file(cls, config_file: str) -> "Config":
        if config_file.endswith(".py"):
            import importlib.util

            spec = importlib.util.spec_from_file_location("config", config_file)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            return cls(
                llm_api_key=_get_value(config_module, "REPORT_ENGINE_API_KEY"),
                llm_base_url=_get_value(config_module, "REPORT_ENGINE_BASE_URL"),
                llm_model_name=_get_value(config_module, "REPORT_ENGINE_MODEL_NAME"),
                max_content_length=int(_get_value(config_module, "MAX_CONTENT_LENGTH", 200000)),
                output_dir=_get_value(config_module, "REPORT_OUTPUT_DIR", "final_reports"),
                template_dir=_get_value(config_module, "TEMPLATE_DIR", "ReportEngine/report_template"),
                api_timeout=float(_get_value(config_module, "REPORT_API_TIMEOUT", 900.0)),
                max_retry_delay=float(_get_value(config_module, "REPORT_MAX_RETRY_DELAY", 180.0)),
                max_retries=int(_get_value(config_module, "REPORT_MAX_RETRIES", 8)),
                log_file=_get_value(config_module, "REPORT_LOG_FILE", "logs/report.log"),
                enable_pdf_export=str(
                    _get_value(config_module, "ENABLE_PDF_EXPORT", "true")
                ).lower()
                in ("true", "1", "yes"),
                chart_style=_get_value(config_module, "CHART_STYLE", "modern"),
            )

        config_dict = {}
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        config_dict[key.strip()] = value.strip()

        return cls(
            llm_api_key=_get_value(config_dict, "REPORT_ENGINE_API_KEY"),
            llm_base_url=_get_value(config_dict, "REPORT_ENGINE_BASE_URL"),
            llm_model_name=_get_value(config_dict, "REPORT_ENGINE_MODEL_NAME"),
            max_content_length=int(_get_value(config_dict, "MAX_CONTENT_LENGTH", 200000)),
            output_dir=_get_value(config_dict, "REPORT_OUTPUT_DIR", "final_reports"),
            template_dir=_get_value(config_dict, "TEMPLATE_DIR", "ReportEngine/report_template"),
            api_timeout=float(_get_value(config_dict, "REPORT_API_TIMEOUT", 900.0)),
            max_retry_delay=float(_get_value(config_dict, "REPORT_MAX_RETRY_DELAY", 180.0)),
            max_retries=int(_get_value(config_dict, "REPORT_MAX_RETRIES", 8)),
            log_file=_get_value(config_dict, "REPORT_LOG_FILE", "logs/report.log"),
            enable_pdf_export=str(
                _get_value(config_dict, "ENABLE_PDF_EXPORT", "true")
            ).lower()
            in ("true", "1", "yes"),
            chart_style=_get_value(config_dict, "CHART_STYLE", "modern"),
        )


def load_config(config_file: Optional[str] = None) -> Config:
    if config_file:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        file_to_load = config_file
    else:
        for candidate in ("config.py", "config.env", ".env"):
            if os.path.exists(candidate):
                file_to_load = candidate
                print(f"已找到配置文件: {candidate}")
                break
        else:
            raise FileNotFoundError("未找到配置文件，请创建 config.py。")

    config = Config.from_file(file_to_load)
    if not config.validate():
        raise ValueError("Report Engine 配置校验失败，请检查 config.py 中的相关配置。")
    return config


def print_config(config: Config):
    print("\n=== Report Engine 配置 ===")
    print(f"LLM 模型: {config.llm_model_name}")
    print(f"LLM Base URL: {config.llm_base_url or '(默认)'}")
    print(f"最大内容长度: {config.max_content_length}")
    print(f"输出目录: {config.output_dir}")
    print(f"模板目录: {config.template_dir}")
    print(f"API 超时时间: {config.api_timeout} 秒")
    print(f"最大重试间隔: {config.max_retry_delay} 秒")
    print(f"最大重试次数: {config.max_retries}")
    print(f"日志文件: {config.log_file}")
    print(f"PDF 导出: {config.enable_pdf_export}")
    print(f"图表样式: {config.chart_style}")
    print(f"LLM API Key: {'已配置' if config.llm_api_key else '未配置'}")
    print("========================\n")
