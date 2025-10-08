"""
Kimi LLM实现
使用Moonshot AI的Kimi API进行文本生成
"""

import os
import sys
from typing import Optional, Dict, Any
from openai import OpenAI
# 假设 .base 模块和 BaseLLM 类已存在
from .base import BaseLLM

DEFAULT_KIMI_BASE_URL = "https://api.moonshot.cn/v1"

# 添加utils目录到Python路径并导入重试模块
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    utils_dir = os.path.join(root_dir, 'utils')
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    from retry_helper import with_retry, with_graceful_retry, LLM_RETRY_CONFIG
except ImportError:
    # 如果无法导入重试模块，使用空装饰器避免报错
    def with_retry(config):
        def decorator(func):
            return func
        return decorator
    LLM_RETRY_CONFIG = None


class KimiLLM(BaseLLM):
    """Kimi LLM实现类"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化Kimi客户端

        Args:
            api_key: Kimi API密钥，如果不提供则从环境变量读取
            model_name: 模型名称，默认使用kimi-k2-0711-preview
            base_url: Kimi API基础地址
        """
        if api_key is None:
            api_key = os.getenv("KIMI_API_KEY")
            if not api_key:
                raise ValueError("Kimi API Key未找到！请设置KIMI_API_KEY环境变量或在初始化时提供")

        super().__init__(api_key, model_name)
        
        self.base_url = base_url or os.getenv("KIMI_BASE_URL") or DEFAULT_KIMI_BASE_URL
        
        # 初始化OpenAI客户端，使用Kimi的endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.default_model = model_name or self.get_default_model()

    def get_default_model(self) -> str:
        """获取默认模型名称"""
        return "kimi-k2-0711-preview"

    @with_retry(LLM_RETRY_CONFIG)
    def invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        调用Kimi API生成回复

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            **kwargs: 其他参数，如temperature、max_tokens等

        Returns:
            Kimi生成的回复文本
        """
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 智能计算max_tokens - 根据输入长度自动调整输出长度
            input_length = len(system_prompt) + len(user_prompt)
            if input_length > 100000:  # 超长文本
                default_max_tokens = 81920
            elif input_length > 50000:  # 超长文本
                default_max_tokens = 40960
            elif input_length > 20000:  # 长文本
                default_max_tokens = 16384
            elif input_length > 5000:  # 中等文本
                default_max_tokens = 8192
            else:  # 短文本
                default_max_tokens = 4096

            # 设置默认参数，针对长文本处理优化
            params = {
                "model": self.default_model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.6),  # Kimi建议使用0.6
                "max_tokens": kwargs.get("max_tokens", default_max_tokens),  # 智能调整token限制
                "stream": False
            }

            # 添加其他可选参数
            if "top_p" in kwargs:
                params["top_p"] = kwargs["top_p"]
            if "presence_penalty" in kwargs:
                params["presence_penalty"] = kwargs["presence_penalty"]
            if "frequency_penalty" in kwargs:
                params["frequency_penalty"] = kwargs["frequency_penalty"]
            if "stop" in kwargs:
                params["stop"] = kwargs["stop"]

            # 输出调试信息（仅在使用Kimi时）
            print(f"[Kimi] 输入长度: {input_length}, 使用max_tokens: {params['max_tokens']}")

            # 调用API
            response = self.client.chat.completions.create(**params)

            # 提取回复内容
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                return self.validate_response(content)
            else:
                return ""

        except Exception as e:
            print(f"Kimi API调用错误: {str(e)}")
            raise e

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前模型信息

        Returns:
            模型信息字典
        """
        return {
            "provider": "Kimi",
            "model": self.default_model,
            "api_base": self.base_url,
            "max_context_length": "长文本支持（200K+ tokens）"
        }

    # ==================== 代码修改部分 ====================
    def invoke_long_context(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        专门用于长文本处理的调用方法 (作为invoke的兼容接口)。
        此方法通过设置推荐的默认参数，然后调用通用的invoke方法来处理请求。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            **kwargs: 其他参数

        Returns:
            Kimi生成的回复文本
        """
        # 为长文本场景，设置一个慷慨的默认 max_tokens，仅当用户未指定时生效。
        # 您原有的16384是一个非常合理的值。
        kwargs.setdefault("max_tokens", 16384)
        
        # 直接调用核心的invoke方法，将所有参数（包括预设的默认值）传递给它。
        return self.invoke(system_prompt, user_prompt, **kwargs)
