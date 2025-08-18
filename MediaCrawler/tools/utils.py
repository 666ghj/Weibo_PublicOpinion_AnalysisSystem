# 声明：本代码仅供学习和研究目的使用。使用者应遵守以下原则：  
# 1. 不得用于任何商业用途。  
# 2. 使用时应遵守目标平台的使用条款和robots.txt规则。  
# 3. 不得进行大规模爬取或对平台造成运营干扰。  
# 4. 应合理控制请求频率，避免给目标平台带来不必要的负担。   
# 5. 不得用于任何非法或不当的用途。
#   
# 详细许可条款请参阅项目根目录下的LICENSE文件。  
# 使用本代码即表示您同意遵守上述原则和LICENSE中的所有条款。  


import argparse
import logging

from .crawler_util import *
from .slider_util import *
from .time_util import *


def init_loging_config():
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s (%(filename)s:%(lineno)d) - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    _logger = logging.getLogger("MediaCrawler")
    _logger.setLevel(level)

    # 根据配置决定是否启用详细日志
    try:
        import config
        if not getattr(config, 'ENABLE_VERBOSE_LOGGING', False):
            # 简化第三方库日志输出，完全禁用或只显示CRITICAL级别
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.CRITICAL)

            asyncio_logger = logging.getLogger("asyncio")
            asyncio_logger.setLevel(logging.CRITICAL)

            playwright_logger = logging.getLogger("playwright")
            playwright_logger.setLevel(logging.CRITICAL)

            # 禁用一些特定的日志记录器
            logging.getLogger("urllib3").setLevel(logging.CRITICAL)
            logging.getLogger("httpcore").setLevel(logging.CRITICAL)
    except ImportError:
        # 如果config模块还未加载，使用默认的简化设置
        httpx_logger = logging.getLogger("httpx")
        httpx_logger.setLevel(logging.CRITICAL)

        asyncio_logger = logging.getLogger("asyncio")
        asyncio_logger.setLevel(logging.CRITICAL)

        playwright_logger = logging.getLogger("playwright")
        playwright_logger.setLevel(logging.CRITICAL)

        logging.getLogger("urllib3").setLevel(logging.CRITICAL)
        logging.getLogger("httpcore").setLevel(logging.CRITICAL)

    return _logger


logger = init_loging_config()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
