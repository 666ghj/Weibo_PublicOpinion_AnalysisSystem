# 声明：本代码仅供学习和研究目的使用。使用者应遵守以下原则：
# 1. 不得用于任何商业用途。
# 2. 使用时应遵守目标平台的使用条款和robots.txt规则。
# 3. 不得进行大规模爬取或对平台造成运营干扰。
# 4. 应合理控制请求频率，避免给目标平台带来不必要的负担。
# 5. 不得用于任何非法或不当的用途。
#
# 详细许可条款请参阅项目根目录下的LICENSE文件。
# 使用本代码即表示您同意遵守上述原则和LICENSE中的所有条款。


import asyncio
import sys
import warnings
from typing import Optional

import cmd_arg
import config
import db

# 抑制特定的警告和异常信息
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 自定义异常钩子，过滤程序结束时的清理异常
def custom_excepthook(exc_type, exc_value, exc_traceback):
    # 过滤掉程序结束时的常见清理异常
    if exc_type == KeyboardInterrupt:
        return

    error_msg = str(exc_value)
    ignore_errors = [
        "Event loop is closed",
        "Target page, context or browser has been closed",
        "I/O operation on closed pipe",
        "Task was destroyed but it is pending",
        "Connection.run()",
        "_ProactorBasePipeTransport.__del__",
        "BaseSubprocessTransport.__del__"
    ]

    if any(ignore_err in error_msg for ignore_err in ignore_errors):
        return

    # 只显示真正的错误
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# 设置自定义异常钩子
sys.excepthook = custom_excepthook
from base.base_crawler import AbstractCrawler
from media_platform.bilibili import BilibiliCrawler
from media_platform.douyin import DouYinCrawler
from media_platform.kuaishou import KuaishouCrawler
from media_platform.tieba import TieBaCrawler
from media_platform.weibo import WeiboCrawler
from media_platform.xhs import XiaoHongShuCrawler
from media_platform.zhihu import ZhihuCrawler


class CrawlerFactory:
    CRAWLERS = {
        "xhs": XiaoHongShuCrawler,
        "dy": DouYinCrawler,
        "ks": KuaishouCrawler,
        "bili": BilibiliCrawler,
        "wb": WeiboCrawler,
        "tieba": TieBaCrawler,
        "zhihu": ZhihuCrawler,
    }

    @staticmethod
    def create_crawler(platform: str) -> AbstractCrawler:
        crawler_class = CrawlerFactory.CRAWLERS.get(platform)
        if not crawler_class:
            raise ValueError(
                "Invalid Media Platform Currently only supported xhs or dy or ks or bili ..."
            )
        return crawler_class()


crawler: Optional[AbstractCrawler] = None


async def main():
    # Init crawler
    global crawler

    # parse cmd
    await cmd_arg.parse_cmd()

    # init db
    if config.SAVE_DATA_OPTION in ["db", "sqlite"]:
        await db.init_db()

    crawler = CrawlerFactory.create_crawler(platform=config.PLATFORM)
    await crawler.start()


def cleanup():
    if crawler:
        # asyncio.run(crawler.close())
        pass
    if config.SAVE_DATA_OPTION in ["db", "sqlite"]:
        asyncio.run(db.close())


if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(main())
    except KeyboardInterrupt:
        # 用户手动中断，静默处理
        pass
    except Exception as e:
        # 只显示真正的错误，不显示程序结束时的清理异常
        if not any(err_type in str(e) for err_type in ["Event loop is closed", "Target page", "I/O operation on closed pipe"]):
            print(f"程序运行出错: {e}")
    finally:
        try:
            cleanup()
        except Exception:
            # 静默处理清理过程中的异常
            pass
