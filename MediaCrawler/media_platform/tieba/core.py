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
import os
import random
from asyncio import Task
from typing import Dict, List, Optional, Tuple

from playwright.async_api import (
    BrowserContext,
    BrowserType,
    Page,
    Playwright,
    async_playwright,
)

import config
from base.base_crawler import AbstractCrawler
from model.m_baidu_tieba import TiebaCreator, TiebaNote
from proxy.proxy_ip_pool import IpInfoModel, create_ip_pool
from store import tieba as tieba_store
from tools import utils
from tools.cdp_browser import CDPBrowserManager
from var import crawler_type_var, source_keyword_var

from .client import BaiduTieBaClient
from .field import SearchNoteType, SearchSortType
from .help import TieBaExtractor
from .login import BaiduTieBaLogin


class TieBaCrawler(AbstractCrawler):
    context_page: Page
    tieba_client: BaiduTieBaClient
    browser_context: BrowserContext
    cdp_manager: Optional[CDPBrowserManager]

    def __init__(self) -> None:
        self.index_url = "https://tieba.baidu.com"
        self.user_agent = utils.get_user_agent()
        self._page_extractor = TieBaExtractor()
        self.cdp_manager = None

    async def start(self) -> None:
        """
        Start the crawler
        Returns:

        """
        ip_proxy_pool, httpx_proxy_format = None, None
        if config.ENABLE_IP_PROXY:
            utils.logger.info(
                "[BaiduTieBaCrawler.start] Begin create ip proxy pool ..."
            )
            ip_proxy_pool = await create_ip_pool(
                config.IP_PROXY_POOL_COUNT, enable_validate_ip=True
            )
            ip_proxy_info: IpInfoModel = await ip_proxy_pool.get_proxy()
            _, httpx_proxy_format = utils.format_proxy_info(ip_proxy_info)
            utils.logger.info(
                f"[BaiduTieBaCrawler.start] Init default ip proxy, value: {httpx_proxy_format}"
            )

        # Create a client to interact with the baidutieba website.
        self.tieba_client = BaiduTieBaClient(
            ip_pool=ip_proxy_pool,
            default_ip_proxy=httpx_proxy_format,
        )
        crawler_type_var.set(config.CRAWLER_TYPE)
        if config.CRAWLER_TYPE == "search":
            # Search for notes and retrieve their comment information.
            await self.search()
            await self.get_specified_tieba_notes()
        elif config.CRAWLER_TYPE == "detail":
            # Get the information and comments of the specified post
            await self.get_specified_notes()
        elif config.CRAWLER_TYPE == "creator":
            # Get creator's information and their notes and comments
            await self.get_creators_and_notes()
        elif config.CRAWLER_TYPE == "trending":
            # Get trending keywords and crawl related posts
            await self.search_trending()
        else:
            pass

        utils.logger.info("[BaiduTieBaCrawler.start] Tieba Crawler finished ...")

    async def search(self) -> None:
        """
        Search for notes and retrieve their comment information.
        Returns:

        """
        utils.logger.info(
            "[BaiduTieBaCrawler.search] Begin search baidu tieba keywords"
        )
        tieba_limit_count = 10  # tieba limit page fixed value
        if config.CRAWLER_MAX_NOTES_COUNT < tieba_limit_count:
            config.CRAWLER_MAX_NOTES_COUNT = tieba_limit_count
        start_page = config.START_PAGE
        for keyword in config.KEYWORDS.split(","):
            source_keyword_var.set(keyword)
            utils.logger.info(
                f"[BaiduTieBaCrawler.search] Current search keyword: {keyword}"
            )
            page = 1
            while (
                page - start_page + 1
            ) * tieba_limit_count <= config.CRAWLER_MAX_NOTES_COUNT:
                if page < start_page:
                    utils.logger.info(f"[BaiduTieBaCrawler.search] Skip page {page}")
                    page += 1
                    continue
                try:
                    utils.logger.info(
                        f"[BaiduTieBaCrawler.search] search tieba keyword: {keyword}, page: {page}"
                    )
                    notes_list: List[TiebaNote] = (
                        await self.tieba_client.get_notes_by_keyword(
                            keyword=keyword,
                            page=page,
                            page_size=tieba_limit_count,
                            sort=SearchSortType.TIME_DESC,
                            note_type=SearchNoteType.FIXED_THREAD,
                        )
                    )
                    if not notes_list:
                        utils.logger.info(
                            f"[BaiduTieBaCrawler.search] Search note list is empty"
                        )
                        break
                    utils.logger.info(
                        f"[BaiduTieBaCrawler.search] Note list len: {len(notes_list)}"
                    )
                    await self.get_specified_notes(
                        note_id_list=[note_detail.note_id for note_detail in notes_list]
                    )
                    page += 1
                except Exception as ex:
                    utils.logger.error(
                        f"[BaiduTieBaCrawler.search] Search keywords error, current page: {page}, current keyword: {keyword}, err: {ex}"
                    )
                    break

    async def get_specified_tieba_notes(self):
        """
        Get the information and comments of the specified post by tieba name
        Returns:

        """
        tieba_limit_count = 50
        if config.CRAWLER_MAX_NOTES_COUNT < tieba_limit_count:
            config.CRAWLER_MAX_NOTES_COUNT = tieba_limit_count
        for tieba_name in config.TIEBA_NAME_LIST:
            utils.logger.info(
                f"[BaiduTieBaCrawler.get_specified_tieba_notes] Begin get tieba name: {tieba_name}"
            )
            page_number = 0
            while page_number <= config.CRAWLER_MAX_NOTES_COUNT:
                note_list: List[TiebaNote] = (
                    await self.tieba_client.get_notes_by_tieba_name(
                        tieba_name=tieba_name, page_num=page_number
                    )
                )
                if not note_list:
                    utils.logger.info(
                        f"[BaiduTieBaCrawler.get_specified_tieba_notes] Get note list is empty"
                    )
                    break

                utils.logger.info(
                    f"[BaiduTieBaCrawler.get_specified_tieba_notes] tieba name: {tieba_name} note list len: {len(note_list)}"
                )
                await self.get_specified_notes([note.note_id for note in note_list])
                page_number += tieba_limit_count

    async def get_specified_notes(
        self, note_id_list: List[str] = config.TIEBA_SPECIFIED_ID_LIST
    ):
        """
        Get the information and comments of the specified post
        Args:
            note_id_list:

        Returns:

        """
        semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY_NUM)
        task_list = [
            self.get_note_detail_async_task(note_id=note_id, semaphore=semaphore)
            for note_id in note_id_list
        ]
        note_details = await asyncio.gather(*task_list)
        note_details_model: List[TiebaNote] = []
        for note_detail in note_details:
            if note_detail is not None:
                note_details_model.append(note_detail)
                await tieba_store.update_tieba_note(note_detail)
        await self.batch_get_note_comments(note_details_model)

    async def get_note_detail_async_task(
        self, note_id: str, semaphore: asyncio.Semaphore
    ) -> Optional[TiebaNote]:
        """
        Get note detail
        Args:
            note_id: baidu tieba note id
            semaphore: asyncio semaphore

        Returns:

        """
        async with semaphore:
            try:
                utils.logger.info(
                    f"[BaiduTieBaCrawler.get_note_detail] Begin get note detail, note_id: {note_id}"
                )
                note_detail: TiebaNote = await self.tieba_client.get_note_by_id(note_id)
                if not note_detail:
                    utils.logger.error(
                        f"[BaiduTieBaCrawler.get_note_detail] Get note detail error, note_id: {note_id}"
                    )
                    return None
                return note_detail
            except Exception as ex:
                utils.logger.error(
                    f"[BaiduTieBaCrawler.get_note_detail] Get note detail error: {ex}"
                )
                return None
            except KeyError as ex:
                utils.logger.error(
                    f"[BaiduTieBaCrawler.get_note_detail] have not fund note detail note_id:{note_id}, err: {ex}"
                )
                return None

    async def batch_get_note_comments(self, note_detail_list: List[TiebaNote]):
        """
        Batch get note comments
        Args:
            note_detail_list:

        Returns:

        """
        if not config.ENABLE_GET_COMMENTS:
            return

        semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY_NUM)
        task_list: List[Task] = []
        for note_detail in note_detail_list:
            task = asyncio.create_task(
                self.get_comments_async_task(note_detail, semaphore),
                name=note_detail.note_id,
            )
            task_list.append(task)
        await asyncio.gather(*task_list)

    async def get_comments_async_task(
        self, note_detail: TiebaNote, semaphore: asyncio.Semaphore
    ):
        """
        Get comments async task
        Args:
            note_detail:
            semaphore:

        Returns:

        """
        async with semaphore:
            utils.logger.info(
                f"[BaiduTieBaCrawler.get_comments] Begin get note id comments {note_detail.note_id}"
            )
            await self.tieba_client.get_note_all_comments(
                note_detail=note_detail,
                crawl_interval=random.random(),
                callback=tieba_store.batch_update_tieba_note_comments,
                max_count=config.CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES,
            )

    async def get_creators_and_notes(self) -> None:
        """
        Get creator's information and their notes and comments
        Returns:

        """
        utils.logger.info(
            "[WeiboCrawler.get_creators_and_notes] Begin get weibo creators"
        )
        for creator_url in config.TIEBA_CREATOR_URL_LIST:
            creator_page_html_content = await self.tieba_client.get_creator_info_by_url(
                creator_url=creator_url
            )
            creator_info: TiebaCreator = self._page_extractor.extract_creator_info(
                creator_page_html_content
            )
            if creator_info:
                utils.logger.info(
                    f"[WeiboCrawler.get_creators_and_notes] creator info: {creator_info}"
                )
                if not creator_info:
                    raise Exception("Get creator info error")

                await tieba_store.save_creator(user_info=creator_info)

                # Get all note information of the creator
                all_notes_list = (
                    await self.tieba_client.get_all_notes_by_creator_user_name(
                        user_name=creator_info.user_name,
                        crawl_interval=0,
                        callback=tieba_store.batch_update_tieba_notes,
                        max_note_count=config.CRAWLER_MAX_NOTES_COUNT,
                        creator_page_html_content=creator_page_html_content,
                    )
                )

                await self.batch_get_note_comments(all_notes_list)

            else:
                utils.logger.error(
                    f"[WeiboCrawler.get_creators_and_notes] get creator info error, creator_url:{creator_url}"
                )

    async def launch_browser(
        self,
        chromium: BrowserType,
        playwright_proxy: Optional[Dict],
        user_agent: Optional[str],
        headless: bool = True,
    ) -> BrowserContext:
        """
        Launch browser and create browser
        Args:
            chromium:
            playwright_proxy:
            user_agent:
            headless:

        Returns:

        """
        utils.logger.info(
            "[BaiduTieBaCrawler.launch_browser] Begin create browser context ..."
        )
        if config.SAVE_LOGIN_STATE:
            # feat issue #14
            # we will save login state to avoid login every time
            user_data_dir = os.path.join(
                os.getcwd(), "browser_data", config.USER_DATA_DIR % config.PLATFORM
            )  # type: ignore
            browser_context = await chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                accept_downloads=True,
                headless=headless,
                proxy=playwright_proxy,  # type: ignore
                viewport={"width": 1920, "height": 1080},
                user_agent=user_agent,
            )
            return browser_context
        else:
            browser = await chromium.launch(headless=headless, proxy=playwright_proxy)  # type: ignore
            browser_context = await browser.new_context(
                viewport={"width": 1920, "height": 1080}, user_agent=user_agent
            )
            return browser_context

    async def launch_browser_with_cdp(
        self,
        playwright: Playwright,
        playwright_proxy: Optional[Dict],
        user_agent: Optional[str],
        headless: bool = True,
    ) -> BrowserContext:
        """
        使用CDP模式启动浏览器
        """
        try:
            self.cdp_manager = CDPBrowserManager()
            browser_context = await self.cdp_manager.launch_and_connect(
                playwright=playwright,
                playwright_proxy=playwright_proxy,
                user_agent=user_agent,
                headless=headless,
            )

            # 显示浏览器信息
            browser_info = await self.cdp_manager.get_browser_info()
            utils.logger.info(f"[TieBaCrawler] CDP浏览器信息: {browser_info}")

            return browser_context

        except Exception as e:
            utils.logger.error(f"[TieBaCrawler] CDP模式启动失败，回退到标准模式: {e}")
            # 回退到标准模式
            chromium = playwright.chromium
            return await self.launch_browser(
                chromium, playwright_proxy, user_agent, headless
            )

    async def close(self):
        """
        Close browser context
        Returns:

        """
        # 如果使用CDP模式，需要特殊处理
        if self.cdp_manager:
            await self.cdp_manager.cleanup()
            self.cdp_manager = None
        else:
            await self.browser_context.close()
        utils.logger.info("[BaiduTieBaCrawler.close] Browser context closed ...")

    async def search_trending(self) -> None:
        """Search trending keywords and retrieve related posts."""
        utils.logger.info("[BaiduTieBaCrawler.search_trending] Begin search tieba trending keywords")

        try:
            # Get trending keywords
            trending_res = await self.tieba_client.get_trending_keywords()
            utils.logger.info(f"[BaiduTieBaCrawler.search_trending] Trending response: {trending_res}")

            # 由于贴吧热搜API较复杂，这里使用一些常见的热门关键词作为示例
            all_trending_keywords = ["热点", "新闻", "娱乐", "科技", "游戏", "体育", "生活", "美食", "旅游", "时尚", "教育", "健康", "汽车", "房产", "财经", "军事", "历史", "文化", "艺术", "音乐"]
            # 如果TRENDING_KEYWORDS_COUNT为0或None，获取所有热搜；否则限制数量
            if config.TRENDING_KEYWORDS_COUNT and config.TRENDING_KEYWORDS_COUNT > 0:
                trending_keywords = all_trending_keywords[:config.TRENDING_KEYWORDS_COUNT]
            else:
                trending_keywords = all_trending_keywords

            if not trending_keywords:
                utils.logger.warning("[BaiduTieBaCrawler.search_trending] No trending keywords found")
                return

            utils.logger.info(f"[BaiduTieBaCrawler.search_trending] Found {len(trending_keywords)} trending keywords: {trending_keywords}")

            # Search posts for each trending keyword
            if config.ENABLE_TRENDING_KEYWORDS_CRAWL:
                for keyword in trending_keywords:
                    source_keyword_var.set(keyword)
                    utils.logger.info(f"[BaiduTieBaCrawler.search_trending] Searching trending keyword: {keyword}")
                    await self.search_keyword_posts(keyword, max_posts=config.TRENDING_KEYWORD_MAX_NOTES_COUNT)

        except Exception as e:
            utils.logger.error(f"[BaiduTieBaCrawler.search_trending] Error getting trending keywords: {e}")

    async def search_keyword_posts(self, keyword: str, max_posts: int = 50) -> None:
        """Search posts for a specific keyword with limited count."""
        tieba_limit_count = 10  # tieba limit page fixed value
        if max_posts < tieba_limit_count:
            max_posts = tieba_limit_count

        page = 1
        crawled_count = 0

        while crawled_count < max_posts:
            try:
                utils.logger.info(f"[BaiduTieBaCrawler.search_keyword_posts] search tieba keyword: {keyword}, page: {page}")
                notes_res = await self.tieba_client.get_notes_by_keyword(
                    keyword=keyword,
                    page=page,
                    page_size=tieba_limit_count,
                )

                if not notes_res:
                    utils.logger.info(f"[BaiduTieBaCrawler.search_keyword_posts] No more content for keyword: {keyword}")
                    break

                note_ids = []
                for note_item in notes_res:
                    if note_item:
                        note_ids.append(note_item.note_id)
                        await tieba_store.update_tieba_note(note_item)

                if not note_ids:
                    utils.logger.info(f"[BaiduTieBaCrawler.search_keyword_posts] No more content for keyword: {keyword}")
                    break

                crawled_count += len(note_ids)
                utils.logger.info(f"[BaiduTieBaCrawler.search_keyword_posts] keyword: {keyword}, crawled: {crawled_count}/{max_posts}")

                await self.batch_get_note_comments(note_ids)
                page += 1

                if crawled_count >= max_posts:
                    break

            except Exception as e:
                utils.logger.error(f"[BaiduTieBaCrawler.search_keyword_posts] Error searching keyword {keyword}: {e}")
                break
