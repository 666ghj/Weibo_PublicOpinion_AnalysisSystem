-- MindSpider AI爬虫项目 - 数据库表结构
-- 基于MediaCrawler表结构扩展，添加BroadTopicExtraction模块所需表

-- ===============================
-- BroadTopicExtraction 模块表结构
-- ===============================

-- ----------------------------
-- Table structure for daily_news
-- 每日新闻表：存储get_today_news.py获取的热点新闻
-- ----------------------------
DROP TABLE IF EXISTS `daily_news`;
CREATE TABLE `daily_news` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `news_id` varchar(128) NOT NULL COMMENT '新闻唯一ID',
    `source_platform` varchar(32) NOT NULL COMMENT '新闻源平台(weibo|zhihu|bilibili|toutiao|douyin等)',
    `title` varchar(500) NOT NULL COMMENT '新闻标题',
    `url` varchar(512) DEFAULT NULL COMMENT '新闻链接',
    `description` text COMMENT '新闻描述或摘要',
    `extra_info` text COMMENT '额外信息(JSON格式存储)',
    `crawl_date` date NOT NULL COMMENT '爬取日期',
    `rank_position` int DEFAULT NULL COMMENT '在热榜中的排名位置',
    `add_ts` bigint NOT NULL COMMENT '记录添加时间戳',
    `last_modify_ts` bigint NOT NULL COMMENT '记录最后修改时间戳',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_daily_news_unique` (`news_id`, `source_platform`, `crawl_date`),
    KEY `idx_daily_news_date` (`crawl_date`),
    KEY `idx_daily_news_platform` (`source_platform`),
    KEY `idx_daily_news_rank` (`rank_position`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='每日热点新闻表';

-- ----------------------------
-- Table structure for daily_topics
-- 每日话题表：存储TopicGPT提取的话题信息
-- ----------------------------
DROP TABLE IF EXISTS `daily_topics`;
CREATE TABLE `daily_topics` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `topic_id` varchar(64) NOT NULL COMMENT '话题唯一ID',
    `topic_name` varchar(255) NOT NULL COMMENT '话题名称',
    `topic_description` text COMMENT '话题描述',
    `keywords` text COMMENT '话题关键词(JSON格式存储)',
    `extract_date` date NOT NULL COMMENT '话题提取日期',
    `relevance_score` float DEFAULT NULL COMMENT '话题相关性得分',
    `news_count` int DEFAULT 0 COMMENT '关联的新闻数量',
    `processing_status` varchar(16) DEFAULT 'pending' COMMENT '处理状态(pending|processing|completed|failed)',
    `add_ts` bigint NOT NULL COMMENT '记录添加时间戳',
    `last_modify_ts` bigint NOT NULL COMMENT '记录最后修改时间戳',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_daily_topics_unique` (`topic_id`, `extract_date`),
    KEY `idx_daily_topics_date` (`extract_date`),
    KEY `idx_daily_topics_status` (`processing_status`),
    KEY `idx_daily_topics_score` (`relevance_score`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='每日提取话题表';

-- ----------------------------
-- Table structure for topic_news_relation
-- 话题新闻关联表：记录话题和新闻的关联关系
-- ----------------------------
DROP TABLE IF EXISTS `topic_news_relation`;
CREATE TABLE `topic_news_relation` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `topic_id` varchar(64) NOT NULL COMMENT '话题ID',
    `news_id` varchar(128) NOT NULL COMMENT '新闻ID',
    `relation_score` float DEFAULT NULL COMMENT '关联度得分',
    `extract_date` date NOT NULL COMMENT '关联提取日期',
    `add_ts` bigint NOT NULL COMMENT '记录添加时间戳',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_topic_news_unique` (`topic_id`, `news_id`, `extract_date`),
    KEY `idx_topic_news_topic` (`topic_id`),
    KEY `idx_topic_news_news` (`news_id`),
    KEY `idx_topic_news_date` (`extract_date`),
    FOREIGN KEY (`topic_id`) REFERENCES `daily_topics`(`topic_id`) ON DELETE CASCADE,
    FOREIGN KEY (`news_id`) REFERENCES `daily_news`(`news_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='话题新闻关联表';

-- ----------------------------
-- Table structure for crawling_tasks
-- 爬取任务表：记录基于话题的平台爬取任务
-- ----------------------------
DROP TABLE IF EXISTS `crawling_tasks`;
CREATE TABLE `crawling_tasks` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `task_id` varchar(64) NOT NULL COMMENT '任务唯一ID',
    `topic_id` varchar(64) NOT NULL COMMENT '关联的话题ID',
    `platform` varchar(32) NOT NULL COMMENT '目标平台(xhs|dy|ks|bili|wb|tieba|zhihu)',
    `search_keywords` text NOT NULL COMMENT '搜索关键词(JSON格式存储)',
    `task_status` varchar(16) DEFAULT 'pending' COMMENT '任务状态(pending|running|completed|failed|paused)',
    `start_time` bigint DEFAULT NULL COMMENT '任务开始时间戳',
    `end_time` bigint DEFAULT NULL COMMENT '任务结束时间戳',
    `total_crawled` int DEFAULT 0 COMMENT '已爬取内容数量',
    `success_count` int DEFAULT 0 COMMENT '成功爬取数量',
    `error_count` int DEFAULT 0 COMMENT '错误数量',
    `error_message` text COMMENT '错误信息',
    `config_params` text COMMENT '爬取配置参数(JSON格式)',
    `scheduled_date` date NOT NULL COMMENT '计划执行日期',
    `add_ts` bigint NOT NULL COMMENT '记录添加时间戳',
    `last_modify_ts` bigint NOT NULL COMMENT '记录最后修改时间戳',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_crawling_tasks_unique` (`task_id`),
    KEY `idx_crawling_tasks_topic` (`topic_id`),
    KEY `idx_crawling_tasks_platform` (`platform`),
    KEY `idx_crawling_tasks_status` (`task_status`),
    KEY `idx_crawling_tasks_date` (`scheduled_date`),
    FOREIGN KEY (`topic_id`) REFERENCES `daily_topics`(`topic_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='爬取任务表';

-- ===============================
-- MediaCrawler表结构扩展字段
-- ===============================

-- 为MediaCrawler现有表添加话题关联字段，支持MindSpider功能
-- 注意：这些字段是可选的，不影响MediaCrawler原有功能

-- 为小红书笔记表添加话题关联字段
ALTER TABLE `xhs_note` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '关联的话题ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '关联的爬取任务ID';

-- 为抖音视频表添加话题关联字段
ALTER TABLE `douyin_aweme` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '关联的话题ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '关联的爬取任务ID';

-- 为快手视频表添加话题关联字段
ALTER TABLE `kuaishou_video` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '关联的话题ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '关联的爬取任务ID';

-- 为B站视频表添加话题关联字段
ALTER TABLE `bilibili_video` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '关联的话题ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '关联的爬取任务ID';

-- 为微博帖子表添加话题关联字段
ALTER TABLE `weibo_note` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '关联的话题ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '关联的爬取任务ID';

-- 为贴吧帖子表添加话题关联字段
ALTER TABLE `tieba_note` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '关联的话题ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '关联的爬取任务ID';

-- 为知乎内容表添加话题关联字段
ALTER TABLE `zhihu_content` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '关联的话题ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '关联的爬取任务ID';

-- ===============================
-- 创建视图用于数据分析
-- ===============================

-- 话题爬取统计视图
CREATE OR REPLACE VIEW `v_topic_crawling_stats` AS
SELECT 
    dt.topic_id,
    dt.topic_name,
    dt.extract_date,
    dt.processing_status,
    COUNT(DISTINCT ct.task_id) as total_tasks,
    SUM(CASE WHEN ct.task_status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
    SUM(CASE WHEN ct.task_status = 'failed' THEN 1 ELSE 0 END) as failed_tasks,
    SUM(ct.total_crawled) as total_content_crawled,
    SUM(ct.success_count) as total_success_count,
    SUM(ct.error_count) as total_error_count
FROM daily_topics dt
LEFT JOIN crawling_tasks ct ON dt.topic_id = ct.topic_id
GROUP BY dt.topic_id, dt.topic_name, dt.extract_date, dt.processing_status;

-- 每日数据统计视图
CREATE OR REPLACE VIEW `v_daily_summary` AS
SELECT 
    crawl_date,
    COUNT(DISTINCT news_id) as total_news,
    COUNT(DISTINCT source_platform) as platforms_covered,
    (SELECT COUNT(*) FROM daily_topics WHERE extract_date = dn.crawl_date) as topics_extracted,
    (SELECT COUNT(*) FROM crawling_tasks WHERE scheduled_date = dn.crawl_date) as tasks_created
FROM daily_news dn
GROUP BY crawl_date
ORDER BY crawl_date DESC;

-- ===============================
-- 初始化索引优化
-- ===============================

-- 为关联查询优化添加复合索引
CREATE INDEX `idx_topic_date_status` ON `daily_topics` (`extract_date`, `processing_status`);
CREATE INDEX `idx_task_topic_platform` ON `crawling_tasks` (`topic_id`, `platform`, `task_status`);
CREATE INDEX `idx_news_date_platform` ON `daily_news` (`crawl_date`, `source_platform`);

-- ===============================
-- 数据库配置优化建议
-- ===============================

-- 设置合适的字符集和排序规则
-- ALTER DATABASE mindspider CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 建议的数据保留策略（可选）
-- 可以根据需要创建事件调度器来清理历史数据
-- 例如：删除90天前的新闻数据，保留话题和爬取结果数据
