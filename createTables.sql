SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- article表
-- ----------------------------
CREATE TABLE `article` (
  `id` bigint(20) DEFAULT NULL,
  `likeNum` bigint(20) DEFAULT NULL,
  `commentsLen` bigint(20) DEFAULT NULL,
  `reposts_count` bigint(20) DEFAULT NULL,
  `region` text,
  `content` text,
  `contentLen` bigint(20) DEFAULT NULL,
  `created_at` text,
  `type` text,
  `detailUrl` text,
  `authorAvatar` text,
  `authorName` text,
  `authorDetail` text,
  `isVip` double DEFAULT NULL,
  `topic` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- comments表
-- ----------------------------
CREATE TABLE `comments` (
  `articleId` bigint(20) DEFAULT NULL,
  `created_at` text,
  `likes_counts` bigint(20) DEFAULT NULL,
  `region` text,
  `content` text,
  `authorName` text,
  `authorGender` text,
  `authorAddress` text,
  `authorAvatar` text,
  `topic` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- user表
-- ----------------------------
CREATE TABLE `user` (
  `username` varchar(255) DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL,
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `createTime` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8;

-- 爬虫模板表
CREATE TABLE IF NOT EXISTS `crawler_templates` (
  `id` VARCHAR(64) NOT NULL COMMENT '模板ID',
  `name` VARCHAR(64) NOT NULL COMMENT '模板名称',
  `description` VARCHAR(255) NULL COMMENT '模板描述',
  `icon` VARCHAR(32) NULL COMMENT '图标',
  `config` JSON NOT NULL COMMENT '配置JSON',
  `created_at` DATETIME NOT NULL COMMENT '创建时间',
  `updated_at` DATETIME NOT NULL COMMENT '更新时间',
  `deleted` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否删除',
  PRIMARY KEY (`id`),
  INDEX `idx_crawler_templates_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='爬虫配置模板表';

-- 分析流程模板表
CREATE TABLE IF NOT EXISTS `analysis_templates` (
  `id` VARCHAR(64) NOT NULL COMMENT '模板ID',
  `name` VARCHAR(64) NOT NULL COMMENT '模板名称',
  `description` VARCHAR(255) NULL COMMENT '模板描述',
  `icon` VARCHAR(32) NULL COMMENT '图标',
  `components` JSON NOT NULL COMMENT '组件JSON',
  `connections` JSON NOT NULL COMMENT '连接JSON',
  `created_at` DATETIME NOT NULL COMMENT '创建时间',
  `updated_at` DATETIME NOT NULL COMMENT '更新时间',
  `deleted` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否删除',
  PRIMARY KEY (`id`),
  INDEX `idx_analysis_templates_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分析流程模板表';

-- 工作流执行任务表
CREATE TABLE IF NOT EXISTS `workflow_tasks` (
  `id` VARCHAR(64) NOT NULL COMMENT '任务ID',
  `template_id` VARCHAR(64) NULL COMMENT '关联模板ID',
  `type` VARCHAR(32) NOT NULL COMMENT '任务类型：crawler/analysis',
  `status` VARCHAR(16) NOT NULL COMMENT '任务状态：pending/running/completed/failed',
  `progress` INT(11) NOT NULL DEFAULT 0 COMMENT '进度百分比',
  `config` JSON NOT NULL COMMENT '任务配置',
  `result` JSON NULL COMMENT '执行结果',
  `error` TEXT NULL COMMENT '错误信息',
  `started_at` DATETIME NULL COMMENT '开始时间',
  `completed_at` DATETIME NULL COMMENT '完成时间',
  `created_at` DATETIME NOT NULL COMMENT '创建时间',
  `updated_at` DATETIME NOT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`),
  INDEX `idx_workflow_tasks_type_status` (`type`, `status`),
  INDEX `idx_workflow_tasks_template` (`template_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工作流执行任务表';