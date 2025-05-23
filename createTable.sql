-- MySQL dump 10.13  Distrib 8.0.36, for macos14 (arm64)
--
-- Host: 127.0.0.1    Database: Weibo_PublicOpinion_AnalysisSystem
-- ------------------------------------------------------
-- Server version	8.0.36

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `analysis_templates`
--

DROP TABLE IF EXISTS `analysis_templates`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `analysis_templates` (
  `id` varchar(64) NOT NULL COMMENT '模板ID',
  `name` varchar(64) NOT NULL COMMENT '模板名称',
  `description` varchar(255) DEFAULT NULL COMMENT '模板描述',
  `icon` varchar(32) DEFAULT NULL COMMENT '图标',
  `components` json NOT NULL COMMENT '组件JSON',
  `connections` json NOT NULL COMMENT '连接JSON',
  `created_at` datetime NOT NULL COMMENT '创建时间',
  `updated_at` datetime NOT NULL COMMENT '更新时间',
  `deleted` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否删除',
  PRIMARY KEY (`id`),
  KEY `idx_analysis_templates_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='分析流程模板表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `article`
--

DROP TABLE IF EXISTS `article`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `article` (
  `id` bigint DEFAULT NULL,
  `likeNum` bigint DEFAULT NULL,
  `commentsLen` bigint DEFAULT NULL,
  `reposts_count` bigint DEFAULT NULL,
  `region` text,
  `content` text,
  `contentLen` bigint DEFAULT NULL,
  `created_at` text,
  `type` text,
  `detailUrl` text,
  `authorAvatar` text,
  `authorName` text,
  `authorDetail` text,
  `isVip` double DEFAULT NULL,
  `topic` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `comments`
--

DROP TABLE IF EXISTS `comments`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `comments` (
  `articleId` bigint DEFAULT NULL,
  `created_at` text,
  `likes_counts` bigint DEFAULT NULL,
  `region` text,
  `content` text,
  `authorName` text,
  `authorGender` text,
  `authorAddress` text,
  `authorAvatar` text,
  `topic` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `crawler_templates`
--

DROP TABLE IF EXISTS `crawler_templates`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `crawler_templates` (
  `id` varchar(64) NOT NULL COMMENT '模板ID',
  `name` varchar(64) NOT NULL COMMENT '模板名称',
  `description` varchar(255) DEFAULT NULL COMMENT '模板描述',
  `icon` varchar(32) DEFAULT NULL COMMENT '图标',
  `config` json NOT NULL COMMENT '配置JSON',
  `created_at` datetime NOT NULL COMMENT '创建时间',
  `updated_at` datetime NOT NULL COMMENT '更新时间',
  `deleted` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否删除',
  PRIMARY KEY (`id`),
  KEY `idx_crawler_templates_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='爬虫配置模板表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `login_history`
--

DROP TABLE IF EXISTS `login_history`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `login_history` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(255) NOT NULL,
  `login_time` datetime NOT NULL,
  `ip_address` varchar(50) DEFAULT NULL,
  `user_agent` text,
  `success` tinyint(1) DEFAULT '0',
  `attempt_count` int DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `idx_login_history_username` (`username`),
  KEY `idx_login_history_time` (`login_time`)
) ENGINE=InnoDB AUTO_INCREMENT=30 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户登录历史记录';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `logout_history`
--

DROP TABLE IF EXISTS `logout_history`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `logout_history` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(255) NOT NULL,
  `logout_time` datetime NOT NULL,
  `ip_address` varchar(50) DEFAULT NULL,
  `user_agent` text,
  `session_id` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_logout_history_username` (`username`),
  KEY `idx_logout_history_time` (`logout_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户登出历史记录';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `register_history`
--

DROP TABLE IF EXISTS `register_history`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `register_history` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(255) NOT NULL,
  `register_time` datetime NOT NULL,
  `ip_address` varchar(50) DEFAULT NULL,
  `user_agent` text,
  `email` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_register_history_username` (`username`),
  KEY `idx_register_history_time` (`register_time`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户注册历史记录';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `user`
--

DROP TABLE IF EXISTS `user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user` (
  `username` varchar(255) DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL,
  `id` int NOT NULL AUTO_INCREMENT,
  `createTime` varchar(255) DEFAULT NULL,
  `email` varchar(255) DEFAULT NULL,
  `status` varchar(20) DEFAULT 'active',
  `last_password_change` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `workflow_tasks`
--

DROP TABLE IF EXISTS `workflow_tasks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `workflow_tasks` (
  `id` varchar(64) NOT NULL COMMENT '任务ID',
  `template_id` varchar(64) DEFAULT NULL COMMENT '关联模板ID',
  `type` varchar(32) NOT NULL COMMENT '任务类型：crawler/analysis',
  `status` varchar(16) NOT NULL COMMENT '任务状态：pending/running/completed/failed',
  `progress` int NOT NULL DEFAULT '0' COMMENT '进度百分比',
  `config` json NOT NULL COMMENT '任务配置',
  `result` json DEFAULT NULL COMMENT '执行结果',
  `error` text COMMENT '错误信息',
  `started_at` datetime DEFAULT NULL COMMENT '开始时间',
  `completed_at` datetime DEFAULT NULL COMMENT '完成时间',
  `created_at` datetime NOT NULL COMMENT '创建时间',
  `updated_at` datetime NOT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_workflow_tasks_type_status` (`type`,`status`),
  KEY `idx_workflow_tasks_template` (`template_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='工作流执行任务表';
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-05-23 17:00:37
