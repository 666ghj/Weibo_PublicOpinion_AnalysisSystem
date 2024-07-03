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