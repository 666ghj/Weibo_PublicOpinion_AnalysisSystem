// 多语言支持文件
const translations = {
    'zh': {
        // 导航菜单
        'home': '首页',
        'hotWord': '热词统计',
        'tableData': '微博舆情统计',
        'articleChar': '文章分析',
        'ipChar': 'IP分析',
        'commentChar': '评论分析',
        'yuqingChar': '舆情分析',
        'yuqingpredict': '舆情预测',
        'articleCloud': '文章内容词云图',
        'dataVisualization': '数据可视化',
        'weiboSystem': '微博舆情分析系统',
        'wordCloud': '词云图',
        
        // 首页
        'articleCount': '文章个数',
        'articleCrawlRule': '文章爬取规则',
        'nextCrawlTime': '下次爬取时间',
        'articlePublishTimeCount': '文章发布时间个数',
        'commentLikeCountTopFore': '评论点赞量 Top Fore',
        'viewAll': '查看全部',
        'articleTypeRatio': '文章类型占比',
        'commentUserWordCloud': '评论用户名词云图',
        'commentUserTimeRatio': '评论用户时间占比',
        
        // 热词页面
        'hotWordStatistics': '热词统计页',
        'hotWordCloud': '热词词云图',
        'hotWordRanking': '热词查询表格',
        'wordFrequency': '词频',
        'hotWordSelection': '热词选择',
        'hotWordName': '热词名称',
        'occurrenceCount': '出现次数',
        'hotWordSentiment': '热词情感',
        'hotWordYearTrend': '热词年份变化趋势',
        'queryCommentsByHotWord': '根据选择的热词从而查询出评论数据',
        'hotWordTimeDistribution': '热词出现时间分布个数',
        
        // 舆情分析页面
        'hotWordSentimentTrendBar': '热词情感趋势柱状图',
        'hotWordSentimentTrendTree': '热词情感趋势树形图',
        'articleCommentSentimentTrendPie': '文章内容与评论内容舆情趋势饼状图',
        
        // 舆情预测页面
        'topicStatisticsPage': '话题统计页',
        
        // 文章分析页面
        'articleCharPage': '文章分析页',
        'typeSelection': '类型选择',
        'articleLikeAnalysis': '文章点赞量分析 👍',
        'articleCommentAnalysis': '文章评论量分析 🔥',
        'articleForwardAnalysis': '文章转发量分析 🥇',
        'likeRangeStatistics': '点赞区间统计',
        'rangeCount': '区间个数',
        
        // 评论分析页面
        'commentLikeRangeChart': '评论点赞次数区间图',
        'commentUserGenderRatio': '评论用户性别占比',
        'userCommentWordCloud': '用户评论词云图',
        
        // IP分析页面
        'articleIpLocationAnalysis': '文章IP位置分析图',
        'commentIpLocationAnalysis': '评论IP位置分析图',
        
        // 评论相关
        'commentUser': '评论用户',
        'commentGender': '评论性别',
        'commentAddress': '评论地址',
        'commentContent': '评论内容',
        'likeCount': '点赞量',
        
        // 微博舆情统计页面
        'weiboArticleStatTable': '微博文章统计表格 - 舆情 情感分类',
        'sentimentClassification': '情感分类',
        'articleId': '文章ID',
        'articleIp': '文章IP',
        'articleTitle': '文章标题',
        'articleLike': '点赞量',
        'articleForward': '转发量',
        'articleComment': '评论量',
        'articleType': '类型',
        'articleContent': '内容',
        'articleTime': '发布时间',
        
        // 通用
        'switchToEnglish': '切换到英文',
        'switchToChinese': '切换到中文',
        'semester': '网安小学期',
        
        // 错误页面
        'pageNotFound': '页面未找到',
        'backToHome': '返回首页',
        'serverError': '服务器错误',
        'forbidden': '禁止访问',
        'badRequest': '错误请求'
    },
    'en': {
        // Navigation menu
        'home': 'Home',
        'hotWord': 'Hot Words',
        'tableData': 'Weibo Public Opinion Stats',
        'articleChar': 'Article Analysis',
        'ipChar': 'IP Analysis',
        'commentChar': 'Comment Analysis',
        'yuqingChar': 'Public Opinion Analysis',
        'yuqingpredict': 'Opinion Prediction',
        'articleCloud': 'Article Content Word Cloud',
        'dataVisualization': 'Data Visualization',
        'weiboSystem': 'Weibo Public Opinion Analysis System',
        'wordCloud': 'Word Cloud',
        
        // Home page
        'articleCount': 'Article Count',
        'articleCrawlRule': 'Article Crawl Rule',
        'nextCrawlTime': 'Next Crawl Time',
        'articlePublishTimeCount': 'Article Publish Time Count',
        'commentLikeCountTopFore': 'Comment Like Count Top Four',
        'viewAll': 'View All',
        'articleTypeRatio': 'Article Type Ratio',
        'commentUserWordCloud': 'Comment User Word Cloud',
        'commentUserTimeRatio': 'Comment User Time Ratio',
        
        // Hot word page
        'hotWordStatistics': 'Hot Word Statistics',
        'hotWordCloud': 'Hot Word Cloud',
        'hotWordRanking': 'Hot Word Ranking',
        'wordFrequency': 'Word Frequency',
        'hotWordSelection': 'Hot Word Selection',
        'hotWordName': 'Hot Word Name',
        'occurrenceCount': 'Occurrence Count',
        'hotWordSentiment': 'Hot Word Sentiment',
        'hotWordYearTrend': 'Hot Word Year Trend',
        'queryCommentsByHotWord': 'Query comments based on selected hot word',
        'hotWordTimeDistribution': 'Hot Word Time Distribution Count',
        
        // Public opinion analysis page
        'hotWordSentimentTrendBar': 'Hot Word Sentiment Trend Bar Chart',
        'hotWordSentimentTrendTree': 'Hot Word Sentiment Trend Tree Chart',
        'articleCommentSentimentTrendPie': 'Article and Comment Sentiment Trend Pie Chart',
        
        // Opinion prediction page
        'topicStatisticsPage': 'Topic Statistics Page',
        
        // Article analysis page
        'articleCharPage': 'Article Analysis Page',
        'typeSelection': 'Type Selection',
        'articleLikeAnalysis': 'Article Like Analysis 👍',
        'articleCommentAnalysis': 'Article Comment Analysis 🔥',
        'articleForwardAnalysis': 'Article Forward Analysis 🥇',
        'likeRangeStatistics': 'Like Range Statistics',
        'rangeCount': 'Range Count',
        
        // Comment analysis page
        'commentLikeRangeChart': 'Comment Like Range Chart',
        'commentUserGenderRatio': 'Comment User Gender Ratio',
        'userCommentWordCloud': 'User Comment Word Cloud',
        
        // IP analysis page
        'articleIpLocationAnalysis': 'Article IP Location Analysis',
        'commentIpLocationAnalysis': 'Comment IP Location Analysis',
        
        // Comment related
        'commentUser': 'Comment User',
        'commentGender': 'Gender',
        'commentAddress': 'Address',
        'commentContent': 'Content',
        'likeCount': 'Likes',
        
        // Weibo public opinion stats page
        'weiboArticleStatTable': 'Weibo Article Statistics Table - Sentiment Classification',
        'sentimentClassification': 'Sentiment Classification',
        'articleId': 'Article ID',
        'articleIp': 'Article IP',
        'articleTitle': 'Article Title',
        'articleLike': 'Likes',
        'articleForward': 'Forwards',
        'articleComment': 'Comments',
        'articleType': 'Type',
        'articleContent': 'Content',
        'articleTime': 'Publish Time',
        
        // Common
        'switchToEnglish': 'Switch to English',
        'switchToChinese': 'Switch to Chinese',
        'semester': 'Network Security Semester',
        
        // Error pages
        'pageNotFound': 'Page Not Found',
        'backToHome': 'Back to Home',
        'serverError': 'Server Error',
        'forbidden': 'Forbidden',
        'badRequest': 'Bad Request'
    }
};

// 获取当前语言
function getCurrentLanguage() {
    return localStorage.getItem('language') || 'zh';
}

// 设置语言
function setLanguage(lang) {
    localStorage.setItem('language', lang);
    location.reload();
}

// 翻译函数
function t(key) {
    const lang = getCurrentLanguage();
    return translations[lang][key] || key;
}

// 页面加载时应用翻译
document.addEventListener('DOMContentLoaded', function() {
    // 应用当前语言
    applyTranslations();
    
    // 添加语言切换按钮事件
    const langSwitcher = document.getElementById('language-switcher');
    if (langSwitcher) {
        langSwitcher.addEventListener('click', function() {
            const currentLang = getCurrentLanguage();
            const newLang = currentLang === 'zh' ? 'en' : 'zh';
            setLanguage(newLang);
        });
    }
});

// 应用翻译到页面元素
function applyTranslations() {
    const elements = document.querySelectorAll('[data-i18n]');
    elements.forEach(el => {
        const key = el.getAttribute('data-i18n');
        el.textContent = t(key);
    });
} 