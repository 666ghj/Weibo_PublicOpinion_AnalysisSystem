// 爬虫控制面板翻译资源文件
// 包含中文(zh-CN)和英文(en-US)的翻译

const spiderI18nResources = {
  'zh-CN': {
    translation: {
      // 页面标题和导航
      'page-title': '爬虫控制面板',
      
      // 卡片标题
      'topic-selection': '选择话题类型',
      'spider-parameters': '爬虫参数配置',
      'content-filters': '内容筛选配置',
      'account-config': '账号配置',
      'parallel-config': '并行配置',
      'db-config': '数据库配置',
      'ai-assistant': 'AI配置助手',
      'spider-status': '爬虫状态',
      
      // 话题选择部分
      'add-custom-topic': '添加自定义话题',
      'custom-topic-placeholder': '输入自定义话题',
      'btn-add': '添加',
      'selected-topics': '已选择的话题：',
      
      // 爬虫参数部分
      'crawl-depth': '爬取深度',
      'crawl-depth-hint': '每个话题爬取的页数（1-10）',
      'interval': '爬取间隔(秒)',
      'interval-hint': '每次请求之间的间隔时间',
      'max-retries': '最大重试次数',
      'timeout': '请求超时时间(秒)',
      
      // 筛选器部分
      'help': '帮助',
      'filter-conditions': '筛选条件说明：',
      'filter-condition-1': '数值条件：设置大于某个值进行筛选，如点赞数>1000',
      'filter-condition-2': '正则匹配：使用正则表达式匹配内容，如包含特定关键词',
      'filter-condition-3': '多个条件之间是"与"的关系，即同时满足才会保留',
      'filter-tip': '提示：合理设置筛选条件可以提高数据质量',
      'interaction-filter': '互动数据筛选',
      'likes-gt': '点赞数大于',
      'comments-gt': '评论数大于',
      'reposts-gt': '转发数大于',
      'reads-gt': '阅读数大于',
      'regex-filter': '内容正则筛选',
      'add-regex-filter': '添加正则筛选',
      'advanced-options': '高级选项',
      'only-original': '仅爬取原创内容',
      'must-have-media': '必须包含图片或视频',
      'only-verified': '仅认证用户的内容',
      
      // 账号配置部分
      'btn-add-account': '添加账号',
      'how-to-get-cookie': '如何获取Cookie？',
      'cookie-step-1': '登录微博网页版',
      'cookie-step-2': '按F12打开开发者工具',
      'cookie-step-3': '切换到Network标签页',
      'cookie-step-4': '刷新页面，找到请求头中的Cookie值',
      'cookie-warning': '注意：请勿泄露您的Cookie信息！',
      'account-tip': '提示：添加多个账号可以提高爬取效率，系统会自动在账号间轮换。',
      'no-account-warning': '请至少添加一个账号',
      'username': '用户名',
      'username-placeholder': '微博用户名',
      'password': '密码',
      'password-placeholder': '微博密码',
      'cookie': 'Cookie',
      'cookie-placeholder': '请输入微博Cookie',
      'save-cookie': '保存Cookie（加密存储）',
      'status-pending': '状态：待验证',
      'btn-validate-account': '验证账号',
      'status-validating': '状态：验证中...',
      'status-success': '状态：验证成功',
      'status-failed': '状态：验证失败 - ',
      'error-empty-cookie': 'Cookie不能为空',
      
      // 正则筛选器
      'regex-pattern': '正则表达式',
      'regex-pattern-placeholder': '输入正则表达式',
      'match-target': '匹配目标',
      'target-content': '微博内容',
      'target-author': '作者名',
      'target-location': '发布位置',
      'inverse-match': '反向匹配（不包含匹配项）',
      
      // 并行配置
      'max-concurrent': '最大并行数',
      'max-concurrent-hint': '同时进行爬取的最大话题数（1-5）',
      'requests-per-minute': '每分钟请求数限制',
      'requests-per-minute-hint': '避免请求过于频繁（30-120）',
      
      // 数据库配置
      'db-type': '数据库类型',
      'host': '主机地址',
      'port': '端口',
      'db-name': '数据库名',
      'username-db': '用户名',
      'password-db': '密码',
      'btn-test-connection': '测试连接',
      'db-connect-success': '数据库连接测试成功！',
      'db-connect-fail': '数据库连接测试失败：',
      'db-connect-error': '测试连接时发生错误：',
      
      // AI配置助手
      'ai-prompt-label': '用自然语言描述您的爬虫需求',
      'ai-prompt-placeholder': '例如：我想爬取最近一周关于人工智能的热门微博，重点关注转发量超过1000的内容，每个话题爬取前5页内容。',
      'btn-generate-config': '生成配置',
      'auto-apply': '自动应用生成的配置',
      'ai-suggestion': 'AI助手建议：',
      'ai-config-applied': 'AI配置已自动应用',
      'ai-config-error': '生成配置时出错：',
      'empty-prompt-error': '请输入您的爬虫需求描述！',
      
      // 操作按钮
      'btn-start': '开始爬取',
      'btn-save-config': '保存配置',
      'config-saved': '配置已保存！',
      'save-failed': '保存失败：',
      'save-error': '保存出错：',
      
      // 爬虫状态
      'task-started': '爬虫任务已启动...',
      'start-failed': '启动失败：',
      'error': '错误：',
      
      // 验证错误提示
      'select-topic-error': '请至少选择一个话题！',
      'invalid-regex-error': '正则表达式 "{0}" 格式无效！',
      'need-account-error': '请至少添加一个账号！',
      'empty-cookie-error': '存在未配置Cookie的账号，请检查！',
      'concurrent-limit-error': '最大并行数必须在1-5之间！',
      'request-limit-error': '每分钟请求数必须在30-120之间！',
      'db-config-error': '请完整填写数据库配置信息！'
    }
  },
  'en-US': {
    translation: {
      // Page title and navigation
      'page-title': 'Spider Control Panel',
      
      // Card titles
      'topic-selection': 'Select Topic Types',
      'spider-parameters': 'Spider Parameters',
      'content-filters': 'Content Filters',
      'account-config': 'Account Configuration',
      'parallel-config': 'Parallel Configuration',
      'db-config': 'Database Configuration',
      'ai-assistant': 'AI Configuration Assistant',
      'spider-status': 'Spider Status',
      
      // Topic selection section
      'add-custom-topic': 'Add Custom Topic',
      'custom-topic-placeholder': 'Enter custom topic',
      'btn-add': 'Add',
      'selected-topics': 'Selected Topics:',
      
      // Spider parameters section
      'crawl-depth': 'Crawl Depth',
      'crawl-depth-hint': 'Number of pages to crawl for each topic (1-10)',
      'interval': 'Interval (seconds)',
      'interval-hint': 'Time between requests',
      'max-retries': 'Maximum Retries',
      'timeout': 'Request Timeout (seconds)',
      
      // Filters section
      'help': 'Help',
      'filter-conditions': 'Filter conditions:',
      'filter-condition-1': 'Numeric conditions: Set values to filter by, e.g., likes > 1000',
      'filter-condition-2': 'Regex matching: Use regular expressions to match content, e.g., contain specific keywords',
      'filter-condition-3': 'Multiple conditions are combined with AND logic',
      'filter-tip': 'Tip: Setting proper filters can improve data quality',
      'interaction-filter': 'Interaction Data Filters',
      'likes-gt': 'Likes greater than',
      'comments-gt': 'Comments greater than',
      'reposts-gt': 'Reposts greater than',
      'reads-gt': 'Reads greater than',
      'regex-filter': 'Content Regex Filters',
      'add-regex-filter': 'Add Regex Filter',
      'advanced-options': 'Advanced Options',
      'only-original': 'Only crawl original content',
      'must-have-media': 'Must contain images or videos',
      'only-verified': 'Only content from verified users',
      
      // Account configuration section
      'btn-add-account': 'Add Account',
      'how-to-get-cookie': 'How to get the Cookie?',
      'cookie-step-1': 'Login to Weibo web version',
      'cookie-step-2': 'Press F12 to open developer tools',
      'cookie-step-3': 'Switch to Network tab',
      'cookie-step-4': 'Refresh page and find Cookie value in request headers',
      'cookie-warning': 'Warning: Do not expose your Cookie information!',
      'account-tip': 'Tip: Adding multiple accounts can improve crawling efficiency, the system will automatically rotate between accounts.',
      'no-account-warning': 'Please add at least one account',
      'username': 'Username',
      'username-placeholder': 'Weibo username',
      'password': 'Password',
      'password-placeholder': 'Weibo password',
      'cookie': 'Cookie',
      'cookie-placeholder': 'Please enter Weibo Cookie',
      'save-cookie': 'Save Cookie (encrypted storage)',
      'status-pending': 'Status: Pending verification',
      'btn-validate-account': 'Validate Account',
      'status-validating': 'Status: Validating...',
      'status-success': 'Status: Validation successful',
      'status-failed': 'Status: Validation failed - ',
      'error-empty-cookie': 'Cookie cannot be empty',
      
      // Regex filters
      'regex-pattern': 'Regular Expression',
      'regex-pattern-placeholder': 'Enter regular expression',
      'match-target': 'Match Target',
      'target-content': 'Weibo content',
      'target-author': 'Author name',
      'target-location': 'Posting location',
      'inverse-match': 'Inverse match (exclude matches)',
      
      // Parallel configuration
      'max-concurrent': 'Maximum Concurrent Tasks',
      'max-concurrent-hint': 'Maximum number of topics to crawl simultaneously (1-5)',
      'requests-per-minute': 'Requests Per Minute Limit',
      'requests-per-minute-hint': 'Avoid too frequent requests (30-120)',
      
      // Database configuration
      'db-type': 'Database Type',
      'host': 'Host',
      'port': 'Port',
      'db-name': 'Database Name',
      'username-db': 'Username',
      'password-db': 'Password',
      'btn-test-connection': 'Test Connection',
      'db-connect-success': 'Database connection test successful!',
      'db-connect-fail': 'Database connection test failed: ',
      'db-connect-error': 'Error while testing connection: ',
      
      // AI assistant
      'ai-prompt-label': 'Describe your crawling requirements in natural language',
      'ai-prompt-placeholder': 'For example: I want to crawl trending Weibo posts about AI from the past week, focusing on content with more than 1000 reposts, crawling the first 5 pages for each topic.',
      'btn-generate-config': 'Generate Configuration',
      'auto-apply': 'Auto-apply generated configuration',
      'ai-suggestion': 'AI Assistant Suggestion:',
      'ai-config-applied': 'AI configuration applied automatically',
      'ai-config-error': 'Error generating configuration: ',
      'empty-prompt-error': 'Please enter your crawler requirements!',
      
      // Action buttons
      'btn-start': 'Start Crawling',
      'btn-save-config': 'Save Configuration',
      'config-saved': 'Configuration saved!',
      'save-failed': 'Save failed: ',
      'save-error': 'Error saving: ',
      
      // Spider status
      'task-started': 'Crawler task started...',
      'start-failed': 'Start failed: ',
      'error': 'Error: ',
      
      // Validation error messages
      'select-topic-error': 'Please select at least one topic!',
      'invalid-regex-error': 'Regular expression "{0}" is invalid!',
      'need-account-error': 'Please add at least one account!',
      'empty-cookie-error': 'There are accounts without Cookie configuration, please check!',
      'concurrent-limit-error': 'Maximum concurrent tasks must be between 1-5!',
      'request-limit-error': 'Requests per minute must be between 30-120!',
      'db-config-error': 'Please complete all database configuration fields!'
    }
  }
}; 