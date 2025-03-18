// 翻译资源文件
// 包含中文(zh-CN)和英文(en-US)的翻译

const i18nResources = {
  'zh-CN': {
    translation: {
      // 页面标题
      'page-title': '工作流编辑器 - 微博舆情分析系统',
      'navbar-brand': '工作流编辑器',
      
      // 导航菜单
      'nav-visual-editor': '可视化编辑',
      'nav-template-mgmt': '模板管理',
      'nav-task-list': '任务列表',
      
      // 按钮
      'btn-save': '保存',
      'btn-run': '运行',
      'btn-cancel': '取消',
      'btn-close': '关闭',
      'btn-create-new': '新建',
      'btn-validate': '验证',
      'btn-undo': '撤销',
      'btn-redo': '重做',
      'btn-zoom-in': '放大',
      'btn-zoom-out': '缩小',
      'btn-fit-view': '适应视图',
      'btn-export': '导出工作流',
      'btn-import': '导入工作流',
      'btn-cancel-task': '取消任务',
      'btn-view-full-result': '查看完整结果',
      
      // 选项卡
      'tab-components': '组件',
      'tab-templates': '模板',
      
      // 组件类别
      'comp-data-source': '数据源',
      'comp-data-processing': '数据处理',
      'comp-model-analysis': '模型分析',
      'comp-visualization': '可视化',
      
      // 组件
      'comp-database': '数据库',
      'comp-file': '文件',
      'comp-crawler': '爬虫',
      'comp-filter': '过滤',
      'comp-sort': '排序',
      'comp-aggregate': '聚合',
      'comp-sentiment': '情感分析',
      'comp-topic': '话题分类',
      'comp-keywords': '关键词提取',
      'comp-summarize': '文本摘要',
      'comp-chart': '图表',
      'comp-table': '表格',
      'comp-wordcloud': '词云',
      
      // 模板相关
      'templates-crawler': '爬虫模板',
      'templates-analysis': '分析流程模板',
      'modal-save-template': '保存为模板',
      'template-name': '模板名称',
      'template-description': '描述',
      'template-icon': '图标',
      
      // 图标名称
      'icon-chart': '图表',
      'icon-filter': '过滤',
      'icon-crawler': '爬虫',
      'icon-ai': 'AI分析',
      'icon-database': '数据库',
      'icon-wordcloud': '词云',
      
      // 属性面板
      'properties-title': '组件属性',
      
      // 工作流状态
      'workflow-status-message': '工作流就绪。拖拽左侧组件到画布创建节点。',
      'nodes': '节点',
      'connections': '连接',
      
      // 运行工作流
      'modal-run-workflow': '运行工作流',
      'run-workflow-confirm': '确认要运行当前工作流吗？',
      'save-before-run': '运行前保存工作流',
      
      // 任务状态
      'modal-task-status': '任务执行状态',
      'task-progress': '进度',
      'task-status-info': '状态信息',
      'task-waiting': '等待中',
      'task-id': '任务ID:',
      'task-status': '状态:',
      'task-start-time': '开始时间:',
      'task-complete-time': '完成时间:',
      'task-current-step': '当前步骤:',
      'waiting-to-start': '等待开始',
      'task-elapsed-time': '耗时:',
      'task-result-preview': '结果预览',
      'refresh-preview': '刷新预览',
      'loading': '加载中...',
      'task-running-preparing': '任务运行中，正在准备预览数据...',
      'preview-after-task': '任务完成后将显示结果预览...',
      'preview-error': '加载预览时发生错误'
    }
  },
  'en-US': {
    translation: {
      // Page title
      'page-title': 'Workflow Editor - Weibo Public Opinion Analysis System',
      'navbar-brand': 'Workflow Editor',
      
      // Navigation menu
      'nav-visual-editor': 'Visual Editor',
      'nav-template-mgmt': 'Template Management',
      'nav-task-list': 'Task List',
      
      // Buttons
      'btn-save': 'Save',
      'btn-run': 'Run',
      'btn-cancel': 'Cancel',
      'btn-close': 'Close',
      'btn-create-new': 'Create New',
      'btn-validate': 'Validate',
      'btn-undo': 'Undo',
      'btn-redo': 'Redo',
      'btn-zoom-in': 'Zoom In',
      'btn-zoom-out': 'Zoom Out',
      'btn-fit-view': 'Fit View',
      'btn-export': 'Export Workflow',
      'btn-import': 'Import Workflow',
      'btn-cancel-task': 'Cancel Task',
      'btn-view-full-result': 'View Full Results',
      
      // Tabs
      'tab-components': 'Components',
      'tab-templates': 'Templates',
      
      // Component categories
      'comp-data-source': 'Data Sources',
      'comp-data-processing': 'Data Processing',
      'comp-model-analysis': 'Model Analysis',
      'comp-visualization': 'Visualization',
      
      // Components
      'comp-database': 'Database',
      'comp-file': 'File',
      'comp-crawler': 'Crawler',
      'comp-filter': 'Filter',
      'comp-sort': 'Sort',
      'comp-aggregate': 'Aggregate',
      'comp-sentiment': 'Sentiment Analysis',
      'comp-topic': 'Topic Classification',
      'comp-keywords': 'Keyword Extraction',
      'comp-summarize': 'Text Summarization',
      'comp-chart': 'Chart',
      'comp-table': 'Table',
      'comp-wordcloud': 'Word Cloud',
      
      // Template related
      'templates-crawler': 'Crawler Templates',
      'templates-analysis': 'Analysis Flow Templates',
      'modal-save-template': 'Save as Template',
      'template-name': 'Template Name',
      'template-description': 'Description',
      'template-icon': 'Icon',
      
      // Icon names
      'icon-chart': 'Chart',
      'icon-filter': 'Filter',
      'icon-crawler': 'Crawler',
      'icon-ai': 'AI Analysis',
      'icon-database': 'Database',
      'icon-wordcloud': 'Word Cloud',
      
      // Properties panel
      'properties-title': 'Component Properties',
      
      // Workflow status
      'workflow-status-message': 'Workflow ready. Drag components from the left panel to create nodes.',
      'nodes': 'Nodes',
      'connections': 'Connections',
      
      // Run workflow
      'modal-run-workflow': 'Run Workflow',
      'run-workflow-confirm': 'Are you sure you want to run the current workflow?',
      'save-before-run': 'Save workflow before running',
      
      // Task status
      'modal-task-status': 'Task Execution Status',
      'task-progress': 'Progress',
      'task-status-info': 'Status Information',
      'task-waiting': 'Waiting',
      'task-id': 'Task ID:',
      'task-status': 'Status:',
      'task-start-time': 'Start Time:',
      'task-complete-time': 'Complete Time:',
      'task-current-step': 'Current Step:',
      'waiting-to-start': 'Waiting to start',
      'task-elapsed-time': 'Elapsed Time:',
      'task-result-preview': 'Result Preview',
      'refresh-preview': 'Refresh Preview',
      'loading': 'Loading...',
      'task-running-preparing': 'Task is running, preparing preview data...',
      'preview-after-task': 'Results preview will be displayed after the task is completed...',
      'preview-error': 'Error loading preview'
    }
  }
}; 