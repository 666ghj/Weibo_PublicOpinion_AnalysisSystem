let workflowEditorInitialized = false;

// 初始化i18next多语言支持
function initializeI18n() {
    // 获取浏览器语言，默认为中文
    const browserLang = navigator.language || 'zh-CN';
    const defaultLang = browserLang.startsWith('zh') ? 'zh-CN' : 'en-US';
    
    // 初始化i18next
    i18next.init({
        lng: localStorage.getItem('preferred_language') || defaultLang,
        resources: i18nResources,
        fallbackLng: 'zh-CN',
    }).then(function(t) {
        // 更新当前语言显示
        updateLanguageDisplay();
        
        // 应用翻译到所有元素
        applyTranslations();
    });
}

// 更新语言显示
function updateLanguageDisplay() {
    const currentLang = i18next.language;
    const displayName = currentLang === 'zh-CN' ? '中文' : 'English';
    document.getElementById('currentLanguage').textContent = displayName;
}

// 应用翻译到所有元素
function applyTranslations() {
    // 翻译data-i18n属性的元素
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        element.textContent = i18next.t(key);
    });
    
    // 翻译title属性
    document.querySelectorAll('[data-i18n-title]').forEach(element => {
        const key = element.getAttribute('data-i18n-title');
        element.title = i18next.t(key);
    });
    
    // 更新页面标题
    document.title = i18next.t('page-title');
}

// 切换语言
function switchLanguage(lang) {
    // 保存语言偏好到本地存储
    localStorage.setItem('preferred_language', lang);
    
    // 更改i18next语言
    i18next.changeLanguage(lang).then(() => {
        // 更新语言显示
        updateLanguageDisplay();
        
        // 应用翻译
        applyTranslations();
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // 检查是否已初始化，防止多次执行
    if (workflowEditorInitialized) {
        console.log('工作流编辑器已初始化，跳过重复初始化');
        return;
    }
    workflowEditorInitialized = true;
    
    // 初始化多语言支持
    initializeI18n();
    
    // 添加语言切换事件
    document.querySelectorAll('.language-option').forEach(option => {
        option.addEventListener('click', function() {
            const lang = this.getAttribute('data-lang');
            switchLanguage(lang);
        });
    });
    
    // 工作流编辑器的主要元素
    const workflowCanvas = document.getElementById('workflowCanvas');
    const connectionsSvg = document.getElementById('connectionsSvg');
    
    // 工作流数据对象
    let workflowData = {
        metadata: {
            name: '新建工作流',
            description: '',
            created: new Date().toISOString(),
            modified: new Date().toISOString()
        },
        nodes: [],
        connections: []
    };
    
    // 拖拽相关变量
    let isDragging = false;
    let dragTarget = null;
    let dragOffset = { x: 0, y: 0 };
    
    // 连接相关变量
    let isConnecting = false;
    let connectionStart = null;
    let connectionPreviewPath = null;
    
    // 记录初始化状态，避免重复初始化导致的组件重复添加
    let isInitialized = false;

    // 历史记录管理
    const MAX_HISTORY = 50; // 最多保存50步历史
    let history = [];
    let currentHistoryIndex = -1;
    
    // 自动保存相关变量
    const AUTO_SAVE_INTERVAL = 3 * 60 * 1000; // 3分钟
    let autoSaveTimer = null;
    
    // 视图缩放相关变量
    let canvasScale = 1;
    let canvasTranslate = { x: 0, y: 0 };

    // 设置编辑器网格背景
    setEditorBackground();
    
    // 初始化组件面板拖拽
    initializeComponentDrag();
    
    function setEditorBackground() {
        workflowCanvas.style.backgroundSize = '20px 20px';
        workflowCanvas.style.backgroundImage = `
            linear-gradient(to right, #f0f0f0 1px, transparent 1px),
            linear-gradient(to bottom, #f0f0f0 1px, transparent 1px)
        `;
    }
    
    function initializeComponentDrag() {
        const components = document.querySelectorAll('.component-item');
        components.forEach(component => {
            component.setAttribute('draggable', 'true');
            component.addEventListener('dragstart', function(e) {
                e.dataTransfer.setData('componentType', this.dataset.type);
                e.dataTransfer.setData('componentSubtype', this.dataset.subtype);
            });
        });
    }
    
    // 创建模板卡片
    function createTemplateCard(template) {
        const div = document.createElement('div');
        div.className = 'template-item';
        
        div.innerHTML = `
            <div class="d-flex align-items-center mb-2">
                <i class="fas fa-${template.icon || 'file-alt'} me-2"></i>
                <span class="template-title">${template.name}</span>
            </div>
            <div class="template-desc">${template.description || '无描述'}</div>
            <div class="mt-2 text-end">
                <button class="btn btn-sm btn-outline-primary load-template-btn">加载</button>
            </div>
        `;
        
        // 添加加载模板的事件
        const loadBtn = div.querySelector('.load-template-btn');
        loadBtn.addEventListener('click', function() {
            loadWorkflow(template.id);
        });
        
        return div;
    }
    
    function loadWorkflow(templateId) {
        // 加载特定工作流模板的逻辑
        fetch(`/api/workflow/${templateId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                clearWorkflow();
                renderWorkflow(data.workflow);
            } else {
                alert('加载工作流失败: ' + data.error);
            }
        })
        .catch(error => {
            console.error('加载工作流出错:', error);
            alert('加载工作流时发生错误');
        });
    }
    
    function clearWorkflow() {
        // 清除画布上的所有节点和连接
        workflowCanvas.querySelectorAll('.workflow-node').forEach(node => {
            node.parentNode.removeChild(node);
        });
        
        connectionsSvg.querySelectorAll('.connection-path').forEach(path => {
            path.parentNode.removeChild(path);
        });
        
        // 清空数据
        workflowData.nodes = [];
        workflowData.connections = [];
    }
    
    function renderWorkflow(workflowToRender) {
        // 设置工作流元数据
        workflowData.metadata = workflowToRender.metadata || {
            name: '未命名工作流',
            description: '',
            created: new Date().toISOString(),
            modified: new Date().toISOString()
        };
        
        // 渲染节点
        if (workflowToRender.nodes && Array.isArray(workflowToRender.nodes)) {
            workflowToRender.nodes.forEach(node => {
                const nodeElement = createNodeFromData(node);
                if (nodeElement) {
                    setupNodeEvents(nodeElement, node);
                }
            });
        }
        
        // 渲染连接
        if (workflowToRender.connections && Array.isArray(workflowToRender.connections)) {
            workflowToRender.connections.forEach(conn => {
                workflowData.connections.push(conn);
                drawConnection(conn.sourceId, conn.targetId, conn.id);
            });
        }
    }
    
    function createNodeFromData(nodeData) {
        // 检查节点是否已存在
        const existingNode = document.getElementById(nodeData.id);
        if (existingNode) {
            console.warn('节点已存在:', nodeData.id);
            return existingNode;
        }
        
        // 从数据创建节点DOM元素
        const nodeElement = document.createElement('div');
        nodeElement.className = 'workflow-node';
        nodeElement.id = nodeData.id;
        
        // 确保坐标有效
        const x = typeof nodeData.x === 'number' ? nodeData.x : 100;
        const y = typeof nodeData.y === 'number' ? nodeData.y : 100;
        
        nodeElement.style.left = x + 'px';
        nodeElement.style.top = y + 'px';
        
        // 根据节点类型设置不同的样式
        nodeElement.classList.add(`node-type-${nodeData.type}`);
        
        // 构建节点内容
        nodeElement.innerHTML = `
            <div class="node-header">
                <span class="node-title">${nodeData.title}</span>
                <div class="node-type-badge">${getComponentTypeLabel(nodeData.type)}</div>
            </div>
            <div class="node-content">
                <div class="node-subtype">${nodeData.subtype}</div>
                <p class="node-description">${nodeData.config ? '已配置' : '点击配置参数'}</p>
            </div>
            <div class="node-ports">
                <div class="port port-in" data-port-type="input" title="输入连接点"></div>
                <div class="port port-out" data-port-type="output" title="输出连接点"></div>
            </div>
            <div class="node-actions">
                <button class="btn btn-sm btn-outline-danger delete-node-btn" title="删除节点">
                    <i class="fas fa-trash-alt"></i>
                </button>
                <button class="btn btn-sm btn-outline-primary config-node-btn" title="配置节点">
                    <i class="fas fa-cog"></i>
                </button>
            </div>
        `;
        
        // 添加入场动画
        nodeElement.classList.add('node-entering');
        setTimeout(() => {
            nodeElement.classList.remove('node-entering');
        }, 300);
        
        workflowCanvas.appendChild(nodeElement);
        
        // 绑定配置按钮事件
        const configBtn = nodeElement.querySelector('.config-node-btn');
        configBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            openNodeConfig(nodeData);
        });
        
        // 添加到节点数据
        const existingNodeIndex = workflowData.nodes.findIndex(node => node.id === nodeData.id);
        if (existingNodeIndex === -1) {
            workflowData.nodes.push(nodeData);
        } else {
            workflowData.nodes[existingNodeIndex] = nodeData;
        }
        
        return nodeElement;
    }
    
    // ====== 运行工作流 ======
    document.getElementById('runWorkflowBtn').addEventListener('click', function() {
        // 先验证工作流是否有效
        const validationResult = validateWorkflow(workflowData);
        
        if (!validationResult.valid) {
            showNotification('错误', `无法运行: ${validationResult.message}`, 'error');
            return;
        }
        
        $('#runWorkflowModal').modal('show');
    });
    
    document.getElementById('confirmRunBtn').addEventListener('click', function() {
        const shouldSave = document.getElementById('saveBeforeRun').checked;
        
        if (shouldSave) {
            // 如果选择了先保存
            workflowData.metadata.modified = new Date().toISOString();
            saveWorkflow(workflowData);
        }
        
        // 关闭确认对话框
        $('#runWorkflowModal').modal('hide');
        
        // 提交工作流执行
        runWorkflow(workflowData);
    });
    
    function runWorkflow(workflow) {
        // 发送工作流到服务器执行
        fetch('/api/workflow/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(workflow)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 显示任务状态监控
                showTaskStatus(data.taskId);
            } else {
                alert('运行工作流失败: ' + data.error);
            }
        })
        .catch(error => {
            console.error('运行工作流出错:', error);
            alert('运行工作流时发生错误，请重试');
        });
    }
    
    // ====== 任务状态监控 ======
    let taskStatusInterval = null;
    
    function showTaskStatus(taskId) {
        // 显示任务状态模态框
        document.getElementById('taskIdDisplay').textContent = taskId;
        document.getElementById('taskStatusDisplay').textContent = '运行中';
        document.getElementById('taskStartTimeDisplay').textContent = new Date().toLocaleString();
        document.getElementById('taskCompleteTimeDisplay').textContent = '-';
        document.getElementById('taskProgressBar').style.width = '0%';
        document.getElementById('taskResultPreview').innerHTML = '<p class="text-muted">任务运行中，请稍候...</p>';
        
        $('#taskStatusModal').modal('show');
        
        // 开始定期检查任务状态
        if (taskStatusInterval) {
            clearInterval(taskStatusInterval);
        }
        
        pollTaskStatus(taskId);
        taskStatusInterval = setInterval(() => pollTaskStatus(taskId), 3000);
    }
    
    function pollTaskStatus(taskId) {
        fetch(`/api/task/${taskId}/status`)
        .then(response => response.json())
        .then(data => {
            updateTaskStatusDisplay(data);
            
            // 如果任务已完成或失败，停止轮询
            if (data.status === 'completed' || data.status === 'failed') {
                if (taskStatusInterval) {
                    clearInterval(taskStatusInterval);
                    taskStatusInterval = null;
                }
            }
        })
        .catch(error => {
            console.error('获取任务状态出错:', error);
        });
    }
    
    function updateTaskStatusDisplay(taskData) {
        const statusDisplay = document.getElementById('taskStatusDisplay');
        const progressBar = document.getElementById('taskProgressBar');
        const resultPreview = document.getElementById('taskResultPreview');
        
        statusDisplay.textContent = getStatusText(taskData.status);
        
        // 更新进度条
        progressBar.style.width = `${taskData.progress || 0}%`;
        
        // 根据状态设置进度条颜色
        progressBar.className = 'progress-bar';
        if (taskData.status === 'completed') {
            progressBar.classList.add('bg-success');
            document.getElementById('taskCompleteTimeDisplay').textContent = new Date().toLocaleString();
        } else if (taskData.status === 'failed') {
            progressBar.classList.add('bg-danger');
            document.getElementById('taskCompleteTimeDisplay').textContent = new Date().toLocaleString();
        } else {
            progressBar.classList.add('bg-primary');
        }
        
        // 显示结果预览
        if (taskData.status === 'completed' && taskData.resultPreview) {
            resultPreview.innerHTML = generateResultPreview(taskData.resultPreview);
        } else if (taskData.status === 'failed' && taskData.error) {
            resultPreview.innerHTML = `<div class="alert alert-danger">${taskData.error}</div>`;
        }
    }
    
    function getStatusText(status) {
        switch (status) {
            case 'pending': return '排队中';
            case 'running': return '运行中';
            case 'completed': return '已完成';
            case 'failed': return '失败';
            default: return status;
        }
    }
    
    function generateResultPreview(resultData) {
        if (!resultData) return '<p class="text-muted">无可用预览</p>';
        
        let html = '';
        
        if (resultData.type === 'text') {
            html = `<pre class="p-2 bg-light rounded">${resultData.content}</pre>`;
        } else if (resultData.type === 'table') {
            html = '<div class="table-responsive"><table class="table table-sm table-bordered">';
            
            // 表头
            if (resultData.headers && resultData.headers.length) {
                html += '<thead><tr>';
                resultData.headers.forEach(header => {
                    html += `<th>${header}</th>`;
                });
                html += '</tr></thead>';
            }
            
            // 表内容
            if (resultData.rows && resultData.rows.length) {
                html += '<tbody>';
                resultData.rows.slice(0, 5).forEach(row => {
                    html += '<tr>';
                    row.forEach(cell => {
                        html += `<td>${cell}</td>`;
                    });
                    html += '</tr>';
                });
                html += '</tbody>';
            }
            
            html += '</table>';
            
            if (resultData.rows && resultData.rows.length > 5) {
                html += `<p class="text-muted">显示前5行，共${resultData.rows.length}行</p>`;
            }
            html += '</div>';
        } else if (resultData.type === 'chart') {
            html = '<div class="text-center"><img src="' + resultData.imageUrl + 
                   '" class="img-fluid" alt="结果图表"></div>';
        }
        
        return html;
    }
    
    document.getElementById('cancelTaskBtn').addEventListener('click', function() {
        const taskId = document.getElementById('taskIdDisplay').textContent;
        
        if (!taskId) return;
        
        // 发送取消任务请求
        fetch(`/api/task/${taskId}/cancel`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('任务已取消');
                if (taskStatusInterval) {
                    clearInterval(taskStatusInterval);
                    taskStatusInterval = null;
                }
                
                document.getElementById('taskStatusDisplay').textContent = '已取消';
            } else {
                alert('取消任务失败: ' + data.error);
            }
        })
        .catch(error => {
            console.error('取消任务出错:', error);
            alert('取消任务时发生错误');
        });
    });
    
    document.getElementById('viewResultBtn').addEventListener('click', function() {
        const taskId = document.getElementById('taskIdDisplay').textContent;
        if (!taskId) return;
        
        // 跳转到结果页面
        window.open(`/result/${taskId}`, '_blank');
    });
    
    // ====== 辅助函数 ======
    function getComponentTypeLabel(type) {
        const typeLabels = {
            'data_source': '数据源',
            'preprocessing': '数据处理',
            'model': '模型分析',
            'visualization': '可视化'
        };
        
        return typeLabels[type] || type;
    }
    
    function getDefaultConfig(type, subtype) {
        // 根据组件类型返回默认配置
        const defaults = {
            'data_source': {
                'database': { connectionString: '', query: '' },
                'file': { filePath: '', format: 'csv' },
                'crawler': { url: '', depth: 1, keywords: '' }
            },
            'preprocessing': {
                'filter': { field: '', operator: 'contains', value: '' },
                'sort': { field: '', order: 'asc' },
                'aggregate': { groupBy: '', function: 'count' }
            },
            'model': {
                'sentiment': { language: 'zh', algorithm: 'bayes' },
                'topic': { numTopics: 5, algorithm: 'lda' },
                'keywords': { topk: 10, algorithm: 'tfidf' },
                'summarize': { ratio: 0.2, algorithm: 'extractive' }
            },
            'visualization': {
                'chart': { type: 'bar', title: '', xField: '', yField: '' },
                'table': { fields: [], pageSize: 10 },
                'wordcloud': { maxWords: 100, colorScheme: 'default' }
            }
        };
        
        return defaults[type] && defaults[type][subtype] ? defaults[type][subtype] : {};
    }
    
    function getComponentConfigs(type, subtype) {
        // 返回特定组件类型的配置选项
        const configs = {
            'data_source': {
                'database': [
                    { id: 'connectionString', label: '连接字符串', type: 'text' },
                    { id: 'query', label: 'SQL查询', type: 'textarea' },
                    { id: 'limit', label: '结果限制', type: 'number' }
                ],
                'file': [
                    { id: 'filePath', label: '文件路径', type: 'text' },
                    { id: 'format', label: '文件格式', type: 'select', options: [
                        { value: 'csv', label: 'CSV' },
                        { value: 'excel', label: 'Excel' },
                        { value: 'json', label: 'JSON' },
                        { value: 'txt', label: '文本文件' }
                    ] }
                ],
                'crawler': [
                    { id: 'url', label: '起始URL', type: 'text' },
                    { id: 'depth', label: '爬取深度', type: 'number' },
                    { id: 'keywords', label: '关键词', type: 'text' },
                    { id: 'maxItems', label: '最大爬取数量', type: 'number' }
                ]
            },
            'preprocessing': {
                'filter': [
                    { id: 'field', label: '字段名', type: 'text' },
                    { id: 'operator', label: '操作符', type: 'select', options: [
                        { value: 'equals', label: '等于' },
                        { value: 'contains', label: '包含' },
                        { value: 'startsWith', label: '开头是' },
                        { value: 'endsWith', label: '结尾是' },
                        { value: 'greaterThan', label: '大于' },
                        { value: 'lessThan', label: '小于' }
                    ] },
                    { id: 'value', label: '值', type: 'text' }
                ],
                'sort': [
                    { id: 'field', label: '排序字段', type: 'text' },
                    { id: 'order', label: '排序方向', type: 'select', options: [
                        { value: 'asc', label: '升序' },
                        { value: 'desc', label: '降序' }
                    ] }
                ],
                'aggregate': [
                    { id: 'groupBy', label: '分组字段', type: 'text' },
                    { id: 'function', label: '聚合函数', type: 'select', options: [
                        { value: 'count', label: '计数' },
                        { value: 'sum', label: '求和' },
                        { value: 'avg', label: '平均值' },
                        { value: 'min', label: '最小值' },
                        { value: 'max', label: '最大值' }
                    ] },
                    { id: 'valueField', label: '值字段', type: 'text' }
                ]
            },
            'model': {
                'sentiment': [
                    { id: 'language', label: '语言', type: 'select', options: [
                        { value: 'zh', label: '中文' },
                        { value: 'en', label: '英文' }
                    ] },
                    { id: 'algorithm', label: '算法', type: 'select', options: [
                        { value: 'bayes', label: '朴素贝叶斯' },
                        { value: 'svm', label: '支持向量机' },
                        { value: 'bert', label: 'BERT' }
                    ] },
                    { id: 'textField', label: '文本字段', type: 'text' }
                ],
                'topic': [
                    { id: 'numTopics', label: '主题数量', type: 'number' },
                    { id: 'algorithm', label: '算法', type: 'select', options: [
                        { value: 'lda', label: 'LDA' },
                        { value: 'nmf', label: 'NMF' }
                    ] },
                    { id: 'textField', label: '文本字段', type: 'text' }
                ],
                'keywords': [
                    { id: 'topk', label: '关键词数量', type: 'number' },
                    { id: 'algorithm', label: '算法', type: 'select', options: [
                        { value: 'tfidf', label: 'TF-IDF' },
                        { value: 'textrank', label: 'TextRank' }
                    ] },
                    { id: 'textField', label: '文本字段', type: 'text' }
                ],
                'summarize': [
                    { id: 'ratio', label: '摘要比例', type: 'number' },
                    { id: 'algorithm', label: '算法', type: 'select', options: [
                        { value: 'extractive', label: '抽取式摘要' },
                        { value: 'abstractive', label: '生成式摘要' }
                    ] },
                    { id: 'textField', label: '文本字段', type: 'text' }
                ]
            },
            'visualization': {
                'chart': [
                    { id: 'type', label: '图表类型', type: 'select', options: [
                        { value: 'bar', label: '柱状图' },
                        { value: 'line', label: '折线图' },
                        { value: 'pie', label: '饼图' },
                        { value: 'scatter', label: '散点图' }
                    ] },
                    { id: 'title', label: '图表标题', type: 'text' },
                    { id: 'xField', label: 'X轴字段', type: 'text' },
                    { id: 'yField', label: 'Y轴字段', type: 'text' },
                    { id: 'colorField', label: '颜色字段', type: 'text' }
                ],
                'table': [
                    { id: 'fields', label: '显示字段(逗号分隔)', type: 'text' },
                    { id: 'pageSize', label: '每页记录数', type: 'number' },
                    { id: 'sortable', label: '允许排序', type: 'checkbox' }
                ],
                'wordcloud': [
                    { id: 'textField', label: '文本字段', type: 'text' },
                    { id: 'maxWords', label: '最大词数', type: 'number' },
                    { id: 'colorScheme', label: '配色方案', type: 'select', options: [
                        { value: 'default', label: '默认' },
                        { value: 'warm', label: '暖色调' },
                        { value: 'cool', label: '冷色调' },
                        { value: 'rainbow', label: '彩虹色' }
                    ] }
                ]
            }
        };
        
        return configs[type] && configs[type][subtype] ? configs[type][subtype] : [];
    }
    
    // 初始加载示例模板
    showSampleTemplates();

    function saveWorkflow(workflowData) {
        // 保存工作流到服务器
        fetch('/api/workflow/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(workflowData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('工作流保存成功');
            } else {
                alert('保存工作流失败: ' + data.error);
            }
        })
        .catch(error => {
            console.error('保存工作流出错:', error);
            alert('保存工作流时发生错误');
        });
    }

    // 显示示例模板（修复CORS错误）
    function showSampleTemplates() {
        // 使用示例模板数据，避免直接从文件系统加载API
        const sampleTemplates = [
            {
                id: 'template_1',
                name: '微博热搜分析模板',
                description: '爬取微博热搜榜数据，分析热点话题和情感倾向',
                icon: 'fire'
            },
            {
                id: 'template_2',
                name: '用户评论情感分析',
                description: '分析用户评论的情感倾向，生成情感分布图表',
                icon: 'heart'
            },
            {
                id: 'template_3',
                name: '话题趋势监测',
                description: '监测特定话题的讨论热度变化及关键词提取',
                icon: 'chart-line'
            }
        ];

        try {
            const container = document.getElementById('analysisTemplatesList');
            if(container) {
                container.innerHTML = '';
                sampleTemplates.forEach(template => {
                    const templateDiv = createTemplateCard(template);
                    container.appendChild(templateDiv);
                });
            } else {
                // 尝试其他容器
                const alternativeContainer = document.getElementById('templateList') || 
                                            document.getElementById('crawlerTemplatesList');
                if (alternativeContainer) {
                    alternativeContainer.innerHTML = '';
                    sampleTemplates.forEach(template => {
                        const templateDiv = createTemplateCard(template);
                        alternativeContainer.appendChild(templateDiv);
                    });
                } else {
                    console.warn('未找到合适的模板容器');
                }
            }
        } catch (error) {
            console.error('加载模板出错:', error);
        }
    }

    // 模板拖放功能
    workflowCanvas.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        
        // 显示可放置区域指示
        this.classList.add('drag-over');
    });

    workflowCanvas.addEventListener('dragleave', function() {
        // 移除可放置区域指示
        this.classList.remove('drag-over');
    });

    // workflowCanvas.addEventListener('drop', function(e) {
    //     e.preventDefault();
    //     this.classList.remove('drag-over');
        
    //     const componentType = e.dataTransfer.getData('componentType');
    //     const componentSubtype = e.dataTransfer.getData('componentSubtype');
        
    //     if (componentType && componentSubtype) {
    //         const rect = workflowCanvas.getBoundingClientRect();
            
    //         // 修正：修复坐标计算，确保准确的放置位置
    //         // 考虑滚动位置和缩放因素
    //         let x = (e.clientX - rect.left) / canvasScale - canvasTranslate.x;
    //         let y = (e.clientY - rect.top) / canvasScale - canvasTranslate.y;
            
    //         // 调整位置，使节点中心与鼠标位置对齐（假设节点宽度约为200px，高度为100px）
    //         x = x - 100; // 使节点中心与鼠标对齐
    //         y = y - 50;
            
    //         // 确保节点完全在可见区域内
    //         x = Math.max(0, Math.min(x, rect.width - 200));
    //         y = Math.max(0, Math.min(y, rect.height - 100));
            
    //         // 添加节点并记录历史
    //         addNode(componentType, componentSubtype, x, y);
    //         addToHistory();
            
    //         // 用户反馈
    //         showNotification('成功', `已添加 ${getComponentTypeLabel(componentType)}-${componentSubtype} 节点`, 'success');
    //     }
    // });

    // 添加其他初始化代码
    initializeWorkflowEditor();
    setupEventListeners();
    showSampleTemplates();

    function initializeWorkflowEditor() {
        // 初始化编辑器的基本设置
        workflowData = {
            metadata: {
                name: '新建工作流',
                description: '',
                created: new Date().toISOString(),
                modified: new Date().toISOString()
            },
            nodes: [],
            connections: []
        };
    }

    function setupEventListeners() {
        // 设置各种事件监听器
        document.getElementById('saveWorkflowBtn').addEventListener('click', function() {
            workflowData.metadata.modified = new Date().toISOString();
            saveWorkflow(workflowData);
        });
    }

    // 添加节点的函数
    function addNode(componentType, componentSubtype, x, y) {
        const nodeId = 'node_' + Date.now();
        const nodeData = {
            id: nodeId,
            type: componentType,
            subtype: componentSubtype,
            title: getComponentTypeLabel(componentType) + '-' + componentSubtype,
            x: x,
            y: y,
            config: getDefaultConfig(componentType, componentSubtype)
        };
        
        const nodeElement = createNodeFromData(nodeData);
        setupNodeEvents(nodeElement, nodeData);
        
        // 更新工作流状态
        onWorkflowChanged();
        
        return nodeElement;
    }

    // 设置节点事件
    function setupNodeEvents(nodeElement, nodeData) {
        // 节点拖动事件
        nodeElement.addEventListener('mousedown', function(e) {
            if (e.target.closest('.port') || e.target.closest('.delete-node-btn')) {
                return; // 如果点击的是端口或删除按钮，不处理拖动
            }
            
            isDragging = true;
            dragTarget = nodeElement;
            const rect = nodeElement.getBoundingClientRect();
            dragOffset = {
                x: e.clientX - rect.left/2,
                y: e.clientY - rect.top/2
            };
            
            nodeElement.style.zIndex = '100';
        });
        
        // 删除节点
        const deleteBtn = nodeElement.querySelector('.delete-node-btn');
        deleteBtn.addEventListener('click', function() {
            deleteNode(nodeData.id);
        });
        
        // 节点配置
        nodeElement.addEventListener('click', function(e) {
            if (!e.target.closest('.port') && !e.target.closest('.delete-node-btn')) {
                openNodeConfig(nodeData);
            }
        });
        
        // 连接处理
        const ports = nodeElement.querySelectorAll('.port');
        ports.forEach(port => {
            port.addEventListener('mousedown', function(e) {
                e.stopPropagation();
                if (port.dataset.portType === 'output') {
                    startConnection(nodeData.id, e);
                }
            });
            
            port.addEventListener('mouseup', function() {
                if (isConnecting && connectionStart && connectionStart.id !== nodeData.id && port.dataset.portType === 'input') {
                    completeConnection(connectionStart.id, nodeData.id);
                }
            });
        });
    }
    
    // 删除节点
    function deleteNode(nodeId) {
        const node = document.getElementById(nodeId);
        if (node) {
            node.parentNode.removeChild(node);
        }
        
        // 删除相关连接
        workflowData.connections = workflowData.connections.filter(conn => {
            if (conn.sourceId === nodeId || conn.targetId === nodeId) {
                const path = document.getElementById('connection_' + conn.id);
                if (path) {
                    path.parentNode.removeChild(path);
                }
                return false;
            }
            return true;
        });
        
        // 从数据中删除节点
        workflowData.nodes = workflowData.nodes.filter(node => node.id !== nodeId);
        
        // 更新工作流状态
        onWorkflowChanged();
        
        showNotification('已删除', '节点已从工作流中移除', 'info');
    }
    
    // 处理全局鼠标事件
    document.addEventListener('mousemove', function(e) {
        if (isDragging && dragTarget) {
            const x = e.clientX - dragOffset.x;
            const y = e.clientY - dragOffset.y;
            
            dragTarget.style.left = x + 'px';
            dragTarget.style.top = y + 'px';
            
            // 更新节点数据
            const nodeId = dragTarget.id;
            const node = workflowData.nodes.find(n => n.id === nodeId);
            if (node) {
                node.x = x;
                node.y = y;
            }
            
            // 更新连接
            updateNodeConnections(nodeId);
        }
        
        // 处理连接预览
        if (isConnecting && connectionStart) {
            updateConnectionPreview(e.clientX, e.clientY);
        }
    });
    
    document.addEventListener('mouseup', function() {
        if (isDragging && dragTarget) {
            dragTarget.style.zIndex = '10';
            isDragging = false;
            dragTarget = null;
        }
        
        if (isConnecting) {
            cancelConnection();
        }
    });
    
    // 初始化侧边栏切换
    const componentsTabBtn = document.getElementById('componentsTabBtn');
    const templatesTabBtn = document.getElementById('templatesTabBtn');
    
    if (componentsTabBtn) {
        componentsTabBtn.addEventListener('click', function() {
            document.getElementById('componentsPanel').style.display = 'block';
            document.getElementById('templatesPanel').style.display = 'none';
            this.classList.add('active');
            templatesTabBtn.classList.remove('active');
        });
    }
    
    if (templatesTabBtn) {
        templatesTabBtn.addEventListener('click', function() {
            document.getElementById('componentsPanel').style.display = 'none';
            document.getElementById('templatesPanel').style.display = 'block';
            this.classList.add('active');
            componentsTabBtn.classList.remove('active');
        });
    }
    
    // 初始加载示例模板
    showSampleTemplates();

    // 绘制连接
    function drawConnection(sourceId, targetId, connectionId) {
        const sourceNode = document.getElementById(sourceId);
        const targetNode = document.getElementById(targetId);
        
        if (!sourceNode || !targetNode) {
            console.error('连接节点不存在:', sourceId, targetId);
            return null;
        }
        
        const sourcePort = sourceNode.querySelector('.port-out');
        const targetPort = targetNode.querySelector('.port-in');
        
        const sourceRect = sourcePort.getBoundingClientRect();
        const targetRect = targetPort.getBoundingClientRect();
        const canvasRect = workflowCanvas.getBoundingClientRect();
        
        const start = {
            x: sourceRect.left + sourceRect.width/2 - canvasRect.left,
            y: sourceRect.top + sourceRect.height/2 - canvasRect.top
        };
        
        const end = {
            x: targetRect.left + targetRect.width/2 - canvasRect.left,
            y: targetRect.top + targetRect.height/2 - canvasRect.top
        };
        
        // 创建连接路径
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'connection-path');
        path.setAttribute('id', 'connection_' + (connectionId || `${sourceId}_${targetId}`));
        
        // 绘制贝塞尔曲线
        const dx = Math.abs(end.x - start.x) * 0.5;
        const pathData = `M ${start.x},${start.y} C ${start.x + dx},${start.y} ${end.x - dx},${end.y} ${end.x},${end.y}`;
        path.setAttribute('d', pathData);
        
        connectionsSvg.appendChild(path);
        return path;
    }
    
    // 更新节点连接
    function updateNodeConnections(nodeId) {
        workflowData.connections.forEach(conn => {
            if (conn.sourceId === nodeId || conn.targetId === nodeId) {
                const path = document.getElementById('connection_' + conn.id);
                if (path) {
                    path.parentNode.removeChild(path);
                }
                drawConnection(conn.sourceId, conn.targetId, conn.id);
            }
        });
    }
    
    // 开始创建连接
    function startConnection(nodeId, e) {
        isConnecting = true;
        connectionStart = { id: nodeId, event: e };
        
        // 创建预览连接线
        connectionPreviewPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        connectionPreviewPath.setAttribute('class', 'connection-path');
        connectionPreviewPath.style.strokeDasharray = '5,5';
        connectionPreviewPath.style.opacity = '0.6';
        connectionsSvg.appendChild(connectionPreviewPath);
        
        updateConnectionPreview(e.clientX, e.clientY);
    }
    
    // 更新连接预览
    function updateConnectionPreview(clientX, clientY) {
        if (!connectionStart || !connectionPreviewPath) return;
        
        const sourceNode = document.getElementById(connectionStart.id);
        if (!sourceNode) return;
        
        const sourcePort = sourceNode.querySelector('.port-out');
        const sourceRect = sourcePort.getBoundingClientRect();
        const canvasRect = workflowCanvas.getBoundingClientRect();
        
        const start = {
            x: sourceRect.left + sourceRect.width/2 - canvasRect.left,
            y: sourceRect.top + sourceRect.height/2 - canvasRect.top
        };
        
        const end = {
            x: clientX - canvasRect.left,
            y: clientY - canvasRect.top
        };
        
        // 高亮可连接的目标端口
        document.querySelectorAll('.workflow-node').forEach(node => {
            if (node.id !== connectionStart.id) {
                const inputPort = node.querySelector('.port-in');
                const inputPortRect = inputPort.getBoundingClientRect();
                
                // 计算鼠标与端口的距离
                const dx = clientX - (inputPortRect.left + inputPortRect.width/2);
                const dy = clientY - (inputPortRect.top + inputPortRect.height/2);
                const distance = Math.sqrt(dx*dx + dy*dy);
                
                // 如果距离小于20像素，高亮端口
                if (distance < 20) {
                    inputPort.classList.add('port-highlight');
                    
                    // 更新预览连接终点到端口中心
                    end.x = inputPortRect.left + inputPortRect.width/2 - canvasRect.left;
                    end.y = inputPortRect.top + inputPortRect.height/2 - canvasRect.top;
                } else {
                    inputPort.classList.remove('port-highlight');
                }
            }
        });
        
        // 绘制预览连接
        const dx = Math.abs(end.x - start.x) * 0.5;
        const pathData = `M ${start.x},${start.y} C ${start.x + dx},${start.y} ${end.x - dx},${end.y} ${end.x},${end.y}`;
        connectionPreviewPath.setAttribute('d', pathData);
    }
    
    // 完成连接
    function completeConnection(sourceId, targetId) {
        // 检查是否是自连接
        if (sourceId === targetId) {
            showNotification('警告', '不能连接到自己', 'warning');
            cancelConnection();
            return;
        }
        
        // 检查连接是否已存在
        const connectionExists = workflowData.connections.some(conn => 
            conn.sourceId === sourceId && conn.targetId === targetId);
            
        if (connectionExists) {
            showNotification('警告', '连接已存在', 'warning');
            cancelConnection();
            return;
        }
        
        // 检查是否会形成循环
        if (wouldCreateCycle(sourceId, targetId)) {
            showNotification('错误', '不能创建循环连接', 'error');
            cancelConnection();
            return;
        }
        
        // 生成连接ID
        const connectionId = `conn_${Date.now()}`;
        
        // 添加到数据中
        workflowData.connections.push({
            id: connectionId,
            sourceId: sourceId,
            targetId: targetId
        });
        
        // 绘制最终连接
        drawConnection(sourceId, targetId, connectionId);
        
        // 清理预览状态
        cancelConnection();
        
        // 更新工作流状态
        onWorkflowChanged();
        
        // 显示成功通知
        showNotification('成功', '已创建连接', 'success');
    }
    
    // 取消连接操作
    function cancelConnection() {
        if (connectionPreviewPath && connectionPreviewPath.parentNode) {
            connectionPreviewPath.parentNode.removeChild(connectionPreviewPath);
        }
        
        // 移除所有高亮端口
        document.querySelectorAll('.port-highlight').forEach(port => {
            port.classList.remove('port-highlight');
        });
        
        isConnecting = false;
        connectionStart = null;
        connectionPreviewPath = null;
    }
    
    // 打开节点配置面板
    function openNodeConfig(nodeData) {
        const propertiesPanel = document.getElementById('propertiesPanel');
        const propertiesContent = document.getElementById('propertiesContent');
        
        // 显示面板
        propertiesPanel.classList.add('open');
        
        // 生成配置表单
        const configOptions = getComponentConfigs(nodeData.type, nodeData.subtype);
        let formHtml = `
            <div class="properties-header">
                <h6 class="mb-0">${nodeData.title} 配置</h6>
                <span class="badge bg-secondary">${nodeData.id}</span>
            </div>
            <form id="nodeConfigForm" class="mt-3">
        `;
        
        configOptions.forEach(option => {
            const value = nodeData.config && nodeData.config[option.id] !== undefined ? 
                          nodeData.config[option.id] : '';
                          
            formHtml += `<div class="mb-3">
                <label for="${option.id}" class="form-label">${option.label}</label>`;
                
            if (option.type === 'select') {
                formHtml += `<select class="form-select" id="${option.id}" name="${option.id}">`;
                option.options.forEach(opt => {
                    const selected = value === opt.value ? 'selected' : '';
                    formHtml += `<option value="${opt.value}" ${selected}>${opt.label}</option>`;
                });
                formHtml += `</select>`;
            } else if (option.type === 'checkbox') {
                const checked = value ? 'checked' : '';
                formHtml += `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="${option.id}" name="${option.id}" ${checked}>
                        <label class="form-check-label" for="${option.id}">${option.label}</label>
                    </div>
                `;
            } else if (option.type === 'textarea') {
                formHtml += `<textarea class="form-control" id="${option.id}" name="${option.id}" rows="3">${value}</textarea>`;
                if (option.placeholder) {
                    formHtml += `<div class="form-text">${option.placeholder}</div>`;
                }
            } else {
                formHtml += `<input type="${option.type}" class="form-control" id="${option.id}" name="${option.id}" value="${value}"`;
                if (option.placeholder) {
                    formHtml += ` placeholder="${option.placeholder}"`;
                }
                if (option.min !== undefined) {
                    formHtml += ` min="${option.min}"`;
                }
                if (option.max !== undefined) {
                    formHtml += ` max="${option.max}"`;
                }
                formHtml += `>`;
                if (option.helpText) {
                    formHtml += `<div class="form-text">${option.helpText}</div>`;
                }
            }
            
            formHtml += `</div>`;
        });
        
        formHtml += `
            <div class="d-flex justify-content-between mt-4">
                <button type="button" class="btn btn-secondary" id="cancelConfigBtn">取消</button>
                <button type="button" class="btn btn-primary" id="saveConfigBtn">应用</button>
            </div>
            </form>
        `;
        
        propertiesContent.innerHTML = formHtml;
        
        // 保存配置事件
        document.getElementById('saveConfigBtn').addEventListener('click', function() {
            const form = document.getElementById('nodeConfigForm');
            if (!form.checkValidity()) {
                form.reportValidity();
                return;
            }
            
            const formData = new FormData(form);
            const config = {};
            
            // 构建配置对象
            configOptions.forEach(option => {
                if (option.type === 'checkbox') {
                    config[option.id] = document.getElementById(option.id).checked;
                } else {
                    config[option.id] = formData.get(option.id);
                }
            });
            
            // 更新节点配置
            const node = workflowData.nodes.find(n => n.id === nodeData.id);
            if (node) {
                node.config = config;
                
                // 更新节点显示
                const nodeElement = document.getElementById(nodeData.id);
                if (nodeElement) {
                    const descElement = nodeElement.querySelector('.node-description');
                    if (descElement) {
                        descElement.textContent = '已配置';
                        descElement.classList.add('configured');
                    }
                }
                
                // 添加到历史记录
                addToHistory();
                
                // 显示成功通知
                showNotification('成功', '节点配置已更新', 'success');
            }
            
            // 关闭面板
            closePropertiesPanel();
        });
        
        // 取消配置事件
        document.getElementById('cancelConfigBtn').addEventListener('click', closePropertiesPanel);
    }
    
    // 关闭属性面板
    function closePropertiesPanel() {
        document.getElementById('propertiesPanel').classList.remove('open');
    }
    
    // 绑定关闭属性面板的事件
    document.getElementById('closePropertiesBtn').addEventListener('click', closePropertiesPanel);

    function initializeAutoSave() {
        // 开始自动保存计时器
        startAutoSaveTimer();
        
        // 添加用户交互检测
        workflowCanvas.addEventListener('mousedown', resetAutoSaveTimer);
        document.addEventListener('keydown', resetAutoSaveTimer);
    }
    
    function startAutoSaveTimer() {
        if (autoSaveTimer) {
            clearTimeout(autoSaveTimer);
        }
        
        autoSaveTimer = setTimeout(function() {
            // 只在有节点时自动保存
            if (workflowData.nodes.length > 0) {
                console.log('自动保存工作流...');
                // 设置自动保存标志
                workflowData.autoSaved = true;
                saveWorkflow(workflowData);
            }
            // 重新开始计时器
            startAutoSaveTimer();
        }, AUTO_SAVE_INTERVAL);
    }
    
    function resetAutoSaveTimer() {
        startAutoSaveTimer();
    }
    
    // 初始化自动保存
    initializeAutoSave();
    
    // 优化拖放功能
    workflowCanvas.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        
        // 显示可放置区域指示
        this.classList.add('drag-over');
    });
    
    workflowCanvas.addEventListener('dragleave', function() {
        // 移除可放置区域指示
        this.classList.remove('drag-over');
    });
    
    workflowCanvas.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('drag-over');
        
        const componentType = e.dataTransfer.getData('componentType');
        const componentSubtype = e.dataTransfer.getData('componentSubtype');
        
        if (componentType && componentSubtype) {
            const rect = workflowCanvas.getBoundingClientRect();
            // 计算相对于画布的坐标
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // 添加节点并记录历史
            addNode(componentType, componentSubtype, x, y);
            addToHistory();
            
            // 用户反馈
            showNotification('成功', `已添加 ${getComponentTypeLabel(componentType)}-${componentSubtype} 节点`, 'success');
        }
    });
    
    // 优化创建节点函数
    function createNodeFromData(nodeData) {
        // 检查节点是否已存在
        const existingNode = document.getElementById(nodeData.id);
        if (existingNode) {
            console.warn('节点已存在:', nodeData.id);
            return existingNode;
        }
        
        // 从数据创建节点DOM元素
        const nodeElement = document.createElement('div');
        nodeElement.className = 'workflow-node';
        nodeElement.id = nodeData.id;
        
        // 确保坐标有效
        const x = typeof nodeData.x === 'number' ? nodeData.x : 100;
        const y = typeof nodeData.y === 'number' ? nodeData.y : 100;
        
        nodeElement.style.left = x + 'px';
        nodeElement.style.top = y + 'px';
        
        // 根据节点类型设置不同的样式
        nodeElement.classList.add(`node-type-${nodeData.type}`);
        
        // 构建节点内容
        nodeElement.innerHTML = `
            <div class="node-header">
                <span class="node-title">${nodeData.title}</span>
                <div class="node-type-badge">${getComponentTypeLabel(nodeData.type)}</div>
            </div>
            <div class="node-content">
                <div class="node-subtype">${nodeData.subtype}</div>
                <p class="node-description">${nodeData.config ? '已配置' : '点击配置参数'}</p>
            </div>
            <div class="node-ports">
                <div class="port port-in" data-port-type="input" title="输入连接点"></div>
                <div class="port port-out" data-port-type="output" title="输出连接点"></div>
            </div>
            <div class="node-actions">
                <button class="btn btn-sm btn-outline-danger delete-node-btn" title="删除节点">
                    <i class="fas fa-trash-alt"></i>
                </button>
                <button class="btn btn-sm btn-outline-primary config-node-btn" title="配置节点">
                    <i class="fas fa-cog"></i>
                </button>
            </div>
        `;
        
        // 添加入场动画
        nodeElement.classList.add('node-entering');
        setTimeout(() => {
            nodeElement.classList.remove('node-entering');
        }, 300);
        
        workflowCanvas.appendChild(nodeElement);
        
        // 绑定配置按钮事件
        const configBtn = nodeElement.querySelector('.config-node-btn');
        configBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            openNodeConfig(nodeData);
        });
        
        // 添加到节点数据
        const existingNodeIndex = workflowData.nodes.findIndex(node => node.id === nodeData.id);
        if (existingNodeIndex === -1) {
            workflowData.nodes.push(nodeData);
        } else {
            workflowData.nodes[existingNodeIndex] = nodeData;
        }
        
        return nodeElement;
    }
    
    // 优化节点配置面板
    function openNodeConfig(nodeData) {
        const propertiesPanel = document.getElementById('propertiesPanel');
        const propertiesContent = document.getElementById('propertiesContent');
        
        // 显示面板
        propertiesPanel.classList.add('open');
        
        // 生成配置表单
        const configOptions = getComponentConfigs(nodeData.type, nodeData.subtype);
        let formHtml = `
            <div class="properties-header">
                <h6 class="mb-0">${nodeData.title} 配置</h6>
                <span class="badge bg-secondary">${nodeData.id}</span>
            </div>
            <form id="nodeConfigForm" class="mt-3">
        `;
        
        configOptions.forEach(option => {
            const value = nodeData.config && nodeData.config[option.id] !== undefined ? 
                          nodeData.config[option.id] : '';
                          
            formHtml += `<div class="mb-3">
                <label for="${option.id}" class="form-label">${option.label}</label>`;
                
            if (option.type === 'select') {
                formHtml += `<select class="form-select" id="${option.id}" name="${option.id}">`;
                option.options.forEach(opt => {
                    const selected = value === opt.value ? 'selected' : '';
                    formHtml += `<option value="${opt.value}" ${selected}>${opt.label}</option>`;
                });
                formHtml += `</select>`;
            } else if (option.type === 'checkbox') {
                const checked = value ? 'checked' : '';
                formHtml += `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="${option.id}" name="${option.id}" ${checked}>
                        <label class="form-check-label" for="${option.id}">${option.label}</label>
                    </div>
                `;
            } else if (option.type === 'textarea') {
                formHtml += `<textarea class="form-control" id="${option.id}" name="${option.id}" rows="3">${value}</textarea>`;
                if (option.placeholder) {
                    formHtml += `<div class="form-text">${option.placeholder}</div>`;
                }
            } else {
                formHtml += `<input type="${option.type}" class="form-control" id="${option.id}" name="${option.id}" value="${value}"`;
                if (option.placeholder) {
                    formHtml += ` placeholder="${option.placeholder}"`;
                }
                if (option.min !== undefined) {
                    formHtml += ` min="${option.min}"`;
                }
                if (option.max !== undefined) {
                    formHtml += ` max="${option.max}"`;
                }
                formHtml += `>`;
                if (option.helpText) {
                    formHtml += `<div class="form-text">${option.helpText}</div>`;
                }
            }
            
            formHtml += `</div>`;
        });
        
        formHtml += `
            <div class="d-flex justify-content-between mt-4">
                <button type="button" class="btn btn-secondary" id="cancelConfigBtn">取消</button>
                <button type="button" class="btn btn-primary" id="saveConfigBtn">应用</button>
            </div>
            </form>
        `;
        
        propertiesContent.innerHTML = formHtml;
        
        // 保存配置事件
        document.getElementById('saveConfigBtn').addEventListener('click', function() {
            const form = document.getElementById('nodeConfigForm');
            if (!form.checkValidity()) {
                form.reportValidity();
                return;
            }
            
            const formData = new FormData(form);
            const config = {};
            
            // 构建配置对象
            configOptions.forEach(option => {
                if (option.type === 'checkbox') {
                    config[option.id] = document.getElementById(option.id).checked;
                } else {
                    config[option.id] = formData.get(option.id);
                }
            });
            
            // 更新节点配置
            const node = workflowData.nodes.find(n => n.id === nodeData.id);
            if (node) {
                node.config = config;
                
                // 更新节点显示
                const nodeElement = document.getElementById(nodeData.id);
                if (nodeElement) {
                    const descElement = nodeElement.querySelector('.node-description');
                    if (descElement) {
                        descElement.textContent = '已配置';
                        descElement.classList.add('configured');
                    }
                }
                
                // 添加到历史记录
                addToHistory();
                
                // 显示成功通知
                showNotification('成功', '节点配置已更新', 'success');
            }
            
            // 关闭面板
            closePropertiesPanel();
        });
        
        // 取消配置事件
        document.getElementById('cancelConfigBtn').addEventListener('click', closePropertiesPanel);
    }
    
    // 改进连接预览
    function updateConnectionPreview(clientX, clientY) {
        if (!connectionStart || !connectionPreviewPath) return;
        
        const sourceNode = document.getElementById(connectionStart.id);
        if (!sourceNode) return;
        
        const sourcePort = sourceNode.querySelector('.port-out');
        const sourceRect = sourcePort.getBoundingClientRect();
        const canvasRect = workflowCanvas.getBoundingClientRect();
        
        const start = {
            x: sourceRect.left + sourceRect.width/2 - canvasRect.left,
            y: sourceRect.top + sourceRect.height/2 - canvasRect.top
        };
        
        const end = {
            x: clientX - canvasRect.left,
            y: clientY - canvasRect.top
        };
        
        // 高亮可连接的目标端口
        document.querySelectorAll('.workflow-node').forEach(node => {
            if (node.id !== connectionStart.id) {
                const inputPort = node.querySelector('.port-in');
                const inputPortRect = inputPort.getBoundingClientRect();
                
                // 计算鼠标与端口的距离
                const dx = clientX - (inputPortRect.left + inputPortRect.width/2);
                const dy = clientY - (inputPortRect.top + inputPortRect.height/2);
                const distance = Math.sqrt(dx*dx + dy*dy);
                
                // 如果距离小于20像素，高亮端口
                if (distance < 20) {
                    inputPort.classList.add('port-highlight');
                    
                    // 更新预览连接终点到端口中心
                    end.x = inputPortRect.left + inputPortRect.width/2 - canvasRect.left;
                    end.y = inputPortRect.top + inputPortRect.height/2 - canvasRect.top;
                } else {
                    inputPort.classList.remove('port-highlight');
                }
            }
        });
        
        // 绘制预览连接
        const dx = Math.abs(end.x - start.x) * 0.5;
        const pathData = `M ${start.x},${start.y} C ${start.x + dx},${start.y} ${end.x - dx},${end.y} ${end.x},${end.y}`;
        connectionPreviewPath.setAttribute('d', pathData);
    }
    
    // 取消连接操作时清除高亮
    function cancelConnection() {
        if (connectionPreviewPath && connectionPreviewPath.parentNode) {
            connectionPreviewPath.parentNode.removeChild(connectionPreviewPath);
        }
        
        // 移除所有高亮端口
        document.querySelectorAll('.port-highlight').forEach(port => {
            port.classList.remove('port-highlight');
        });
        
        isConnecting = false;
        connectionStart = null;
        connectionPreviewPath = null;
    }
    
    // 优化连接完成处理
    function completeConnection(sourceId, targetId) {
        // 检查是否是自连接
        if (sourceId === targetId) {
            showNotification('警告', '不能连接到自己', 'warning');
            cancelConnection();
            return;
        }
        
        // 检查连接是否已存在
        const connectionExists = workflowData.connections.some(conn => 
            conn.sourceId === sourceId && conn.targetId === targetId);
            
        if (connectionExists) {
            showNotification('警告', '连接已存在', 'warning');
            cancelConnection();
            return;
        }
        
        // 检查是否会形成循环
        if (wouldCreateCycle(sourceId, targetId)) {
            showNotification('错误', '不能创建循环连接', 'error');
            cancelConnection();
            return;
        }
        
        // 生成连接ID
        const connectionId = `conn_${Date.now()}`;
        
        // 添加到数据中
        workflowData.connections.push({
            id: connectionId,
            sourceId: sourceId,
            targetId: targetId
        });
        
        // 绘制最终连接
        drawConnection(sourceId, targetId, connectionId);
        
        // 清理预览状态
        cancelConnection();
        
        // 更新工作流状态
        onWorkflowChanged();
        
        // 显示成功通知
        showNotification('成功', '已创建连接', 'success');
    }
    
    // 检查是否会形成循环
    function wouldCreateCycle(sourceId, targetId) {
        // 如果目标节点可以到达源节点，那么添加这条边会导致循环
        return canReach(targetId, sourceId, new Set());
    }
    
    // 检查从startId是否可以到达endId
    function canReach(startId, endId, visited) {
        if (startId === endId) return true;
        
        // 标记当前节点为已访问
        visited.add(startId);
        
        // 获取startId的所有出边
        const outConnections = workflowData.connections.filter(conn => conn.sourceId === startId);
        
        // 检查每条出边
        for (const conn of outConnections) {
            const nextId = conn.targetId;
            
            // 如果下一个节点未访问，继续搜索
            if (!visited.has(nextId)) {
                if (canReach(nextId, endId, visited)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    // 验证工作流
    function validateWorkflow(workflow) {
        // 检查是否有节点
        if (!workflow.nodes || workflow.nodes.length === 0) {
            return { valid: false, message: '工作流没有节点' };
        }
        
        // 检查是否有连接
        if (!workflow.connections || workflow.connections.length === 0) {
            return { valid: false, message: '工作流没有连接' };
        }
        
        // 检查是否存在没有配置的节点
        const unconfiguredNodes = workflow.nodes.filter(node => !node.config);
        if (unconfiguredNodes.length > 0) {
            const nodeTitles = unconfiguredNodes.map(n => n.title).join(', ');
            return { valid: false, message: `以下节点未配置: ${nodeTitles}` };
        }
        
        // 检查是否存在没有输入的节点（除了数据源类型）
        const nonSourceNodes = workflow.nodes.filter(node => node.type !== 'data_source');
        for (const node of nonSourceNodes) {
            const hasInput = workflow.connections.some(conn => conn.targetId === node.id);
            if (!hasInput) {
                return { valid: false, message: `节点 ${node.title} 没有输入连接` };
            }
        }
        
        // 检查是否存在没有输出的节点（除了可视化类型）
        const nonVisNodes = workflow.nodes.filter(node => node.type !== 'visualization');
        for (const node of nonVisNodes) {
            const hasOutput = workflow.connections.some(conn => conn.sourceId === node.id);
            if (!hasOutput) {
                return { valid: false, message: `节点 ${node.title} 没有输出连接` };
            }
        }
        
        return { valid: true };
    }
    
    // 初始化撤销/重做按钮
    function initializeToolbarButtons() {
        const undoBtn = document.createElement('button');
        undoBtn.id = 'undoBtn';
        undoBtn.className = 'btn btn-sm btn-outline-secondary toolbar-btn';
        undoBtn.title = '撤销';
        undoBtn.innerHTML = '<i class="fas fa-undo"></i>';
        undoBtn.disabled = true;
        undoBtn.addEventListener('click', undo);
        
        const redoBtn = document.createElement('button');
        redoBtn.id = 'redoBtn';
        redoBtn.className = 'btn btn-sm btn-outline-secondary toolbar-btn';
        redoBtn.title = '重做';
        redoBtn.innerHTML = '<i class="fas fa-redo"></i>';
        redoBtn.disabled = true;
        redoBtn.addEventListener('click', redo);
        
        // 查找工具栏容器
        const toolbarContainer = document.getElementById('workflowToolbar');
        if (toolbarContainer) {
            toolbarContainer.prepend(redoBtn);
            toolbarContainer.prepend(undoBtn);
        } else {
            // 如果没有找到工具栏，创建一个浮动工具栏
            const floatingToolbar = document.createElement('div');
            floatingToolbar.id = 'workflowToolbar';
            floatingToolbar.className = 'workflow-floating-toolbar';
            floatingToolbar.appendChild(undoBtn);
            floatingToolbar.appendChild(redoBtn);
            
            document.body.appendChild(floatingToolbar);
        }
    }
    
    // 改进显示示例模板逻辑
    function showSampleTemplates() {
        // 使用示例模板数据
        const sampleTemplates = [
            {
                id: 'template_1',
                name: '微博热搜分析模板',
                description: '爬取微博热搜榜数据，分析热点话题和情感倾向',
                icon: 'fire'
            },
            {
                id: 'template_2',
                name: '用户评论情感分析',
                description: '分析用户评论的情感倾向，生成情感分布图表',
                icon: 'heart'
            },
            {
                id: 'template_3',
                name: '话题趋势监测',
                description: '监测特定话题的讨论热度变化及关键词提取',
                icon: 'chart-line'
            },
            {
                id: 'template_4',
                name: '舆情预警分析',
                description: '实时监测并预警负面舆情，生成应对建议',
                icon: 'bell'
            }
        ];
    
        try {
            // 尝试寻找合适的容器
            const containers = [
                document.getElementById('analysisTemplatesList'),
                document.getElementById('templateList'),
                document.getElementById('crawlerTemplatesList'),
                document.querySelector('.templates-container')
            ];
            
            const container = containers.find(el => el !== null);
            
            if (container) {
                container.innerHTML = '';
                sampleTemplates.forEach(template => {
                    const templateDiv = createTemplateCard(template);
                    container.appendChild(templateDiv);
                });
            } else {
                console.warn('未找到合适的模板容器');
                
                // 如果找不到容器，尝试创建一个模板区域
                const templatesPanel = document.getElementById('templatesPanel');
                if (templatesPanel) {
                    const newContainer = document.createElement('div');
                    newContainer.className = 'templates-container';
                    templatesPanel.appendChild(newContainer);
                    
                    sampleTemplates.forEach(template => {
                        const templateDiv = createTemplateCard(template);
                        newContainer.appendChild(templateDiv);
                    });
                }
            }
        } catch (error) {
            console.error('加载模板出错:', error);
        }
    }
    
    // 添加键盘快捷键支持
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', function(e) {
            // Ctrl+Z: 撤销
            if (e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                undo();
            }
            
            // Ctrl+Y: 重做
            if (e.ctrlKey && e.key === 'y') {
                e.preventDefault();
                redo();
            }
            
            // Ctrl+S: 保存
            if (e.ctrlKey && e.key === 's') {
                e.preventDefault();
                saveWorkflow(workflowData);
            }
            
            // Delete: 删除选中的节点
            if (e.key === 'Delete') {
                const selectedNode = document.querySelector('.workflow-node.selected');
                if (selectedNode) {
                    deleteNode(selectedNode.id);
                    addToHistory();
                }
            }
            
            // Escape: 取消连接或关闭配置面板
            if (e.key === 'Escape') {
                if (isConnecting) {
                    cancelConnection();
                } else if (document.getElementById('propertiesPanel').classList.contains('open')) {
                    closePropertiesPanel();
                }
            }
        });
    }
    
    // 添加节点选择功能
    function setupNodeSelection() {
        workflowCanvas.addEventListener('click', function(e) {
            if (e.target === workflowCanvas || e.target === connectionsSvg) {
                // 如果点击的是画布本身，清除所有选中
                clearNodeSelection();
            }
        });
    }
    
    // 清除节点选择
    function clearNodeSelection() {
        document.querySelectorAll('.workflow-node.selected').forEach(node => {
            node.classList.remove('selected');
        });
    }
    
    // 为节点添加选择功能
    function setupNodeSelectEvents(nodeElement) {
        nodeElement.addEventListener('click', function(e) {
            // 如果没有按下Ctrl键，先清除其他节点的选择
            if (!e.ctrlKey) {
                clearNodeSelection();
            }
            
            // 选择当前节点
            nodeElement.classList.add('selected');
            e.stopPropagation();
        });
    }
    
    // 启用键盘快捷键和节点选择功能
    setupKeyboardShortcuts();
    setupNodeSelection();
    
    // 初始化
    initializeToolbarButtons();
    // 初始化：将当前状态添加到历史记录
    addToHistory();
    
    // 添加导出/导入功能
    function setupExportImport() {
        const exportBtn = document.createElement('button');
        exportBtn.id = 'exportWorkflowBtn';
        exportBtn.className = 'btn btn-sm btn-outline-secondary toolbar-btn';
        exportBtn.title = '导出工作流';
        exportBtn.innerHTML = '<i class="fas fa-file-export"></i>';
        exportBtn.addEventListener('click', exportWorkflow);
        
        const importBtn = document.createElement('button');
        importBtn.id = 'importWorkflowBtn';
        importBtn.className = 'btn btn-sm btn-outline-secondary toolbar-btn';
        importBtn.title = '导入工作流';
        importBtn.innerHTML = '<i class="fas fa-file-import"></i>';
        importBtn.addEventListener('click', importWorkflow);
        
        // 添加到工具栏
        const toolbarContainer = document.getElementById('workflowToolbar');
        if (toolbarContainer) {
            toolbarContainer.appendChild(exportBtn);
            toolbarContainer.appendChild(importBtn);
        }
    }
    
    function exportWorkflow() {
        // 创建下载内容
        const workflowJson = JSON.stringify(workflowData, null, 2);
        const blob = new Blob([workflowJson], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        
        // 创建下载链接
        const a = document.createElement('a');
        a.href = url;
        a.download = `${workflowData.metadata.name || 'workflow'}_${new Date().toISOString().slice(0,10)}.json`;
        document.body.appendChild(a);
        a.click();
        
        // 清理
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 0);
        
        showNotification('成功', '工作流已导出为JSON文件', 'success');
    }
    
    function importWorkflow() {
        // 创建文件输入元素
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'application/json';
        fileInput.style.display = 'none';
        
        fileInput.addEventListener('change', function(e) {
            if (!e.target.files.length) return;
            
            const file = e.target.files[0];
            const reader = new FileReader();
            
            reader.onload = function(event) {
                try {
                    const importedWorkflow = JSON.parse(event.target.result);
                    
                    // 验证导入的数据
                    if (!importedWorkflow.nodes || !importedWorkflow.metadata) {
                        throw new Error('无效的工作流文件格式');
                    }
                    
                    // 提示用户确认
                    if (confirm('确定要导入此工作流？这将替换当前的工作流。')) {
                        clearWorkflow();
                        renderWorkflow(importedWorkflow);
                        addToHistory();
                        showNotification('成功', '工作流已导入', 'success');
                    }
                } catch (err) {
                    console.error('导入工作流出错:', err);
                    showNotification('错误', '导入失败：无效的工作流文件', 'error');
                }
            };
            
            reader.readAsText(file);
        });
        
        document.body.appendChild(fileInput);
        fileInput.click();
        
        // 清理
        setTimeout(() => {
            document.body.removeChild(fileInput);
        }, 0);
    }
    
    // 设置导出/导入功能
    setupExportImport();
    
    // ====== 历史记录管理 ======
    function addToHistory() {
        // 如果当前不是历史的最后一步，截断历史
        if (currentHistoryIndex < history.length - 1) {
            history = history.slice(0, currentHistoryIndex + 1);
        }
        
        // 深拷贝当前状态
        const stateCopy = JSON.parse(JSON.stringify(workflowData));
        history.push(stateCopy);
        
        // 限制历史记录大小
        if (history.length > MAX_HISTORY) {
            history.shift();
        } else {
            currentHistoryIndex++;
        }
        
        // 更新撤销/重做按钮状态
        updateHistoryButtonStates();
    }
    
    function undo() {
        if (currentHistoryIndex > 0) {
            currentHistoryIndex--;
            restoreFromHistory();
            showNotification('撤销', '已撤销上一步操作', 'info');
        }
    }
    
    function redo() {
        if (currentHistoryIndex < history.length - 1) {
            currentHistoryIndex++;
            restoreFromHistory();
            showNotification('重做', '已重做操作', 'info');
        }
    }
    
    function restoreFromHistory() {
        // 从历史记录恢复工作流状态
        const historicalState = history[currentHistoryIndex];
        
        // 清除画布
        clearWorkflow();
        
        // 恢复状态
        workflowData = JSON.parse(JSON.stringify(historicalState));
        renderWorkflow(workflowData);
        
        // 更新按钮状态
        updateHistoryButtonStates();
    }
    
    function updateHistoryButtonStates() {
        const undoBtn = document.getElementById('undoBtn');
        const redoBtn = document.getElementById('redoBtn');
        
        if (undoBtn) {
            undoBtn.disabled = currentHistoryIndex <= 0;
        }
        
        if (redoBtn) {
            redoBtn.disabled = currentHistoryIndex >= history.length - 1;
        }
    }

    // ====== 通知系统 ======
    function showNotification(title, message, type = 'info') {
        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = `workflow-notification notification-${type}`;
        
        notification.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas ${getNotificationIcon(type)} me-2"></i>
                <div>
                    <div class="fw-bold">${title}</div>
                    <div class="small">${message}</div>
                </div>
                <button type="button" class="btn-close ms-3" aria-label="关闭"></button>
            </div>
        `;
        
        // 添加到通知容器
        const container = document.getElementById('notificationContainer');
        if (!container) {
            // 如果容器不存在，创建一个
            const notificationContainer = document.createElement('div');
            notificationContainer.id = 'notificationContainer';
            notificationContainer.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1050;
                max-width: 350px;
            `;
            document.body.appendChild(notificationContainer);
            notificationContainer.appendChild(notification);
        } else {
            container.appendChild(notification);
        }
        
        // 显示通知
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // 绑定关闭按钮事件
        const closeBtn = notification.querySelector('.btn-close');
        closeBtn.addEventListener('click', () => {
            hideNotification(notification);
        });
        
        // 设置自动隐藏
        setTimeout(() => {
            hideNotification(notification);
        }, 5000);
    }
    
    function hideNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
    
    function getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'fa-check-circle';
            case 'warning': return 'fa-exclamation-triangle';
            case 'error': return 'fa-times-circle';
            case 'info':
            default: return 'fa-info-circle';
        }
    }
    
    // 优化工作流验证功能
    function validateWorkflow(workflow) {
        // 检查是否有节点
        if (!workflow.nodes || workflow.nodes.length === 0) {
            return { valid: false, message: '工作流没有节点' };
        }
        
        // 检查是否有连接
        if (!workflow.connections || workflow.connections.length === 0) {
            return { valid: false, message: '工作流没有连接' };
        }
        
        // 检查是否存在没有配置的节点
        const unconfiguredNodes = workflow.nodes.filter(node => !node.config);
        if (unconfiguredNodes.length > 0) {
            const nodeTitles = unconfiguredNodes.map(n => n.title).join(', ');
            return { valid: false, message: `以下节点未配置: ${nodeTitles}` };
        }
        
        // 检查是否存在没有输入的节点（除了数据源类型）
        const nonSourceNodes = workflow.nodes.filter(node => node.type !== 'data_source');
        for (const node of nonSourceNodes) {
            const hasInput = workflow.connections.some(conn => conn.targetId === node.id);
            if (!hasInput) {
                return { valid: false, message: `节点 "${node.title}" 没有输入连接` };
            }
        }
        
        // 检查是否存在没有输出的节点（除了可视化类型和预测类型的某些子类型）
        const nonOutputNodeTypes = ['visualization'];
        const nonOutputNodeSubtypes = {
            'prediction': ['report', 'alert'] // 这些预测子类型不需要输出
        };
        
        const shouldHaveOutput = node => {
            if (nonOutputNodeTypes.includes(node.type)) return false;
            return !(nonOutputNodeSubtypes[node.type] && 
                   nonOutputNodeSubtypes[node.type].includes(node.subtype));
        };
        
        const nonVisNodes = workflow.nodes.filter(shouldHaveOutput);
        
        for (const node of nonVisNodes) {
            const hasOutput = workflow.connections.some(conn => conn.sourceId === node.id);
            if (!hasOutput) {
                return { valid: false, message: `节点 "${node.title}" 没有输出连接` };
            }
        }
        
        // 检查是否有环
        const nodeIds = workflow.nodes.map(node => node.id);
        for (const nodeId of nodeIds) {
            if (hasCycle(nodeId, new Set(), workflow.connections)) {
                return { valid: false, message: '工作流中存在循环连接' };
            }
        }
        
        // 检查是否有悬空连接（连接指向不存在的节点）
        for (const conn of workflow.connections) {
            if (!workflow.nodes.some(node => node.id === conn.sourceId)) {
                return { valid: false, message: `存在连接指向不存在的源节点ID: ${conn.sourceId}` };
            }
            if (!workflow.nodes.some(node => node.id === conn.targetId)) {
                return { valid: false, message: `存在连接指向不存在的目标节点ID: ${conn.targetId}` };
            }
        }
        
        return { valid: true };
    }
    
    // 检查是否有环
    function hasCycle(currentId, visited, connections) {
        if (visited.has(currentId)) {
            return true; // 发现环
        }
        
        visited.add(currentId);
        
        // 获取从currentId出发的所有连接
        const outgoingConnections = connections.filter(conn => conn.sourceId === currentId);
        
        for (const conn of outgoingConnections) {
            const nextId = conn.targetId;
            
            // 创建一个新的已访问集合副本
            const newVisited = new Set(visited);
            if (hasCycle(nextId, newVisited, connections)) {
                return true;
            }
        }
        
        return false;
    }
    
    // 验证工作流按钮事件
    document.getElementById('validateWorkflowBtn')?.addEventListener('click', function() {
        const result = validateWorkflow(workflowData);
        if (result.valid) {
            showNotification('验证通过', '工作流有效，可以运行', 'success');
        } else {
            showNotification('验证失败', result.message, 'error');
        }
    });
    
    // 工作流工具栏事件绑定
    function bindToolbarEvents() {
        // 撤销/重做
        document.getElementById('undoBtn')?.addEventListener('click', undo);
        document.getElementById('redoBtn')?.addEventListener('click', redo);
        
        // 缩放控制
        document.getElementById('zoomInBtn')?.addEventListener('click', () => {
            zoomCanvas(0.1);
        });
        
        document.getElementById('zoomOutBtn')?.addEventListener('click', () => {
            zoomCanvas(-0.1);
        });
        
        document.getElementById('fitViewBtn')?.addEventListener('click', fitCanvasView);
    }
    
    // 画布缩放功能
    function zoomCanvas(delta) {
        let newScale = canvasScale + delta;
        
        // 限制缩放范围
        newScale = Math.max(0.5, Math.min(2, newScale));
        
        if (newScale !== canvasScale) {
            canvasScale = newScale;
            applyCanvasTransform();
            
            // 更新连接线
            workflowData.connections.forEach(conn => {
                const path = document.getElementById('connection_' + conn.id);
                if (path) {
                    path.parentNode.removeChild(path);
                }
                drawConnection(conn.sourceId, conn.targetId, conn.id);
            });
            
            // 显示当前缩放比例
            showNotification('视图', `缩放比例: ${Math.round(canvasScale * 100)}%`, 'info');
        }
    }
    
    // 适应视图
    function fitCanvasView() {
        if (workflowData.nodes.length === 0) {
            return; // 没有节点，不需要调整
        }
        
        // 重置缩放和平移
        canvasScale = 1;
        canvasTranslate = { x: 0, y: 0 };
        applyCanvasTransform();
        
        // 重新绘制所有连接
        workflowData.connections.forEach(conn => {
            const path = document.getElementById('connection_' + conn.id);
            if (path) {
                path.parentNode.removeChild(path);
            }
            drawConnection(conn.sourceId, conn.targetId, conn.id);
        });
        
        showNotification('视图', '已重置视图', 'info');
    }
    
    // 应用画布变换
    function applyCanvasTransform() {
        const transform = `scale(${canvasScale}) translate(${canvasTranslate.x}px, ${canvasTranslate.y}px)`;
        workflowCanvas.style.transform = transform;
    }
    
    // 更新工作流状态信息
    function updateWorkflowStatus() {
        const nodeCount = document.getElementById('nodeCount');
        const connectionCount = document.getElementById('connectionCount');
        const statusBar = document.getElementById('workflowStatusBar');
        
        if (nodeCount) nodeCount.textContent = workflowData.nodes.length;
        if (connectionCount) connectionCount.textContent = workflowData.connections.length;
        
        if (statusBar) {
            // 根据工作流状态更新状态栏
            if (workflowData.nodes.length === 0) {
                statusBar.style.display = 'flex';
                statusBar.querySelector('#workflowStatusMessage').textContent = 
                    '工作流就绪。拖拽左侧组件到画布创建节点。';
            } else if (workflowData.connections.length === 0) {
                statusBar.style.display = 'flex';
                statusBar.querySelector('#workflowStatusMessage').textContent = 
                    '已添加节点。请连接节点以创建完整工作流。';
            } else {
                const validationResult = validateWorkflow(workflowData);
                if (!validationResult.valid) {
                    statusBar.style.display = 'flex';
                    statusBar.querySelector('#workflowStatusMessage').textContent = 
                        `工作流需要修正: ${validationResult.message}`;
                    statusBar.classList.add('bg-warning-subtle');
                    statusBar.classList.remove('bg-light', 'bg-success-subtle');
                } else {
                    statusBar.style.display = 'flex';
                    statusBar.querySelector('#workflowStatusMessage').textContent = 
                        '工作流有效，可以运行。';
                    statusBar.classList.add('bg-success-subtle');
                    statusBar.classList.remove('bg-light', 'bg-warning-subtle');
                }
            }
        }
    }
    
    // 每次工作流变化时更新状态
    function onWorkflowChanged() {
        updateWorkflowStatus();
        addToHistory();
    }
    
    // 增强添加节点函数
    function addNode(componentType, componentSubtype, x, y) {
        const nodeId = 'node_' + Date.now();
        const nodeData = {
            id: nodeId,
            type: componentType,
            subtype: componentSubtype,
            title: getComponentTypeLabel(componentType) + '-' + componentSubtype,
            x: x,
            y: y,
            config: getDefaultConfig(componentType, componentSubtype)
        };
        
        const nodeElement = createNodeFromData(nodeData);
        setupNodeEvents(nodeElement, nodeData);
        
        // 更新工作流状态
        onWorkflowChanged();
        
        return nodeElement;
    }
    
    // 增强删除节点函数
    function deleteNode(nodeId) {
        const node = document.getElementById(nodeId);
        if (node) {
            node.parentNode.removeChild(node);
        }
        
        // 删除相关连接
        workflowData.connections = workflowData.connections.filter(conn => {
            if (conn.sourceId === nodeId || conn.targetId === nodeId) {
                const path = document.getElementById('connection_' + conn.id);
                if (path) {
                    path.parentNode.removeChild(path);
                }
                return false;
            }
            return true;
        });
        
        // 从数据中删除节点
        workflowData.nodes = workflowData.nodes.filter(node => node.id !== nodeId);
        
        // 更新工作流状态
        onWorkflowChanged();
        
        showNotification('已删除', '节点已从工作流中移除', 'info');
    }
    
    // 增强完成连接函数
    function completeConnection(sourceId, targetId) {
        // 检查是否是自连接
        if (sourceId === targetId) {
            showNotification('警告', '不能连接到自己', 'warning');
            cancelConnection();
            return;
        }
        
        // 检查连接是否已存在
        const connectionExists = workflowData.connections.some(conn => 
            conn.sourceId === sourceId && conn.targetId === targetId);
            
        if (connectionExists) {
            showNotification('警告', '连接已存在', 'warning');
            cancelConnection();
            return;
        }
        
        // 检查是否会形成循环
        if (wouldCreateCycle(sourceId, targetId)) {
            showNotification('错误', '不能创建循环连接', 'error');
            cancelConnection();
            return;
        }
        
        // 生成连接ID
        const connectionId = `conn_${Date.now()}`;
        
        // 添加到数据中
        workflowData.connections.push({
            id: connectionId,
            sourceId: sourceId,
            targetId: targetId
        });
        
        // 绘制最终连接
        drawConnection(sourceId, targetId, connectionId);
        
        // 清理预览状态
        cancelConnection();
        
        // 更新工作流状态
        onWorkflowChanged();
        
        // 显示成功通知
        showNotification('成功', '已创建连接', 'success');
    }
    
    // 绑定工具栏事件
    bindToolbarEvents();
    
    // 初始化：将当前状态添加到历史记录
    addToHistory();
    
    // 初始更新工作流状态
    updateWorkflowStatus();
});
