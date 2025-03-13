document.addEventListener('DOMContentLoaded', function() {
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
        // 从数据创建节点DOM元素
        const nodeElement = document.createElement('div');
        nodeElement.className = 'workflow-node';
        nodeElement.id = nodeData.id;
        nodeElement.style.left = nodeData.x + 'px';
        nodeElement.style.top = nodeData.y + 'px';
        
        // 构建节点内容
        nodeElement.innerHTML = `
            <div class="node-header">
                <span class="node-title">${nodeData.title}</span>
                <span class="node-type">${getComponentTypeLabel(nodeData.type)}</span>
            </div>
            <div class="node-content">
                <p class="node-description">${nodeData.config ? '已配置' : '点击配置参数'}</p>
            </div>
            <div class="node-ports">
                <div class="port port-in" data-port-type="input"></div>
                <div class="port port-out" data-port-type="output"></div>
            </div>
            <div class="node-actions mt-2">
                <button class="btn btn-sm btn-outline-danger delete-node-btn">
                    <i class="fas fa-trash-alt"></i>
                </button>
            </div>
        `;
        
        workflowCanvas.appendChild(nodeElement);
        
        // 添加到节点数据
        workflowData.nodes.push(nodeData);
        
        return nodeElement;
    }
    
    // ====== 运行工作流 ======
    document.getElementById('runWorkflowBtn').addEventListener('click', function() {
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
    });

    workflowCanvas.addEventListener('drop', function(e) {
        e.preventDefault();
        const componentType = e.dataTransfer.getData('componentType');
        const componentSubtype = e.dataTransfer.getData('componentSubtype');
        
        if (componentType && componentSubtype) {
            const rect = workflowCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            addNode(componentType, componentSubtype, x, y);
        }
    });

    // 添加其他初始化代码
    document.addEventListener('DOMContentLoaded', function() {
        initializeWorkflowEditor();
        setupEventListeners();
        showSampleTemplates();
    });

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
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
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
        
        // 绘制预览连接
        const dx = Math.abs(end.x - start.x) * 0.5;
        const pathData = `M ${start.x},${start.y} C ${start.x + dx},${start.y} ${end.x - dx},${end.y} ${end.x},${end.y}`;
        connectionPreviewPath.setAttribute('d', pathData);
    }
    
    // 完成连接
    function completeConnection(sourceId, targetId) {
        // 检查连接是否已存在
        const connectionExists = workflowData.connections.some(conn => 
            conn.sourceId === sourceId && conn.targetId === targetId);
            
        if (connectionExists) {
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
    }
    
    // 取消连接操作
    function cancelConnection() {
        if (connectionPreviewPath && connectionPreviewPath.parentNode) {
            connectionPreviewPath.parentNode.removeChild(connectionPreviewPath);
        }
        
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
            <h6 class="mb-3">${nodeData.title} 配置</h6>
            <form id="nodeConfigForm">
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
            } else {
                formHtml += `<input type="${option.type}" class="form-control" id="${option.id}" name="${option.id}" value="${value}">`;
            }
            
            formHtml += `</div>`;
        });
        
        formHtml += `
            <div class="d-flex justify-content-end">
                <button type="button" class="btn btn-primary" id="saveConfigBtn">应用</button>
            </div>
            </form>
        `;
        
        propertiesContent.innerHTML = formHtml;
        
        // 保存配置事件
        document.getElementById('saveConfigBtn').addEventListener('click', function() {
            const form = document.getElementById('nodeConfigForm');
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
                    }
                }
            }
            
            // 关闭面板
            closePropertiesPanel();
        });
    }
    
    // 关闭属性面板
    function closePropertiesPanel() {
        document.getElementById('propertiesPanel').classList.remove('open');
    }
    
    // 绑定关闭属性面板的事件
    document.getElementById('closePropertiesBtn').addEventListener('click', closePropertiesPanel);
});
