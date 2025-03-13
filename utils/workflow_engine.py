#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import uuid
import logging
import traceback
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from utils.db_manager import DatabaseManager
from utils.cache_manager import CacheManager
from utils.model_router import ModelRouter
from utils.sensitive_filter import SensitiveDataFilter
from spider.weibo_crawler import WeiboCrawler
from utils.ai_analyzer import AIAnalyzer

# 配置日志
from utils.logger import setup_logger
logger = setup_logger('workflow_engine', 'logs/workflow_engine.log')

class WorkflowEngine:
    """工作流引擎 - 负责执行数据爬取和分析工作流"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WorkflowEngine, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.db = DatabaseManager()
        self.cache = CacheManager(memory_capacity=50, cache_duration=3600)
        self.model_router = ModelRouter()
        self.sensitive_filter = SensitiveDataFilter()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.running_tasks = {}
        
        # 创建必要的目录
        self.data_dir = Path('data/workflow')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
        logger.info("工作流引擎初始化完成")
    
    def execute_crawler_workflow(self, task_id, config):
        """
        执行爬虫工作流
        
        Args:
            task_id: 任务ID
            config: 爬虫配置
        """
        logger.info(f"开始执行爬虫工作流: {task_id}")
        
        try:
            # 更新任务状态为运行中
            self._update_task_status(task_id, 'running', 0)
            
            # 创建爬虫实例
            crawler = WeiboCrawler()
            
            # 设置爬虫参数
            source = config.get('source', 'hot_topics')
            depth = config.get('crawl_depth', 1)
            interval = config.get('interval', 5)
            filters = config.get('filters', {})
            
            # 执行爬取
            result = crawler.crawl(
                source=source,
                depth=depth,
                interval=interval,
                filters=filters,
                callback=lambda progress: self._update_task_progress(task_id, progress)
            )
            
            # 更新任务状态为已完成
            self._update_task_status(task_id, 'completed', 100, result=result)
            logger.info(f"爬虫工作流完成: {task_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"爬虫工作流出错: {str(e)}")
            logger.error(traceback.format_exc())
            self._update_task_status(task_id, 'failed', 0, error=str(e))
            return None
    
    def execute_analysis_workflow(self, task_id, workflow):
        """
        执行分析工作流
        
        Args:
            task_id: 任务ID
            workflow: 工作流配置
        """
        logger.info(f"开始执行分析工作流: {task_id}")
        
        try:
            # 更新任务状态为运行中
            self._update_task_status(task_id, 'running', 0)
            
            components = workflow.get('components', [])
            connections = workflow.get('connections', [])
            
            # 验证工作流
            if not components or not connections:
                raise ValueError("工作流配置不完整，缺少组件或连接")
                
            # 构建组件依赖图
            component_map, dependency_graph = self._build_dependency_graph(components, connections)
            
            # 进行拓扑排序
            execution_order = self._topological_sort(dependency_graph)
            
            # 执行组件
            result_map = {}
            total_components = len(execution_order)
            
            for idx, component_id in enumerate(execution_order):
                component = component_map.get(component_id)
                if not component:
                    continue
                    
                # 计算总体进度
                progress = int((idx / total_components) * 100)
                self._update_task_progress(task_id, progress)
                
                # 获取输入数据
                input_data = self._get_component_input_data(component_id, connections, result_map)
                
                # 执行组件
                result = self._execute_component(component, input_data)
                
                # 存储结果
                result_map[component_id] = result
            
            # 获取最终输出
            final_outputs = self._get_final_outputs(dependency_graph, result_map)
            
            # 应用敏感信息过滤
            if final_outputs and self.sensitive_filter.is_enabled():
                if isinstance(final_outputs, dict):
                    final_outputs = self.sensitive_filter.filter_dict(final_outputs)
                elif isinstance(final_outputs, list):
                    final_outputs = self.sensitive_filter.filter_list(final_outputs)
                    
            # 更新任务状态为已完成
            self._update_task_status(task_id, 'completed', 100, result=final_outputs)
            logger.info(f"分析工作流完成: {task_id}")
            
            return final_outputs
            
        except Exception as e:
            logger.error(f"分析工作流出错: {str(e)}")
            logger.error(traceback.format_exc())
            self._update_task_status(task_id, 'failed', 0, error=str(e))
            return None
    
    def start_workflow(self, workflow_type, config, template_id=None):
        """
        异步启动工作流
        
        Args:
            workflow_type: 工作流类型 (crawler/analysis)
            config: 工作流配置
            template_id: 关联的模板ID
            
        Returns:
            task_id: 工作流任务ID
        """
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 保存任务信息到数据库
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                """
                INSERT INTO workflow_tasks 
                (id, template_id, type, status, progress, config, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    task_id, 
                    template_id, 
                    workflow_type, 
                    'pending', 
                    0, 
                    json.dumps(config, ensure_ascii=False), 
                    now, 
                    now
                )
            )
            conn.commit()
            
            # 异步执行工作流
            if workflow_type == 'crawler':
                self.running_tasks[task_id] = self.executor.submit(
                    self.execute_crawler_workflow, task_id, config
                )
            elif workflow_type == 'analysis':
                self.running_tasks[task_id] = self.executor.submit(
                    self.execute_analysis_workflow, task_id, config
                )
            else:
                logger.error(f"未知的工作流类型: {workflow_type}")
                return None
            
            return task_id
            
        except Exception as e:
            logger.error(f"启动工作流失败: {str(e)}")
            conn.rollback()
            return None
        finally:
            cursor.close()
    
    def get_task_status(self, task_id):
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            task: 任务信息
        """
        # 先检查缓存
        cache_key = f"task_status:{task_id}"
        cached_task = self.cache.get(cache_key)
        if cached_task:
            return cached_task
            
        # 从数据库获取
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT * FROM workflow_tasks WHERE id = %s",
                (task_id,)
            )
            task = cursor.fetchone()
            
            if task:
                # 将JSON字符串转为Python对象
                if task.get('config'):
                    task['config'] = json.loads(task['config'])
                if task.get('result'):
                    task['result'] = json.loads(task['result'])
                    
                # 缓存结果
                self.cache.set(cache_key, task)
                
            return task
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {str(e)}")
            return None
        finally:
            cursor.close()
    
    def cancel_task(self, task_id):
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            success: 是否成功
        """
        # 检查任务是否存在并正在运行
        if task_id in self.running_tasks:
            # 尝试取消任务
            future = self.running_tasks[task_id]
            if not future.done():
                future.cancel()
            
            # 从运行列表中移除
            del self.running_tasks[task_id]
        
        # 更新数据库状态
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                """
                UPDATE workflow_tasks 
                SET status = %s, updated_at = %s
                WHERE id = %s
                """,
                ('cancelled', now, task_id)
            )
            conn.commit()
            
            # 清理缓存
            cache_key = f"task_status:{task_id}"
            self.cache.delete(cache_key)
            
            return True
            
        except Exception as e:
            logger.error(f"取消任务失败: {str(e)}")
            conn.rollback()
            return False
        finally:
            cursor.close()
    
    def _update_task_status(self, task_id, status, progress, result=None, error=None):
        """更新任务状态"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            update_fields = ["status = %s", "progress = %s", "updated_at = %s"]
            params = [status, progress, now]
            
            # 添加开始时间
            if status == 'running' and progress == 0:
                update_fields.append("started_at = %s")
                params.append(now)
            
            # 添加完成时间
            if status in ['completed', 'failed']:
                update_fields.append("completed_at = %s")
                params.append(now)
            
            # 添加结果
            if result is not None:
                update_fields.append("result = %s")
                params.append(json.dumps(result, ensure_ascii=False))
            
            # 添加错误
            if error is not None:
                update_fields.append("error = %s")
                params.append(error)
            
            # 构建SQL
            sql = f"""
                UPDATE workflow_tasks 
                SET {', '.join(update_fields)}
                WHERE id = %s
            """
            params.append(task_id)
            
            cursor.execute(sql, tuple(params))
            conn.commit()
            
            # 清理缓存
            cache_key = f"task_status:{task_id}"
            self.cache.delete(cache_key)
            
        except Exception as e:
            logger.error(f"更新任务状态失败: {str(e)}")
            conn.rollback()
        finally:
            cursor.close()
    
    def _update_task_progress(self, task_id, progress):
        """更新任务进度"""
        self._update_task_status(task_id, 'running', progress)
    
    def _build_dependency_graph(self, components, connections):
        """构建组件依赖图"""
        component_map = {comp['id']: comp for comp in components}
        dependency_graph = {comp['id']: [] for comp in components}
        
        # 构建依赖关系
        for conn in connections:
            source = conn.get('source')
            target = conn.get('target')
            
            if source and target and source in component_map and target in component_map:
                dependency_graph[target].append(source)
        
        return component_map, dependency_graph
    
    def _topological_sort(self, graph):
        """拓扑排序，确定组件执行顺序"""
        visited = set()
        temp = set()
        order = []
        
        def visit(node):
            if node in temp:
                raise ValueError(f"工作流存在循环依赖: {node}")
            if node in visited:
                return
                
            temp.add(node)
            for neighbor in graph.get(node, []):
                visit(neighbor)
                
            temp.remove(node)
            visited.add(node)
            order.append(node)
        
        for node in graph:
            if node not in visited:
                visit(node)
                
        return list(reversed(order))
    
    def _get_component_input_data(self, component_id, connections, result_map):
        """获取组件的输入数据"""
        input_data = {}
        
        for conn in connections:
            if conn.get('target') == component_id:
                source_id = conn.get('source')
                if source_id in result_map:
                    input_name = conn.get('targetInput', 'default')
                    input_data[input_name] = result_map[source_id]
        
        return input_data
    
    def _execute_component(self, component, input_data):
        """执行单个组件"""
        component_type = component.get('type')
        config = component.get('config', {})
        
        if component_type == 'data_source':
            return self._execute_data_source(config, input_data)
        elif component_type == 'preprocessing':
            return self._execute_preprocessing(config, input_data)
        elif component_type == 'model':
            return self._execute_model(config, input_data)
        elif component_type == 'visualization':
            return self._execute_visualization(config, input_data)
        else:
            logger.warning(f"未知的组件类型: {component_type}")
            return None
    
    def _execute_data_source(self, config, input_data):
        """执行数据源组件"""
        source_type = config.get('source_type')
        
        if source_type == 'database':
            # 从数据库获取数据
            table = config.get('table')
            filters = config.get('filters', {})
            limit = config.get('limit', 1000)
            
            query_conditions = []
            query_params = []
            
            for key, value in filters.items():
                if value:
                    query_conditions.append(f"{key} = %s")
                    query_params.append(value)
            
            where_clause = f"WHERE {' AND '.join(query_conditions)}" if query_conditions else ""
            
            sql = f"SELECT * FROM {table} {where_clause} LIMIT {limit}"
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute(sql, tuple(query_params))
                return cursor.fetchall()
            except Exception as e:
                logger.error(f"数据库查询出错: {str(e)}")
                return []
            finally:
                cursor.close()
                
        elif source_type == 'file':
            # 从文件加载数据
            file_path = config.get('file_path')
            if not file_path or not os.path.exists(file_path):
                return []
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.json'):
                        return json.load(f)
                    else:
                        return f.read()
            except Exception as e:
                logger.error(f"文件读取出错: {str(e)}")
                return []
                
        elif source_type == 'api':
            # 这里需要实现API调用逻辑
            # 由于涉及复杂的HTTP请求，暂不实现
            logger.warning("API数据源暂未实现")
            return []
            
        else:
            logger.warning(f"未知的数据源类型: {source_type}")
            return []
    
    def _execute_preprocessing(self, config, input_data):
        """执行数据预处理组件"""
        preprocessing_type = config.get('preprocessing_type')
        data = input_data.get('default', [])
        
        if not data:
            return []
            
        if preprocessing_type == 'filter':
            # 过滤数据
            field = config.get('field')
            value = config.get('value')
            operator = config.get('operator', 'eq')
            
            if not field:
                return data
                
            result = []
            for item in data:
                if operator == 'eq' and item.get(field) == value:
                    result.append(item)
                elif operator == 'neq' and item.get(field) != value:
                    result.append(item)
                elif operator == 'contains' and value in str(item.get(field, '')):
                    result.append(item)
                elif operator == 'not_contains' and value not in str(item.get(field, '')):
                    result.append(item)
            
            return result
            
        elif preprocessing_type == 'sort':
            # 排序数据
            field = config.get('field')
            order = config.get('order', 'asc')
            
            if not field:
                return data
                
            return sorted(
                data, 
                key=lambda x: x.get(field, ''), 
                reverse=(order == 'desc')
            )
            
        elif preprocessing_type == 'aggregate':
            # 聚合数据
            group_by = config.get('group_by')
            aggregate_field = config.get('aggregate_field')
            aggregate_type = config.get('aggregate_type', 'count')
            
            if not group_by:
                return data
                
            result = {}
            for item in data:
                key = item.get(group_by)
                if key not in result:
                    result[key] = {
                        'count': 0,
                        'sum': 0,
                        'values': []
                    }
                
                result[key]['count'] += 1
                
                if aggregate_field:
                    value = item.get(aggregate_field, 0)
                    if isinstance(value, (int, float)):
                        result[key]['sum'] += value
                        result[key]['values'].append(value)
            
            # 计算最终结果
            final_result = []
            for key, values in result.items():
                item = {group_by: key}
                
                if aggregate_type == 'count':
                    item['value'] = values['count']
                elif aggregate_type == 'sum':
                    item['value'] = values['sum']
                elif aggregate_type == 'avg':
                    item['value'] = values['sum'] / values['count'] if values['count'] > 0 else 0
                
                final_result.append(item)
            
            return final_result
            
        else:
            logger.warning(f"未知的预处理类型: {preprocessing_type}")
            return data
    
    def _execute_model(self, config, input_data):
        """执行模型组件"""
        model_type = config.get('model_type')
        data = input_data.get('default', [])
        
        if not data:
            return []
            
        analyzer = AIAnalyzer()
        
        if model_type == 'sentiment':
            # 情感分析
            texts = []
            if isinstance(data, list):
                # 如果是列表，从指定字段获取文本
                field = config.get('text_field', 'content')
                texts = [item.get(field, '') for item in data if item.get(field)]
            elif isinstance(data, str):
                # 如果是字符串，直接使用
                texts = [data]
                
            # 获取合适的模型
            model = self.model_router.select_model_for_text(texts[0] if texts else "", "sentiment")
            
            # 执行分析
            results = []
            for text in texts:
                result = analyzer.analyze_sentiment(text, model=model)
                results.append(result)
                
            # 如果输入是列表，将结果合并回原始数据
            if isinstance(data, list):
                field = config.get('text_field', 'content')
                for i, item in enumerate(data):
                    if i < len(results) and item.get(field):
                        item['sentiment'] = results[i]
                return data
            else:
                return results[0] if results else None
                
        elif model_type == 'topic':
            # 主题分类
            texts = []
            if isinstance(data, list):
                field = config.get('text_field', 'content')
                texts = [item.get(field, '') for item in data if item.get(field)]
            elif isinstance(data, str):
                texts = [data]
                
            # 获取合适的模型
            model = self.model_router.select_model_for_text(texts[0] if texts else "", "topic")
            
            # 执行分析
            results = []
            for text in texts:
                result = analyzer.analyze_topic(text, model=model)
                results.append(result)
                
            # 如果输入是列表，将结果合并回原始数据
            if isinstance(data, list):
                field = config.get('text_field', 'content')
                for i, item in enumerate(data):
                    if i < len(results) and item.get(field):
                        item['topic'] = results[i]
                return data
            else:
                return results[0] if results else None
                
        elif model_type == 'keywords':
            # 关键词提取
            texts = []
            if isinstance(data, list):
                field = config.get('text_field', 'content')
                texts = [item.get(field, '') for item in data if item.get(field)]
            elif isinstance(data, str):
                texts = [data]
                
            # 获取合适的模型
            model = self.model_router.select_model_for_text(texts[0] if texts else "", "keyword")
            
            # 执行分析
            results = []
            for text in texts:
                result = analyzer.extract_keywords(text, model=model)
                results.append(result)
                
            # 如果输入是列表，将结果合并回原始数据
            if isinstance(data, list):
                field = config.get('text_field', 'content')
                for i, item in enumerate(data):
                    if i < len(results) and item.get(field):
                        item['keywords'] = results[i]
                return data
            else:
                return results[0] if results else None
                
        elif model_type == 'summarize':
            # 文本摘要
            texts = []
            if isinstance(data, list):
                field = config.get('text_field', 'content')
                texts = [item.get(field, '') for item in data if item.get(field)]
            elif isinstance(data, str):
                texts = [data]
                
            # 获取合适的模型
            model = self.model_router.select_model_for_text(texts[0] if texts else "", "summarization")
            
            # 执行分析
            results = []
            for text in texts:
                result = analyzer.summarize_text(text, model=model)
                results.append(result)
                
            # 如果输入是列表，将结果合并回原始数据
            if isinstance(data, list):
                field = config.get('text_field', 'content')
                for i, item in enumerate(data):
                    if i < len(results) and item.get(field):
                        item['summary'] = results[i]
                return data
            else:
                return results[0] if results else None
                
        else:
            logger.warning(f"未知的模型类型: {model_type}")
            return data
    
    def _execute_visualization(self, config, input_data):
        """执行可视化组件"""
        visualization_type = config.get('visualization_type')
        data = input_data.get('default', [])
        
        if not data:
            return {}
            
        if visualization_type == 'chart':
            # 图表可视化
            chart_type = config.get('chart_type', 'bar')
            x_field = config.get('x_field')
            y_field = config.get('y_field')
            title = config.get('title', '数据可视化')
            
            if not x_field or not y_field:
                return {'error': '缺少x或y字段'}
                
            # 提取数据
            chart_data = {
                'type': chart_type,
                'title': title,
                'xAxis': {'type': 'category', 'data': []},
                'yAxis': {'type': 'value'},
                'series': [{'data': []}]
            }
            
            for item in data:
                x_value = item.get(x_field)
                y_value = item.get(y_field)
                
                if x_value is not None and y_value is not None:
                    chart_data['xAxis']['data'].append(x_value)
                    chart_data['series'][0]['data'].append(y_value)
            
            return chart_data
            
        elif visualization_type == 'table':
            # 表格可视化
            columns = config.get('columns', [])
            title = config.get('title', '数据表格')
            
            # 如果没有指定列，使用数据中的所有字段
            if not columns and isinstance(data, list) and data:
                columns = list(data[0].keys())
                
            # 构建表格数据
            table_data = {
                'type': 'table',
                'title': title,
                'columns': columns,
                'data': data
            }
            
            return table_data
            
        elif visualization_type == 'wordcloud':
            # 词云可视化
            word_field = config.get('word_field')
            value_field = config.get('value_field')
            title = config.get('title', '词云图')
            
            if not word_field:
                return {'error': '缺少词字段'}
                
            # 构建词云数据
            wordcloud_data = {
                'type': 'wordcloud',
                'title': title,
                'data': []
            }
            
            for item in data:
                word = item.get(word_field)
                value = item.get(value_field, 1)
                
                if word:
                    wordcloud_data['data'].append({
                        'name': word,
                        'value': value
                    })
            
            return wordcloud_data
            
        else:
            logger.warning(f"未知的可视化类型: {visualization_type}")
            return {}
    
    def _get_final_outputs(self, dependency_graph, result_map):
        """获取最终输出结果"""
        # 找出没有后继节点的叶子节点
        leaf_nodes = []
        all_targets = set()
        
        for node, deps in dependency_graph.items():
            all_targets.update(deps)
        
        for node in dependency_graph:
            if node not in all_targets:
                leaf_nodes.append(node)
        
        # 收集所有叶子节点的结果
        outputs = {}
        for node in leaf_nodes:
            if node in result_map:
                outputs[node] = result_map[node]
        
        return outputs 