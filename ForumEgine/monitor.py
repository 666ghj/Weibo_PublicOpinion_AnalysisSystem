"""
日志监控器 - 实时监控三个log文件中的SummaryNode输出
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime
import re
import json
from typing import Dict, Optional, List
from threading import Lock

class LogMonitor:
    """基于文件变化的智能日志监控器"""
   
    def __init__(self, log_dir: str = "logs"):
        """初始化日志监控器"""
        self.log_dir = Path(log_dir)
        self.forum_log_file = self.log_dir / "forum.log"
       
        # 要监控的日志文件
        self.monitored_logs = {
            'insight': self.log_dir / 'insight.log',
            'media': self.log_dir / 'media.log',
            'query': self.log_dir / 'query.log'
        }
       
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.file_positions = {}  # 记录每个文件的读取位置
        self.file_line_counts = {}  # 记录每个文件的行数
        self.is_searching = False  # 是否正在搜索
        self.search_inactive_count = 0  # 搜索非活跃计数器
        self.write_lock = Lock()  # 写入锁，防止并发写入冲突
       
        # 目标节点名称 - 直接匹配字符串
        self.target_nodes = [
            'FirstSummaryNode',
            'ReflectionSummaryNode'
        ]
        
        # 多行内容捕获状态
        self.capturing_json = {}  # 每个app的JSON捕获状态
        self.json_buffer = {}     # 每个app的JSON缓冲区
        self.json_start_line = {} # 每个app的JSON开始行
       
        # 确保logs目录存在
        self.log_dir.mkdir(exist_ok=True)
   
    def clear_forum_log(self):
        """清空forum.log文件"""
        try:
            if self.forum_log_file.exists():
                self.forum_log_file.unlink()
           
            # 创建新的forum.log文件并写入开始标记
            with open(self.forum_log_file, 'w', encoding='utf-8') as f:
                start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== ForumEgine 监控开始 - {start_time} ===\n")
               
            print(f"ForumEgine: forum.log 已清空并初始化")
            
            # 重置JSON捕获状态
            self.capturing_json = {}
            self.json_buffer = {}
            self.json_start_line = {}
           
        except Exception as e:
            print(f"ForumEgine: 清空forum.log失败: {e}")
   
    def write_to_forum_log(self, content: str):
        """写入内容到forum.log（线程安全）"""
        try:
            with self.write_lock:  # 使用锁确保线程安全
                with open(self.forum_log_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    f.write(f"[{timestamp}] {content}\n")
                    f.flush()
        except Exception as e:
            print(f"ForumEgine: 写入forum.log失败: {e}")
   
    def is_target_log_line(self, line: str) -> bool:
        """检查是否是目标日志行（SummaryNode）"""
        # 简单字符串包含检查，更可靠
        for node_name in self.target_nodes:
            if node_name in line:
                return True
        return False
    
    def is_valuable_content(self, line: str) -> bool:
        """判断是否是有价值的内容（排除短小的提示信息）"""
        # 如果包含"清理后的输出"，则认为是有价值的
        if "清理后的输出" in line:
            return True
        
        # 排除常见的短小提示信息
        exclude_patterns = [
            "JSON解析成功",
            "成功生成",
            "已更新段落",
            "正在生成",
            "开始处理",
            "处理完成"
        ]
        
        for pattern in exclude_patterns:
            if pattern in line:
                return False
        
        # 如果行长度过短，也认为不是有价值的内容
        clean_line = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', line).strip()
        if len(clean_line) < 30:  # 阈值可以调整
            return False
            
        return True
    
    def is_json_start_line(self, line: str) -> bool:
        """判断是否是JSON开始行"""
        return "清理后的输出: {" in line
    
    def is_json_end_line(self, line: str) -> bool:
        """判断是否是JSON结束行"""
        stripped = line.strip()
        return stripped == "}" or (stripped.startswith("[") and stripped.endswith("] }"))
    
    def extract_json_content(self, json_lines: List[str]) -> Optional[str]:
        """从多行中提取并解析JSON内容"""
        try:
            # 找到JSON开始的位置
            json_start_idx = -1
            for i, line in enumerate(json_lines):
                if "清理后的输出: {" in line:
                    json_start_idx = i
                    break
            
            if json_start_idx == -1:
                return None
            
            # 提取JSON部分
            first_line = json_lines[json_start_idx]
            json_start_pos = first_line.find("清理后的输出: {")
            if json_start_pos == -1:
                return None
            
            json_part = first_line[json_start_pos + len("清理后的输出: "):]
            
            # 如果第一行就包含完整JSON，直接处理
            if json_part.strip().endswith("}") and json_part.count("{") == json_part.count("}"):
                try:
                    json_obj = json.loads(json_part.strip())
                    return self.format_json_content(json_obj)
                except json.JSONDecodeError:
                    pass
            
            # 处理多行JSON
            json_text = json_part
            for line in json_lines[json_start_idx + 1:]:
                # 移除时间戳
                clean_line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', line)
                json_text += clean_line
            
            # 尝试解析JSON
            json_obj = json.loads(json_text.strip())
            return self.format_json_content(json_obj)
            
        except json.JSONDecodeError as e:
            print(f"ForumEgine: JSON解析失败: {e}")
            # 如果JSON解析失败，返回原始文本（去除时间戳）
            if json_lines:
                first_line = json_lines[0]
                if "清理后的输出:" in first_line:
                    json_start_pos = first_line.find("清理后的输出:")
                    return first_line[json_start_pos:].strip()
            return None
        except Exception as e:
            print(f"ForumEgine: 提取JSON内容时出错: {e}")
            return None
    
    def format_json_content(self, json_obj: dict) -> str:
        """格式化JSON内容为可读形式"""
        try:
            # 提取主要内容
            content_parts = []
            
            if "paragraph_latest_state" in json_obj:
                content_parts.append(f"首次总结: {json_obj['paragraph_latest_state']}")
            
            if "updated_paragraph_latest_state" in json_obj:
                content_parts.append(f"反思总结: {json_obj['updated_paragraph_latest_state']}")
            
            # 如果没有找到预期的字段，返回整个JSON的字符串表示
            if not content_parts:
                return f"清理后的输出: {json.dumps(json_obj, ensure_ascii=False, indent=2)}"
            
            return "\n".join(content_parts)
            
        except Exception as e:
            print(f"ForumEgine: 格式化JSON时出错: {e}")
            return f"清理后的输出: {json.dumps(json_obj, ensure_ascii=False, indent=2)}"

    def extract_node_content(self, line: str) -> Optional[str]:
        """提取节点内容"""
        # 移除时间戳部分，保留节点名称和消息
        # 格式: [HH:MM:SS] [NodeName] message
        match = re.search(r'\[\d{2}:\d{2}:\d{2}\]\s*(.+)', line)
        if match:
            return match.group(1).strip()
        return line.strip()
   
    def get_file_size(self, file_path: Path) -> int:
        """获取文件大小"""
        try:
            return file_path.stat().st_size if file_path.exists() else 0
        except:
            return 0
   
    def get_file_line_count(self, file_path: Path) -> int:
        """获取文件行数"""
        try:
            if not file_path.exists():
                return 0
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
   
    def read_new_lines(self, file_path: Path, app_name: str) -> List[str]:
        """读取文件中的新行"""
        new_lines = []
       
        try:
            if not file_path.exists():
                return new_lines
           
            current_size = self.get_file_size(file_path)
            last_position = self.file_positions.get(app_name, 0)
           
            # 如果文件变小了，说明被清空了，重新从头开始
            if current_size < last_position:
                last_position = 0
                # 重置JSON捕获状态
                self.capturing_json[app_name] = False
                self.json_buffer[app_name] = []
           
            if current_size > last_position:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    new_lines = new_content.split('\n')
                   
                    # 更新位置
                    self.file_positions[app_name] = f.tell()
                   
                    # 过滤空行
                    new_lines = [line.strip() for line in new_lines if line.strip()]
                   
        except Exception as e:
            print(f"ForumEgine: 读取{app_name}日志失败: {e}")
       
        return new_lines
   
    def process_lines_for_json(self, lines: List[str], app_name: str) -> List[str]:
        """处理行以捕获多行JSON内容"""
        captured_contents = []
        
        # 初始化状态
        if app_name not in self.capturing_json:
            self.capturing_json[app_name] = False
            self.json_buffer[app_name] = []
        
        for line in lines:
            if not line.strip():
                continue
                
            # 检查是否是目标节点行
            if self.is_target_log_line(line):
                if self.is_json_start_line(line):
                    # 开始捕获JSON
                    self.capturing_json[app_name] = True
                    self.json_buffer[app_name] = [line]
                    self.json_start_line[app_name] = line
                    
                    # 检查是否是单行JSON
                    if line.strip().endswith("}"):
                        # 单行JSON，立即处理
                        content = self.extract_json_content([line])
                        if content:
                            captured_contents.append(content)
                        self.capturing_json[app_name] = False
                        self.json_buffer[app_name] = []
                        
                elif self.is_valuable_content(line):
                    # 其他有价值的SummaryNode内容
                    formatted_content = f"[{app_name.upper()}] {self.extract_node_content(line)}"
                    captured_contents.append(formatted_content)
                    
            elif self.capturing_json[app_name]:
                # 正在捕获JSON的后续行
                self.json_buffer[app_name].append(line)
                
                # 检查是否是JSON结束
                if self.is_json_end_line(line):
                    # JSON结束，处理完整的JSON
                    content = self.extract_json_content(self.json_buffer[app_name])
                    if content:
                        captured_contents.append(f"[{app_name.upper()}] {content}")
                    
                    # 重置状态
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []
        
        return captured_contents
   
    def monitor_logs(self):
        """智能监控日志文件"""
        print("ForumEgine: 开始智能监控日志文件...")
       
        # 初始化文件行数和位置 - 记录当前状态作为基线
        for app_name, log_file in self.monitored_logs.items():
            self.file_line_counts[app_name] = self.get_file_line_count(log_file)
            self.file_positions[app_name] = self.get_file_size(log_file)
            self.capturing_json[app_name] = False
            self.json_buffer[app_name] = []
            print(f"ForumEgine: {app_name} 基线行数: {self.file_line_counts[app_name]}")
       
        while self.is_monitoring:
            try:
                # 同时检测三个log文件的变化
                any_growth = False
                any_shrink = False
                captured_any = False
               
                # 为每个log文件独立处理
                for app_name, log_file in self.monitored_logs.items():
                    current_lines = self.get_file_line_count(log_file)
                    previous_lines = self.file_line_counts.get(app_name, 0)
                   
                    if current_lines > previous_lines:
                        any_growth = True
                        # 立即读取新增内容
                        new_lines = self.read_new_lines(log_file, app_name)
                       
                        # 先检查是否需要触发搜索（只触发一次）
                        if not self.is_searching:
                            for line in new_lines:
                                if line.strip() and 'FirstSummaryNode' in line:
                                    print(f"ForumEgine: 在{app_name}中检测到FirstSummaryNode，开始监控记录")
                                    self.is_searching = True
                                    self.search_inactive_count = 0
                                    # 清空forum.log开始新会话
                                    self.clear_forum_log()
                                    break  # 找到一个就够了，跳出循环
                       
                        # 处理所有新增内容（如果正在搜索状态）
                        if self.is_searching:
                            # 使用新的处理逻辑
                            captured_contents = self.process_lines_for_json(new_lines, app_name)
                            
                            for content in captured_contents:
                                self.write_to_forum_log(content)
                                print(f"ForumEgine: 捕获 - {content}")
                                captured_any = True
                   
                    elif current_lines < previous_lines:
                        any_shrink = True
                        print(f"ForumEgine: 检测到 {app_name} 日志缩短，将重置基线")
                        # 重置文件位置到新的文件末尾
                        self.file_positions[app_name] = self.get_file_size(log_file)
                        # 重置JSON捕获状态
                        self.capturing_json[app_name] = False
                        self.json_buffer[app_name] = []
                   
                    # 更新行数记录
                    self.file_line_counts[app_name] = current_lines
               
                # 检查是否应该结束当前搜索会话
                if self.is_searching:
                    if any_shrink:
                        # log变短，结束当前搜索会话，重置为等待状态
                        print("ForumEgine: 日志缩短，结束当前搜索会话，回到等待状态")
                        self.is_searching = False
                        self.search_inactive_count = 0
                        # 写入结束标记
                        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.write_to_forum_log(f"=== ForumEgine 搜索会话结束 - {end_time} ===")
                        print("ForumEgine: 已重置基线，等待下次FirstSummaryNode触发")
                    elif not any_growth and not captured_any:
                        # 没有增长也没有捕获内容，增加非活跃计数
                        self.search_inactive_count += 1
                        if self.search_inactive_count >= 30:  # 30秒无活动才结束
                            print("ForumEgine: 长时间无活动，结束搜索会话")
                            self.is_searching = False
                            self.search_inactive_count = 0
                            # 写入结束标记
                            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            self.write_to_forum_log(f"=== ForumEgine 搜索会话超时结束 - {end_time} ===")
                    else:
                        self.search_inactive_count = 0  # 重置计数器
               
                # 短暂休眠
                time.sleep(1)
               
            except Exception as e:
                print(f"ForumEgine: 监控过程中出错: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(2)
       
        print("ForumEgine: 停止监控日志文件")
   
    def start_monitoring(self):
        """开始智能监控"""
        if self.is_monitoring:
            print("ForumEgine: 监控已经在运行中")
            return False
       
        try:
            # 启动监控
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_logs, daemon=True)
            self.monitor_thread.start()
           
            print("ForumEgine: 智能监控已启动")
            return True
           
        except Exception as e:
            print(f"ForumEgine: 启动监控失败: {e}")
            self.is_monitoring = False
            return False
   
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            print("ForumEgine: 监控未运行")
            return
       
        try:
            self.is_monitoring = False
           
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)
           
            # 写入结束标记
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.write_to_forum_log(f"=== ForumEgine 监控结束 - {end_time} ===")
           
            print("ForumEgine: 监控已停止")
           
        except Exception as e:
            print(f"ForumEgine: 停止监控失败: {e}")
   
    def get_forum_log_content(self) -> List[str]:
        """获取forum.log的内容"""
        try:
            if not self.forum_log_file.exists():
                return []
           
            with open(self.forum_log_file, 'r', encoding='utf-8') as f:
                return [line.rstrip('\n\r') for line in f.readlines()]
               
        except Exception as e:
            print(f"ForumEgine: 读取forum.log失败: {e}")
            return []

# 全局监控器实例
_monitor_instance = None

def get_monitor() -> LogMonitor:
    """获取全局监控器实例"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = LogMonitor()
    return _monitor_instance

def start_forum_monitoring():
    """启动ForumEgine智能监控"""
    return get_monitor().start_monitoring()

def stop_forum_monitoring():
    """停止ForumEgine监控"""
    get_monitor().stop_monitoring()

def get_forum_log():
    """获取forum.log内容"""
    return get_monitor().get_forum_log_content()