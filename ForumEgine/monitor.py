"""
日志监控器 - 实时监控三个log文件中的SummaryNode和ReportFormattingNode输出
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime
import re
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
            'ReflectionSummaryNode', 
            'ReportFormattingNode'
        ]
        
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
        """检查是否是目标日志行（SummaryNode或ReportFormattingNode）"""
        # 简单字符串包含检查，更可靠
        for node_name in self.target_nodes:
            if node_name in line:
                return True
        return False
    
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
    
    # 移除这个方法，逻辑已经合并到monitor_logs中
    
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
    
    def monitor_logs(self):
        """智能监控日志文件"""
        print("ForumEgine: 开始智能监控日志文件...")
        
        # 初始化文件行数和位置 - 记录当前状态作为基线
        for app_name, log_file in self.monitored_logs.items():
            self.file_line_counts[app_name] = self.get_file_line_count(log_file)
            self.file_positions[app_name] = self.get_file_size(log_file)
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
                            for line in new_lines:
                                if line.strip() and self.is_target_log_line(line):
                                    # 立即记录目标节点输出
                                    formatted_content = f"[{app_name.upper()}] {line.strip()}"
                                    self.write_to_forum_log(formatted_content)
                                    print(f"ForumEgine: 捕获 - {formatted_content}")
                                    captured_any = True
                    
                    elif current_lines < previous_lines:
                        any_shrink = True
                        print(f"ForumEgine: 检测到 {app_name} 日志缩短，将重置基线")
                        # 重置文件位置到新的文件末尾
                        self.file_positions[app_name] = self.get_file_size(log_file)
                    
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
