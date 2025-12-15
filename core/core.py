#from __future__ import annotations
#标准库
import os
import time
import threading
import queue
import logging
from dataclasses import dataclass
from datetime import datetime

#数据处理与科学计算

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import tkinter as tk

#网络与并发

#其他工具
import psutil
from collections import defaultdict
#import logging

######################
#地图生成与布局
#####################

######################
# 地图生成与布局 - 高性能实现
######################

# 导入 GPU 加速库，如果可用
try:
    import cupy as cp
    import pycuda.driver as cuda
    HAS_GPU = False # 这个数组就是用不了，一直报non-scalar numpy.ndarray cannot be used for fill
    logging.info("GPU acceleration enabled")
except ImportError:
    cp=None
    HAS_GPU = False
    logging.info("GPU acceleration not available, falling back to CPU")

# 地图生成配置
@dataclass
class MapGenerationConfig:
    """地图生成配置，封装所有参数以提高可维护性和可配置性"""
    width: int
    height: int
    export_model: bool = False
    threads: int = max(1, os.cpu_count() - 1)  # 默认线程数
    chunk_size: int = 64  # 块处理大小
    use_gpu: bool = HAS_GPU  # 是否使用GPU加速
    memory_limit: int = int(psutil.virtual_memory().available * 0.8)  # 80%的可用内存
    lod_levels: int = 3  # 细节级别数量
    visualization_mode: str = "standard"  # 可视化模式：standard, high_quality, performance
    cache_dir: str = ".map_cache"  # 缓存目录

# 性能监控器
class PerformanceMonitor:
    """跟踪和报告生成过程的性能指标"""
    
    def __init__(self):
        self.start_times = {}
        self.durations = defaultdict(float)
        self.memory_usage = []
        self.start_memory = psutil.Process().memory_info().rss
        
    def start(self, task_name):
        self.start_times[task_name] = time.time()
        return task_name
        
    def end(self, task_name):
        if task_name in self.start_times:
            duration = time.time() - self.start_times[task_name]
            self.durations[task_name] += duration
            current_memory = psutil.Process().memory_info().rss
            self.memory_usage.append((task_name, current_memory - self.start_memory))
            return duration
        return 0
        
    def report(self):
        """生成详细的性能报告"""
        total_time = sum(self.durations.values())
        report = "=== Performance Report ===\n"
        report += f"Total execution time: {total_time:.2f}s\n\n"
        report += "Task breakdown:\n"
        
        # 按耗时排序
        sorted_tasks = sorted(self.durations.items(), key=lambda x: x[1], reverse=True)
        for task, duration in sorted_tasks:
            percentage = (duration / total_time) * 100
            report += f"- {task}: {duration:.2f}s ({percentage:.1f}%)\n"
        
        # 内存使用峰值
        if self.memory_usage:
            peak_task, peak_memory = max(self.memory_usage, key=lambda x: x[1])
            report += f"\nPeak memory usage: {peak_memory/1048576:.1f}MB during {peak_task}\n"
        
        return report

class ThreadSafeLogger:
    """线程安全的日志记录器，可以安全地从任何线程更新GUI"""
    
    def __init__(self, text_widget, max_lines=1000, log_file=None):
        self.text_widget = text_widget
        self.max_lines = max_lines
        self.queue = queue.Queue()
        self.log_file = log_file
        
        # 配置标准日志
        self.logger = logging.getLogger("MapGenerator")
        self.logger.setLevel(logging.DEBUG)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 如果指定了日志文件，添加文件处理器
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # 启动处理队列的任务
        if self.text_widget:
            self.text_widget.after(100, self._process_queue)
    
    def _process_queue(self):
        """处理消息队列，更新GUI"""
        try:
            while True:  # 处理队列中所有消息
                message, level = self.queue.get_nowait()
                
                # 更新文本部件
                if self.text_widget:
                    # 临时启用文本框编辑
                    self.text_widget.config(state=tk.NORMAL)
                    
                    # 添加带时间戳的消息
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    formatted_msg = f"[{timestamp}] [{level}] {message}\n"
                    
                    # 设置消息颜色
                    tag = f"level_{level.lower()}"
                    self.text_widget.insert(tk.END, formatted_msg, tag)
                    
                    # 配置颜色标签
                    if level == "ERROR":
                        self.text_widget.tag_config(tag, foreground="red")
                    elif level == "WARNING":
                        self.text_widget.tag_config(tag, foreground="orange")
                    elif level == "INFO":
                        self.text_widget.tag_config(tag, foreground="blue")
                    
                    # 限制行数
                    lines = self.text_widget.get('1.0', tk.END).count('\n')
                    if lines > self.max_lines:
                        self.text_widget.delete('1.0', f"{lines - self.max_lines}.0")
                    
                    # 滚动到底部
                    self.text_widget.see(tk.END)
                    
                    # 重新禁用文本框
                    self.text_widget.config(state=tk.DISABLED)
                
                self.queue.task_done()
        except queue.Empty:
            # 队列空了，等待一会儿再检查
            pass
        
        # 安排下一次检查
        if self.text_widget:
            self.text_widget.after(100, self._process_queue)
    
    def log(self, message, level="INFO"):
        """记录消息"""
        # 标准日志记录
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "DEBUG":
            self.logger.debug(message)
        else:
            self.logger.info(message)
        
        # 将消息添加到队列
        if self.text_widget:
            self.queue.put((message, level))
    
    def error(self, message):
        """记录错误消息"""
        self.log(message, "ERROR")
    
    def warning(self, message):
        """记录警告消息"""
        self.log(message, "WARNING")
    
    def debug(self, message):
        """记录调试消息"""
        self.log(message, "DEBUG")
        
    def info(self, message):
        """记录信息消息"""
        self.log(message, "INFO")