#from __future__ import annotations
#标准库
import hashlib
import time
import tkinter as tk
import threading
from datetime import datetime

#数据处理与科学计算
import numpy as np

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from matplotlib.gridspec import GridSpec

#网络与并发
import concurrent.futures

#项目文件
from core.evolution.evolve_generation import *


def analyze_terrain_quality(height_map, temp_map, humid_map):
    """分析地形质量并提供改进建议
    
    Args:
        height_map: 高度图
        temp_map: 温度图
        humid_map: 湿度图
        
    Returns:
        quality_metrics: 质量度量字典
        suggestions: 改进建议列表
    """
    from scipy import ndimage
    import numpy as np
    
    height, width = height_map.shape
    suggestions = []
    quality_metrics = {}
    
    # 计算地形复杂度
    gradient_y, gradient_x = np.gradient(height_map)
    slope = np.sqrt(gradient_x**2 + gradient_y**2)
    complexity = np.mean(slope)
    quality_metrics["地形复杂度"] = complexity
    
    # 高度分布分析
    height_hist, _ = np.histogram(height_map, bins=10)
    height_entropy = -np.sum((height_hist/height_hist.sum()) * 
                             np.log2(height_hist/height_hist.sum() + 1e-10))
    quality_metrics["高度多样性"] = height_entropy
    
    # 检测不自然的平坦区域
    flat_threshold = 0.05
    flat_areas = slope < flat_threshold
    # 寻找大面积平坦区域
    labeled_flats, num_flats = ndimage.label(flat_areas)
    flat_sizes = ndimage.sum(flat_areas, labeled_flats, range(1, num_flats+1))
    large_flat_count = np.sum(flat_sizes > (width * height * 0.05))
    
    if large_flat_count > 3:
        suggestions.append("检测到大面积平坦区域，建议添加一些高度变化")
    
    # 检测过于陡峭的山脉
    steep_threshold = 1.5
    steep_areas = slope > steep_threshold
    # 寻找大面积陡峭区域
    labeled_steeps, num_steeps = ndimage.label(steep_areas)
    steep_sizes = ndimage.sum(steep_areas, labeled_steeps, range(1, num_steeps+1))
    large_steep_count = np.sum(steep_sizes > (width * height * 0.02))
    
    if large_steep_count > 5:
        suggestions.append("检测到过多陡峭区域，建议适当平滑部分陡峭山脉")
    
    # 温度与高度相关性分析
    temp_height_corr = np.corrcoef(height_map.flatten(), temp_map.flatten())[0, 1]
    if abs(temp_height_corr) < 0.3:
        suggestions.append("温度与高度关联性较弱，建议调整温度分布")
    
    # 湿度与地形特征分析
    humid_gradient_corr = np.corrcoef(humid_map.flatten(), slope.flatten())[0, 1]
    if humid_gradient_corr > 0:
        suggestions.append("湿度分布不够自然，通常陡峭区域湿度应较低")
    
    # 检测边界锐化问题
    edge_height = np.concatenate([
        height_map[0, :], height_map[-1, :], 
        height_map[:, 0], height_map[:, -1]
    ])
    if np.std(edge_height) < 0.5 * np.std(height_map):
        suggestions.append("地图边缘过于平坦，建议增加边界变化")
    
    # 检查水域分布
    potential_water = height_map < np.percentile(height_map, 15)
    labeled_water, num_water = ndimage.label(potential_water)
    water_sizes = ndimage.sum(potential_water, labeled_water, range(1, num_water+1))
    
    if len(water_sizes) < 2:
        suggestions.append("水域分布不足，建议添加更多低洼区域形成湖泊")
    
    return quality_metrics, suggestions

def manually_adjust_height(map_data, map_params, logger, seed=None, parent_frame=None, on_complete=None):
    """
    提供交互式界面让用户手动调整高度图，并自动更新温度和湿度
    
    Args:
        map_data: 地图数据对象
        map_params: 地图参数
        logger: 日志记录器
        seed: 随机种子
        parent_frame: Tkinter父容器(如果为None则创建独立窗口)
        on_complete: 完成编辑后的回调函数
    
    Returns:
        更新后的map_data、height_map, temp_map, humid_map
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider, RadioButtons, RectangleSelector
    import numpy as np
    from matplotlib.patches import Rectangle
    from scipy.ndimage import gaussian_filter
    from core.generation.generate_height_temp_humid import biome_temperature, moisture_map
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    
    logger.log("启动高度手动调整界面...\n")
    
    # 种子处理
    if seed is None:
        seed = np.random.randint(0, 999999)
    
    # 获取当前高度、温度和湿度图
    height_map = map_data.get_layer("height").copy()
    temp_map = map_data.get_layer("temperature").copy()
    humid_map = map_data.get_layer("humidity").copy()
    
    height, width = height_map.shape
    
    # 根据是否有父容器决定创建独立窗口还是嵌入
    if parent_frame is None:
        # 创建独立的交互式绘图界面
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("高度图手动编辑工具", fontsize=16)
        is_embedded = False
    else:
        # 使用Grid布局配置父容器
        parent_frame.grid_columnconfigure(0, weight=1)
        parent_frame.grid_rowconfigure(0, weight=1)  # 图表区域可扩展
        parent_frame.grid_rowconfigure(1, weight=0)  # 工具栏区域固定高度
        
        # 创建嵌入到Tkinter中的matplotlib图形
        fig = plt.Figure(figsize=(14, 10))
        fig.suptitle("高度图手动编辑工具", fontsize=16)
        
        # 创建Canvas区域并放在第一行
        canvas_frame = tk.Frame(parent_frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # 添加导航工具栏并放在第二行
        toolbar_frame = tk.Frame(parent_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # 创建子图布局
        gs = fig.add_gridspec(2, 2)
        axes = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])], 
                [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]]
        is_embedded = True
        
        # 增加底部边距，为工具和按钮留出空间
        fig.subplots_adjust(bottom=0.3)
    
    # 高度图显示
    height_ax = axes[0][0]
    height_img = height_ax.imshow(height_map, cmap='terrain')
    height_ax.set_title("高度图")
    fig.colorbar(height_img, ax=height_ax, label="高度")
    
    # 温度图显示
    temp_ax = axes[0][1]
    temp_img = temp_ax.imshow(temp_map, cmap='plasma')
    temp_ax.set_title("温度图")
    fig.colorbar(temp_img, ax=temp_ax, label="温度")
    
    # 湿度图显示
    humid_ax = axes[1][0]
    humid_img = humid_ax.imshow(humid_map, cmap='Blues')
    humid_ax.set_title("湿度图")
    fig.colorbar(humid_img, ax=humid_ax, label="湿度")
    
    # 预览图显示
    preview_ax = axes[1][1]
    # 创建RGB预览图
    preview_img = np.zeros((height, width, 3))
    # 红色通道：温度
    preview_img[:, :, 0] = temp_map
    # 绿色通道：高度
    preview_img[:, :, 1] = height_map / height_map.max()
    # 蓝色通道：湿度
    preview_img[:, :, 2] = humid_map
    
    preview_display = preview_ax.imshow(preview_img)
    preview_ax.set_title("预览图 (R:温度, G:高度, B:湿度)")
    
    # 控制面板
    plt.subplots_adjust(bottom=0.3)
    
    # 修改滑块位置，确保不重叠且位置合理
    # 添加滑块区域
    slider_ax = plt.axes([0.2, 0.20, 0.65, 0.03])  # 确保位置合理不重叠
    brush_size_slider = Slider(slider_ax, '笔刷大小', 1, 20, valinit=5, valstep=1)

    intensity_ax = plt.axes([0.2, 0.15, 0.65, 0.03])  # 确保位置合理不重叠
    intensity_slider = Slider(intensity_ax, '效果强度', 0.1, 1.0, valinit=0.5)

    # 确保初始值设置
    brush_size = 5  # 确保这个值等于滑块的valinit
    intensity = 0.5  # 确保这个值等于滑块的valinit
    
    # 添加编辑工具单选按钮
    tools_ax = plt.axes([0.2, 0.05, 0.2, 0.08])
    edit_tools = RadioButtons(tools_ax, ('提升', '降低', '平滑', '随机化'))
    
    # 添加按钮
    save_ax = plt.axes([0.45, 0.05, 0.15, 0.05])
    save_button = Button(save_ax, '保存修改')
    
    cancel_ax = plt.axes([0.65, 0.05, 0.15, 0.05])
    cancel_button = Button(cancel_ax, '取消')
    
    # 记录原始数据以允许取消
    original_height = height_map.copy()
    
    # 当前工具和参数
    current_tool = '提升'
    brush_size = 5
    intensity = 0.5
    drawing = False
    last_pos = None
    
    # 区域选择器状态
    selection_active = False
    selected_region = None
    selection_rect = None  # 矩形选择视觉对象
    
    # 新增：地形工具状态
    terrain_tool_active = None  # 当前激活的地形工具 (None, 'mountain', 'valley', 'plateau')
    
    # 回调函数
    def update_climate():
        """根据修改后的高度图更新温度和湿度"""
        nonlocal temp_map, humid_map, preview_img
        
        # 获取风向参数
        prevailing_wind_x = map_params.get("prevailing_wind_x", 1.0)
        prevailing_wind_y = map_params.get("prevailing_wind_y", 0.0)
        
        # 重新计算温度图
        new_temp = biome_temperature(
            height_map,
            latitude_effect=map_params.get("latitude_effect", 0.5),
            seed=seed,
            use_frequency_optimization=map_params.get("use_frequency_optimization", True)
        )
        
        # 重新计算湿度图
        new_humid = moisture_map(
            height_map,
            new_temp,
            prevailing_wind=(prevailing_wind_x, prevailing_wind_y),
            seed=seed,
            use_frequency_optimization=map_params.get("use_frequency_optimization", True)
        )
        
        # 更新图像
        temp_map = new_temp
        humid_map = new_humid
        
        temp_img.set_data(temp_map)
        humid_img.set_data(humid_map)
        
        # 更新预览图
        preview_img[:, :, 0] = temp_map
        preview_img[:, :, 1] = height_map / height_map.max()
        preview_img[:, :, 2] = humid_map
        preview_display.set_data(preview_img)
        
        fig.canvas.draw_idle()

    # 修改这部分代码以根据嵌入模式调整布局
    if is_embedded:
        # 为嵌入模式调整控制面板的位置，将其上移
        plt.subplots_adjust(bottom=0.4)  # 增加底部空间，使控件可见
        
        # 智能地形工具面板位置调整 - 上移到图形中间区域
        advanced_tools_ax = plt.axes([0.85, 0.45, 0.13, 0.2])  # Y坐标从0.05改为0.45
        advanced_tools_ax.set_title("智能地形工具")
        advanced_tools_ax.axis('off')
        
        # 地形按钮位置也相应上移
        mountain_ax = plt.axes([0.87, 0.61, 0.1, 0.04])  # 上移
        valley_ax = plt.axes([0.87, 0.57, 0.1, 0.04])    # 上移
        plateau_ax = plt.axes([0.87, 0.53, 0.1, 0.04])   # 上移
        plains_ax = plt.axes([0.87, 0.49, 0.1, 0.04])    # 上移
        hills_ax = plt.axes([0.87, 0.45, 0.1, 0.04])     # 上移
        basin_ax = plt.axes([0.87, 0.41, 0.1, 0.04])     # 上移
        desert_ax = plt.axes([0.76, 0.61, 0.1, 0.04])    # 上移
        
        # 添加这些代码 - 创建按钮对象
        mountain_button = Button(mountain_ax, '添加山脉')
        valley_button = Button(valley_ax, '添加河谷')
        plateau_button = Button(plateau_ax, '添加高原')
        plains_button = Button(plains_ax, '添加平原')
        hills_button = Button(hills_ax, '添加丘陵')
        basin_button = Button(basin_ax, '添加盆地')
        desert_button = Button(desert_ax, '添加沙漠')
        
        # 撤销/重做按钮位置调整
        undo_ax = plt.axes([0.05, 0.45, 0.05, 0.05])     # 上移
        redo_ax = plt.axes([0.11, 0.45, 0.05, 0.05])     # 上移
        
        # 添加这些代码 - 创建撤销/重做按钮
        undo_button = Button(undo_ax, '撤销')
        redo_button = Button(redo_ax, '重做')
        
        # 选择按钮位置调整
        select_ax = plt.axes([0.05, 0.52, 0.12, 0.04])   # 上移
        clear_selection_ax = plt.axes([0.05, 0.58, 0.12, 0.04])  # 上移
        
        # 添加这些代码 - 创建选择按钮
        select_button = Button(select_ax, '区域选择')
        clear_selection_button = Button(clear_selection_ax, '清除选择')
        
        # 添加历史栈初始化
        history_stack = []
        redo_stack = []
        select_active = False
        
        # 地形尺寸滑块位置调整
        terrain_size_ax = plt.axes([0.87, 0.38, 0.1, 0.03])  # 上移
    else:
        # 独立窗口模式使用原来的位置
        plt.subplots_adjust(bottom=0.3)
        
        # 原有代码的位置设置...
        advanced_tools_ax.set_title("智能地形工具")
        advanced_tools_ax.axis('off')
        
        # 保持原有的按钮并调整位置
        mountain_ax = plt.axes([0.87, 0.21, 0.1, 0.04])
        mountain_button = Button(mountain_ax, '添加山脉')
        
        valley_ax = plt.axes([0.87, 0.17, 0.1, 0.04])
        valley_button = Button(valley_ax, '添加河谷')
        
        plateau_ax = plt.axes([0.87, 0.13, 0.1, 0.04])
        plateau_button = Button(plateau_ax, '添加高原')
        
        # 添加新的地形模板按钮
        plains_ax = plt.axes([0.87, 0.09, 0.1, 0.04])
        plains_button = Button(plains_ax, '添加平原')
        
        hills_ax = plt.axes([0.87, 0.05, 0.1, 0.04])
        hills_button = Button(hills_ax, '添加丘陵')
        
        basin_ax = plt.axes([0.87, 0.01, 0.1, 0.04])
        basin_button = Button(basin_ax, '添加盆地')
        
        # 添加第二列按钮
        desert_ax = plt.axes([0.76, 0.21, 0.1, 0.04])
        desert_button = Button(desert_ax, '添加沙漠')
        
        # 添加撤销/重做功能
        history_stack = []
        redo_stack = []
        
        undo_ax = plt.axes([0.05, 0.05, 0.05, 0.05])
        undo_button = Button(undo_ax, '撤销')
        
        redo_ax = plt.axes([0.11, 0.05, 0.05, 0.05])
        redo_button = Button(redo_ax, '重做')
        
        # 区域选择模式切换按钮
        select_ax = plt.axes([0.05, 0.12, 0.12, 0.04])
        select_button = Button(select_ax, '区域选择')
        select_active = False
        
        # 添加清除选择按钮
        clear_selection_ax = plt.axes([0.05, 0.18, 0.12, 0.04])
        clear_selection_button = Button(clear_selection_ax, '清除选择')

    def clear_selection(event):
        nonlocal selected_region, selection_rect
        if selected_region:
            selected_region = None
            if selection_rect and selection_rect in height_ax.patches:
                selection_rect.remove()
                fig.canvas.draw_idle()
            status_text.set_text("已清除选择区域")
        else:
            status_text.set_text("没有活动的选择区域")

    # 添加地形工具尺寸滑块
    terrain_size_ax = plt.axes([0.87, 0.03, 0.1, 0.03])
    terrain_size_slider = Slider(terrain_size_ax, '地形尺寸', 0.5, 2.0, valinit=1.0, valstep=0.1)

    def on_terrain_size_change(val):
        nonlocal status_text
        status_text.set_text(f"地形工具尺寸: {val:.1f}x")

    terrain_size_slider.on_changed(on_terrain_size_change)

    # 修改地形生成函数以使用尺寸系数
    def add_mountain_range(event=None):
        nonlocal height_map, terrain_tool_active, status_text, last_pos, selected_region
        
        # 获取当前地形尺寸系数
        size_factor = terrain_size_slider.val
        
        # 保存历史
        save_history()
        
        # 如果没有选择区域，提示用户选择
        if not selected_region and not last_pos:
            status_text.set_text("请先点击地图选择山脉中心点，或使用区域选择工具选择范围")
            return
        
        # 确定山脉范围和中心
        if selected_region:
            x1, y1, x2, y2 = selected_region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            mountain_width = int((x2 - x1) // 2 * size_factor)
            mountain_height = int((y2 - y1) // 2 * size_factor)
        else:
            center_y, center_x = last_pos
            # 默认尺寸
            mountain_width = int(width // 6 * size_factor)
            mountain_height = int(height // 6 * size_factor)
        
        # 生成山脉高度
        status_text.set_text("正在生成山脉...")
        for y in range(max(0, center_y - mountain_height), min(height, center_y + mountain_height)):
            for x in range(max(0, center_x - mountain_width), min(width, center_x + mountain_width)):
                # 计算到中心的距离
                distance = np.sqrt(((y - center_y) / mountain_height)**2 + ((x - center_x) / mountain_width)**2)
                if distance < 1:
                    # 使用柯西函数创建尖锐山峰
                    elevation = 30 * (1 / (1 + (distance * 3)**2))
                    height_map[y, x] += elevation
        
        # 更新显示
        height_img.set_data(height_map)
        fig.canvas.draw_idle()
        update_climate()
        status_text.set_text(f"山脉添加完成(尺寸:{size_factor:.1f}x) - 可继续使用山脉工具")
        
        # 重置选择区域以允许新选择
        selected_region = None
        if selection_rect and selection_rect in height_ax.patches:
            selection_rect.remove()
            fig.canvas.draw_idle()

    
    # 状态信息文本
    status_text = plt.figtext(0.5, 0.01, "准备就绪", ha="center", va="bottom", 
                    bbox=dict(boxstyle="round", fc="lightblue", alpha=0.8))
    
    
    # 记录历史状态
    def save_history():
        history_stack.append(height_map.copy())
        redo_stack.clear()  # 新的编辑动作会清空重做栈
        if len(history_stack) > 20:  # 限制历史记录数量
            history_stack.pop(0)
    
    # 定义区域选择函数
    def toggle_selector(event):
        nonlocal select_active, selection_active, selected_region, selection_rect
        
        select_active = not select_active
        selection_active = select_active
        
        if select_active:
            select_button.label.set_text('取消选择')
            status_text.set_text("区域选择模式：拖拽鼠标选择区域")
            selector.set_active(True)
        else:
            select_button.label.set_text('区域选择')
            # 如果有选区，则不清除选区和矩形
            if not selected_region:
                status_text.set_text("准备就绪")
            else:
                status_text.set_text("区域已选择(Esc键清除)")
            
            selector.set_active(False)
        
        plt.draw()


    # 优化区域选择回调
    def onselect(eclick, erelease):
        nonlocal selected_region, selection_rect, selection_active
        
        if eclick.xdata is None or erelease.xdata is None:
            return
        
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # 确保坐标在有效范围内
        x1 = max(0, min(width-1, x1))
        y1 = max(0, min(height-1, y1))
        x2 = max(0, min(width-1, x2))
        y2 = max(0, min(height-1, y2))
        
        # 确保x1<=x2, y1<=y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 存储选择区域
        selected_region = (x1, y1, x2, y2)
        
        # 更新状态文本
        w, h = x2-x1, y2-y1
        status_text.set_text(f"已选择区域: 左上({x1},{y1}) 右下({x2},{y2}) 尺寸:{w}x{h}")
        
        # 创建或更新选择矩形
        if selection_rect and selection_rect in height_ax.patches:
            selection_rect.remove()
        
        selection_rect = Rectangle((x1-0.5, y1-0.5), w, h, 
                            fill=False, edgecolor='yellow', linewidth=2)
        height_ax.add_patch(selection_rect)
        fig.canvas.draw_idle()
        
        # 如果当前有活动的地形工具，提示用户
        if terrain_tool_active:
            status_text.set_text(f"区域已选择，点击高度图应用{terrain_tool_active}工具")
        
        # 自动关闭选择模式，便于立即使用所选区域
        if selection_active:
            toggle_selector(None)
    
    # 创建选择器
    selector = RectangleSelector(height_ax, onselect,
                            useblit=True, button=[1], 
                            minspanx=5, minspany=5, 
                            spancoords='pixels', interactive=True)
    selector.set_active(False)  # 初始状态为非激活
    
    # 预设地形函数改进版
    # 优化add_mountain_range函数
    def add_mountain_range(event=None):
        nonlocal height_map, terrain_tool_active, status_text, last_pos, selected_region
        
        # 保存历史
        save_history()
        
        # 如果没有选择区域，提示用户选择
        if not selected_region and not last_pos:
            status_text.set_text("请先点击地图选择山脉中心点，或使用区域选择工具选择范围")
            return
        
        # 确定山脉范围和中心
        if selected_region:
            x1, y1, x2, y2 = selected_region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            mountain_width = (x2 - x1) // 2
            mountain_height = (y2 - y1) // 2
        else:
            center_y, center_x = last_pos
            # 默认尺寸
            mountain_width = width // 6
            mountain_height = height // 6
        
        # 生成山脉高度
        status_text.set_text("正在生成山脉...")
        for y in range(max(0, center_y - mountain_height), min(height, center_y + mountain_height)):
            for x in range(max(0, center_x - mountain_width), min(width, center_x + mountain_width)):
                # 计算到中心的距离
                distance = np.sqrt(((y - center_y) / mountain_height)**2 + ((x - center_x) / mountain_width)**2)
                if distance < 1:
                    # 使用柯西函数创建尖锐山峰
                    elevation = 30 * (1 / (1 + (distance * 3)**2))
                    height_map[y, x] += elevation
        
        # 更新显示
        height_img.set_data(height_map)
        fig.canvas.draw_idle()
        update_climate()
        status_text.set_text(f"山脉添加完成 - 可继续使用山脉工具")
        
        # 重置选择区域以允许新选择
        selected_region = None
        if selection_rect and selection_rect in height_ax.patches:
            selection_rect.remove()
            fig.canvas.draw_idle()
    
    def add_river_valley(event=None):
        nonlocal height_map, terrain_tool_active, status_text, selected_region, selection_rect
        
        # 保存历史
        save_history()
        
        # 如果没有选择区域，提示用户选择
        if not selected_region and not last_pos:
            terrain_tool_active = 'valley'
            status_text.set_text("请先点击地图选择河谷起点，或使用区域选择工具选择范围")
            return
        
        # 确定河谷起点和路径
        if selected_region:
            x1, y1, x2, y2 = selected_region
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            valley_width = max(3, min((x2 - x1) // 4, width // 10))
            valley_length = y2 - y1
            
            # 计算方向 - 从选择区域的上边缘到下边缘
            dir_x = 0
            dir_y = 1
        else:
            center_y, center_x = last_pos
            # 默认尺寸
            valley_width = width // 10
            valley_length = height // 3
            
            # 随机河流方向
            angle = np.random.uniform(0, 2 * np.pi)
            dir_x, dir_y = np.cos(angle), np.sin(angle)
        
        status_text.set_text("正在生成河谷...")
        # 生成曲折河谷路径
        points = []
        x, y = center_x, center_y
        for _ in range(20):
            points.append((int(y), int(x)))
            # 添加一些随机性使河流曲折
            x += dir_x * valley_length/20 + np.random.normal(0, 2)
            y += dir_y * valley_length/20 + np.random.normal(0, 2)
            if x < 0 or x >= width or y < 0 or y >= height:
                break
        
        # 沿路径挖掘河谷
        for y, x in points:
            if 0 <= y < height and 0 <= x < width:
                # 在路径周围降低高度
                for dy in range(-valley_width//2, valley_width//2 + 1):
                    for dx in range(-valley_width//2, valley_width//2 + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            # 距离中心越远，降低越少
                            dist = np.sqrt(dy**2 + dx**2) / (valley_width/2)
                            if dist < 1:
                                # 使用平滑的高斯函数创建U形河谷
                                depth = 15 * (1 - dist**2)
                                height_map[ny, nx] = max(0, height_map[ny, nx] - depth)
        
        # 更新显示
        height_img.set_data(height_map)
        fig.canvas.draw_idle()
        update_climate()
        status_text.set_text("河谷添加完成 - 可继续使用河谷工具")
        
        # 重置选择区域以允许新选择
        selected_region = None
        if selection_rect and selection_rect in height_ax.patches:
            selection_rect.remove()
            fig.canvas.draw_idle()
        
        # 不重置地形工具状态，允许继续使用
        # terrain_tool_active = None
    
    # 添加平原生成函数
    def add_plains(event=None):
        nonlocal height_map, terrain_tool_active, status_text, last_pos, selected_region
        
        # 获取当前地形尺寸系数
        size_factor = terrain_size_slider.val
        
        # 保存历史
        save_history()
        
        # 如果没有选择区域，提示用户选择
        if not selected_region and not last_pos:
            status_text.set_text("请先点击地图选择平原中心点，或使用区域选择工具选择范围")
            return
        
        # 确定平原范围和中心
        if selected_region:
            x1, y1, x2, y2 = selected_region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            plains_width = int((x2 - x1) * size_factor)
            plains_height = int((y2 - y1) * size_factor)
        else:
            center_y, center_x = last_pos
            # 默认尺寸 - 平原通常比较大
            plains_width = int(width // 4 * size_factor)
            plains_height = int(height // 4 * size_factor)
        
        # 生成平原高度
        status_text.set_text("正在生成平原...")
        
        # 获取区域当前的平均高度
        region_sum = 0
        region_count = 0
        y_start = max(0, center_y - plains_height//2)
        y_end = min(height, center_y + plains_height//2)
        x_start = max(0, center_x - plains_width//2)
        x_end = min(width, center_x + plains_width//2)
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                region_sum += height_map[y, x]
                region_count += 1
        
        # 计算目标高度 - 使用较低的高度值
        target_height = region_sum / max(1, region_count) * 0.7
        # 平原高度不应过高
        target_height = min(target_height, 30)
        
        # 应用平原高度，边缘平滑过渡
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # 计算到中心的归一化距离
                dx = (x - center_x) / (plains_width/2)
                dy = (y - center_y) / (plains_height/2)
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < 1:
                    # 中心区域保持平坦
                    if distance < 0.7:
                        # 添加轻微随机起伏
                        noise = np.random.normal(0, 0.5)
                        height_map[y, x] = target_height + noise
                    else:
                        # 边缘平滑过渡
                        t = (distance - 0.7) / 0.3
                        height_map[y, x] = target_height * (1-t) + height_map[y, x] * t
        
        # 平滑整个区域以确保自然过渡
        temp = height_map[y_start:y_end, x_start:x_end].copy()
        temp = gaussian_filter(temp, sigma=2)
        height_map[y_start:y_end, x_start:x_end] = temp
        
        # 更新显示
        height_img.set_data(height_map)
        fig.canvas.draw_idle()
        update_climate()
        status_text.set_text(f"平原添加完成(尺寸:{size_factor:.1f}x) - 可继续使用平原工具")
        
        # 重置选择区域以允许新选择
        selected_region = None
        if selection_rect and selection_rect in height_ax.patches:
            selection_rect.remove()
            fig.canvas.draw_idle()
    
    # 添加丘陵生成函数
    def add_hills(event=None):
        nonlocal height_map, terrain_tool_active, status_text, last_pos, selected_region
        
        # 获取当前地形尺寸系数
        size_factor = terrain_size_slider.val
        
        # 保存历史
        save_history()
        
        # 如果没有选择区域，提示用户选择
        if not selected_region and not last_pos:
            status_text.set_text("请先点击地图选择丘陵中心点，或使用区域选择工具选择范围")
            return
        
        # 确定丘陵范围和中心
        if selected_region:
            x1, y1, x2, y2 = selected_region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            hills_width = int((x2 - x1) * size_factor)
            hills_height = int((y2 - y1) * size_factor)
        else:
            center_y, center_x = last_pos
            # 默认尺寸
            hills_width = int(width // 5 * size_factor)
            hills_height = int(height // 5 * size_factor)
        
        status_text.set_text("正在生成丘陵...")
        
        # 生成多个柔和的小山丘
        num_hills = int(hills_width * hills_height / 400)  # 丘陵密度
        for _ in range(num_hills):
            # 随机位置，在选定区域内
            hill_x = center_x + np.random.randint(-hills_width//2, hills_width//2)
            hill_y = center_y + np.random.randint(-hills_height//2, hills_height//2)
            
            # 确保在地图范围内
            if 0 <= hill_y < height and 0 <= hill_x < width:
                # 随机丘陵大小
                hill_size = np.random.randint(5, 15)
                hill_height = np.random.uniform(5, 15)  # 较低的高度
                
                # 创建圆润的丘陵
                for y in range(max(0, hill_y-hill_size), min(height, hill_y+hill_size)):
                    for x in range(max(0, hill_x-hill_size), min(width, hill_x+hill_size)):
                        # 计算到丘陵中心的距离
                        distance = np.sqrt(((y - hill_y) / hill_size)**2 + ((x - hill_x) / hill_size)**2)
                        if distance < 1:
                            # 使用余弦函数创建圆润的丘陵
                            elevation = hill_height * (np.cos(distance * np.pi) + 1) / 2
                            height_map[y, x] += elevation
        
        # 平滑整个区域
        y_start = max(0, center_y - hills_height//2)
        y_end = min(height, center_y + hills_height//2)
        x_start = max(0, center_x - hills_width//2)
        x_end = min(width, center_x + hills_width//2)
        
        temp = height_map[y_start:y_end, x_start:x_end].copy()
        temp = gaussian_filter(temp, sigma=1.5)
        height_map[y_start:y_end, x_start:x_end] = temp
        
        # 更新显示
        height_img.set_data(height_map)
        fig.canvas.draw_idle()
        update_climate()
        status_text.set_text(f"丘陵添加完成(尺寸:{size_factor:.1f}x) - 可继续使用丘陵工具")
        
        # 重置选择区域以允许新选择
        selected_region = None
        if selection_rect and selection_rect in height_ax.patches:
            selection_rect.remove()
            fig.canvas.draw_idle()
    
    # 添加盆地生成函数
    def add_basin(event=None):
        nonlocal height_map, terrain_tool_active, status_text, last_pos, selected_region
        
        # 获取当前地形尺寸系数
        size_factor = terrain_size_slider.val
        
        # 保存历史
        save_history()
        
        # 如果没有选择区域，提示用户选择
        if not selected_region and not last_pos:
            status_text.set_text("请先点击地图选择盆地中心点，或使用区域选择工具选择范围")
            return
        
        # 确定盆地范围和中心
        if selected_region:
            x1, y1, x2, y2 = selected_region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            basin_radius = min(x2 - x1, y2 - y1) // 2
            basin_radius = int(basin_radius * size_factor)
        else:
            center_y, center_x = last_pos
            # 默认尺寸
            basin_radius = int(min(width, height) // 6 * size_factor)
        
        status_text.set_text("正在生成盆地...")
        
        # 获取周边的平均高度作为参考
        rim_sum = 0
        rim_count = 0
        for angle in np.linspace(0, 2*np.pi, 16):
            rim_x = int(center_x + basin_radius * np.cos(angle))
            rim_y = int(center_y + basin_radius * np.sin(angle))
            if 0 <= rim_y < height and 0 <= rim_x < width:
                rim_sum += height_map[rim_y, rim_x]
                rim_count += 1
        
        rim_height = rim_sum / max(1, rim_count)
        basin_depth = rim_height - 15  # 盆地中心比边缘低约15单位
        basin_depth = max(5, basin_depth)  # 确保最低高度
        
        # 创建盆地
        for y in range(max(0, center_y - basin_radius), min(height, center_y + basin_radius)):
            for x in range(max(0, center_x - basin_radius), min(width, center_x + basin_radius)):
                # 计算到中心的距离
                distance = np.sqrt(((y - center_y) / basin_radius)**2 + ((x - center_x) / basin_radius)**2)
                if distance < 1:
                    # 盆地深度从中心到边缘
                    t = distance**0.8  # 非线性过渡，使边缘更陡
                    target_height = basin_depth + (rim_height - basin_depth) * t
                    # 添加轻微噪声
                    noise = np.random.normal(0, 0.7)
                    height_map[y, x] = target_height + noise
        
        # 更新显示
        height_img.set_data(height_map)
        fig.canvas.draw_idle()
        update_climate()
        status_text.set_text(f"盆地添加完成(尺寸:{size_factor:.1f}x) - 可继续使用盆地工具")
        
        # 重置选择区域以允许新选择
        selected_region = None
        if selection_rect and selection_rect in height_ax.patches:
            selection_rect.remove()
            fig.canvas.draw_idle()
    
    # 添加沙漠/沙丘生成函数
    def add_desert(event=None):
        nonlocal height_map, terrain_tool_active, status_text, last_pos, selected_region
        
        # 获取当前地形尺寸系数
        size_factor = terrain_size_slider.val
        
        # 保存历史
        save_history()
        
        # 如果没有选择区域，提示用户选择
        if not selected_region and not last_pos:
            status_text.set_text("请先点击地图选择沙漠中心点，或使用区域选择工具选择范围")
            return
        
        # 确定沙漠范围和中心
        if selected_region:
            x1, y1, x2, y2 = selected_region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            desert_width = int((x2 - x1) * size_factor)
            desert_height = int((y2 - y1) * size_factor)
        else:
            center_y, center_x = last_pos
            # 默认尺寸
            desert_width = int(width // 4 * size_factor)
            desert_height = int(height // 4 * size_factor)
        
        status_text.set_text("正在生成沙漠...")
        
        # 获取区域当前的平均高度
        region_sum = 0
        region_count = 0
        y_start = max(0, center_y - desert_height//2)
        y_end = min(height, center_y + desert_height//2)
        x_start = max(0, center_x - desert_width//2)
        x_end = min(width, center_x + desert_width//2)
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                region_sum += height_map[y, x]
                region_count += 1
        
        # 计算目标高度 - 沙漠通常比较低
        base_height = region_sum / max(1, region_count) * 0.8
        base_height = min(base_height, 25)
        
        # 创建沙漠基础地形
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # 计算到中心的归一化距离
                dx = (x - center_x) / (desert_width/2)
                dy = (y - center_y) / (desert_height/2)
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < 1:
                    if distance < 0.8:
                        # 设置基础沙漠高度
                        height_map[y, x] = base_height
                    else:
                        # 边缘过渡
                        t = (distance - 0.8) / 0.2
                        height_map[y, x] = base_height * (1-t) + height_map[y, x] * t
        
        # 添加沙丘
        num_dunes = int(desert_width * desert_height / 500)  # 沙丘密度
        for _ in range(num_dunes):
            # 随机位置，在选定区域内
            dune_x = center_x + np.random.randint(-desert_width//2 * 0.8, desert_width//2 * 0.8)
            dune_y = center_y + np.random.randint(-desert_height//2 * 0.8, desert_height//2 * 0.8)
            
            # 确保在地图范围内
            if 0 <= dune_y < height and 0 <= dune_x < width:
                # 随机沙丘大小
                dune_size = np.random.randint(3, 10)
                dune_height = np.random.uniform(2, 6)  # 较低的沙丘
                # 随机沙丘形状（椭圆）
                stretch_x = np.random.uniform(0.7, 1.3)
                stretch_y = np.random.uniform(0.7, 1.3)
                # 随机旋转角度
                angle = np.random.uniform(0, np.pi)
                
                # 创建椭圆形沙丘
                for y in range(max(0, dune_y-dune_size), min(height, dune_y+dune_size)):
                    for x in range(max(0, dune_x-dune_size), min(width, dune_x+dune_size)):
                        # 旋转和变形距离计算
                        dx = x - dune_x
                        dy = y - dune_y
                        # 应用旋转
                        rx = dx * np.cos(angle) - dy * np.sin(angle)
                        ry = dx * np.sin(angle) + dy * np.cos(angle)
                        # 应用拉伸
                        rx = rx / stretch_x
                        ry = ry / stretch_y
                        # 计算变形后的距离
                        distance = np.sqrt((rx/dune_size)**2 + (ry/dune_size)**2)
                        
                        if distance < 1:
                            # 使用指数函数创建尖锐的沙丘
                            elevation = dune_height * (1 - distance**1.5)
                            height_map[y, x] += elevation
        
        # 最后对整个区域应用细微噪声，模拟沙纹
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                dx = (x - center_x) / (desert_width/2)
                dy = (y - center_y) / (desert_height/2)
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < 0.95:
                    # 添加细小沙纹
                    ripple = np.sin(x*0.5) * np.sin(y*0.5) * 0.5
                    noise = np.random.normal(0, 0.3)
                    height_map[y, x] += ripple + noise
        
        # 更新显示
        height_img.set_data(height_map)
        fig.canvas.draw_idle()
        update_climate()
        status_text.set_text(f"沙漠添加完成(尺寸:{size_factor:.1f}x) - 可继续使用沙漠工具")
        
        # 重置选择区域以允许新选择
        selected_region = None
        if selection_rect and selection_rect in height_ax.patches:
            selection_rect.remove()
            fig.canvas.draw_idle()
    
    def add_plateau(event=None):
        nonlocal height_map, terrain_tool_active, status_text, selected_region, selection_rect
        
        # 保存历史
        save_history()
        
        # 如果没有选择区域，提示用户选择
        if not selected_region and not last_pos:
            terrain_tool_active = 'plateau'
            status_text.set_text("请先点击地图选择高原中心点，或使用区域选择工具选择范围")
            return
        
        # 确定高原范围和中心
        if selected_region:
            x1, y1, x2, y2 = selected_region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            plateau_radius = min(x2 - x1, y2 - y1) // 2
        else:
            center_y, center_x = last_pos
            # 默认尺寸
            plateau_radius = min(width, height) // 5
        
        plateau_height = 20  # 高原高度
        
        status_text.set_text("正在生成高原...")
        # 生成高原
        for y in range(max(0, center_y - plateau_radius), min(height, center_y + plateau_radius)):
            for x in range(max(0, center_x - plateau_radius), min(width, center_x + plateau_radius)):
                # 计算到中心的距离
                distance = np.sqrt(((y - center_y) / plateau_radius)**2 + ((x - center_x) / plateau_radius)**2)
                if distance < 0.7:  # 内部区域保持平坦
                    height_map[y, x] = max(height_map[y, x], height_map[center_y, center_x] + plateau_height)
                elif distance < 1:  # 边缘区域逐渐过渡
                    # 平滑过渡到周围地形
                    t = (distance - 0.7) / 0.3  # 0到1的过渡值
                    target_height = height_map[center_y, center_x] + plateau_height
                    height_map[y, x] = max(height_map[y, x], 
                                        target_height * (1-t) + height_map[y, x] * t)
        
        # 更新显示
        height_img.set_data(height_map)
        fig.canvas.draw_idle()
        update_climate()
        status_text.set_text("高原添加完成 - 可继续使用高原工具")
        
        # 重置选择区域以允许新选择
        selected_region = None
        if selection_rect and selection_rect in height_ax.patches:
            selection_rect.remove()
            fig.canvas.draw_idle()
        
        # 不重置地形工具状态，允许继续使用
        # terrain_tool_active = None
    
    def undo_action(event):
        nonlocal height_map, terrain_tool_active
        if history_stack:
            # 保存当前状态到重做栈
            redo_stack.append(height_map.copy())
            # 恢复上一状态
            height_map = history_stack.pop()
            # 更新显示
            height_img.set_data(height_map)
            fig.canvas.draw_idle()
            update_climate()
            
            # 保持工具状态，但更新提示
            if terrain_tool_active:
                status_text.set_text(f"已撤销上一步操作 - 当前工具: {terrain_tool_active}")
            else:
                status_text.set_text("已撤销上一步操作")

    def redo_action(event):
        nonlocal height_map, terrain_tool_active
        if redo_stack:
            # 保存当前状态到历史栈
            history_stack.append(height_map.copy())
            # 恢复重做状态
            height_map = redo_stack.pop()
            # 更新显示
            height_img.set_data(height_map)
            fig.canvas.draw_idle()
            update_climate()
            
            # 保持工具状态，但更新提示
            if terrain_tool_active:
                status_text.set_text(f"已重做操作 - 当前工具: {terrain_tool_active}")
            else:
                status_text.set_text("已重做操作")
                
    # 改进apply_tool函数，确保所有工具正确应用
    def apply_tool(y, x):
        """在指定位置应用当前选择的工具"""
        nonlocal height_map
        
        # 添加工具应用前的调试信息
        print(f"应用工具: {current_tool}, 位置: ({x},{y}), 笔刷大小: {brush_size}, 强度: {intensity}")
        
        # 确定影响范围
        y_start = max(0, y - brush_size)
        y_end = min(height, y + brush_size + 1)
        x_start = max(0, x - brush_size)
        x_end = min(width, x + brush_size + 1)
        
        # 创建权重掩码（圆形渐变）
        mask = np.zeros((y_end - y_start, x_end - x_start))
        for i in range(y_end - y_start):
            for j in range(x_end - x_start):
                distance = np.sqrt((i - (y - y_start))**2 + (j - (x - x_start))**2)
                if distance <= brush_size:
                    mask[i, j] = max(0, 1 - distance / brush_size)
        
        # 记录原始高度用于对比
        original_area = height_map[y_start:y_end, x_start:x_end].copy()
        
        # 应用工具
        if current_tool == '提升':
            # 提升高度
            height_map[y_start:y_end, x_start:x_end] += mask * intensity * 5
            status_text.set_text(f"提升地形：坐标({x},{y})")
        elif current_tool == '降低':
            # 降低高度
            height_map[y_start:y_end, x_start:x_end] -= mask * intensity * 5
            status_text.set_text(f"降低地形：坐标({x},{y})")
        elif current_tool == '平滑':
            # 平滑高度
            area = height_map[y_start:y_end, x_start:x_end].copy()
            blurred = gaussian_filter(area, sigma=1)
            height_map[y_start:y_end, x_start:x_end] = area * (1 - mask * intensity) + blurred * (mask * intensity)
            status_text.set_text(f"平滑地形：坐标({x},{y})")
        elif current_tool == '随机化':
            # 添加随机噪声
            noise = np.random.normal(0, 5, size=mask.shape)
            height_map[y_start:y_end, x_start:x_end] += noise * mask * intensity
            status_text.set_text(f"随机化地形：坐标({x},{y})")
        
        # 确保高度在合理范围内
        height_map = np.clip(height_map, 0, 100)
        
        # 添加智能平滑
        if current_tool == '平滑':
            area = height_map[y_start:y_end, x_start:x_end].copy()
            # 使用不同sigma值的多层平滑以保留更多细节
            blurred1 = gaussian_filter(area, sigma=1.5)
            blurred2 = gaussian_filter(area, sigma=0.5)
            # 混合两种平滑结果，保留更多地形细节
            blended = blurred1 * 0.7 + blurred2 * 0.3
            # 应用渐变掩码
            height_map[y_start:y_end, x_start:x_end] = area * (1 - mask * intensity) + blended * (mask * intensity)
        
        # 验证工具是否有效应用（通过检查区域是否发生变化）
        changed = not np.array_equal(original_area, height_map[y_start:y_end, x_start:x_end])
        print(f"工具应用{'成功' if changed else '失败'} - 区域{'已' if changed else '未'}发生变化")
        
        # 应用变更后自动保存历史记录
        save_history()
        
        # 更新高度图显示
        height_img.set_data(height_map)
        fig.canvas.draw_idle()
    
    # 重写点击事件处理函数
    def on_click(event):
        """鼠标点击回调"""
        nonlocal drawing, last_pos, terrain_tool_active, selected_region
        
        # 如果点击不在高度图上，忽略
        if event.inaxes != height_ax:
            return
            
        # 获取鼠标位置
        y, x = int(event.ydata), int(event.xdata)
        
        # 如果区域选择模式激活，由RectangleSelector处理
        if selection_active:
            return
            
        # 如果有活跃的地形工具
        if terrain_tool_active:
            last_pos = (y, x)
            
            # 根据当前工具执行对应操作
            if terrain_tool_active == 'mountain':
                add_mountain_range()
            elif terrain_tool_active == 'valley':
                add_river_valley()
            elif terrain_tool_active == 'plateau':
                add_plateau()
            elif terrain_tool_active == 'plains':
                add_plains()
            elif terrain_tool_active == 'hills':
                add_hills()
            elif terrain_tool_active == 'basin':
                add_basin()
            elif terrain_tool_active == 'desert':
                add_desert()
            
            # 地形工具不重置，允许多次使用
            return
            
        # 常规绘制模式
        drawing = True
        last_pos = (y, x)
        apply_tool(y, x)
    
    def on_release(event):
        """鼠标释放回调"""
        nonlocal drawing
        
        # 如果不是在绘制中或不是在高度图上释放，忽略
        if not drawing or event.inaxes != height_ax:
            return
            
        drawing = False
        # 更新气候数据
        update_climate()
    
    def on_motion(event):
        """鼠标移动回调"""
        nonlocal last_pos
        
        # 如果不是在绘制中或不是在高度图上移动，忽略
        if not drawing or event.inaxes != height_ax:
            return
            
        y, x = int(event.ydata), int(event.xdata)
        
        # 防止重复处理同一点
        if last_pos == (y, x):
            return
            
        # 绘制从上一个点到当前点的线
        if last_pos is not None:
            y0, x0 = last_pos
            
            # 计算步骤数量
            steps = max(abs(y - y0), abs(x - x0)) * 2
            if steps > 0:
                # 在两点之间创建补间点
                for i in range(steps + 1):
                    t = i / steps
                    yi = int(y0 + (y - y0) * t)
                    xi = int(x0 + (x - x0) * t)
                    apply_tool(yi, xi)
        else:
            apply_tool(y, x)
            
        last_pos = (y, x)
    
    def on_tool_change(label):
        """工具改变回调"""
        nonlocal current_tool, terrain_tool_active
        current_tool = label
        # 切换工具时取消地形工具状态
        terrain_tool_active = None
        status_text.set_text(f"当前工具：{label}")
        # 添加调试日志
        print(f"工具已切换为: {label}")
        # 强制更新UI
        fig.canvas.draw_idle()
    
    def on_brush_change(val):
        """笔刷大小改变回调"""
        nonlocal brush_size
        brush_size = int(val)
        status_text.set_text(f"笔刷大小：{brush_size}")
        print(f"笔刷大小已更新为: {brush_size}")
        # 确保状态显示更新
        fig.canvas.draw_idle()
    
    def on_intensity_change(val):
        """强度改变回调"""
        nonlocal intensity
        intensity = val
        status_text.set_text(f"效果强度：{intensity:.1f}")
    
    def on_save(event=None):
        """保存修改回调"""
        # 更新地图数据
        map_data.layers["height"] = height_map
        map_data.layers["temperature"] = temp_map
        map_data.layers["humidity"] = humid_map
        logger.log("高度调整已保存\n")
        
        if is_embedded:
            if on_complete:
                # 调用完成回调，这将继续生成过程
                on_complete(map_data, height_map, temp_map, humid_map)
        else:
            plt.close(fig)
        
        return map_data, height_map, temp_map, humid_map
    
    def on_cancel(_):
        """取消修改回调"""
        logger.log("高度调整已取消\n")
        if is_embedded:
            if on_complete:
                on_complete(map_data, map_data.get_layer("height"), map_data.get_layer("temperature"), map_data.get_layer("humidity"))
        else:
            plt.close(fig)
    # 改进键盘事件处理函数
    def on_key_press(event):
        nonlocal terrain_tool_active, selected_region, selection_rect, tool_indicator
        
        # 添加调试输出
        print(f"收到键盘事件: key={event.key}, 当前工具={terrain_tool_active}")
        
        # 按Esc键取消当前工具或选择
        if event.key == 'escape':
            if terrain_tool_active:
                # 先清除工具状态
                terrain_tool_active = None
                # 清除工具指示器
                tool_indicator.set_text("")
                tool_indicator.set_bbox(dict(boxstyle="round", fc="white", alpha=0.0))
                status_text.set_text("地形工具已取消")
                fig.canvas.draw_idle()
            elif selected_region:
                # 清除选择区域
                selected_region = None
                if selection_rect and selection_rect in height_ax.patches:
                    selection_rect.remove()
                    fig.canvas.draw_idle()
                status_text.set_text("区域选择已清除")
            else:
                status_text.set_text("准备就绪")
        
        # 添加快捷键: M=山脉, V=山谷, P=高原
        elif event.key == 'm':
            set_terrain_tool('mountain', None)
        elif event.key == 'v':
            set_terrain_tool('valley', None)
        elif event.key == 'p':
            set_terrain_tool('plateau', None)
        # 新增快捷键: L=平原(Land), H=丘陵(Hills), B=盆地(Basin), D=沙漠(Desert)
        elif event.key == 'l':
            set_terrain_tool('plains', None)
        elif event.key == 'h':
            set_terrain_tool('hills', None)
        elif event.key == 'b':
            set_terrain_tool('basin', None)
        elif event.key == 'd':
            set_terrain_tool('desert', None)
        # 添加快捷键: S=选择工具
        elif event.key == 's':
            toggle_selector(None)
        
        # 其他快捷键可以在这里添加

    # 在文件末尾添加此连接
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # 连接按钮事件
    # 连接按钮事件，直接设置活动工具而不立即执行
    # 创建更好的工具状态指示器
    tool_indicator = plt.figtext(0.85, 0.25, "", ha="center", va="center", 
                    bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.0))

    # 更新set_terrain_tool函数
    def set_terrain_tool(tool_name, event):
        nonlocal terrain_tool_active, status_text, tool_indicator
        
        print(f"尝试设置地形工具: {tool_name}, 当前工具: {terrain_tool_active}")
        
        # 如果当前工具与新选择相同，则视为取消选择
        if terrain_tool_active == tool_name:
            terrain_tool_active = None
            tool_indicator.set_text("")
            tool_indicator.set_bbox(dict(boxstyle="round", fc="white", alpha=0.0))
            status_text.set_text("已取消工具选择")
            fig.canvas.draw_idle()
            return
        
        terrain_tool_active = tool_name
        
        # 更新工具状态指示器
        if tool_name == 'mountain':
            tool_indicator.set_text("当前工具:\n山脉")
            status_text.set_text("已选择山脉工具: 请点击高度图选择中心点或使用区域选择工具")
        elif tool_name == 'valley':
            tool_indicator.set_text("当前工具:\n河谷")
            status_text.set_text("已选择河谷工具: 请点击高度图选择起点或使用区域选择工具")
        elif tool_name == 'plateau':
            tool_indicator.set_text("当前工具:\n高原")
            status_text.set_text("已选择高原工具: 请点击高度图选择中心点或使用区域选择工具")
        elif tool_name == 'plains':
            tool_indicator.set_text("当前工具:\n平原")
            status_text.set_text("已选择平原工具: 请点击高度图选择中心点或使用区域选择工具")
        elif tool_name == 'hills':
            tool_indicator.set_text("当前工具:\n丘陵")
            status_text.set_text("已选择丘陵工具: 请点击高度图选择中心点或使用区域选择工具")
        elif tool_name == 'basin':
            tool_indicator.set_text("当前工具:\n盆地")
            status_text.set_text("已选择盆地工具: 请点击高度图选择中心点或使用区域选择工具")
        elif tool_name == 'desert':
            tool_indicator.set_text("当前工具:\n沙漠")
            status_text.set_text("已选择沙漠工具: 请点击高度图选择中心点或使用区域选择工具")
        
        # 确保工具指示器可见
        tool_indicator.set_bbox(dict(boxstyle="round", fc="lightyellow", alpha=0.8))
        fig.canvas.draw_idle()

        # 输出确认信息
        print(f"已激活地形工具: {tool_name}")
        
    # 连接所有按钮事件
    mountain_button.on_clicked(lambda event: set_terrain_tool('mountain', event))
    valley_button.on_clicked(lambda event: set_terrain_tool('valley', event))
    plateau_button.on_clicked(lambda event: set_terrain_tool('plateau', event))
    plains_button.on_clicked(lambda event: set_terrain_tool('plains', event))
    hills_button.on_clicked(lambda event: set_terrain_tool('hills', event))
    basin_button.on_clicked(lambda event: set_terrain_tool('basin', event))
    desert_button.on_clicked(lambda event: set_terrain_tool('desert', event))
    undo_button.on_clicked(undo_action)
    redo_button.on_clicked(redo_action)
    select_button.on_clicked(toggle_selector)
    
    # 连接其他事件
    edit_tools.on_clicked(on_tool_change)
    brush_size_slider.on_changed(on_brush_change)
    intensity_slider.on_changed(on_intensity_change)
    save_button.on_clicked(on_save)
    cancel_button.on_clicked(on_cancel)
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    
    clear_selection_button.on_clicked(clear_selection)
    
    # 如果是嵌入模式，确保绘图更新并调整画布大小
    if is_embedded:
        canvas.draw()
        fig.canvas = canvas
        
        # 强制更新布局
        parent_frame.update_idletasks()
        
        # 确保工具栏可见
        toolbar_frame.update()
        
        # 将图使用更大的底部边距
        fig.subplots_adjust(bottom=0.35)
        canvas.draw_idle()
    
    # 如果是独立模式，显示窗口
    if not is_embedded:
        plt.tight_layout(rect=[0, 0.3, 1, 0.95])
        plt.show()
    
    return map_data, height_map, temp_map, humid_map

#############################################################################################

def visualize_biome_map(ax, biome_map, color_mapping):
    """可视化生物群系地图"""
    img = np.zeros((len(biome_map), len(biome_map[0]), 3))
    for y in range(len(biome_map)):
        for x in range(len(biome_map[0])):
            img[y, x] = color_mapping.get(biome_map[y][x]['name'], (0.5, 0.5, 0.5))
    ax.imshow(img)
    ax.axis('off')

def get_visual_scores(engine, parent_frame=None, on_complete=None):
    """增强的可视化评分界面，支持嵌入到GUI中
    
    Args:
        engine: BiomeEvolutionEngine实例
        parent_frame: Tkinter父容器(如果为None则创建独立窗口)
        on_complete: 评分完成后的回调函数
    
    Returns:
        评分列表或事件对象
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import threading
    
    # 添加一个事件标志
    scoring_completed = threading.Event()
    
    # 判断是否嵌入模式
    is_embedded = parent_frame is not None
    
    selected = set()
    color_mapping = BiomeEvolutionEngine.BIOME_COLORS
    axes = []  # 存储axes列表
    
    # 完成选择的处理函数
    def process_selections():
        # 返回评分列表：选中的给9分，未选中的给6-7分之间的随机值
        scores = [9 if i in selected else np.random.uniform(6, 7) for i in range(engine.population_size)]
        return scores
    
    # 清理前一个界面的所有子部件
    if is_embedded:
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        # 确保父容器使用一致的布局管理器
        parent_frame.pack_propagate(True)
        
        # 创建Figure，避免使用全局plt
        fig = plt.Figure(figsize=(15, 10), constrained_layout=True)
        fig.suptitle("请点击最喜欢的3个方案（左键选择，右键取消）", fontsize=14)
        
        # 创建Canvas
        canvas_frame = tk.Frame(parent_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建按钮框架
        btn_frame = tk.Frame(parent_frame)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        
        # 添加提示标签
        tk.Label(
            btn_frame,
            text="选择完成后请点击此按钮继续",
            fg="red",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        # 添加确认按钮
        def on_confirm_clicked():
            # 获取评分
            scores = process_selections()
            
            # 状态更新
            status_text.set_text("评分已确认，正在继续地图生成...")
            canvas.draw_idle()
            
            # 设置事件标志
            scoring_completed.set()
            
            # 调用回调函数
            if on_complete:
                print(f"评分已确认: 共{len(selected)}个选择，设置事件标志")
                on_complete(scores)
        
        confirm_btn = tk.Button(
            btn_frame, 
            text="确认选择并继续", 
            command=on_confirm_clicked,
            bg="lightgreen",
            fg="black",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5
        )
        confirm_btn.pack(side=tk.RIGHT, padx=20, pady=5)
    else:
        # 独立窗口模式
        plt.close('all')  # 关闭所有现有matplotlib窗口，避免干扰
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("请点击最喜欢的3个方案（左键选择，右键取消）", fontsize=14)
    
    # 计算网格行列数
    n_cols = 2
    n_rows = (engine.population_size + n_cols - 1) // n_cols
    grid = GridSpec(n_rows+1, n_cols, figure=fig, height_ratios=[0.15] + [1]*n_rows)
    
    # 添加说明面板
    info_ax = fig.add_subplot(grid[0, :])
    info_ax.axis('off')
    info_text = (
        "选择标准：\n"
        "1. 生物群系分布是否自然合理\n"
        "2. 过渡区是否平滑\n"
        "3. 是否具有多样性和游戏探索价值\n"
        "4. 水域与陆地分布是否平衡"
    )
    info_ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12)
    
    # 创建子图
    for i in range(engine.population_size):
        ax = fig.add_subplot(grid[i//n_cols+1, i%n_cols])
        axes.append(ax)
        
        biome_map = engine.population[i]
        assert len(biome_map.shape) == 2, "biome_map必须是二维数组"
        
        # 可视化
        ax.imshow(biome_map, cmap=matplotlib.colors.ListedColormap(list(color_mapping.values())))
        ax.set_title(f"方案 {i+1}", y=-0.15)
        ax.axis('off')
        
        # 初始化边框样式
        ax.patch.set_linewidth(3)
        ax.patch.set_edgecolor('white')
        
        # 添加选择状态指示器
        ax.text(0.02, 0.02, "", transform=ax.transAxes, 
                backgroundcolor='white', color='black',
                fontsize=12, fontweight='bold', ha='left', va='bottom')
        
        # 添加生物群系统计信息
        biome_counts = np.bincount(biome_map.flatten(), minlength=len(color_mapping))
        top_biomes = np.argsort(-biome_counts)[:3]  # 获取前三多的生物群系
        
        biome_names = list(color_mapping.keys())
        biome_info = "\n".join([f"{biome_names[i]}: {biome_counts[i]/biome_map.size:.1%}" 
                              for i in top_biomes if biome_counts[i] > 0])
        
        ax.text(0.98, 0.98, biome_info, transform=ax.transAxes,
                fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))
    
    # 添加状态信息面板
    status_text = fig.text(0.5, 0.01, 
                          "已选择: 0/3 方案", 
                          ha='center', fontsize=12, 
                          bbox=dict(boxstyle="round,pad=0.3", fc='lightblue', alpha=0.8))
    
    # 警告闪烁效果
    def flash_warning(text_element, flashes=3):
        for i in range(flashes):
            text_element.set_bbox(dict(boxstyle="round,pad=0.3", fc='red', alpha=0.8))
            text_element.set_text("已达到最大选择数量(3)！")
            if is_embedded:
                canvas.draw_idle()
            else:
                fig.canvas.draw_idle()
            
            if is_embedded:
                parent_frame.update()
            else:
                plt.pause(0.1)
            
            text_element.set_bbox(dict(boxstyle="round,pad=0.3", fc='lightblue', alpha=0.8))
            text_element.set_text(f"已选择: {len(selected)}/3 方案")
            if is_embedded:
                canvas.draw_idle()
            else:
                fig.canvas.draw_idle()
            
            if is_embedded:
                parent_frame.update()
            else:
                plt.pause(0.1)
    
    # 点击事件处理函数
    def on_click(event):
        if event.inaxes and event.inaxes in axes:
            ax_index = axes.index(event.inaxes)
            
            # 根据鼠标按键处理
            if event.button == 1:  # 左键 - 选择
                # 切换选择状态
                if ax_index in selected:
                    selected.remove(ax_index)
                    border_color = 'white'
                    selection_label = ""
                    bg_color = 'white'
                else:
                    if len(selected) >= 3: 
                        # 已达到最大选择数量，显示闪烁提醒
                        flash_warning(status_text)
                        return
                    selected.add(ax_index)
                    border_color = 'red'
                    selection_label = "已选择"
                    bg_color = '#ffffcc'
            
                # 更新视觉反馈
                event.inaxes.patch.set_linewidth(5)
                event.inaxes.patch.set_edgecolor(border_color)
                event.inaxes.set_facecolor(bg_color)
                event.inaxes.texts[0].set_text(selection_label)
                
                # 更新状态信息
                status_text.set_text(f"已选择: {len(selected)}/3 方案")
                
                # 重绘
                if is_embedded:
                    canvas.draw_idle()
                else:
                    fig.canvas.draw_idle()
                    plt.pause(0.01)
            
            elif event.button == 3:  # 右键 - 取消选择
                if ax_index in selected:
                    selected.remove(ax_index)
                    event.inaxes.patch.set_linewidth(3)
                    event.inaxes.patch.set_edgecolor('white')
                    event.inaxes.set_facecolor('white')
                    event.inaxes.texts[0].set_text("")
                    
                    # 更新状态信息
                    status_text.set_text(f"已选择: {len(selected)}/3 方案")
                    
                    # 重绘
                    if is_embedded:
                        canvas.draw_idle()
                    else:
                        fig.canvas.draw_idle()
                        plt.pause(0.01)
    
    # 键盘事件处理
    def on_key(event):
        if event.key == 'escape':  # ESC键关闭窗口
            if not is_embedded:
                plt.close(fig)
        elif event.key == 'r':  # R键重置选择
            selected.clear()
            for ax in axes:
                ax.patch.set_linewidth(3)
                ax.patch.set_edgecolor('white')
                ax.set_facecolor('white')
                ax.texts[0].set_text("")
            
            status_text.set_text("已选择: 0/3 方案")
            if is_embedded:
                canvas.draw_idle()
            else:
                fig.canvas.draw_idle()
    
    # 嵌入式模式下的最终处理
    if is_embedded:
        # 连接事件
        canvas.mpl_connect('button_press_event', on_click)
        canvas.mpl_connect('key_press_event', on_key)
        
        # 立即绘制
        canvas.draw()
        return scoring_completed  # 返回事件对象
    else:
        # 独立窗口模式
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 添加确认按钮
        confirm_ax = plt.axes([0.70, 0.01, 0.25, 0.06])
        confirm_button = plt.Button(
            confirm_ax, 
            '确认选择并继续', 
            color='lightgreen',
            hovercolor='green'
        )
        
        def on_confirm_clicked(event=None):
            # 获取评分
            scores = process_selections()
            
            # 先设置事件，确保在回调前就准备好了
            scoring_completed.set()
            
            # 调用回调函数
            if on_complete:
                print(f"评分已确认: 共{len(selected)}个选择，设置事件标志")
                on_complete(scores)
            
            # 关闭窗口
            plt.close(fig)
        
        confirm_button.on_clicked(on_confirm_clicked)
        
        # 连接事件
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show(block=True)  # 确保阻塞，避免代码继续执行
        return process_selections()

############################################################################################################

#from __future__ import annotations
#标准库
import random
import hashlib

#数据处理与科学计算
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, minimum_filter, maximum_filter
from numba import jit, prange, int32, float32, float64
import joblib
from scipy.signal import convolve2d

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#网络与并发

#其他工具
from opensimplex import OpenSimplex

#项目文件
from utils.tools import *

import numpy as np
from numba import jit, prange, float64, int32
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import convolve, convolve2d
import math

@jit(nopython=True)
def clip_value(value, min_val, max_val):
    """在Numba中替代np.clip，用于标量值"""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

# 修改 normalize_array 函数以处理 NaN 值
@jit(nopython=True, parallel=True)
def normalize_array(array):
    """增强版标准化数组函数，使用队列驱动的洪水填充算法更高效地处理NaN和极端值"""
    # 创建数组副本以避免修改原始数据
    result = array.copy()
    height, width = result.shape
    
    # 检测并标记无效值位置
    invalid_mask = np.zeros((height, width), dtype=np.bool_)
    valid_points = []  # 存储有效值的点（靠近无效区域边界的点）
    
    for y in range(height):
        for x in range(width):
            if np.isnan(result[y, x]) or np.isinf(result[y, x]) or result[y, x] == 0.0:
                invalid_mask[y, x] = True
                # 检查周围是否有有效点，如有则添加到种子列表
                for dy in range(max(0, y-1), min(height, y+2)):
                    for dx in range(max(0, x-1), min(width, x+2)):
                        if 0 <= dy < height and 0 <= dx < width:
                            if not (np.isnan(result[dy, dx]) or np.isinf(result[dy, dx]) or result[dy, dx] == 0.0):
                                valid_points.append((dy, dx))
    
    # 如果有无效值且有有效边界点，使用洪水填充算法
    if np.any(invalid_mask) and valid_points:
        # 去除重复的有效点
        unique_valid_points = []
        visited = set()
        
        for y, x in valid_points:
            if (y, x) not in visited:
                visited.add((y, x))
                unique_valid_points.append((y, x))
        
        # 从边界有效点开始扩散填充
        queue = list(unique_valid_points)
        visited_fill = np.zeros((height, width), dtype=np.bool_)
        
        while queue:
            y, x = queue.pop(0)
            
            # 对所有相邻的无效像素执行填充
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and invalid_mask[ny, nx] and not visited_fill[ny, nx]:
                        # 计算周围8个方向的有效值的加权平均
                        valid_sum = 0.0
                        weight_sum = 0.0
                        
                        for ndy in range(-2, 3):
                            for ndx in range(-2, 3):
                                nny, nnx = ny + ndy, nx + ndx
                                if 0 <= nny < height and 0 <= nnx < width and not invalid_mask[nny, nnx]:
                                    # 使用反距离加权
                                    dist = max(0.01, np.sqrt(ndy*ndy + ndx*ndx))
                                    weight = 1.0 / dist
                                    valid_sum += result[nny, nnx] * weight
                                    weight_sum += weight
                        
                        if weight_sum > 0:
                            result[ny, nx] = valid_sum / weight_sum
                            invalid_mask[ny, nx] = False
                            visited_fill[ny, nx] = True
                            queue.append((ny, nx))
    
    # 处理任何剩余的无效值（孤立区域）
    if np.any(invalid_mask):
        # 计算有效值的全局平均值
        valid_values = []
        for y in range(height):
            for x in range(width):
                if not invalid_mask[y, x]:
                    valid_values.append(result[y, x])
        
        if valid_values:
            global_avg = sum(valid_values) / len(valid_values)
            
            # 将所有剩余无效值设为全局平均值
            for y in range(height):
                for x in range(width):
                    if invalid_mask[y, x]:
                        result[y, x] = global_avg
    
    # 裁剪极端异常值（超过3个标准差）
    valid_values = []
    for i in range(result.size):
        if not (np.isnan(result.flat[i]) or np.isinf(result.flat[i])):
            valid_values.append(result.flat[i])
    
    if valid_values:
        mean_val = sum(valid_values) / len(valid_values)
        
        # 计算标准差 - 修改这里的生成器表达式为显式循环
        sum_squared_diff = 0.0
        for x in valid_values:
            sum_squared_diff += (x - mean_val) ** 2
        std_val = np.sqrt(sum_squared_diff / len(valid_values))
        
        # 裁剪极端值
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        
        for i in range(result.size):
            if result.flat[i] < lower_bound:
                result.flat[i] = lower_bound
            elif result.flat[i] > upper_bound:
                result.flat[i] = upper_bound
    
    # 标准化处理
    min_val = np.min(result)
    max_val = np.max(result)
    range_val = max_val - min_val
    
    # 防止除以零或极小值
    if range_val > 1e-6:  # 使用更保守的阈值
        return (result - min_val) / range_val
    else:
        # 如果范围非常小，返回常数中值
        return np.ones_like(result) * 0.5

# 优化的2D专用Simplex噪声实现，增加更多梯度向量和边缘平滑处理
@jit(float64(float64, float64, int32), nopython=True)
def simplex_noise(x, y, seed):
    """优化的2D专用Simplex噪声实现，提高随机性和自然度"""
    # 动态生成梯度向量表，提高多样性
    def generate_gradients(base_seed):
        # 生成32个梯度向量而非16个，增加随机性
        grad_table = np.zeros((32, 2), dtype=np.float64)
        
        # 基础正交和对角梯度 - 保证基本方向覆盖
        for i in range(8):
            angle = (i * np.pi / 4) + (base_seed % 100) * 0.01
            grad_table[i, 0] = np.cos(angle)
            grad_table[i, 1] = np.sin(angle)
        
        # 使用低差异序列生成更均匀分布的角度，替代纯随机
        for i in range(8, 32):
            # 使用修正的黄金比例法生成均匀分布的角度，避免角度聚集
            gold_ratio = 0.618033988749895  # 黄金比例
            angle = 2.0 * np.pi * ((i * gold_ratio) % 1.0)
            
            # 添加受控的随机扰动
            rand_seed = (base_seed * (i+1) * 1103515245) % (2**31)
            angle_jitter = (rand_seed % 1000) / 1000.0 * 0.2  # 小范围扰动
            angle = (angle + angle_jitter) % (2.0 * np.pi)
            
            # 随机长度梯度，增加多样性，但保持合理范围
            length = 0.7 + (rand_seed % 100) / 100.0 * 0.6  # 范围缩小到0.7-1.3
            grad_table[i, 0] = np.cos(angle) * length
            grad_table[i, 1] = np.sin(angle) * length
            
            # 归一化以保持一致性
            norm = np.sqrt(grad_table[i, 0]**2 + grad_table[i, 1]**2)
            if norm > 0.0001:
                grad_table[i, 0] /= norm
                grad_table[i, 1] /= norm
        
        return grad_table
    
    # 生成种子特定的梯度表
    _GRADIENTS_2D = generate_gradients(seed)
    
    # 使用32位Hash进一步打乱种子影响
    def hash_coords(ix, iy, seed):
        # Jenkins One-at-a-time hash
        hash_val = seed & 0xFFFFFFFF
        
        hash_val += ix & 0xFFFFFFFF
        hash_val += (hash_val << 10)
        hash_val ^= (hash_val >> 6)
        
        hash_val += iy & 0xFFFFFFFF
        hash_val += (hash_val << 10)
        hash_val ^= (hash_val >> 6)
        
        hash_val += (hash_val << 3)
        hash_val ^= (hash_val >> 11)
        hash_val += (hash_val << 15)
        
        return hash_val & 0x1F  # 返回0-31之间的索引
    
    # Skew因子 (2D simplex网格变换)
    F2 = 0.5 * (np.sqrt(3.0) - 1.0)
    G2 = (3.0 - np.sqrt(3.0)) / 6.0
    
    # 加入细微扰动提高自然性
    x += np.sin(y * 0.01) * 0.01
    y += np.sin(x * 0.01) * 0.01
    
    # 优化的坐标计算
    s = (x + y) * F2
    i = np.floor(x + s)
    j = np.floor(y + s)
    
    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t
    
    x0 = x - X0
    y0 = y - Y0
    
    # 确定单形内点的位置
    i1 = 1 if x0 > y0 else 0
    j1 = 0 if x0 > y0 else 1
    
    # 相对坐标
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2
    
    # 修改哈希索引计算
    ii = int(i) & 0xFFFFFFFF
    jj = int(j) & 0xFFFFFFFF
    
    # 使用增强哈希函数
    idx0 = hash_coords(ii, jj, seed)
    idx1 = hash_coords(ii + i1, jj + j1, seed)
    idx2 = hash_coords(ii + 1, jj + 1, seed)
    
    # 计算贡献值，使用更精确的梯度计算
    n0 = 0.0
    t0 = 0.5 - x0*x0 - y0*y0
    if t0 > 0:
        t0 *= t0
        t0 *= t0  # 使用t0^4而非t0^2提高平滑度
        n0 = t0 * (_GRADIENTS_2D[idx0, 0] * x0 + _GRADIENTS_2D[idx0, 1] * y0)
    
    n1 = 0.0
    t1 = 0.5 - x1*x1 - y1*y1
    if t1 > 0:
        t1 *= t1
        t1 *= t1
        n1 = t1 * (_GRADIENTS_2D[idx1, 0] * x1 + _GRADIENTS_2D[idx1, 1] * y1)
    
    n2 = 0.0
    t2 = 0.5 - x2*x2 - y2*y2
    if t2 > 0:
        t2 *= t2
        t2 *= t2
        n2 = t2 * (_GRADIENTS_2D[idx2, 0] * x2 + _GRADIENTS_2D[idx2, 1] * y2)
    
    # 将结果缩放到[-1,1]范围，增加幅度提高细节
    return 45.0 * (n0 + n1 + n2)

# 定义新的Worley噪声（元胞噪声）函数用于模拟断裂地形
@jit(nopython=True)
def worley_noise(x, y, seed):
    """Worley噪声实现，生成细胞状结构，适合模拟断裂带、岩石纹理等"""
    nx = int(x)
    ny = int(y)
    
    # 使用种子生成随机数
    seed_x = (seed * 1103515245 + 12345) % (2**31)
    seed_y = (seed * 134775813 + 1) % (2**31)
    
    min_dist = 1.0
    
    # 检查周围3x3个单元格
    for i in range(-1, 2):
        for j in range(-1, 2):
            # 确定当前单元格的特征点
            cx = nx + i
            cy = ny + j
            
            # 使用种子生成单元格内的随机点
            cell_seed_x = (cx * 1103515245 + seed_x) % (2**31)
            cell_seed_y = (cy * 134775813 + seed_y) % (2**31)
            
            # 单元格内随机点的偏移
            px = cx + (cell_seed_x % 1024) / 1024.0
            py = cy + (cell_seed_y % 1024) / 1024.0
            
            # 计算距离
            dx = x - px
            dy = y - py
            dist = np.sqrt(dx*dx + dy*dy)
            
            min_dist = min(min_dist, dist)
    
    return min_dist

# 域扭曲函数增强，提供更自然的地形流动感
@jit(nopython=True)
def advanced_domain_warping(x, y, seed, strength=1.0):
    """高级域扭曲函数，添加频率衰减控制"""
    # 基础扭曲 - 大尺度变形
    base_freq = 0.4
    dx = simplex_noise(x*base_freq, y*base_freq, seed) * 2.0 * strength
    dy = simplex_noise(x*base_freq + 5.2, y*base_freq + 3.7, seed) * 2.0 * strength
    
    # 二次扭曲 - 中尺度细节，频率衰减
    second_freq = base_freq * 0.75  # 频率衰减
    second_strength = strength * 0.5  # 强度衰减
    d2x = simplex_noise((x + dx)*second_freq, (y + dy)*second_freq, seed+1) * 1.0 * second_strength
    d2y = simplex_noise((x + dx)*second_freq + 1.7, (y + dy)*second_freq + 8.3, seed+1) * 1.0 * second_strength
    
    # 细节扭曲 - 小尺度纹理，进一步频率衰减
    detail_freq = second_freq * 0.6  # 更小的频率
    detail_strength = second_strength * 0.3  # 更小的强度
    flow_x = simplex_noise((x + dx + d2x)*detail_freq*2, (y + dy + d2y)*detail_freq*2, seed+2) * 0.3 * detail_strength
    flow_y = simplex_noise((x + dx + d2x)*detail_freq*2 + 2.7, (y + dy + d2y)*detail_freq*2 + 4.3, seed+2) * 0.3 * detail_strength
    
    # 组合所有扭曲，应用非线性抑制以防止极端变形
    # 使用tanh函数限制最终位移幅度
    total_dx = dx + d2x + flow_x
    total_dy = dy + d2y + flow_y
    
    # 非线性抑制极端扭曲
    max_displacement = 3.0 * strength
    total_dx = np.tanh(total_dx / max_displacement) * max_displacement
    total_dy = np.tanh(total_dy / max_displacement) * max_displacement
    
    warped_x = x + total_dx
    warped_y = y + total_dy
    
    return warped_x, warped_y

# 高级噪声组合函数，结合多种噪声以增加自然度
@jit(nopython=True)
def natural_terrain_noise(x, y, seed, terrain_type="mixed", elevation=0.5, gradient=0.1):
    """具有动态权重的地形噪声函数，根据高度和梯度自适应调整不同噪声的权重"""
    # 基本权重 - 与原始函数相同
    if terrain_type == "mountains":
        simplex_weight = 0.70
        worley_weight = 0.15
        ridge_weight = 0.15
        warp_strength = 1.2
    elif terrain_type == "hills":
        simplex_weight = 0.65
        worley_weight = 0.05
        ridge_weight = 0.30
        warp_strength = 0.8
    elif terrain_type == "plains":
        simplex_weight = 0.55
        worley_weight = 0.05
        ridge_weight = 0.40
        warp_strength = 0.5
    elif terrain_type == "badlands":
        simplex_weight = 0.50
        worley_weight = 0.35
        ridge_weight = 0.15
        warp_strength = 1.0
    elif terrain_type == "canyon":
        simplex_weight = 0.40
        worley_weight = 0.50
        ridge_weight = 0.10
        warp_strength = 1.5
    else:  # mixed
        simplex_weight = 0.60
        worley_weight = 0.20
        ridge_weight = 0.20
        warp_strength = 1.0
    
    # 动态调整权重
    # 高度越高，增加山脊特征以提高山峰锐利度
    if elevation > 0.7:
        ridge_factor = min(1.0, (elevation - 0.7) * 3.0)
        ridge_boost = ridge_factor * 0.2
        simplex_weight = max(0.3, simplex_weight - ridge_boost * 0.5)
        worley_weight = max(0.05, worley_weight - ridge_boost * 0.5)
        ridge_weight += ridge_boost
    
    # 坡度越大，增加Worley噪声比例以模拟崎岖地形
    if gradient > 0.3:
        worley_factor = min(1.0, (gradient - 0.3) * 2.0)
        worley_boost = worley_factor * 0.15
        simplex_weight = max(0.3, simplex_weight - worley_boost)
        worley_weight += worley_boost
    
    # 归一化权重确保总和为1
    total_weight = simplex_weight + worley_weight + ridge_weight
    simplex_weight /= total_weight
    worley_weight /= total_weight
    ridge_weight /= total_weight
    
    # 应用域扭曲
    warped_x, warped_y = advanced_domain_warping(x, y, seed, strength=warp_strength)
    
    # 基础Simplex噪声
    simplex_value = simplex_noise(warped_x, warped_y, seed)
    
    # Worley噪声（元胞噪声）
    worley_value = 1.0 - worley_noise(warped_x * 0.5, warped_y * 0.5, seed+10)
    
    # 增加脊线噪声（Ridge noise）模拟山脊
    ridge_noise = abs(simplex_noise(warped_x * 0.7, warped_y * 0.7, seed+20))
    ridge_value = 1.0 - ridge_noise
    # 使用平方根而非平方，减小差异放大
    ridge_value = np.sqrt(ridge_value)  # 或使用 ridge_value**0.75 温和放大
    
    # 组合噪声
    combined = (simplex_value * simplex_weight + 
                worley_value * worley_weight + 
                ridge_value * ridge_weight)
    
    return combined

# 将原来的函数拆分为两个独立函数
@jit(nopython=True)
def advanced_fractal_noise_scalar(width, height, seed, octaves=6, persistence=0.5, lacunarity=2.0, terrain_type="mixed"):
    """使用标量种子的分形噪声生成函数"""
    result = np.zeros((height, width), dtype=np.float64)
    
    # 使用粉红噪声（1/f噪声）的频率分布
    freq = 1.0
    amp = 1.0
    max_val = 0.0
    
    # 根据地形类型调整参数
    if terrain_type == "mountains":
        base_persistence = persistence * 1.1
        base_lacunarity = lacunarity * 0.9
        octave_weight_factor = 0.8  # 低频权重更高
    elif terrain_type == "hills":
        base_persistence = persistence * 0.9
        base_lacunarity = lacunarity * 1.0
        octave_weight_factor = 0.9
    elif terrain_type == "plains":
        base_persistence = persistence * 0.8
        base_lacunarity = lacunarity * 1.1
        octave_weight_factor = 1.0  # 平衡分布
    elif terrain_type == "badlands":
        base_persistence = persistence * 1.0
        base_lacunarity = lacunarity * 1.2
        octave_weight_factor = 1.1  # 高频权重更高
    else:
        base_persistence = persistence
        base_lacunarity = lacunarity
        octave_weight_factor = 1.0
    
    # 应用八度音阶
    for i in range(octaves):
        # 为每个像素生成噪声
        noise_layer = np.zeros((height, width), dtype=np.float64)
        for y in range(height):
            for x in range(width):
                # 每个八度使用不同的噪声类型，增加多样性
                if i % 3 == 0:
                    noise_val = natural_terrain_noise(x * freq / width, y * freq / height, seed + i, terrain_type)
                elif i % 3 == 1:
                    # 使用域扭曲创建流体状纹理
                    warped_x, warped_y = advanced_domain_warping(x * freq / width, y * freq / height, seed + i, 0.7)
                    noise_val = simplex_noise(warped_x, warped_y, seed + i)
                else:
                    # 混合Simplex和Worley创建复杂纹理
                    simplex_val = simplex_noise(x * freq / width, y * freq / height, seed + i)
                    worley_val = 1.0 - worley_noise(x * freq / width * 0.5, y * freq / height * 0.5, seed + i + 42)
                    noise_val = simplex_val * 0.7 + worley_val * 0.3
                
                noise_layer[y, x] = noise_val
        
        # 使用调整后的持续性计算振幅
        current_amp = amp * (1.0 / (1.0 + i * octave_weight_factor))  # 模拟1/f粉红噪声频谱
        result += noise_layer * current_amp
        max_val += current_amp
        
        # 更新频率和振幅
        freq *= base_lacunarity
        amp *= base_persistence
    
    if max_val > 0:
        result /= max_val
    
    # 在每个主要计算函数末尾添加
    # 检查结果中的NaN并替换
    for y in range(height):
        for x in range(width):
            if np.isnan(result[y, x]):
                # 尝试使用周围值的平均值
                valid_neighbors = []
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and not np.isnan(result[ny, nx]):
                            valid_neighbors.append(result[ny, nx])
                
                if valid_neighbors:
                    result[y, x] = sum(valid_neighbors) / len(valid_neighbors)
                else:
                    result[y, x] = 0.5  # 默认安全值
    
    return result

@jit(nopython=True)
def advanced_fractal_noise_scales(width, height, scales, seed, terrain_type="mixed"):
    """使用尺度列表的分形噪声生成函数"""
    result = np.zeros((height, width), dtype=np.float64)
    total_weight = 0.0
    
    # 假设scales是一个包含(scale, weight)元组的数组
    for i in range(len(scales)):
        scale = scales[i][0]  # 第一个元素是尺度
        weight = scales[i][1]  # 第二个元素是权重
        
        # 为每个尺度生成噪声
        scale_noise = np.zeros((height, width), dtype=np.float64)
        for y in range(height):
            for x in range(width):
                # 使用不同尺度的坐标
                nx = x / scale
                ny = y / scale
                scale_noise[y, x] = natural_terrain_noise(nx, ny, seed, terrain_type)
        
        # 按权重累加
        result += scale_noise * weight
        total_weight += weight
    
    if total_weight > 0:
        result /= total_weight
    
    # 在每个主要计算函数末尾添加
    # 检查结果中的NaN并替换
    for y in range(height):
        for x in range(width):
            if np.isnan(result[y, x]):
                # 尝试使用周围值的平均值
                valid_neighbors = []
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and not np.isnan(result[ny, nx]):
                            valid_neighbors.append(result[ny, nx])
                
                if valid_neighbors:
                    result[y, x] = sum(valid_neighbors) / len(valid_neighbors)
                else:
                    result[y, x] = 0.5  # 默认安全值
    
    return result

@jit(nopython=True, parallel=True)
def generate_noise_map_naturalistic(width, height, scales, seed, terrain_types_array):
    """批量优化的噪声生成函数，一次性处理整个区域以提高性能"""
    result = np.zeros((height, width), dtype=np.float64)
    
    # 预计算不同地形类型的参数
    terrain_params = {}
    terrain_types = ["mountains", "hills", "plains", "badlands", "canyon"]
    
    for terrain_idx, terrain_type in enumerate(terrain_types):
        if terrain_type == "mountains":
            octaves = 8
            persistence = 0.65
            lacunarity = 2.2
        elif terrain_type == "hills":
            octaves = 6
            persistence = 0.55
            lacunarity = 2.0
        elif terrain_type == "plains":
            octaves = 5
            persistence = 0.45
            lacunarity = 1.8
        elif terrain_type == "badlands":
            octaves = 7
            persistence = 0.6
            lacunarity = 2.4
        elif terrain_type == "canyon":
            octaves = 8
            persistence = 0.7
            lacunarity = 2.5
        else:
            octaves = 6
            persistence = 0.5
            lacunarity = 2.0
        
        terrain_params[terrain_idx] = (octaves, persistence, lacunarity)
    
    # 分块处理 - 将地图分为小块以提高并行效率和安全性
    block_size = 32  # 选择合适的块大小
    num_blocks_y = (height + block_size - 1) // block_size
    num_blocks_x = (width + block_size - 1) // block_size
    
    # 并行处理块
    for block_y in prange(num_blocks_y):
        # 计算当前块的y范围
        y_start = block_y * block_size
        y_end = min(y_start + block_size, height)
        
        # 每个线程有自己的块，避免数据竞争
        for block_x in range(num_blocks_x):
            # 计算当前块的x范围
            x_start = block_x * block_size
            x_end = min(x_start + block_size, width)
            
            # 为当前块创建临时结果数组
            block_result = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float64)
            
            # 每个八度音阶的噪声计算 - 非并行处理单个块
            for octave in range(8):
                octave_seed = seed + octave
                
                # 处理当前块的所有像素
                for y_local in range(y_end - y_start):
                    y = y_start + y_local
                    
                    for x_local in range(x_end - x_start):
                        x = x_start + x_local
                        
                        # 获取地形类型参数
                        terrain_idx = terrain_types_array[y, x]
                        if terrain_idx >= len(terrain_types):
                            terrain_idx = len(terrain_types) - 1
                        
                        octaves, persistence, lacunarity = terrain_params[terrain_idx]
                        
                        # 如果当前八度在此地形范围内
                        if octave < octaves:
                            # 计算噪声
                            nx = x * (2.0**octave) / width * scales[0, 0]
                            ny = y * (2.0**octave) / width * scales[0, 0]
                            
                            # 根据八度选择噪声类型
                            if octave % 3 == 0:
                                noise_val = simplex_noise(nx, ny, octave_seed)
                            elif octave % 3 == 1:
                                warped_x, warped_y = advanced_domain_warping(nx, ny, octave_seed, 0.7)
                                noise_val = simplex_noise(warped_x, warped_y, octave_seed)
                            else:
                                simplex_val = simplex_noise(nx, ny, octave_seed)
                                worley_val = 1.0 - worley_noise(nx * 0.5, ny * 0.5, octave_seed + 42)
                                noise_val = simplex_val * 0.7 + worley_val * 0.3
                            
                            # 计算八度权重
                            octave_weight = (1.0 / (1.0 + octave * 0.8))
                            if terrain_idx == 0:  # 山地
                                octave_weight *= 1.2  # 增强低频山脉形态
                            elif terrain_idx == 2:  # 平原
                                octave_weight *= 0.8  # 减弱平原起伏
                            
                            # 应用衰减
                            amp = persistence ** octave
                            
                            # 累加到块结果
                            block_result[y_local, x_local] += noise_val * amp * octave_weight
            
            # 将块结果复制到主结果数组
            for y_local in range(y_end - y_start):
                for x_local in range(x_end - x_start):
                    result[y_start + y_local, x_start + x_local] = block_result[y_local, x_local]
    
    # 归一化结果
    return normalize_array(result)

@jit(nopython=True)
def frequency_domain_noise(width, height, seed, octaves=6, persistence=0.5, lacunarity=2.0):
    """使用频域合成直接生成分形噪声，避免逐八度叠加"""
    # 创建输出数组
    result = np.zeros((height, width), dtype=np.float64)
    
    # 计算总共需要的点数量
    total_points = 0
    freq = 1.0
    for i in range(octaves):
        # 根据频率计算每个八度的采样点数
        points_x = max(2, int(width * freq * 0.1))
        points_y = max(2, int(height * freq * 0.1))
        total_points += points_x * points_y
        freq *= lacunarity
    
    # 一次性生成所有采样点的噪声值
    noise_values = np.zeros(total_points, dtype=np.float64)
    
    # 填充噪声值
    point_index = 0
    freq = 1.0
    
    for octave in range(octaves):
        # 计算此频率下的采样点数
        points_x = max(2, int(width * freq * 0.1))
        points_y = max(2, int(height * freq * 0.1))
        
        # 为每个采样点生成一个随机值
        octave_seed = seed + octave
        for py in range(points_y):
            y_pos = py / (points_y - 1)
            for px in range(points_x):
                x_pos = px / (points_x - 1)
                
                # 生成噪声值
                noise_values[point_index] = simplex_noise(x_pos * 10, y_pos * 10, octave_seed)
                point_index += 1
        
        freq *= lacunarity
    
    # 使用生成的值插值计算最终噪声
    point_index = 0
    freq = 1.0
    amp = 1.0
    max_val = 0.0
    
    for octave in range(octaves):
        # 计算此频率下的采样点
        points_x = max(2, int(width * freq * 0.1))
        points_y = max(2, int(height * freq * 0.1))
        
        # 对每个像素进行双线性插值
        for y in range(height):
            y_norm = y / height
            # 计算插值的网格点
            py = y_norm * (points_y - 1)
            py_i = int(py)
            py_f = py - py_i
            py_i1 = min(py_i + 1, points_y - 1)
            
            for x in range(width):
                x_norm = x / width
                # 计算插值的网格点
                px = x_norm * (points_x - 1)
                px_i = int(px)
                px_f = px - px_i
                px_i1 = min(px_i + 1, points_x - 1)
                
                # 获取4个邻近采样点的值
                idx00 = point_index + py_i * points_x + px_i
                idx01 = point_index + py_i * points_x + px_i1
                idx10 = point_index + py_i1 * points_x + px_i
                idx11 = point_index + py_i1 * points_x + px_i1
                
                v00 = noise_values[idx00]
                v01 = noise_values[idx01]
                v10 = noise_values[idx10]
                v11 = noise_values[idx11]
                
                # 双线性插值
                vx0 = v00 * (1 - px_f) + v01 * px_f
                vx1 = v10 * (1 - px_f) + v11 * px_f
                noise_val = vx0 * (1 - py_f) + vx1 * py_f
                
                # 累加到结果
                result[y, x] += noise_val * amp
            
        # 累加幅度以便归一化
        max_val += amp
        
        # 更新频率和振幅
        point_index += points_x * points_y
        freq *= lacunarity
        amp *= persistence
    
    # 归一化
    if max_val > 0:
        result /= max_val
    
    return result

##################################################################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import os

class MapGenerationVisualizer:
    """地图生成过程的可视化工具，支持嵌入到GUI中"""
    
    def __init__(self, save_path=None, show_plots=True, logger=None, gui_frame=None):
        """
        初始化可视化工具
        
        Args:
            save_path: 保存图像的路径，如果为None则不保存
            show_plots: 是否显示图像
            logger: 日志记录器
            gui_frame: Tkinter框架，用于嵌入可视化
        """
        self.save_path = save_path
        self.show_plots = show_plots
        self.logger = logger
        self.gui_frame = gui_frame
        self.step_count = 0
        self.canvas = None
        self.fig = None
        self.ax = None
        
        # 如果需要保存图像且路径不存在，则创建
        if self.save_path and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        # 存储生成的所有图像，以便最终可视化报告
        self.all_visualizations = []
        
        # 如果提供了GUI框架，设置画布
        if self.gui_frame:
            self._setup_gui_canvas()
    
    def _setup_gui_canvas(self):
        """在GUI框架中设置matplotlib画布"""
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import tkinter as tk
            
            print(f"初始化GUI画布: frame类型={type(self.gui_frame).__name__}")
            
            # 创建图形和画布
            self.fig = Figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.gui_frame)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # 添加关闭按钮
            close_button = tk.Button(self.gui_frame, text="关闭预览", 
                                    command=self._close_preview)
            close_button.pack(side=tk.BOTTOM, pady=5)
            
            # 标题区域
            self.fig.suptitle("地图生成可视化", fontsize=14)
            
            print(f"GUI画布初始化成功: canvas={self.canvas is not None}")
            
        except Exception as e:
            print(f"GUI画布初始化错误: {str(e)}\n{traceback.format_exc()}")
            self.canvas = None
            self.fig = None
            self.ax = None

    def _close_preview(self):
        """关闭预览画布"""
        if self.canvas:
            for widget in self.gui_frame.winfo_children():
                widget.destroy()
            # 通知应用预览已关闭
            if self.logger:
                print("预览已关闭")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
    
    def visualize_height_map(self, height_map, title="高度图", step_name=None):
        """可视化高度图"""
        if height_map is None:
            if self.logger:
                print("无法可视化高度图：数据为空")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            return

        self.step_count += 1
        step_title = f"步骤 {self.step_count}: {title}" if step_name is None else f"步骤 {self.step_count}: {step_name} - {title}"
        
        # 如果我们有GUI画布，使用它
        if self.gui_frame and self.canvas:
            # 清除之前的图形
            self.ax.clear()
            
            # 绘制新图形
            img = self.ax.imshow(height_map, cmap='terrain')
            self.fig.colorbar(img, ax=self.ax, label='高度')
            self.ax.set_title(step_title)
            
            # 更新画布 - 添加强制更新
            self.canvas.draw()
            self.canvas.flush_events()  # 尝试强制刷新事件
            
            # 确保主窗口更新
            if hasattr(self.gui_frame, 'update'):
                self.gui_frame.update_idletasks()
        else:
            # 传统的独立窗口模式
            plt.figure(figsize=(10, 8))
            plt.imshow(height_map, cmap='terrain')
            plt.colorbar(label='高度')
            plt.title(step_title)
            
            # 保存图像
            if self.save_path:
                filename = f"{self.step_count:02d}_{step_name or title}.png"
                filepath = os.path.join(self.save_path, filename)
                plt.savefig(filepath, dpi=150)
                if self.logger:
                    print(f"保存高度图到: {filepath}")
                else:
                    print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            
            # 记录可视化内容
            self.all_visualizations.append({
                'step': self.step_count,
                'title': step_title,
                'type': 'height_map',
                'data': height_map.copy() if isinstance(height_map, np.ndarray) else height_map
            })
            
            # 显示图像
            if self.show_plots:
                plt.show()
            else:
                plt.close()
    
    def visualize_temperature_map(self, temp_map, title="温度图", step_name=None):
        """可视化温度图"""
        if temp_map is None:
            if self.logger:
                print("无法可视化温度图：数据为空")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            return

        self.step_count += 1
        step_title = f"步骤 {self.step_count}: {title}" if step_name is None else f"步骤 {self.step_count}: {step_name} - {title}"
        
        # 如果我们有GUI画布，使用它
        if self.gui_frame and self.canvas:
            # 清除之前的图形
            self.ax.clear()
            
            # 绘制新图形
            img = self.ax.imshow(temp_map, cmap='coolwarm')
            self.fig.colorbar(img, ax=self.ax, label='温度')
            self.ax.set_title(step_title)
            
            # 更新画布
            self.canvas.draw()
        else:
            # 传统的独立窗口模式
            plt.figure(figsize=(10, 8))
            plt.imshow(temp_map, cmap='coolwarm')
            plt.colorbar(label='温度')
            plt.title(step_title)
            
            # 保存图像
            if self.save_path:
                filename = f"{self.step_count:02d}_{step_name or title}.png" 
                filepath = os.path.join(self.save_path, filename)
                plt.savefig(filepath, dpi=150)
                if self.logger:
                    print(f"保存温度图到: {filepath}")
                else:
                    print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            
            # 记录可视化内容
            self.all_visualizations.append({
                'step': self.step_count,
                'title': step_title,
                'type': 'temperature_map',
                'data': temp_map.copy() if isinstance(temp_map, np.ndarray) else temp_map
            })
            
            # 显示图像
            if self.show_plots:
                plt.show()
            else:
                plt.close()
    
    def visualize_humidity_map(self, humid_map, title="湿度图", step_name=None):
        """可视化湿度图"""
        if humid_map is None:
            if self.logger:
                print("无法可视化湿度图：数据为空")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            return

        self.step_count += 1
        step_title = f"步骤 {self.step_count}: {title}" if step_name is None else f"步骤 {self.step_count}: {step_name} - {title}"
        
        # 如果我们有GUI画布，使用它
        if self.gui_frame and self.canvas:
            # 清除之前的图形
            self.ax.clear()
            
            # 绘制新图形
            img = self.ax.imshow(humid_map, cmap='Blues')
            self.fig.colorbar(img, ax=self.ax, label='湿度')
            self.ax.set_title(step_title)
            
            # 更新画布
            self.canvas.draw()
        else:
            # 传统的独立窗口模式
            plt.figure(figsize=(10, 8))
            plt.imshow(humid_map, cmap='Blues')
            plt.colorbar(label='湿度')
            plt.title(step_title)
            
            # 保存图像
            if self.save_path:
                filename = f"{self.step_count:02d}_{step_name or title}.png"
                filepath = os.path.join(self.save_path, filename)
                plt.savefig(filepath, dpi=150)
                if self.logger:
                    print(f"保存湿度图到: {filepath}")
                else:
                    print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            
            # 记录可视化内容
            self.all_visualizations.append({
                'step': self.step_count,
                'title': step_title,
                'type': 'humidity_map',
                'data': humid_map.copy() if isinstance(humid_map, np.ndarray) else humid_map
            })
            
            # 显示图像
            if self.show_plots:
                plt.show()
            else:
                plt.close()

    def visualize_rivers(self, height_map, rivers_map, title="河流系统", step_name=None):
        """可视化河流系统"""
        if height_map is None or rivers_map is None:
            if self.logger:
                print("无法可视化河流系统：数据为空")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            return

        self.step_count += 1
        step_title = f"步骤 {self.step_count}: {title}" if step_name is None else f"步骤 {self.step_count}: {step_name} - {title}"
        
        # 添加调试信息，帮助诊断问题
        if self.logger:
            print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
        else:
            print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
        
        # 嵌入式模式检查 - 确保canvas已初始化
        if self.gui_frame is not None:
            # 重新尝试初始化canvas，如果它尚未创建
            if self.canvas is None:
                self._setup_gui_canvas()
            
            # 再次检查canvas是否成功创建
            if self.canvas is not None:
                # 清除之前的图形
                if self.ax:
                    self.ax.clear()
                
                # 绘制基础高度图作为背景
                self.ax.imshow(height_map, cmap='terrain', alpha=0.7)
                
                # 叠加河流
                river_cmap = ListedColormap(['none', 'blue'])
                river_overlay = np.zeros_like(height_map)
                river_overlay[rivers_map > 0] = 1
                self.ax.imshow(river_overlay, cmap=river_cmap, alpha=0.7)
                
                self.ax.set_title(step_title)
                
                # 确保绘图更新
                self.fig.tight_layout()
                self.canvas.draw_idle()
                
                # 记录成功嵌入
                if self.logger:
                    print("河流可视化已成功嵌入到GUI")
                else:
                    print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
                return
            else:
                # canvas创建失败，记录警告
                if self.logger:
                    print("无法创建嵌入式画布，将使用独立窗口", "WARNING")
                else:
                    print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
        
        # 独立窗口模式
        plt.figure(figsize=(10, 8))
        
        # 基础高度图作为背景
        plt.imshow(height_map, cmap='terrain', alpha=0.7)
        
        # 叠加河流
        river_overlay = np.zeros_like(height_map)
        river_overlay[rivers_map > 0] = 1
        plt.imshow(river_overlay, cmap=ListedColormap(['none', 'blue']), alpha=0.7)
        
        plt.title(step_title)
        
        # 保存图像
        if self.save_path:
            filename = f"{self.step_count:02d}_{step_name or title}.png"
            filepath = os.path.join(self.save_path, filename)
            plt.savefig(filepath, dpi=150)
            if self.logger:
                print(f"保存河流系统图到: {filepath}")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
        
        # 记录可视化内容
        self.all_visualizations.append({
            'step': self.step_count,
            'title': step_title,
            'type': 'rivers',
            'data': {
                'height_map': height_map.copy() if isinstance(height_map, np.ndarray) else height_map,
                'rivers_map': rivers_map.copy() if isinstance(rivers_map, np.ndarray) else rivers_map
            }
        })
        
        # 显示图像
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def visualize_biome_map(self, biome_map, biome_data, title="生物群系分布", step_name=None):
        """可视化生物群系分布"""
        if biome_map is None:
            if self.logger:
                print("无法可视化生物群系：数据为空")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            return

        self.step_count += 1
        step_title = f"步骤 {self.step_count}: {title}" if step_name is None else f"步骤 {self.step_count}: {step_name} - {title}"
        
        # 创建生物群系颜色映射
        max_biome_id = np.max(biome_map) + 1
        colors = plt.cm.tab20(np.linspace(0, 1, max_biome_id))
        biome_cmap = ListedColormap(colors)
        
        # 如果我们有GUI画布，使用它
        if self.gui_frame and self.canvas:
            # 清除之前的图形
            self.ax.clear()
            
            # 显示生物群系图
            img = self.ax.imshow(biome_map, cmap=biome_cmap)
            self.ax.set_title(step_title)
            
            # 创建图例
            unique_biomes = np.unique(biome_map)
            legend_entries = []
            legend_labels = []
            
            for biome_id in unique_biomes:
                if biome_id >= 0 and biome_id < len(colors):
                    color = colors[biome_id]
                    legend_entries.append(plt.Rectangle((0,0), 1, 1, fc=color))
                    
                    # 获取生物群系名称
                    biome_name = f"生物群系 {biome_id}"
                    if biome_data and 'biomes' in biome_data:
                        for biome_info in biome_data['biomes']:
                            if biome_info.get('id') == biome_id:
                                biome_name = biome_info.get('name', biome_name)
                                break
                    
                    legend_labels.append(biome_name)
            
            # 添加图例
            self.ax.legend(legend_entries, legend_labels, loc='upper right', 
                        bbox_to_anchor=(1.2, 1), fontsize='small')
            
            # 更新画布
            self.canvas.draw()
        else:
            # 传统的独立窗口模式
            plt.figure(figsize=(12, 10))
            
            # 显示生物群系图
            img = plt.imshow(biome_map, cmap=biome_cmap)
            plt.title(step_title)
            
            # 创建图例
            unique_biomes = np.unique(biome_map)
            legend_entries = []
            legend_labels = []
            
            for biome_id in unique_biomes:
                if biome_id >= 0 and biome_id < len(colors):
                    color = colors[biome_id]
                    legend_entries.append(plt.Rectangle((0,0), 1, 1, fc=color))
                    
                    # 获取生物群系名称
                    biome_name = f"生物群系 {biome_id}"
                    if biome_data and 'biomes' in biome_data:
                        for biome_info in biome_data['biomes']:
                            if biome_info.get('id') == biome_id:
                                biome_name = biome_info.get('name', biome_name)
                                break
                    
                    legend_labels.append(biome_name)
            
            # 添加图例
            plt.legend(legend_entries, legend_labels, loc='upper right', bbox_to_anchor=(1.2, 1))
            
            # 保存图像
            if self.save_path:
                filename = f"{self.step_count:02d}_{step_name or title}.png"
                filepath = os.path.join(self.save_path, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                if self.logger:
                    print(f"保存生物群系图到: {filepath}")
                else:
                    print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            
            # 记录可视化内容
            self.all_visualizations.append({
                'step': self.step_count,
                'title': step_title,
                'type': 'biome_map',
                'data': biome_map.copy() if isinstance(biome_map, np.ndarray) else biome_map
            })
            
            # 显示图像
            if self.show_plots:
                plt.show()
            else:
                plt.close()

    def visualize_objects(self, height_map, vegetation=None, buildings=None, roads=None, title="对象分布", step_name=None):
        """可视化对象分布（植被、建筑、道路等）"""
        if height_map is None:
            if self.logger:
                print("无法可视化对象分布：高度图数据为空")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            return

        self.step_count += 1
        step_title = f"步骤 {self.step_count}: {title}" if step_name is None else f"步骤 {self.step_count}: {step_name} - {title}"
        
        # 如果我们有GUI画布，使用它
        if self.gui_frame and self.canvas:
            # 清除之前的图形
            self.ax.clear()
            
            # 绘制高度图作为背景
            self.ax.imshow(height_map, cmap='terrain', alpha=0.7)
            
            # 绘制植被
            if vegetation and len(vegetation) > 0:
                veg_x = [v.get('x', 0) for v in vegetation if 'x' in v]
                veg_y = [v.get('y', 0) for v in vegetation if 'y' in v]
                self.ax.scatter(veg_x, veg_y, c='green', s=10, alpha=0.6, label='植被')
            
            # 绘制建筑
            if buildings and len(buildings) > 0:
                build_x = [b.get('x', 0) for b in buildings if 'x' in b]
                build_y = [b.get('y', 0) for b in buildings if 'y' in b]
                self.ax.scatter(build_x, build_y, c='red', s=25, alpha=0.8, label='建筑')
            
            # 绘制道路
            if roads and len(roads) > 0:
                road_x = [r.get('x', 0) for r in roads if 'x' in r]
                road_y = [r.get('y', 0) for r in roads if 'y' in r]
                self.ax.scatter(road_x, road_y, c='black', s=3, alpha=0.8, label='道路')
            
            self.ax.set_title(step_title)
            self.ax.legend()
            
            # 更新画布
            self.canvas.draw()
        else:
            # 传统的独立窗口模式
            plt.figure(figsize=(12, 10))
            
            # 绘制高度图作为背景
            plt.imshow(height_map, cmap='terrain', alpha=0.7)
            
            # 绘制植被
            if vegetation and len(vegetation) > 0:
                veg_x = [v.get('x', 0) for v in vegetation if 'x' in v]
                veg_y = [v.get('y', 0) for v in vegetation if 'y' in v]
                plt.scatter(veg_x, veg_y, c='green', s=10, alpha=0.6, label='植被')
            
            # 绘制建筑
            if buildings and len(buildings) > 0:
                build_x = [b.get('x', 0) for b in buildings if 'x' in b]
                build_y = [b.get('y', 0) for b in buildings if 'y' in b]
                plt.scatter(build_x, build_y, c='red', s=25, alpha=0.8, label='建筑')
            
            # 绘制道路
            if roads and len(roads) > 0:
                road_x = [r.get('x', 0) for r in roads if 'x' in r]
                road_y = [r.get('y', 0) for r in roads if 'y' in r]
                plt.scatter(road_x, road_y, c='black', s=3, alpha=0.8, label='道路')
            
            plt.title(step_title)
            plt.legend()
            
            # 保存图像
            if self.save_path:
                filename = f"{self.step_count:02d}_{step_name or title}.png"
                filepath = os.path.join(self.save_path, filename)
                plt.savefig(filepath, dpi=150)
                if self.logger:
                    print(f"保存对象分布图到: {filepath}")
                else:
                    print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            
            # 记录可视化内容
            self.all_visualizations.append({
                'step': self.step_count,
                'title': step_title,
                'type': 'objects',
                'data': {
                    'height_map': height_map.copy() if isinstance(height_map, np.ndarray) else height_map,
                    'vegetation': vegetation,
                    'buildings': buildings,
                    'roads': roads
                }
            })
            
            # 显示图像
            if self.show_plots:
                plt.show()
            else:
                plt.close()

    def visualize_microclimate(self, microclimate_map, height_map=None, title="微气候区域", step_name=None):
        """可视化微气候区域"""
        if microclimate_map is None:
            if self.logger:
                print("无法可视化微气候：数据为空")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            return

        self.step_count += 1
        step_title = f"步骤 {self.step_count}: {title}" if step_name is None else f"步骤 {self.step_count}: {step_name} - {title}"
        
        # 如果我们有GUI画布，使用它
        if self.gui_frame and self.canvas:
            # 清除之前的图形
            self.ax.clear()
            
            if height_map is not None:
                # 使用高度图作为背景
                self.ax.imshow(height_map, cmap='terrain', alpha=0.5)
                
                # 叠加微气候区域
                img = self.ax.imshow(microclimate_map, cmap='plasma', alpha=0.5)
            else:
                # 只显示微气候区域
                img = self.ax.imshow(microclimate_map, cmap='plasma')
            
            self.fig.colorbar(img, ax=self.ax, label='微气候指数')
            self.ax.set_title(step_title)
            
            # 更新画布
            self.canvas.draw()
        else:
            # 传统的独立窗口模式
            plt.figure(figsize=(10, 8))
            
            if height_map is not None:
                # 使用高度图作为背景
                plt.imshow(height_map, cmap='terrain', alpha=0.5)
                
                # 叠加微气候区域
                plt.imshow(microclimate_map, cmap='plasma', alpha=0.5)
            else:
                # 只显示微气候区域
                plt.imshow(microclimate_map, cmap='plasma')
            
            plt.colorbar(label='微气候指数')
            plt.title(step_title)
            
            # 保存图像
            if self.save_path:
                filename = f"{self.step_count:02d}_{step_name or title}.png"
                filepath = os.path.join(self.save_path, filename)
                plt.savefig(filepath, dpi=150)
                if self.logger:
                    print(f"保存微气候图到: {filepath}")
                else:
                    print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            
            # 记录可视化内容
            self.all_visualizations.append({
                'step': self.step_count,
                'title': step_title,
                'type': 'microclimate',
                'data': {
                    'microclimate_map': microclimate_map.copy() if isinstance(microclimate_map, np.ndarray) else microclimate_map,
                    'height_map': height_map.copy() if isinstance(height_map, np.ndarray) else height_map
                }
            })
            
            # 显示图像
            if self.show_plots:
                plt.show()
            else:
                plt.close()
    
    def generate_summary_visualization(self, title="地图生成过程总结"):
        """生成整个地图生成过程的总结可视化"""
        if not self.all_visualizations:
            if self.logger:
                print("无可视化数据用于生成总结")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
            return
        
        # 计算需要的子图数量
        n = len(self.all_visualizations)
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        plt.figure(figsize=(cols*4, rows*3))
        plt.suptitle(title, fontsize=16)
        
        for i, vis in enumerate(self.all_visualizations):
            plt.subplot(rows, cols, i+1)
            
            data_type = vis.get('type')
            data = vis.get('data')
            
            if data_type == 'height_map':
                plt.imshow(data, cmap='terrain')
            elif data_type == 'temperature_map':
                plt.imshow(data, cmap='coolwarm')
            elif data_type == 'humidity_map':
                plt.imshow(data, cmap='Blues')
            elif data_type == 'rivers':
                plt.imshow(data['height_map'], cmap='terrain', alpha=0.7)
                river_overlay = np.zeros_like(data['height_map'])
                river_overlay[data['rivers_map'] > 0] = 1
                plt.imshow(river_overlay, cmap=ListedColormap(['none', 'blue']), alpha=0.7)
            elif data_type == 'biome_map':
                max_biome_id = np.max(data) + 1
                colors = plt.cm.tab20(np.linspace(0, 1, max_biome_id))
                biome_cmap = ListedColormap(colors)
                plt.imshow(data, cmap=biome_cmap)
            elif data_type == 'microclimate':
                if 'height_map' in data and data['height_map'] is not None:
                    plt.imshow(data['height_map'], cmap='terrain', alpha=0.5)
                    plt.imshow(data['microclimate_map'], cmap='plasma', alpha=0.5)
                else:
                    plt.imshow(data['microclimate_map'], cmap='plasma')
            elif data_type == 'objects':
                plt.imshow(data['height_map'], cmap='terrain', alpha=0.7)
                
                if data.get('vegetation'):
                    veg_x = [v.get('x', 0) for v in data['vegetation'] if 'x' in v]
                    veg_y = [v.get('y', 0) for v in data['vegetation'] if 'y' in v]
                    if veg_x and veg_y:
                        plt.scatter(veg_x[:100], veg_y[:100], c='green', s=3, alpha=0.6)
                
                if data.get('buildings'):
                    build_x = [b.get('x', 0) for b in data['buildings'] if 'x' in b]
                    build_y = [b.get('y', 0) for b in data['buildings'] if 'y' in b]
                    if build_x and build_y:
                        plt.scatter(build_x, build_y, c='red', s=5, alpha=0.8)
            
            # 简化的标题
            simple_title = vis.get('title', '').split(': ')[-1]
            plt.title(simple_title, fontsize=10)
            plt.xticks([])
            plt.yticks([])
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # 保存总结图像
        if self.save_path:
            filename = "00_map_generation_summary.png"
            filepath = os.path.join(self.save_path, filename)
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            if self.logger:
                print(f"保存地图生成总结图到: {filepath}")
            else:
                print(f"河流可视化: gui_frame存在={self.gui_frame is not None}, canvas存在={self.canvas is not None}")
        
        # 显示图像
        if self.show_plots:
            plt.show()
        else:
            plt.close()