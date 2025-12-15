#from __future__ import annotations
#标准库

#数据处理与科学计算
import numpy as np

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Patch
from matplotlib.colors import LightSource
from utils.preview_map_3d import preview_map_3d
from PIL.Image import Resampling

#网络与并发

#其他工具

def preview_map(map_data, master):
    """
    高性能地图预览系统，支持分层渲染与交互式探索
    
    利用空间数据结构、矢量化计算和多层渲染架构提供高效可视化
    
    Args:
        height_map: 2D高度图数据
        biome_map: 生物群系数据
        vegetation: 植被数据列表 [(x,y,type,..),...]
        buildings: 建筑数据列表 [(x,y,type,..),...]
        rivers: 河流布尔矩阵
        caves: 洞穴数据
        roads: 聚落道路数据列表 [(x,y,..),...]
        roads_map: 建筑道路元组 (road_network, road_types)
            road_network - 布尔矩阵表示道路存在
            road_types - 整数矩阵表示道路类型
        master: Tkinter控件容器
        content_layout: 内容布局字典(可选)
        
    Returns:
        canvas: 返回渲染画布对象，支持进一步交互
    """
    height_map, biome_map, vegetation, buildings, rivers, content_layout, caves, params, biome_data, roads, roads_map = map_data.unpack()    
    # === 1. 高效数据处理与空间索引 ===
    h, w = len(height_map), len(height_map[0])
    
    # 直接创建结果数组，避免多次转换
    render_array = np.zeros((h, w, 4), dtype=np.float32)  # RGBA格式
    
    # 将高度图转换为numpy数组用于光照计算
    height_np = np.array(height_map, dtype=np.float32)
    
    # 创建空间索引 - 使用NumPy的布尔掩码代替集合，极大提高性能
    cave_mask = np.zeros((h, w), dtype=bool)
    road_mask = np.zeros((h, w), dtype=bool)
    veg_mask = np.zeros((h, w), dtype=bool)
    bld_mask = np.zeros((h, w), dtype=bool)
    story_mask = np.zeros((h, w), dtype=bool)
    creature_mask = np.zeros((h, w), dtype=bool)
    
    # === 2. 矢量化数据处理 ===
    # 初始化道路掩码
    settlement_roads_mask = np.zeros((h, w), dtype=bool)  # 聚落道路
    building_main_road = np.zeros((h, w), dtype=bool)     # 建筑主干道
    building_secondary = np.zeros((h, w), dtype=bool)     # 建筑次干道
    building_paths = np.zeros((h, w), dtype=bool)         # 建筑小路
    
    # 处理聚落间道路
    if roads:
        road_coords = np.array([(r[0], r[1]) for r in roads 
                              if 0 <= r[0] < w and 0 <= r[1] < h], dtype=int)
        if road_coords.size > 0:
            settlement_roads_mask[road_coords[:,1], road_coords[:,0]] = True
    
    # 处理建筑道路系统
    if roads_map is not None:
        road_net, road_types = roads_map
        if road_net.shape == (h, w) and road_types.shape == (h, w):
            # 矢量化解码道路类型
            building_main_road = (road_net == True) & (road_types == 1)
            building_secondary = (road_net == True) & (road_types == 2)
            building_paths = (road_net == True) & (road_types == 3)
    
    # 处理植被数据
    if vegetation:
        veg_coords = [(v[0], v[1]) for v in vegetation if 0 <= v[0] < w and 0 <= v[1] < h]
        if veg_coords:
            veg_coords = np.array(veg_coords, dtype=int)
            veg_mask[veg_coords[:, 1], veg_coords[:, 0]] = True
    
    # 处理建筑数据
    if buildings:
        bld_coords = [(b[0], b[1]) for b in buildings if 0 <= b[0] < w and 0 <= b[1] < h]
        if bld_coords:
            bld_coords = np.array(bld_coords, dtype=int)
            bld_mask[bld_coords[:, 1], bld_coords[:, 0]] = True
    
    # 处理故事和生物数据
    # 确保content_layout是字典
    content_layout = content_layout or {}  # 如果None，设为空字典
    if isinstance(content_layout, dict):
        if "story_events" in content_layout:
            for e in content_layout["story_events"]:
                x, y = e.get("x", -1), e.get("y", -1)
                if 0 <= x < w and 0 <= y < h:
                    story_mask[y, x] = True
                    
        if "creatures" in content_layout:
            for cr in content_layout["creatures"]:
                x, y = cr.get("x", -1), cr.get("y", -1)
                if 0 <= x < w and 0 <= y < h:
                    creature_mask[y, x] = True
    else:
        print(f"Invalid content_layout type: {type(content_layout)}")
    
    # 高效处理洞穴数据
    if caves:
        # 检查caves是否为字典格式（修复后的格式）
        if isinstance(caves, dict) and "caves" in caves:
            cave_points = caves.get("caves", [])
            for point in cave_points:
                if isinstance(point, dict) and "x" in point and "y" in point:
                    x, y = int(point["x"]), int(point["y"])
                    if 0 <= x < w and 0 <= y < h:
                        cave_mask[y, x] = True
        else:
            # 原有逻辑，处理其他可能的格式
            for item in caves:
                if isinstance(item, np.ndarray):
                    # 确保坐标值为整数
                    item = item.astype(int)  # 强制转换为整数
                    valid_indices = (item[:, 0] < w) & (item[:, 1] < h) & (item[:, 0] >= 0) & (item[:, 1] >= 0)
                    if np.any(valid_indices):
                        valid_caves = item[valid_indices]
                        cave_mask[valid_caves[:, 1], valid_caves[:, 0]] = True
                elif isinstance(item, list) and item:
                    for point in item:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            x, y = int(point[0]), int(point[1])  # 确保转换为整数
                            if 0 <= x < w and 0 <= y < h:
                                cave_mask[y, x] = True
    
    # 高效处理河流数据
    river_mask = np.zeros((h, w), dtype=bool)
    if rivers is not None:
        if isinstance(rivers, np.ndarray):
            if rivers.shape == (h, w):
                river_mask = rivers > 0
            elif rivers.shape[:2] == (h, w):
                river_mask = np.any(rivers > 0, axis=-1) if rivers.ndim > 2 else rivers > 0
        else:
            # 处理嵌套列表
            for j in range(min(len(rivers), h)):
                row = rivers[j]
                for i in range(min(len(row), w)):
                    val = row[i]
                    if isinstance(val, np.ndarray):
                        river_mask[j, i] = np.any(val)
                    else:
                        river_mask[j, i] = bool(val)
    
    # === 3. 高级渲染与可视化 ===
    
    # 使用LightSource创建地形光照效果
    ls = LightSource(azdeg=315, altdeg=45)
    
    # 创建颜色映射 - 改进的生物群系颜色处理
    biome_colors = np.zeros((h, w, 3), dtype=np.float32)
    
    # 尝试加载生物群系配置文件
    try:
        import json
        import os
        biome_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                        "data", "configs", "biomes_config.json")
        with open(biome_config_path, "r", encoding="utf-8") as f:
            biomes_config = json.load(f)
        
        # 创建生物群系ID到颜色的映射
        biome_id_to_color = {}
        for idx, biome in enumerate(biomes_config.get("biomes", [])):
            biome_id_to_color[idx] = biome.get("color", [0.5, 0.5, 0.5])
            biome_id_to_color[biome.get("name", f"unknown_{idx}")] = biome.get("color", [0.5, 0.5, 0.5])
    except Exception as e:
        print(f"无法加载生物群系配置: {e}")
        biome_id_to_color = {}
    # 添加更详细的调试信息
    print(f"生物群系数据样例完整内容: {biome_map[0][0]}")
    print(f"应用颜色前，检查前10个单元格的生物群系数据:")
    for j in range(min(3, h)):
        for i in range(min(3, w)):
            print(f"位置[{j},{i}]: {biome_map[j][i]}, 类型: {type(biome_map[j][i])}")

    # 统计生物群系类型
    biome_types = {}
    for j in range(h):
        for i in range(w):
            biome_type = type(biome_map[j][i])
            biome_types[str(biome_type)] = biome_types.get(str(biome_type), 0) + 1
    print(f"生物群系数据类型统计: {biome_types}")         
    # 在加载配置文件后，添加以下代码
    print(f"已加载的生物群系配置ID: {sorted([i for i in biome_id_to_color.keys() if isinstance(i, int)])}")

    # 替换现有的颜色处理循环
    for j in range(h):
        for i in range(w):
            biome_id = biome_map[j][i]
            color = [0.5, 0.5, 0.5]  # 默认灰色
            
            # 对numpy整数特殊处理
            if isinstance(biome_id, (int, np.integer)):
                # 先尝试直接映射
                if biome_id in biome_id_to_color:
                    color = biome_id_to_color[biome_id]
                # 如果是偏移的ID，尝试调整（例如如果配置从0开始而数据从1开始）
                elif biome_id-1 in biome_id_to_color and biome_id > 0:
                    color = biome_id_to_color[biome_id-1]
                # 对于未知ID使用一系列备用颜色
                else:
                    # 为未知生物群系创建一个确定性但不同的颜色
                    backup_colors = [
                        [0.8, 0.2, 0.2],  # 红色
                        [0.2, 0.8, 0.2],  # 绿色
                        [0.2, 0.2, 0.8],  # 蓝色
                        [0.8, 0.8, 0.2],  # 黄色
                        [0.8, 0.2, 0.8],  # 紫色
                        [0.2, 0.8, 0.8],  # 青色
                    ]
                    color = backup_colors[biome_id % len(backup_colors)]
                    
            # 其他数据类型的处理保持不变
            elif isinstance(biome_data, dict) and "color" in biome_data:
                color = biome_data["color"]
            elif isinstance(biome_data, dict) and "name" in biome_data and biome_data["name"] in biome_id_to_color:
                color = biome_id_to_color[biome_data["name"]]
            elif isinstance(biome_data, str) and biome_data in biome_id_to_color:
                color = biome_id_to_color[biome_data]
                
            biome_colors[j, i] = color
    
    # 应用地形光照
    illuminated_biomes = ls.shade_rgb(
        biome_colors, 
        height_np,
        blend_mode='soft',
        fraction=0.6,  # 光照强度
        vert_exag=0.3  # 垂直夸张系数
    )
    
    # 设置基础RGBA值 (基于照明后的生物群系)
    render_array[:, :, :3] = illuminated_biomes
    render_array[:, :, 3] = 1.0  # Alpha通道
    
    # 定义增强视觉对比度的特征颜色
    COLORS = {
        'river': (0.0, 0.5, 1.0, 0.9),    # 蓝色，半透明
        'cave': (0.1, 0.1, 0.2, 1.0),     # 深灰，洞穴
        'settlement_road': (0.9, 0.7, 0.3, 1.0),  # 橙色聚落道路
        'main_road': (0.6, 0.2, 0.2, 1.0),       # 深红建筑主干道
        'secondary': (0.7, 0.6, 0.5, 1.0),       # 土黄建筑次干道
        'path': (0.6, 0.6, 0.6, 1.0),             # 灰色建筑小路
        'vegetation': (0.2, 0.7, 0.2, 0.9), # 绿色，植被
        'building': (0.9, 0.3, 0.2, 1.0), # 砖红色，建筑
        'story': (0.8, 0.3, 0.8, 1.0),    # 紫色，故事事件
        'creature': (0.9, 0.9, 0.1, 1.0)  # 黄色，生物
    }
    
    # 分层应用特征 - 优先级顺序应用
    # 使用NumPy广播高效应用，避免循环
    
    # 1. 河流层 
    render_array[river_mask, :] = COLORS['river']
    
    # 2. 道路层
    # 渲染顺序（从低到高优先级）：
    # 1. 建筑小路
    render_array[building_paths] = COLORS['path']
    
    # 2. 建筑次干道
    render_array[building_secondary] = COLORS['secondary']
    
    # 3. 建筑主干道
    render_array[building_main_road] = COLORS['main_road']
    
    # 4. 聚落间道路（最高道路优先级）
    render_array[settlement_roads_mask] = COLORS['settlement_road']
    
    # 3. 植被层
    render_array[veg_mask, :] = COLORS['vegetation']
    
    # 4. 洞穴层 
    render_array[cave_mask, :] = COLORS['cave']
    
    # 5. 建筑层
    render_array[bld_mask, :] = COLORS['building']
    
    # 6. 故事层
    render_array[story_mask, :] = COLORS['story']
    
    # 7. 生物层 (最高优先级)
    render_array[creature_mask, :] = COLORS['creature']
    
    # === 4. 创建增强的可视化 ===
    
    # 使用Figure而非pyplot直接创建图形，避免内存泄露
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # 渲染主图像
    img = ax.imshow(render_array, origin='lower', interpolation='bilinear')

    # 创建两组图例
    biome_legend = [Patch(facecolor=biome["color"], label=biome["name"]) 
                    for biome in biomes_config.get("biomes", [])]
    # 添加图例
    feature_legend = [
        Patch(facecolor=COLORS['river'][:3], label='河流'),
        Patch(facecolor=COLORS['settlement_road'][:3], label='聚落道路'),
        Patch(facecolor=COLORS['main_road'][:3], label='建筑主干道'),
        Patch(facecolor=COLORS['secondary'][:3], label='建筑次干道'),
        Patch(facecolor=COLORS['path'][:3], label='建筑小路'),
        Patch(facecolor=COLORS['vegetation'][:3], label='植被'),
        Patch(facecolor=COLORS['cave'][:3], label='洞穴'),
        Patch(facecolor=COLORS['building'][:3], label='建筑'),
        Patch(facecolor=COLORS['story'][:3], label='事件'),
        Patch(facecolor=COLORS['creature'][:3], label='生物')
    ]
    # 添加生物群系图例（左上角）
    first_legend = ax.legend(handles=biome_legend, loc='upper left', 
                            title="生物群系", fontsize='small',
                            framealpha=0.7, facecolor='white')
    ax.add_artist(first_legend)

    # 添加特征图例（右上角）
    ax.legend(handles=feature_legend, loc='upper right', 
            title="地形特征", fontsize='small',
            framealpha=0.7, facecolor='white')
    
    # 设置图像标题和坐标轴
    ax.set_title("地图预览 - 多层地理特征", fontsize=12)
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    
    # 增加网格线以提高可读性
    ax.grid(False)
    
    # 为大地图设置刻度间隔
    if max(w, h) > 100:
        # 根据地图大小动态计算理想刻度数
        max_dimension = max(w, h)
        print(f"地图尺寸: {w}x{h}，最大维度: {max_dimension}")  # 调试信息
        
        if max_dimension <= 200:
            ideal_ticks = 10  # 小地图用10个刻度
        elif max_dimension <= 500:
            ideal_ticks = 20  # 中等地图用20个刻度
        else:
            ideal_ticks = 50  # 大地图用50个刻度
        
        print(f"选择的理想刻度数: {ideal_ticks}")  # 调试信息
        
        # 直接计算刻度间隔，不再使用复杂的舍入逻辑
        tick_interval = max_dimension // ideal_ticks
        
        # 美化间隔为5或10的倍数
        if tick_interval > 10:
            tick_interval = round(tick_interval / 10) * 10
        else:
            tick_interval = max(5, tick_interval)
        
        print(f"计算的刻度间隔: {tick_interval}")  # 调试信息
        
        # 设置刻度
        ax.set_xticks(np.arange(0, w, tick_interval))
        ax.set_yticks(np.arange(0, h, tick_interval))
        
        # 防止大地图标签重叠
        if max_dimension >= 1000:
            plt.setp(ax.get_xticklabels()[::2], visible=False)
            plt.setp(ax.get_yticklabels()[::2], visible=False)
        
        # 添加更清晰的比例尺信息
        ax.set_xlabel(f"X 坐标 (每格{tick_interval}单位)")
        ax.set_ylabel(f"Y 坐标 (每格{tick_interval}单位)")
    
    # === 5. 创建独立的可调整大小窗口 ===
    import tkinter as tk
    from tkinter import ttk
    
    try:
        # 打印调试信息
        print("开始创建预览窗口...")
        
        # 创建一个独立的顶级窗口
        preview_window = tk.Toplevel(master)
        preview_window.title("地图预览")
        preview_window.geometry("800x600")  # 设置初始大小
        preview_window.minsize(400, 300)    # 设置最小大小
        
        # 确保窗口显示在顶层
        preview_window.attributes('-topmost', True)
        preview_window.update()
        preview_window.attributes('-topmost', False)
        
        # 允许窗口调整大小
        preview_window.resizable(True, True)
        
        # 创建主框架以容纳所有元素
        main_frame = ttk.Frame(preview_window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建画布容器框架
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Matplotlib画布
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 创建工具栏
        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar_frame = ttk.Frame(main_frame)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
        except Exception as e:
            print(f"工具栏创建失败: {e}")
        
        # 添加状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_bar = ttk.Label(status_frame, text="鼠标悬停查看信息", relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X)
        
        # 添加关闭按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        close_button = ttk.Button(button_frame, text="关闭", command=preview_window.destroy)
        close_button.pack(side=tk.RIGHT, padx=10)
        # 添加切换到3D视图按钮
        from utils.preview_map_3d import preview_map_3d
        view_3d_button = ttk.Button(
            button_frame, 
            text="切换到3D视图", 
            command=lambda: preview_window.after(100, lambda: preview_map_3d(map_data, master))
        )
        view_3d_button.pack(side=tk.LEFT, padx=10)
        
        # 添加交互式地图信息显示
        def on_hover(event):
            if event.inaxes == ax:
                x, y = int(event.xdata), int(event.ydata)
                if 0 <= x < w and 0 <= y < h:
                    # 收集鼠标位置的信息
                    info = []
                    info.append(f"坐标: ({x}, {y})")
                    info.append(f"高度: {height_map[y][x]:.1f}")
                    
                    # 修复生物群系显示 - 支持多种数据类型
                    biome_value = biome_map[y][x]
                    if isinstance(biome_value, dict) and "name" in biome_value:
                        biome_name = biome_value["name"]
                    elif isinstance(biome_value, (int, np.integer)):
                        # 对于整数类型的生物群系ID，直接显示ID
                        biome_name = f"ID:{biome_value}"
                    elif isinstance(biome_value, str):
                        biome_name = biome_value
                    else:
                        biome_name = "未知"
                    info.append(f"生物群系: {biome_name}")
                    
                    # 更新状态栏文本
                    status_text = " | ".join(info)
                    status_bar.config(text=status_text)
        
        # 连接hover事件
        canvas.mpl_connect('motion_notify_event', on_hover)
        
        # 强制绘制画布
        canvas.draw()
        
        # 强制更新窗口
        preview_window.update_idletasks()
        preview_window.deiconify()  # 确保窗口没有被最小化
        
        print("预览窗口创建完成")
        
        # 返回canvas，但保留window引用以防止被销毁
        canvas._preview_window = preview_window  # 保存引用防止被垃圾回收
        return canvas
        
    except Exception as e:
        import traceback
        print(f"创建预览窗口时发生错误: {e}")
        traceback.print_exc()
        # 出错时返回None
        return None
    
def get_combined_preview_image(map_data, max_size=(800, 600)):
    """生成地图的组合预览图像，包含高度图、生物群系和其他关键图层
    
    Args:
        map_data: MapData对象，包含地图数据
        max_size: 最大输出图像尺寸，格式为(宽度, 高度)
    
    Returns:
        PIL.Image: 组合后的地图预览图像，如处理失败返回None
    """
    from PIL import Image, ImageDraw, ImageEnhance, ImageOps
    import numpy as np
    
    if not map_data or not map_data.is_valid():
        return None
    
    try:
        # 获取必要的地图数据层
        height_map = map_data.get_layer("height")
        biome_map = map_data.get_layer("biome")
        
        if height_map is None:
            return None
        
        # 确定图像尺寸
        map_height, map_width = height_map.shape
        
        # 计算缩放因子
        scale_x = max_size[0] / map_width
        scale_y = max_size[1] / map_height
        scale = min(scale_x, scale_y)
        
        new_width = int(map_width * scale)
        new_height = int(map_height * scale)
        
        # 创建高度图基础图像
        height_normalized = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map) + 1e-10)
        height_img_array = (height_normalized * 255).astype(np.uint8)
        height_img = Image.fromarray(height_img_array).convert('L')
        
        # 应用地形颜色映射
        terrain_img = Image.new('RGB', (map_width, map_height))
        
        # 简单地形着色
        terrain_colors = {
            0: (0, 0, 128),     # 深水
            25: (65, 105, 225),  # 浅水
            40: (210, 180, 140), # 沙滩
            50: (34, 139, 34),   # 平原
            70: (0, 100, 0),     # 森林
            90: (139, 137, 137), # 山脚
            150: (169, 169, 169), # 山地
            200: (255, 250, 250), # 山顶/雪
        }
        
        # 创建颜色映射表
        colors = np.zeros((256, 3), dtype=np.uint8)
        heights = sorted(terrain_colors.keys())
        
        for i in range(256):
            normalized_height = i / 255.0 * 100.0
            
            # 找到相邻的高度值
            lower_idx = 0
            for j, h in enumerate(heights):
                if normalized_height >= h:
                    lower_idx = j
            
            upper_idx = min(lower_idx + 1, len(heights) - 1)
            
            if lower_idx == upper_idx:
                colors[i] = terrain_colors[heights[lower_idx]]
            else:
                # 线性插值
                lower_h = heights[lower_idx]
                upper_h = heights[upper_idx]
                t = (normalized_height - lower_h) / (upper_h - lower_h) if upper_h != lower_h else 0
                
                lower_color = np.array(terrain_colors[lower_h])
                upper_color = np.array(terrain_colors[upper_h])
                colors[i] = lower_color + t * (upper_color - lower_color)
        
        # 应用颜色映射
        color_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        for y in range(map_height):
            for x in range(map_width):
                height_val = height_img_array[y, x]
                color_map[y, x] = colors[height_val]
        
        # 创建最终图像
        terrain_img = Image.fromarray(color_map)
        
        # 如果有生物群系图层，叠加它的信息
        if biome_map is not None:
            biome_overlay = Image.new('RGBA', (map_width, map_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(biome_overlay)
            
            # 定义生物群系颜色映射
            biome_colors = {
                0: (0, 0, 0, 0),         # 未定义
                1: (0, 128, 0, 100),     # 森林
                2: (240, 230, 140, 100), # 沙漠
                3: (65, 105, 225, 100),  # 河流/水域
                4: (139, 137, 137, 100), # 山地
                5: (34, 139, 34, 100),   # 平原
                6: (0, 206, 209, 100),   # 沼泽
                7: (220, 20, 60, 100),   # 火山
                8: (255, 250, 250, 100), # 雪地
                9: (139, 69, 19, 100),   # 丘陵
                10: (46, 139, 87, 100),  # 丛林
            }
            
            # 对生物群系进行上色
            if hasattr(biome_map, 'shape'):  # NumPy数组
                for biome_id in np.unique(biome_map):
                    if biome_id in biome_colors:
                        mask = (biome_map == biome_id)
                        color = biome_colors[biome_id]
                        for y in range(map_height):
                            for x in range(map_width):
                                if mask[y, x]:
                                    draw.point((x, y), color)
            
            # 叠加生物群系
            terrain_img = Image.alpha_composite(
                terrain_img.convert('RGBA'),
                biome_overlay
            ).convert('RGB')
        
        # 增强图像对比度
        enhancer = ImageEnhance.Contrast(terrain_img)
        terrain_img = enhancer.enhance(1.2)
        
        # 调整图像大小
        if scale < 1:
            terrain_img = terrain_img.resize((new_width, new_height), Resampling.LANCZOS)
        
        return terrain_img
        
    except Exception as e:
        import traceback
        print(f"生成地图预览图像出错: {e}")
        print(traceback.format_exc())
        return None