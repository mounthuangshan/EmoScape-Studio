import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from mpl_toolkits.mplot3d import Axes3D

def preview_map_3d(map_data, master):
    """
    创建地图的3D交互式预览
    
    Args:
        map_data: 地图数据对象
        master: Tkinter控件容器
        
    Returns:
        canvas: 返回渲染画布对象
    """
    # 尝试从map_data中获取数据
    try:
        if hasattr(map_data, 'unpack'):
            height_map, biome_map, vegetation, buildings, rivers, content_layout, caves, params, biome_data, roads, roads_map = map_data.unpack()
        else:
            # 尝试直接访问属性
            height_map = map_data.get_layer("height") if hasattr(map_data, 'get_layer') else map_data.layers["height"]
            biome_map = map_data.get_layer("biome") if hasattr(map_data, 'get_layer') else map_data.layers["biome"]
            vegetation = map_data.get_object_layer("vegetation") if hasattr(map_data, 'get_object_layer') else []
            buildings = map_data.get_object_layer("buildings") if hasattr(map_data, 'get_object_layer') else []
            rivers = map_data.get_layer("rivers") if hasattr(map_data, 'get_layer') else []
            caves = map_data.get_layer("caves") if hasattr(map_data, 'get_layer') else []
            roads = map_data.get_object_layer("roads") if hasattr(map_data, 'get_object_layer') else []
    except Exception as e:
        print(f"获取地图数据失败: {e}")
        return None
    
    # 确保height_map是numpy数组并获取尺寸
    if not isinstance(height_map, np.ndarray):
        try:
            height_map = np.array(height_map)
        except:
            print("无法将高度图转换为numpy数组")
            return None
    
    h, w = height_map.shape
    
    try:
        # 创建一个独立的顶级窗口
        preview_window = tk.Toplevel(master)
        preview_window.title("3D地图预览")
        preview_window.geometry("900x700")  # 设置初始大小
        preview_window.minsize(600, 500)    # 设置最小大小
        
        # 确保窗口显示在顶层
        preview_window.attributes('-topmost', True)
        preview_window.update()
        preview_window.attributes('-topmost', False)
        
        # 允许窗口调整大小
        preview_window.resizable(True, True)
        
        # 创建主框架以容纳所有元素
        main_frame = ttk.Frame(preview_window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建控制面板框架
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 创建视角控制区域
        view_frame = ttk.LabelFrame(control_frame, text="视角控制")
        view_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 添加预设视角按钮
        ttk.Button(view_frame, text="俯视图", 
                  command=lambda: set_view(90, -90)).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(view_frame, text="侧视图", 
                  command=lambda: set_view(0, -90)).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(view_frame, text="等距视图", 
                  command=lambda: set_view(30, -45)).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 添加渲染控制
        render_frame = ttk.LabelFrame(control_frame, text="渲染控制")
        render_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # 拉伸因子控制
        stretch_var = tk.DoubleVar(value=1.0)
        ttk.Label(render_frame, text="高度拉伸:").grid(row=0, column=0, padx=5, pady=2)
        stretch_scale = ttk.Scale(render_frame, from_=0.1, to=5.0, 
                                 variable=stretch_var, orient=tk.HORIZONTAL, length=150)
        stretch_scale.grid(row=0, column=1, padx=5, pady=2)
        stretch_scale.bind("<ButtonRelease-1>", lambda e: update_stretch(stretch_var.get()))
        
        # 分辨率控制
        resolution_var = tk.IntVar(value=1)
        ttk.Label(render_frame, text="采样率:").grid(row=1, column=0, padx=5, pady=2)
        resolution_scale = ttk.Scale(render_frame, from_=1, to=10, 
                                    variable=resolution_var, orient=tk.HORIZONTAL, length=150)
        resolution_scale.grid(row=1, column=1, padx=5, pady=2)
        resolution_scale.bind("<ButtonRelease-1>", lambda e: update_resolution(resolution_var.get()))
        
        # 创建画布容器框架
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建带有3D轴的图形
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # 初始化网格
        X, Y = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))
        Z = height_map
        
        # 处理生物群系颜色
        colors = np.zeros(height_map.shape + (3,))
        
        # 加载生物群系配置
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
            biomes_config = {"biomes": []}
        
        # 默认颜色映射
        default_colors = [
            [0.8, 0.2, 0.2],  # 红色
            [0.2, 0.8, 0.2],  # 绿色
            [0.2, 0.2, 0.8],  # 蓝色
            [0.8, 0.8, 0.2],  # 黄色
            [0.8, 0.2, 0.8],  # 紫色
            [0.2, 0.8, 0.8],  # 青色
        ]
        
        # 处理生物群系颜色
        try:
            for y in range(h):
                for x in range(w):
                    biome_id = biome_map[y, x]
                    
                    # 尝试获取颜色
                    if isinstance(biome_id, (int, np.integer)):
                        if biome_id in biome_id_to_color:
                            colors[y, x] = biome_id_to_color[biome_id]
                        else:
                            colors[y, x] = default_colors[biome_id % len(default_colors)]
                    else:
                        colors[y, x] = [0.5, 0.5, 0.5]  # 灰色默认值
        except Exception as e:
            print(f"生物群系颜色映射失败: {e}")
            # 使用渐变色作为备用
            from matplotlib.colors import LinearSegmentedColormap
            terrain_cmap = LinearSegmentedColormap.from_list('terrain', 
                           [(0, [0.0, 0.5, 1.0]),     # 深蓝色(低地)
                            (0.3, [0.4, 0.8, 0.2]),   # 绿色(平原)
                            (0.6, [0.8, 0.8, 0.0]),   # 黄色(丘陵)
                            (0.8, [0.6, 0.4, 0.2]),   # 棕色(山地)
                            (1.0, [1.0, 1.0, 1.0])])  # 白色(山顶)
            
            for y in range(h):
                for x in range(w):
                    colors[y, x] = terrain_cmap(height_map[y, x])[:3]
        
        # 采样以提高大地图的性能
        current_sample = 1
        
        # 更新采样率的函数
        def update_samples():
            nonlocal current_sample
            # 根据地图大小动态调整默认采样率，避免大地图渲染过慢
            if max(h, w) > 500:
                current_sample = max(current_sample, 5)
            elif max(h, w) > 200:
                current_sample = max(current_sample, 2)
        
        # 初始样本采样率
        update_samples()
        
        # 采样数据
        sampled_X = X[::current_sample, ::current_sample]
        sampled_Y = Y[::current_sample, ::current_sample]
        sampled_Z = Z[::current_sample, ::current_sample]
        sampled_colors = colors[::current_sample, ::current_sample]

        # 创建Matplotlib画布
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 创建工具栏
        try:
            toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
            toolbar.update()
            toolbar.pack(side=tk.TOP, fill=tk.X)
        except Exception as e:
            print(f"工具栏创建失败: {e}")
        
        # 渲染3D地形
        surf = None
        current_stretch = 1.0
        
        def render_surface():
            nonlocal surf
            
            # 清除现有的地形
            if surf:
                surf.remove()
            
            # 创建3D表面
            surf = ax.plot_surface(
                sampled_X, 
                sampled_Y, 
                sampled_Z * current_stretch,  # 应用拉伸
                facecolors=sampled_colors,
                rstride=1, 
                cstride=1,
                shade=True,
                antialiased=True,
                alpha=0.9
            )
            
            # 设置轴标签
            ax.set_xlabel('X 坐标')
            ax.set_ylabel('Y 坐标')
            ax.set_zlabel('高度')
            
            # 设置边界
            ax.set_xlim(0, w)
            ax.set_ylim(0, h)
            
            # 更新画布
            canvas.draw()
        
        # 初始化
        render_surface()
        
        # 添加特征点云
        feature_points = []
        
        def add_features():
            # 清除现有特征点
            for point in feature_points:
                if point in ax.collections:
                    ax.collections.remove(point)
            feature_points.clear()
            
            # 采样率跟随当前设置
            sample_rate = max(1, current_sample)
            
            # 添加河流 (蓝色点)
            if isinstance(rivers, list) and rivers:
                river_coords = []
                for r in rivers:
                    if isinstance(r, (list, tuple)) and len(r) >= 2:
                        x, y = r[0], r[1]
                        if 0 <= x < w and 0 <= y < h:
                            river_coords.append((x, y, Z[int(y)][int(x)] * current_stretch + 0.05))
                
                if river_coords and len(river_coords) > 0:
                    # 采样河流点以避免过多点导致性能问题
                    river_coords = river_coords[::sample_rate]
                    if river_coords:
                        river_x, river_y, river_z = zip(*river_coords)
                        river_point = ax.scatter(river_x, river_y, river_z, 
                                             color='blue', s=10, alpha=0.7, label='河流')
                        feature_points.append(river_point)
            
            # 添加建筑 (红色方块)
            if buildings:
                bld_coords = []
                for b in buildings:
                    try:
                        if hasattr(b, 'x') and hasattr(b, 'y'):
                            x, y = b.x, b.y
                        elif isinstance(b, (list, tuple)) and len(b) >= 2:
                            x, y = b[0], b[1]
                        elif isinstance(b, dict) and 'x' in b and 'y' in b:
                            x, y = b['x'], b['y']
                        else:
                            continue
                            
                        if 0 <= x < w and 0 <= y < h:
                            bld_coords.append((x, y, Z[int(y)][int(x)] * current_stretch + 0.5))
                    except:
                        continue
                
                if bld_coords:
                    # 采样建筑点
                    bld_coords = bld_coords[::sample_rate]
                    if bld_coords:
                        bld_x, bld_y, bld_z = zip(*bld_coords)
                        bld_point = ax.scatter(bld_x, bld_y, bld_z, 
                                           color='red', s=20, marker='s', label='建筑')
                        feature_points.append(bld_point)
            
            # 添加洞穴 (黑色圆形)
            if caves:
                cave_coords = []
                try:
                    if isinstance(caves, list):
                        for cave in caves:
                            if isinstance(cave, dict) and 'x' in cave and 'y' in cave:
                                x, y = cave['x'], cave['y']
                                if 0 <= x < w and 0 <= y < h:
                                    cave_coords.append((x, y, Z[int(y)][int(x)] * current_stretch + 0.2))
                    elif isinstance(caves, dict) and "caves" in caves:
                        for cave in caves["caves"]:
                            if isinstance(cave, dict) and 'x' in cave and 'y' in cave:
                                x, y = cave['x'], cave['y']
                                if 0 <= x < w and 0 <= y < h:
                                    cave_coords.append((x, y, Z[int(y)][int(x)] * current_stretch + 0.2))
                except:
                    pass
                
                if cave_coords:
                    # 采样洞穴点
                    cave_coords = cave_coords[::sample_rate]
                    if cave_coords:
                        cave_x, cave_y, cave_z = zip(*cave_coords)
                        cave_point = ax.scatter(cave_x, cave_y, cave_z, 
                                            color='black', s=20, marker='o', label='洞穴')
                        feature_points.append(cave_point)
            
            # 添加道路 (黄色线)
            if roads:
                road_coords = []
                for r in roads:
                    try:
                        if hasattr(r, 'x') and hasattr(r, 'y'):
                            x, y = r.x, r.y
                        elif isinstance(r, (list, tuple)) and len(r) >= 2:
                            x, y = r[0], r[1]
                        elif isinstance(r, dict) and 'x' in r and 'y' in r:
                            x, y = r['x'], r['y']
                        else:
                            continue
                            
                        if 0 <= x < w and 0 <= y < h:
                            road_coords.append((x, y, Z[int(y)][int(x)] * current_stretch + 0.1))
                    except:
                        continue
                
                if road_coords:
                    # 采样道路点
                    road_coords = road_coords[::sample_rate]
                    if road_coords:
                        road_x, road_y, road_z = zip(*road_coords)
                        road_point = ax.scatter(road_x, road_y, road_z, 
                                            color='orange', s=5, alpha=0.7, label='道路')
                        feature_points.append(road_point)
            
            # 更新画布
            canvas.draw()
        
        # 视角设置函数
        def set_view(elev, azim):
            ax.view_init(elev=elev, azim=azim)
            canvas.draw()
        
        # 拉伸因子更新函数
        def update_stretch(value):
            nonlocal current_stretch
            current_stretch = value
            
            # 更新Z值
            render_surface()
            add_features()
        
        # 分辨率更新函数
        def update_resolution(value):
            nonlocal current_sample, sampled_X, sampled_Y, sampled_Z, sampled_colors
            
            # 更新采样率，确保至少为1
            current_sample = max(1, int(value))
            
            # 重新采样
            sampled_X = X[::current_sample, ::current_sample]
            sampled_Y = Y[::current_sample, ::current_sample]
            sampled_Z = Z[::current_sample, ::current_sample]
            sampled_colors = colors[::current_sample, ::current_sample]
            
            # 重新渲染
            render_surface()
            add_features()
        
        # 添加状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_bar = ttk.Label(status_frame, text="3D地图预览 - 可拖动旋转查看不同角度", relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X)
        
        # 添加控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # 添加图例按钮
        show_legend_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            button_frame, 
            text="显示图例", 
            variable=show_legend_var,
            command=lambda: (ax.legend() if show_legend_var.get() else ax.legend().remove(), canvas.draw())
        ).pack(side=tk.LEFT, padx=10)
        
        # 添加特征切换
        show_features_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            button_frame, 
            text="显示特征", 
            variable=show_features_var,
            command=lambda: (add_features() if show_features_var.get() else 
                           (lambda: [p.remove() for p in feature_points if p in ax.collections] or 
                             feature_points.clear() or canvas.draw())())
        ).pack(side=tk.LEFT, padx=10)
        
        # 切换回2D视图按钮
        from utils.preview_map import preview_map
        ttk.Button(
            button_frame, 
            text="切换到2D视图", 
            command=lambda: preview_window.after(100, lambda: preview_map(map_data, master))
        ).pack(side=tk.LEFT, padx=10)
        
        # 关闭按钮
        ttk.Button(button_frame, text="关闭", command=preview_window.destroy).pack(side=tk.RIGHT, padx=10)
        
        # 初始添加特征
        add_features()
        
        # 设置默认视角
        set_view(30, -45)
        
        # 安全地更新窗口
        try:
            # 先检查窗口是否有效
            if preview_window.winfo_exists():
                preview_window.deiconify()
                preview_window.update_idletasks()
        except Exception as e:
            print(f"更新窗口时出错: {e}")
            # 如果出错，尝试直接返回画布
            if 'canvas' in locals():
                canvas._preview_window = preview_window
                return canvas
            return None
        
        # 返回canvas，但保留window引用以防止被销毁
        canvas._preview_window = preview_window
        return canvas
        
    except Exception as e:
        import traceback
        print(f"创建3D预览窗口时发生错误: {e}")
        traceback.print_exc()
        return None