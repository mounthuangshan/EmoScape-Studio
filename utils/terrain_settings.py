import tkinter as tk
from tkinter import ttk
import numpy as np
import os

class TerrainGeneratorDialog:
    """地形生成器高级设置对话框"""
    
    def __init__(self, parent, initial_params=None):
        self.parent = parent
        self.result = None
        
        # 使用传入的初始参数或使用默认值
        self.params = initial_params or {}
        self._setup_defaults()
        
        # 创建对话框
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("地形生成器高级设置")
        self.dialog.grab_set()  # 设置为模态对话框
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self._create_widgets()
        self._center_window()
    
    def _setup_defaults(self):
        """设置默认参数值"""
        # 基础参数
        if "seed" not in self.params:
            self.params["seed"] = np.random.randint(0, 9999)
        if "scale_factor" not in self.params:
            self.params["scale_factor"] = 1.0
        if "erosion_iterations" not in self.params:
            self.params["erosion_iterations"] = 3
        if "river_density" not in self.params:
            self.params["river_density"] = 1.0
        if "mountain_sharpness" not in self.params:
            self.params["mountain_sharpness"] = 1.2
        if "use_tectonic" not in self.params:
            self.params["use_tectonic"] = True
        if "detail_level" not in self.params:
            self.params["detail_level"] = 1.0
        
        # 侵蚀参数
        if "erosion_type" not in self.params:
            self.params["erosion_type"] = "advanced"
        if "erosion_strength" not in self.params:
            self.params["erosion_strength"] = 0.8
        if "talus_angle" not in self.params:
            self.params["talus_angle"] = 0.05
        if "sediment_capacity" not in self.params:
            self.params["sediment_capacity"] = 0.15
        if "rainfall" not in self.params:
            self.params["rainfall"] = 0.01
        if "evaporation" not in self.params:
            self.params["evaporation"] = 0.5
        
        # 河流参数
        if "min_watershed_size" not in self.params:
            self.params["min_watershed_size"] = 50
        if "precipitation_factor" not in self.params:
            self.params["precipitation_factor"] = 1.0
        if "meander_factor" not in self.params:
            self.params["meander_factor"] = 0.3
        
        # 噪声参数
        if "use_frequency_optimization" not in self.params:
            self.params["use_frequency_optimization"] = True
        if "octaves" not in self.params:
            self.params["octaves"] = 6
        if "persistence" not in self.params:
            self.params["persistence"] = 0.5
        if "lacunarity" not in self.params:
            self.params["lacunarity"] = 2.0
        
        # 地形分布参数
        if "plain_ratio" not in self.params:
            self.params["plain_ratio"] = 0.3
        if "hill_ratio" not in self.params:
            self.params["hill_ratio"] = 0.3
        if "mountain_ratio" not in self.params:
            self.params["mountain_ratio"] = 0.2
        if "plateau_ratio" not in self.params:
            self.params["plateau_ratio"] = 0.1
        
        # 气候参数
        if "latitude_effect" not in self.params:
            self.params["latitude_effect"] = 0.5
        if "prevailing_wind_x" not in self.params:
            self.params["prevailing_wind_x"] = 1.0
        if "prevailing_wind_y" not in self.params:
            self.params["prevailing_wind_y"] = 0.0
    
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建选项卡控件
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建各个选项卡页面
        basic_frame = ttk.Frame(notebook, padding=10)
        erosion_frame = ttk.Frame(notebook, padding=10)
        river_frame = ttk.Frame(notebook, padding=10)
        noise_frame = ttk.Frame(notebook, padding=10)
        terrain_frame = ttk.Frame(notebook, padding=10)
        climate_frame = ttk.Frame(notebook, padding=10)
        
        notebook.add(basic_frame, text="基本设置")
        notebook.add(erosion_frame, text="侵蚀设置")
        notebook.add(river_frame, text="河流设置")
        notebook.add(noise_frame, text="噪声设置")
        notebook.add(terrain_frame, text="地形分布")
        notebook.add(climate_frame, text="气候设置")
        
        # 填充基本设置选项卡
        self._create_basic_settings(basic_frame)
        
        # 填充侵蚀设置选项卡
        self._create_erosion_settings(erosion_frame)
        
        # 填充河流设置选项卡
        self._create_river_settings(river_frame)
        
        # 填充噪声设置选项卡
        self._create_noise_settings(noise_frame)
        
        # 填充地形分布选项卡
        self._create_terrain_distribution_settings(terrain_frame)
        
        # 填充气候设置选项卡
        self._create_climate_settings(climate_frame)
        
        # 添加预设地形类型按钮
        presets_frame = ttk.LabelFrame(main_frame, text="地形预设")
        presets_frame.pack(fill=tk.X, pady=5, before=notebook)
        
        presets_grid = ttk.Frame(presets_frame)
        presets_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建预设按钮
        preset_types = [
            ("平原地形", self._preset_plains),
            ("高山地形", self._preset_mountains),
            ("丘陵地形", self._preset_hills),
            ("峡谷地形", self._preset_canyons),
            ("岛屿地形", self._preset_islands),
            ("火山地形", self._preset_volcanic),
            ("沙漠地形", self._preset_desert),
            ("随机地形", self._preset_random)
        ]
        
        # 设置按钮网格
        row = 0
        col = 0
        max_cols = 4
        
        for name, command in preset_types:
            ttk.Button(presets_grid, text=name, command=command, width=12).grid(
                row=row, column=col, padx=5, pady=5, sticky=tk.W
            )
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # 创建底部按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="应用", command=self.on_apply).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="重置为默认值", command=self.on_reset).pack(side=tk.LEFT, padx=5)
        
    # 添加预设方法
    def _preset_plains(self):
        """平原地形预设"""
        self.plain_ratio_var.set(0.6)
        self.hill_ratio_var.set(0.3)
        self.mountain_ratio_var.set(0.08)
        self.plateau_ratio_var.set(0.02)
        self.mountain_sharpness_var.set(0.9)
        self.scale_factor_var.set(0.8)
        self.erosion_type_var.set("simple")
        self.erosion_strength_var.set(0.5)
        self.erosion_iterations_var.set(2)
        self.river_density_var.set(1.2)
        self._update_total_ratio()

    def _preset_mountains(self):
        """高山地形预设"""
        self.plain_ratio_var.set(0.2)
        self.hill_ratio_var.set(0.3)
        self.mountain_ratio_var.set(0.4)
        self.plateau_ratio_var.set(0.1)
        self.mountain_sharpness_var.set(1.8)
        self.scale_factor_var.set(1.5)
        self.erosion_type_var.set("advanced")
        self.erosion_strength_var.set(0.9)
        self.erosion_iterations_var.set(4)
        self.talus_angle_var.set(0.07)
        self.river_density_var.set(1.5)
        self._update_total_ratio()

    def _preset_hills(self):
        """丘陵地形预设"""
        self.plain_ratio_var.set(0.25)
        self.hill_ratio_var.set(0.5)
        self.mountain_ratio_var.set(0.2)
        self.plateau_ratio_var.set(0.05)
        self.mountain_sharpness_var.set(1.0)
        self.scale_factor_var.set(1.2)
        self.erosion_type_var.set("thermal")
        self.erosion_iterations_var.set(3)
        self.river_density_var.set(0.8)
        self._update_total_ratio()

    def _preset_canyons(self):
        """峡谷地形预设"""
        self.plain_ratio_var.set(0.3)
        self.hill_ratio_var.set(0.2)
        self.mountain_ratio_var.set(0.3)
        self.plateau_ratio_var.set(0.2)
        self.mountain_sharpness_var.set(1.4)
        self.scale_factor_var.set(1.6)
        self.erosion_type_var.set("hydraulic")
        self.erosion_strength_var.set(1.5)
        self.erosion_iterations_var.set(6)
        self.rainfall_var.set(0.02)
        self.river_density_var.set(1.8)
        self._update_total_ratio()

    def _preset_islands(self):
        """岛屿地形预设"""
        self.plain_ratio_var.set(0.1)
        self.hill_ratio_var.set(0.3)
        self.mountain_ratio_var.set(0.5)
        self.plateau_ratio_var.set(0.1)
        self.mountain_sharpness_var.set(1.3)
        self.scale_factor_var.set(2.0)
        self.erosion_type_var.set("combined")
        self.erosion_iterations_var.set(3)
        self.river_density_var.set(0.6)
        self._update_total_ratio()

    def _preset_volcanic(self):
        """火山地形预设"""
        self.plain_ratio_var.set(0.4)
        self.hill_ratio_var.set(0.2)
        self.mountain_ratio_var.set(0.37)
        self.plateau_ratio_var.set(0.03)
        self.mountain_sharpness_var.set(2.0)
        self.scale_factor_var.set(1.3)
        self.erosion_type_var.set("thermal")
        self.erosion_strength_var.set(0.6)
        self.erosion_iterations_var.set(3)
        self.river_density_var.set(0.4)
        self._update_total_ratio()

    def _preset_desert(self):
        """沙漠地形预设"""
        self.plain_ratio_var.set(0.45)
        self.hill_ratio_var.set(0.35)
        self.mountain_ratio_var.set(0.15)
        self.plateau_ratio_var.set(0.05)
        self.mountain_sharpness_var.set(1.1)
        self.scale_factor_var.set(0.9)
        self.erosion_type_var.set("thermal")
        self.erosion_strength_var.set(1.2)
        self.erosion_iterations_var.set(5)
        self.river_density_var.set(0.3)
        self._update_total_ratio()

    def _preset_random(self):
        """随机地形预设"""
        import random
        self.seed_var.set(random.randint(0, 9999))
        self.plain_ratio_var.set(random.uniform(0.2, 0.5))
        self.hill_ratio_var.set(random.uniform(0.2, 0.4))
        self.mountain_ratio_var.set(random.uniform(0.1, 0.4))
        plateau = max(0.0, 1.0 - float(self.plain_ratio_var.get()) - 
                    float(self.hill_ratio_var.get()) - 
                    float(self.mountain_ratio_var.get()))
        self.plateau_ratio_var.set(plateau)
        self.mountain_sharpness_var.set(random.uniform(0.8, 1.8))
        self.scale_factor_var.set(random.uniform(0.8, 2.0))
        
        erosion_types = ["thermal", "hydraulic", "combined", "advanced", "simple"]
        self.erosion_type_var.set(random.choice(erosion_types))
        self.erosion_strength_var.set(random.uniform(0.5, 1.5))
        self.erosion_iterations_var.set(random.randint(2, 6))
        self.river_density_var.set(random.uniform(0.4, 1.6))
        self._update_total_ratio()
    
    def _create_basic_settings(self, parent):
        """创建基本设置选项卡内容"""
        # 随机种子
        seed_frame = ttk.Frame(parent)
        seed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(seed_frame, text="随机种子:").pack(side=tk.LEFT, padx=5)
        self.seed_var = tk.IntVar(value=self.params["seed"])
        seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_var, width=10)
        seed_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(seed_frame, text="随机", 
                  command=lambda: self.seed_var.set(np.random.randint(0, 9999))
                  ).pack(side=tk.LEFT, padx=5)
        
        # 地形尺度
        scale_frame = ttk.LabelFrame(parent, text="地形尺度")
        scale_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scale_frame, text="尺度因子:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.scale_factor_var = tk.DoubleVar(value=self.params["scale_factor"])
        scale_slider = ttk.Scale(scale_frame, from_=0.5, to=5.0, 
                               variable=self.scale_factor_var, orient=tk.HORIZONTAL)
        scale_slider.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(scale_frame, textvariable=self.scale_factor_var, width=5).grid(row=0, column=2, padx=5, pady=2)
        
        # 山地锐度
        ttk.Label(scale_frame, text="山地锐度:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.mountain_sharpness_var = tk.DoubleVar(value=self.params["mountain_sharpness"])
        mountain_slider = ttk.Scale(scale_frame, from_=0.5, to=3.0, 
                                  variable=self.mountain_sharpness_var, orient=tk.HORIZONTAL)
        mountain_slider.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(scale_frame, textvariable=self.mountain_sharpness_var, width=5).grid(row=1, column=2, padx=5, pady=2)
        
        # 细节级别
        ttk.Label(scale_frame, text="细节级别:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.detail_level_var = tk.DoubleVar(value=self.params["detail_level"])
        detail_slider = ttk.Scale(scale_frame, from_=0.5, to=2.0, 
                                variable=self.detail_level_var, orient=tk.HORIZONTAL)
        detail_slider.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(scale_frame, textvariable=self.detail_level_var, width=5).grid(row=2, column=2, padx=5, pady=2)
        
        # 地质构造
        tectonic_frame = ttk.LabelFrame(parent, text="地质构造")
        tectonic_frame.pack(fill=tk.X, pady=5)
        
        self.use_tectonic_var = tk.BooleanVar(value=self.params["use_tectonic"])
        ttk.Checkbutton(tectonic_frame, text="启用板块构造模拟", 
                       variable=self.use_tectonic_var).pack(anchor=tk.W, padx=5, pady=2)
    
    def _create_erosion_settings(self, parent):
        """创建侵蚀设置选项卡内容"""
        # 侵蚀类型
        type_frame = ttk.Frame(parent)
        type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(type_frame, text="侵蚀类型:").pack(side=tk.LEFT, padx=5)
        self.erosion_type_var = tk.StringVar(value=self.params["erosion_type"])
        erosion_types = ["thermal", "hydraulic", "combined", "advanced", "simple"]
        type_combo = ttk.Combobox(type_frame, textvariable=self.erosion_type_var, 
                                values=erosion_types, state="readonly", width=15)
        type_combo.pack(side=tk.LEFT, padx=5)
        
        # 侵蚀迭代和强度
        erosion_frame = ttk.LabelFrame(parent, text="侵蚀参数")
        erosion_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(erosion_frame, text="侵蚀迭代次数:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.erosion_iterations_var = tk.IntVar(value=self.params["erosion_iterations"])
        ttk.Spinbox(erosion_frame, from_=1, to=10, textvariable=self.erosion_iterations_var, 
                   width=5).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(erosion_frame, text="侵蚀强度:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.erosion_strength_var = tk.DoubleVar(value=self.params["erosion_strength"])
        erosion_slider = ttk.Scale(erosion_frame, from_=0.1, to=2.0, 
                                 variable=self.erosion_strength_var, orient=tk.HORIZONTAL)
        erosion_slider.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(erosion_frame, textvariable=self.erosion_strength_var, width=5).grid(row=1, column=3, padx=5, pady=2)
        
        # 热侵蚀参数
        thermal_frame = ttk.LabelFrame(parent, text="热侵蚀参数")
        thermal_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(thermal_frame, text="滑坡角度:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.talus_angle_var = tk.DoubleVar(value=self.params["talus_angle"])
        talus_slider = ttk.Scale(thermal_frame, from_=0.01, to=0.2, 
                               variable=self.talus_angle_var, orient=tk.HORIZONTAL)
        talus_slider.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(thermal_frame, textvariable=self.talus_angle_var, width=5).grid(row=0, column=2, padx=5, pady=2)
        
        # 水侵蚀参数
        hydraulic_frame = ttk.LabelFrame(parent, text="水侵蚀参数")
        hydraulic_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(hydraulic_frame, text="降雨量:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.rainfall_var = tk.DoubleVar(value=self.params["rainfall"])
        rainfall_slider = ttk.Scale(hydraulic_frame, from_=0.001, to=0.05, 
                                  variable=self.rainfall_var, orient=tk.HORIZONTAL)
        rainfall_slider.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(hydraulic_frame, textvariable=self.rainfall_var, width=5).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(hydraulic_frame, text="蒸发率:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.evaporation_var = tk.DoubleVar(value=self.params["evaporation"])
        evaporation_slider = ttk.Scale(hydraulic_frame, from_=0.1, to=0.9, 
                                     variable=self.evaporation_var, orient=tk.HORIZONTAL)
        evaporation_slider.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(hydraulic_frame, textvariable=self.evaporation_var, width=5).grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(hydraulic_frame, text="沉积物容量:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.sediment_capacity_var = tk.DoubleVar(value=self.params["sediment_capacity"])
        sediment_slider = ttk.Scale(hydraulic_frame, from_=0.05, to=0.5, 
                                   variable=self.sediment_capacity_var, orient=tk.HORIZONTAL)
        sediment_slider.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(hydraulic_frame, textvariable=self.sediment_capacity_var, width=5).grid(row=2, column=2, padx=5, pady=2)
    
    def _create_river_settings(self, parent):
        """创建河流设置选项卡内容"""
        river_frame = ttk.LabelFrame(parent, text="河流参数")
        river_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(river_frame, text="河流密度:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.river_density_var = tk.DoubleVar(value=self.params["river_density"])
        river_density_slider = ttk.Scale(river_frame, from_=0.1, to=3.0, 
                                       variable=self.river_density_var, orient=tk.HORIZONTAL)
        river_density_slider.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(river_frame, textvariable=self.river_density_var, width=5).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(river_frame, text="最小集水区大小:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.min_watershed_size_var = tk.IntVar(value=self.params["min_watershed_size"])
        ttk.Spinbox(river_frame, from_=10, to=200, textvariable=self.min_watershed_size_var, 
                   width=5).grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(river_frame, text="降水因子:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.precipitation_factor_var = tk.DoubleVar(value=self.params["precipitation_factor"])
        precip_slider = ttk.Scale(river_frame, from_=0.5, to=2.0, 
                                variable=self.precipitation_factor_var, orient=tk.HORIZONTAL)
        precip_slider.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(river_frame, textvariable=self.precipitation_factor_var, width=5).grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(river_frame, text="蜿蜒因子:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.meander_factor_var = tk.DoubleVar(value=self.params["meander_factor"])
        meander_slider = ttk.Scale(river_frame, from_=0.0, to=1.0, 
                                 variable=self.meander_factor_var, orient=tk.HORIZONTAL)
        meander_slider.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(river_frame, textvariable=self.meander_factor_var, width=5).grid(row=3, column=2, padx=5, pady=2)
    
    def _create_noise_settings(self, parent):
        """创建噪声设置选项卡内容"""
        noise_frame = ttk.LabelFrame(parent, text="噪声参数")
        noise_frame.pack(fill=tk.X, pady=5)
        
        self.use_freq_opt_var = tk.BooleanVar(value=self.params["use_frequency_optimization"])
        ttk.Checkbutton(noise_frame, text="使用频域优化", 
                       variable=self.use_freq_opt_var).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(noise_frame, text="八度数量:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.octaves_var = tk.IntVar(value=self.params["octaves"])
        ttk.Spinbox(noise_frame, from_=1, to=12, textvariable=self.octaves_var, 
                   width=5).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(noise_frame, text="持续度:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.persistence_var = tk.DoubleVar(value=self.params["persistence"])
        persistence_slider = ttk.Scale(noise_frame, from_=0.1, to=1.0, 
                                     variable=self.persistence_var, orient=tk.HORIZONTAL)
        persistence_slider.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(noise_frame, textvariable=self.persistence_var, width=5).grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(noise_frame, text="频率增长:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.lacunarity_var = tk.DoubleVar(value=self.params["lacunarity"])
        lacunarity_slider = ttk.Scale(noise_frame, from_=1.5, to=3.0, 
                                    variable=self.lacunarity_var, orient=tk.HORIZONTAL)
        lacunarity_slider.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(noise_frame, textvariable=self.lacunarity_var, width=5).grid(row=3, column=2, padx=5, pady=2)
    
    def _create_terrain_distribution_settings(self, parent):
        """创建地形分布设置选项卡内容"""
        terrain_frame = ttk.LabelFrame(parent, text="地形分布比例")
        terrain_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(terrain_frame, text="平原比例:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.plain_ratio_var = tk.DoubleVar(value=self.params["plain_ratio"])
        plain_slider = ttk.Scale(terrain_frame, from_=0.0, to=0.6, 
                               variable=self.plain_ratio_var, orient=tk.HORIZONTAL)
        plain_slider.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(terrain_frame, textvariable=self.plain_ratio_var, width=5).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(terrain_frame, text="丘陵比例:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.hill_ratio_var = tk.DoubleVar(value=self.params["hill_ratio"])
        hill_slider = ttk.Scale(terrain_frame, from_=0.0, to=0.6, 
                              variable=self.hill_ratio_var, orient=tk.HORIZONTAL)
        hill_slider.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(terrain_frame, textvariable=self.hill_ratio_var, width=5).grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(terrain_frame, text="山地比例:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.mountain_ratio_var = tk.DoubleVar(value=self.params["mountain_ratio"])
        mountain_slider = ttk.Scale(terrain_frame, from_=0.0, to=0.6, 
                                  variable=self.mountain_ratio_var, orient=tk.HORIZONTAL)
        mountain_slider.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(terrain_frame, textvariable=self.mountain_ratio_var, width=5).grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(terrain_frame, text="高原比例:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.plateau_ratio_var = tk.DoubleVar(value=self.params["plateau_ratio"])
        plateau_slider = ttk.Scale(terrain_frame, from_=0.0, to=0.4, 
                                 variable=self.plateau_ratio_var, orient=tk.HORIZONTAL)
        plateau_slider.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(terrain_frame, textvariable=self.plateau_ratio_var, width=5).grid(row=3, column=2, padx=5, pady=2)
        
        # 显示总比例
        self.total_ratio_var = tk.StringVar(value="总计: 0.9")
        ttk.Label(terrain_frame, textvariable=self.total_ratio_var).grid(row=4, column=0, columnspan=3, pady=5)
        
        # 添加值变化事件绑定以更新总比例
        self.plain_ratio_var.trace_add("write", self._update_total_ratio)
        self.hill_ratio_var.trace_add("write", self._update_total_ratio)
        self.mountain_ratio_var.trace_add("write", self._update_total_ratio)
        self.plateau_ratio_var.trace_add("write", self._update_total_ratio)
        
        # 初始化更新一次总比例
        self._update_total_ratio()
    
    def _create_climate_settings(self, parent):
        """创建气候设置选项卡内容"""
        climate_frame = ttk.LabelFrame(parent, text="气候参数")
        climate_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(climate_frame, text="纬度影响:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.latitude_effect_var = tk.DoubleVar(value=self.params["latitude_effect"])
        latitude_slider = ttk.Scale(climate_frame, from_=0.0, to=1.0, 
                                  variable=self.latitude_effect_var, orient=tk.HORIZONTAL)
        latitude_slider.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(climate_frame, textvariable=self.latitude_effect_var, width=5).grid(row=0, column=2, padx=5, pady=2)
        
        # 风向参数
        wind_frame = ttk.LabelFrame(parent, text="主导风向")
        wind_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(wind_frame, text="风向X分量:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.wind_x_var = tk.DoubleVar(value=self.params["prevailing_wind_x"])
        wind_x_slider = ttk.Scale(wind_frame, from_=-2.0, to=2.0, 
                                variable=self.wind_x_var, orient=tk.HORIZONTAL)
        wind_x_slider.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(wind_frame, textvariable=self.wind_x_var, width=5).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(wind_frame, text="风向Y分量:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.wind_y_var = tk.DoubleVar(value=self.params["prevailing_wind_y"])
        wind_y_slider = ttk.Scale(wind_frame, from_=-2.0, to=2.0, 
                                variable=self.wind_y_var, orient=tk.HORIZONTAL)
        wind_y_slider.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Entry(wind_frame, textvariable=self.wind_y_var, width=5).grid(row=1, column=2, padx=5, pady=2)
        
        # 添加风向表示
        self.wind_canvas = tk.Canvas(wind_frame, width=100, height=100, bg='white')
        self.wind_canvas.grid(row=2, column=0, columnspan=3, padx=5, pady=5)
        
        # 绘制风向指示器
        self._draw_wind_direction()
        
        # 添加风向变化事件绑定
        self.wind_x_var.trace_add("write", lambda *args: self._draw_wind_direction())
        self.wind_y_var.trace_add("write", lambda *args: self._draw_wind_direction())
    
    def _update_total_ratio(self, *args):
        """更新地形分布总比例显示"""
        try:
            plain = float(self.plain_ratio_var.get())
            hill = float(self.hill_ratio_var.get())
            mountain = float(self.mountain_ratio_var.get())
            plateau = float(self.plateau_ratio_var.get())
            
            total = plain + hill + mountain + plateau
            self.total_ratio_var.set(f"总计: {total:.2f}")
            
            # 如果总比例超过1.0，给予视觉提示
            if total > 1.0:
                self.total_ratio_var.set(f"总计: {total:.2f} (超出1.0)")
        except (ValueError, tk.TclError):
            self.total_ratio_var.set("总计: 计算错误")
    
    def _draw_wind_direction(self):
        """绘制风向指示器"""
        try:
            x_val = float(self.wind_x_var.get())
            y_val = float(self.wind_y_var.get())
            
            # 清空画布
            self.wind_canvas.delete("all")
            
            # 绘制圆形边界
            self.wind_canvas.create_oval(10, 10, 90, 90, outline="gray")
            self.wind_canvas.create_line(50, 50, 50, 10, fill="gray", dash=(2, 2))  # 北
            self.wind_canvas.create_line(50, 50, 90, 50, fill="gray", dash=(2, 2))  # 东
            self.wind_canvas.create_line(50, 50, 50, 90, fill="gray", dash=(2, 2))  # 南
            self.wind_canvas.create_line(50, 50, 10, 50, fill="gray", dash=(2, 2))  # 西
            
            # 标记方向
            self.wind_canvas.create_text(50, 5, text="N")
            self.wind_canvas.create_text(95, 50, text="E")
            self.wind_canvas.create_text(50, 95, text="S")
            self.wind_canvas.create_text(5, 50, text="W")
            
            # 计算方向向量终点
            # 注意：画布坐标系与数学坐标系Y轴相反
            mag = np.sqrt(x_val**2 + y_val**2)
            if mag > 0.0001:  # 避免除以零
                norm_x = x_val / mag
                norm_y = y_val / mag
                
                # 确定箭头长度，基于大小缩放
                arrow_length = min(40.0, 40.0 * mag / 2.0)
                
                end_x = 50 + norm_x * arrow_length
                end_y = 50 - norm_y * arrow_length  # 注意Y轴反向
                
                # 绘制风向箭头
                self.wind_canvas.create_line(50, 50, end_x, end_y, 
                                          arrow=tk.LAST, width=2, fill="blue")
                
                # 显示风向强度
                self.wind_canvas.create_text(50, 70, text=f"强度: {mag:.2f}")
        except (ValueError, tk.TclError):
            # 处理数值转换错误
            self.wind_canvas.delete("all")
            self.wind_canvas.create_text(50, 50, text="无效值")
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry('650x550+{}+{}'.format(x, y))
    
    def on_apply(self):
        """应用按钮事件处理"""
        # 收集所有参数
        self.result = {
            # 基本参数
            "seed": self.seed_var.get(),
            "scale_factor": self.scale_factor_var.get(),
            "mountain_sharpness": self.mountain_sharpness_var.get(),
            "detail_level": self.detail_level_var.get(),
            "use_tectonic": self.use_tectonic_var.get(),
            
            # 侵蚀参数
            "erosion_type": self.erosion_type_var.get(),
            "erosion_iterations": self.erosion_iterations_var.get(),
            "erosion_strength": self.erosion_strength_var.get(),
            "talus_angle": self.talus_angle_var.get(),
            "rainfall": self.rainfall_var.get(),
            "evaporation": self.evaporation_var.get(),
            "sediment_capacity": self.sediment_capacity_var.get(),
            
            # 河流参数
            "river_density": self.river_density_var.get(),
            "min_watershed_size": self.min_watershed_size_var.get(),
            "precipitation_factor": self.precipitation_factor_var.get(),
            "meander_factor": self.meander_factor_var.get(),
            
            # 噪声参数
            "use_frequency_optimization": self.use_freq_opt_var.get(),
            "octaves": self.octaves_var.get(),
            "persistence": self.persistence_var.get(),
            "lacunarity": self.lacunarity_var.get(),
            
            # 地形分布参数
            "plain_ratio": self.plain_ratio_var.get(),
            "hill_ratio": self.hill_ratio_var.get(),
            "mountain_ratio": self.mountain_ratio_var.get(),
            "plateau_ratio": self.plateau_ratio_var.get(),
            
            # 气候参数
            "latitude_effect": self.latitude_effect_var.get(),
            "prevailing_wind_x": self.wind_x_var.get(),
            "prevailing_wind_y": self.wind_y_var.get()
        }
        
        self.dialog.destroy()
    
    def on_cancel(self):
        """取消按钮事件处理"""
        self.result = None
        self.dialog.destroy()
    
    def on_reset(self):
        """重置为默认值按钮事件处理"""
        # 重新设置默认参数
        self._setup_defaults()
        
        # 更新UI控件值
        self.seed_var.set(self.params["seed"])
        self.scale_factor_var.set(self.params["scale_factor"])
        self.mountain_sharpness_var.set(self.params["mountain_sharpness"])
        self.detail_level_var.set(self.params["detail_level"])
        self.use_tectonic_var.set(self.params["use_tectonic"])
        
        self.erosion_type_var.set(self.params["erosion_type"])
        self.erosion_iterations_var.set(self.params["erosion_iterations"])
        self.erosion_strength_var.set(self.params["erosion_strength"])
        self.talus_angle_var.set(self.params["talus_angle"])
        self.rainfall_var.set(self.params["rainfall"])
        self.evaporation_var.set(self.params["evaporation"])
        self.sediment_capacity_var.set(self.params["sediment_capacity"])
        
        self.river_density_var.set(self.params["river_density"])
        self.min_watershed_size_var.set(self.params["min_watershed_size"])
        self.precipitation_factor_var.set(self.params["precipitation_factor"])
        self.meander_factor_var.set(self.params["meander_factor"])
        
        self.use_freq_opt_var.set(self.params["use_frequency_optimization"])
        self.octaves_var.set(self.params["octaves"])
        self.persistence_var.set(self.params["persistence"])
        self.lacunarity_var.set(self.params["lacunarity"])
        
        self.plain_ratio_var.set(self.params["plain_ratio"])
        self.hill_ratio_var.set(self.params["hill_ratio"])
        self.mountain_ratio_var.set(self.params["mountain_ratio"])
        self.plateau_ratio_var.set(self.params["plateau_ratio"])
        
        self.latitude_effect_var.set(self.params["latitude_effect"])
        self.wind_x_var.set(self.params["prevailing_wind_x"])
        self.wind_y_var.set(self.params["prevailing_wind_y"])

def show_terrain_settings(parent, initial_params=None):
    """显示地形设置对话框并返回设置参数"""
    dialog = TerrainGeneratorDialog(parent, initial_params)
    parent.wait_window(dialog.dialog)
    return dialog.result