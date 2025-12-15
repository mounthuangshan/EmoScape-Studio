#from __future__ import annotations
#标准库
import os
import json
import random
import shutil
import threading
import queue
import subprocess
import sys
import shlex
import io
#import logging

#数据处理与科学计算
import numpy as np
#import cupy as cp
#import pycuda.driver as cuda

#图形界面与绘图
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LightSource
from PIL import Image, ImageTk

#网络与并发
import concurrent.futures
import threading

#其他工具
from openai import OpenAI
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable
from enum import Enum, auto
import enum
from collections import defaultdict
import traceback

#项目文件
from utils.llm import *
from core.update_map import *
from core.evolution.evolve_generation import *
from utils.tools import *
from utils.export import *
from core.core import *
from utils.preview_map import *
from utils.preview_map_3d import preview_map_3d
from utils.terrain_settings import show_terrain_settings
from core.evolution.evolve_generation import *
from core.core import *
from core.generation.build_hierarchical_road_network import *
from core.generation.calculate_distance_to_water import *
from core.generation.carve_caves_and_ravines import *
from core.generation.generate_height_temp_humid import *
from core.generation.generate_rivers import *
from core.services.analyze_map_emotions import *
from core.services.run_neat_for_content_layout import *
from core.evolution.evolve_generation import *
from utils.classify_biome import *
from utils.llm import *
from utils.export import *
from core.geo_data_importer import GeoDataImporter
from core.services.map_tools import *
from core.generation.generate_humidity_map import *
from core.generation.generate_temperature_map import *
from core.generation.generate_undergroud import *
from level_generator import LevelGenerator
from core.emotional.emotion_manger import *
from core.emotional.story_emotion_analyzer import StoryEmotionAnalyzer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core.emotional.story_emotion_map import *
from core.dynamic_music_generator import *
from core.generate_music import *
from plugin.autotrain.autotrain import *
from plugin.hunyuan.hunyuan_gui import Hunyuan3DDialog
from plugin.inspiremusic.inspiremusic_client import InspireMusicDialog

# 若无OpenAI密钥请注释
OpenAI.api_key = os.environ.get("OPENAI_API_KEY", None)
OpenAI.base_urls  = "https://api.moonshot.cn/v1"

ATTR_RANGES = {
    "MoveSpeed": (1,10),
    "Range": (1,15),
    "PhysAttack": (1,50),
    "PhysDefense": (1,50),
    "AttackSpeed": (1,5),
    "RespawnTime": (1,60),
    "MagAttack": (1,50),
    "MagDefense": (1,50),
    "strength": (1, 100),
    "agility": (1, 100),
    "intelligence": (1, 100),
    "vitality": (10, 200),
    "armor": (0, 50)
}
ATTR_NAMES = list(ATTR_RANGES.keys())
TARGET_RATIO = 2.0  # 用于生态平衡目标


####################################################
#APP（UI）、MapController（控制器）、dataclass（数据）
####################################################
# 设置日志
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')# 时间戳，日志级别，信息
#logger = logging.getLogger(__name__)


# 高效地图数据结构
class MapData:
    """统一的地图数据容器，使用结构化数组和内存映射优化大地图处理"""
    
    def __init__(self, width, height, use_gpu=False, cache_dir=".map_cache"):
        self.width = width
        self.height = height
        self.use_gpu = use_gpu
        self.generation_complete = False
        self.cache_dir = cache_dir
        self.layers = {}
        self.object_indices = {}
        self.chunk_status = {}
        self.params = {}
        self.biome_data = None
        self.content_layout = {} # 存储整张地图的情感信息
        self.story_content = None  # 用于存储扩展的故事剧情内容
        self.story_analysis = None  # 用于存储故事情感分析结果
        self.emotion_map = None  # 用于存储情感地图对象
        
        # 新增: 生成状态跟踪
        self.generation_state = {
            'resume_point': 'start',
            'completed_gens': 0
        }
        
        # 新增: 编辑器状态
        self.pending_editor = None
        self.editor_state = None
        
        # 新增：地下层数据存储
        self.underground_layers = {}
        self.underground_depth = 0  # 当前配置的地下层数量
        
        # 矿物分布数据
        self.mineral_layers = {}
        
        # 地下洞穴连通网络
        self.cave_networks = []
        
        # 地下水系统
        self.underground_water = {}
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 检查GPU支持
        try:
            import cupy as cp
            self.cp = cp
            self.HAS_GPU = True
            logging.info("GPU加速可用")
        except ImportError:
            self.cp = None
            self.HAS_GPU = False
            logging.info("GPU加速不可用，使用CPU模式")
        
    def is_valid(self):
        """检查地图数据是否有效"""
        if not self.layers:
            logging.warning("地图数据无效：没有任何图层")
            return False
        
        # 检查必需的核心图层是否存在
        essential_layers = ["height"]
        missing_layers = [layer for layer in essential_layers if layer not in self.layers]
        if missing_layers:
            logging.warning(f"地图数据无效：缺少关键图层 {missing_layers}")
            return False
        
        # 检查高度图层
        height_layer = self.layers.get("height")
        if not isinstance(height_layer, (np.ndarray)):
            logging.warning(f"地图数据无效：高度图层类型错误，期望np.ndarray，实际为{type(height_layer)}")
            return False
        
        if height_layer.ndim != 2:
            logging.warning(f"地图数据无效：高度图层维度错误，期望2维，实际为{height_layer.ndim}维")
            return False
        
        return True
        
    def create_layer(self, name, dtype=np.float32, fill_value=0):
        """创建新的地图层"""
        # 确保fill_value是标量并进行类型转换
        if isinstance(fill_value, np.ndarray) and fill_value.size == 1:
            fill_value = fill_value.item()
            
        try:
            fill_value = np.dtype(dtype).type(fill_value)
        except (TypeError, ValueError):
            raise ValueError(f"无法将 {type(fill_value)} 转换为 {dtype} 类型")

        # 确保宽高为整数
        self.height = int(self.height)
        self.width = int(self.width)

        # 使用GPU还是CPU创建数组
        if self.use_gpu and self.HAS_GPU and dtype != np.bool_:
            try:
                self.layers[name] = self.cp.full((self.height, self.width), fill_value, dtype=dtype)
                return self.layers[name]
            except Exception as e:
                logging.warning(f"GPU数组创建失败: {str(e)}，回退到CPU")
        
        # 大型数组使用内存映射
        if self.width * self.height > 1_000_000:
            filename = f"{self.cache_dir}/{name}_layer.npy"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.layers[name] = np.memmap(filename, dtype=dtype, mode='w+', 
                                        shape=(self.height, self.width))
            self.layers[name][:] = fill_value
        else:
            self.layers[name] = np.full((self.height, self.width), fill_value, dtype=dtype)
        
        return self.layers[name]
    
    def add_object_layer(self, name, objects, skip_invalid=False):
        """添加对象层时立即验证元素有效性"""
        valid_objects = []
        invalid_count = 0
        
        for idx, obj in enumerate(objects):
            try:
                # 这里可以添加对象验证逻辑
                valid_objects.append(obj)
            except Exception as e:
                invalid_count += 1
                if not skip_invalid:
                    raise ValueError(f"对象 {idx} 无效: {str(e)}")
        
        if invalid_count > 0:
            logging.warning(f"对象层 '{name}' 中有 {invalid_count} 个无效对象被跳过")
        
        self.layers[name] = valid_objects
        self._build_spatial_index(name)
        return valid_objects
        
    def _build_spatial_index(self, layer_name):
        """为对象层构建空间索引，加速空间查询"""
        objects = self.layers[layer_name]
        index = defaultdict(list)
        chunk_size = 16  # 索引单元格大小
        
        for i, obj in enumerate(objects):
            try:
                if "x" in obj and "y" in obj:
                    cell_x = int(obj["x"]) // chunk_size
                    cell_y = int(obj["y"]) // chunk_size
                    key = (cell_x, cell_y)
                    index[key].append(i)
            except Exception as e:
                logging.warning(f"构建空间索引时跳过对象 {i}: {str(e)}")
                
        self.object_indices[layer_name] = index
    
    def query_region(self, layer_name, x1, y1, x2, y2):
        """查询区域内的对象"""
        if layer_name not in self.object_indices:
            return []
            
        index = self.object_indices[layer_name]
        objects = self.layers[layer_name]
        chunk_size = 16
        result = []
        
        # 计算覆盖的网格单元
        min_cell_x, min_cell_y = x1 // chunk_size, y1 // chunk_size
        max_cell_x, max_cell_y = x2 // chunk_size, y2 // chunk_size
        
        # 遍历相关网格单元
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                for obj_idx in index.get((cell_x, cell_y), []):
                    obj = objects[obj_idx]
                    if x1 <= obj["x"] <= x2 and y1 <= obj["y"] <= y2:
                        result.append(obj)
        
        return result
    
    def get_layer(self, name):
        """安全获取图层数据"""
        if name not in self.layers:
            return None
            
        data = self.layers[name]
        # 动态类型检查
        if self.HAS_GPU and hasattr(data, 'get'):
            # 如果是GPU数组，转换为numpy数组
            try:
                return self.cp.asnumpy(data)
            except:
                pass
        
        # 修复：确保只在数据不为None时才尝试复制
        if data is None:
            return None
        return data if isinstance(data, np.ndarray) else data.copy()
    
    def to_cpu(self, name=None):
        """将指定层或所有层从GPU转移到CPU"""
        if not self.HAS_GPU:
            return
            
        if name:
            if name in self.layers and hasattr(self.layers[name], 'get'):
                self.layers[name] = self.cp.asnumpy(self.layers[name])
        else:
            for key in list(self.layers.keys()):
                if hasattr(self.layers[key], 'get'):
                    self.layers[key] = self.cp.asnumpy(self.layers[key])
    
    def unpack(self):
        """提取所有层数据的标准接口"""
        return (
            self.get_layer("height"),
            self.get_layer("biome"),
            self.layers.get("vegetation", []),
            self.layers.get("buildings", []),
            self.get_layer("rivers"),
            self.content_layout,
            self.get_layer("caves"),
            self.params,
            self.biome_data,
            self.get_layer("roads"),
            (self.get_layer("roads_map"), self.get_layer("roads_types")),
        )
        
    def create_underground_layers(self, depth=3, layer_height=20):
        """创建指定深度的地下层系统
        
        Args:
            depth: 地下层数量
            layer_height: 每层厚度（垂直单位）
        """
        self.underground_depth = depth
        
        # 为每个地下层创建高度图和内容图
        for i in range(depth):
            layer_name = f"underground_{i}"
            self.underground_layers[layer_name] = {
                "height": self.create_layer(f"{layer_name}_height", dtype=np.float32, fill_value=0),
                "content": self.create_layer(f"{layer_name}_content", dtype=np.int32, fill_value=0),
                "detail": {
                    "depth": i * layer_height,
                    "thickness": layer_height,
                    "major_caves": [],
                    "structures": []
                }
            }
            
            # 创建矿物分布层
            self.mineral_layers[layer_name] = self.create_layer(f"{layer_name}_minerals", dtype=np.int32, fill_value=0)
        
        return self.underground_layers

    def get_underground_layer(self, depth_index):
        """获取指定深度的地下层数据"""
        layer_name = f"underground_{depth_index}"
        if layer_name in self.underground_layers:
            return self.underground_layers[layer_name]
        return None

    def add_underground_structure(self, depth_index, structure_data):
        """添加地下结构（洞穴、神殿、矿井等）"""
        layer_name = f"underground_{depth_index}"
        if layer_name in self.underground_layers:
            self.underground_layers[layer_name]["detail"]["structures"].append(structure_data)
            return True
        return False

    def add_cave_network(self, cave_network):
        """添加洞穴网络，连接多个地下层"""
        self.cave_networks.append(cave_network)
        
# 数据类定义（地图参数）
@dataclass
class MapParameters:
    achievement: float = 0.5 #  成就
    exploration: float = 0.5 # 探险
    social: float = 0.5 # 社交
    combat: float = 0.5 # 战斗
    map_width: int = 100 # 地图长度
    map_height: int = 100 # 地图宽度
    vegetation_coverage: float = 0.3 # 植被覆盖率
    river_count: int = 3 # 河流数量
    city_count: int = 3 # 城市数量
    cave_density: int = 0.5 # 洞穴密度
    evolution_generations: int = 1 # 交互式进化代数
    
    # 基础参数
    width: int = 256
    height: int = 256
    seed: int = None
    
    # 高级地形参数
    scale_factor: float = 1.0        # 地形尺度因子
    mountain_sharpness: float = 1.0  # 山地锐度
    erosion_iterations: int = 3      # 侵蚀迭代次数
    river_density: float = 1.0       # 河流密度
    use_tectonic: bool = True        # 是否使用板块构造
    detail_level: float = 1.0        # 细节程度
    
    # 侵蚀参数
    erosion_type: str = 'advanced'   # 侵蚀类型(thermal/hydraulic/combined/advanced/simple)
    erosion_strength: float = 0.8    # 侵蚀强度
    
    # 热力侵蚀参数
    talus_angle: float = 0.05        # 滑坡角度
    angle_of_repose: float = 35.0    # 安息角
    sediment_capacity: float = 0.15  # 沉积物容量
    
    # 水力侵蚀参数
    rainfall: float = 0.01           # 降雨量
    evaporation: float = 0.5         # 蒸发率
    water_capacity: float = 0.05     # 水容量
    water_erosion: float = 0.3       # 水侵蚀率
    deposition: float = 0.5          # 沉积率
    
    # 河流参数
    river_threshold: float = 0.7     # 河流起点高度阈值
    min_river_length: int = 10       # 最小河流长度
    min_watershed_size: int = 50     # 最小集水区面积
    precipitation_factor: float = 1.0 # 降水因子
    meander_factor: float = 0.3      # 河流蜿蜒因子
    
    # 气候参数
    latitude_effect: float = 0.5     # 纬度对温度的影响
    prevailing_wind_x: float = 1.0   # 主导风向X分量
    prevailing_wind_y: float = 0.0   # 主导风向Y分量
    
    # 噪声参数
    use_frequency_optimization: bool = True  # 使用频域优化
    octaves: int = 6                 # 八度数
    persistence: float = 0.5         # 持续度
    lacunarity: float = 2.0          # 频率增长因子
    
    # 地形分布参数
    plain_ratio: float = 0.3         # 平原比例
    hill_ratio: float = 0.3          # 丘陵比例
    mountain_ratio: float = 0.2      # 山地比例
    plateau_ratio: float = 0.1       # 高原比例
    
    # 板块构造参数
    num_plates: int = 7              # 板块数量
    plate_iterations: int = 100      # 板块迭代次数
    
    # 新增地形生成性能优化参数
    enable_micro_detail: bool = True      # 是否启用微细节生成
    enable_extreme_detection: bool = True # 是否启用异常点检测
    optimization_level: int = 1           # 优化级别：0=低(快速)，1=中(均衡)，2=高(高质量)
    # 新增真实地貌特征参数
    enable_realistic_landforms: bool = True
    dominant_landform: str = "auto"

    # 新增大地形参数
    large_map_mode: bool = False      # 开启大地图模式
    province_count: int = 3           # 地质省份数量
    macro_feature_scale: float = 2.0  # 宏观特征缩放因子
    auto_detect_large_map: bool = True  # 自动检测大地图模式
    
    # 地下系统参数
    enable_underground: bool = False      # 是否启用地下系统生成
    underground_depth: int = 3           # 地下层深度
    underground_water_prevalence: float = 0.5  # 地下水系统丰富度
    underground_structure_density: float = 0.5  # 地下结构密度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "achievement": self.achievement,
            "exploration": self.exploration,
            "social": self.social,
            "combat": self.combat,
            "map_width": self.map_width,
            "map_height": self.map_height,
            "vegetation_coverage": self.vegetation_coverage,
            "river_count": self.river_count,
            "city_count": self.city_count,
            "cave_density": self.cave_density
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MapParameters':
        return cls(
            achievement=data.get("achievement", 0.5),
            exploration=data.get("exploration", 0.5),
            social=data.get("social", 0.5),
            combat=data.get("combat", 0.5),
            map_width=data.get("map_width", 100),
            map_height=data.get("map_height", 100),
            vegetation_coverage=data.get("vegetation_coverage", 0.3),
            river_count=data.get("river_count", 3),
            city_count=data.get("city_count", 3),
            cave_density=data.get("cave_density", 1)
        )
        
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['MapParameters']:
        """从文件加载参数，返回MapParameters实例或None（出错时）"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logging.error(f"加载参数失败: {str(e)}")
            return None
    
    def save_to_file(self, filepath: str) -> bool:
        """将参数保存到文件，成功返回True，失败返回False"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logging.error(f"保存参数失败: {str(e)}")
            return False       
        
        
class ConfigManager:
    """配置管理器，处理应用配置的加载、保存和默认值"""
    
    def __init__(self, config_file: str = "./data/configs/app_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            "theme": {
                "name": "default",
                "style": "light",
                "accent_color": "#4a86e8",
                "font_family": "Arial"
            },
            "recent_files": [],
            "auto_save": True,
            "auto_save_interval": 5,  # 分钟
            "max_undo": 20,
            "default_export_path": "./exports",
            "ui": {
                "window_width": 1200,
                "window_height": 800,
                "paned_position": 300,
                "font_size": 10
            },
            "presets": {
                "默认": {
                    "achievement": 0.5,
                    "exploration": 0.5,
                    "social": 0.5,
                    "combat": 0.5,
                    "map_width": 512,
                    "map_height": 512,
                    "vegetation_coverage": 0.3,
                    "river_count": 3,
                    "city_count": 3,
                    "cave_density": 0.5,
                    "scale_factor": 1.0,
                    "mountain_sharpness": 1.0,
                    "river_density": 1.0
                },
                "山地地形": {
                    "achievement": 0.7,
                    "exploration": 0.8,
                    "social": 0.3,
                    "combat": 0.6,
                    "map_width": 800,
                    "map_height": 800,
                    "vegetation_coverage": 0.4,
                    "river_count": 5,
                    "city_count": 2,
                    "cave_density": 0.7,
                    "scale_factor": 1.5,
                    "mountain_sharpness": 2.0,
                    "river_density": 1.2
                },
                "平原地形": {
                    "achievement": 0.4,
                    "exploration": 0.3,
                    "social": 0.7,
                    "combat": 0.3,
                    "map_width": 1024,
                    "map_height": 1024,
                    "vegetation_coverage": 0.6,
                    "river_count": 8,
                    "city_count": 10,
                    "cave_density": 0.2,
                    "scale_factor": 0.8,
                    "mountain_sharpness": 0.5,
                    "river_density": 1.5
                },
                "岛屿地形": {
                    "achievement": 0.6,
                    "exploration": 0.9,
                    "social": 0.4,
                    "combat": 0.4,
                    "map_width": 640,
                    "map_height": 640,
                    "vegetation_coverage": 0.8,
                    "river_count": 2,
                    "city_count": 1,
                    "cave_density": 0.3,
                    "scale_factor": 0.7,
                    "mountain_sharpness": 1.5,
                    "river_density": 0.7
                },
                "峡谷地形": {
                    "achievement": 0.8,
                    "exploration": 0.7,
                    "social": 0.2,
                    "combat": 0.8,
                    "map_width": 900,
                    "map_height": 900,
                    "vegetation_coverage": 0.2,
                    "river_count": 1,
                    "city_count": 2,
                    "cave_density": 0.9,
                    "scale_factor": 1.8,
                    "mountain_sharpness": 2.5,
                    "river_density": 0.5
                }
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 使用递归合并确保所有必要的配置都存在
                    return self._merge_configs(loaded_config, default_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"加载配置文件失败: {e}")
        
        return default_config
    
    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            print(f"保存配置文件失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，增强类型安全性"""
        path = key.split('.')
        config = self.config
        
        for part in path:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default
        
        # 类型安全检查：如果默认值是字典，但结果不是，则返回默认值
        if isinstance(default, dict) and not isinstance(config, dict):
            return default
        
        return config
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        path = key.split('.')
        config = self.config
        
        # 导航到最后一个键的父级
        for part in path[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        # 设置值
        config[path[-1]] = value
        
        # 自动保存配置
        if self.get("auto_save", True):
            self.save_config()
            
    def _merge_configs(self, target, source):
        """递归合并配置字典，增强健壮性"""
        # 类型安全检查
        if not isinstance(target, dict) or not isinstance(source, dict):
            # 如果目标不是字典，直接使用源字典
            if not isinstance(target, dict):
                return source.copy() if isinstance(source, dict) else {}
            return target
            
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                self._merge_configs(target[key], value)
            # 当源是字典但目标不是时，用字典替换
            elif isinstance(value, dict) and not isinstance(target[key], dict):
                target[key] = value.copy()
        return target

class TaskManager:
    """任务管理器 - 处理长时间运行的任务，避免UI阻塞"""
    
    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self._pending_tasks = {}
        self._completed_queue = queue.Queue()
        
    def submit_task(self, name: str, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """提交任务到线程池"""
        future = self._executor.submit(func, *args, **kwargs)
        self._pending_tasks[name] = future
        future.add_done_callback(lambda f: self._task_completed(name, f))
        return future
    
    def _task_completed(self, name: str, future: concurrent.futures.Future):
        """任务完成回调"""
        self._completed_queue.put((name, future))
        if name in self._pending_tasks:
            del self._pending_tasks[name]
    
    def get_completed_tasks(self) -> List[Tuple[str, concurrent.futures.Future]]:
        """获取所有已完成的任务"""
        results = []
        while not self._completed_queue.empty():
            results.append(self._completed_queue.get_nowait())
        return results
    
    def shutdown(self):
        """关闭任务管理器"""
        self._executor.shutdown(wait=False)

# 枚举类型定义
class MapFeature(enum.Enum):
    HEIGHT_MAP = "height_map"
    BIOME = "biome"
    VEGETATION = "vegetation"
    BUILDINGS = "buildings"
    RIVERS = "rivers"
    CAVES = "caves"
    ROADS = "roads"

class TaskStatus(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class CommandHistory:
    """命令历史类，支持撤销/重做功能"""
    
    def __init__(self, max_history: int = 20):
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = max_history
        
    def add_command(self, command: Dict[str, Any]):
        """添加命令到历史"""
        self.undo_stack.append(command)
        self.redo_stack.clear()  # 新命令会清除重做栈
        
        # 限制历史大小
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
    
    def can_undo(self) -> bool:
        """检查是否可以撤销"""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """检查是否可以重做"""
        return len(self.redo_stack) > 0
    
    def undo(self) -> Optional[Dict[str, Any]]:
        """撤销上一个命令"""
        if not self.can_undo():
            return None
        
        command = self.undo_stack.pop()
        self.redo_stack.append(command)
        return command
    
    def redo(self) -> Optional[Dict[str, Any]]:
        """重做上一个撤销的命令"""
        if not self.can_redo():
            return None
        
        command = self.redo_stack.pop()
        self.undo_stack.append(command)
        return command
    
    def clear(self):
        """清除历史"""
        self.undo_stack.clear()
        self.redo_stack.clear()

# =============================================================================
# 领域模型：高级游戏设计理论集成
# =============================================================================

class ContentType(enum.Enum):
    """内容类型枚举，支持游戏设计理论中的MDA框架"""
    CHALLENGE = auto()    # 挑战性内容
    RESOURCE = auto()     # 资源类内容
    STORY = auto()        # 叙事内容
    AESTHETIC = auto()    # 美学内容
    SOCIAL = auto()       # 社交内容
    MYSTERY = auto()      # 神秘内容
    GUIDANCE = auto()     # 引导性内容

@dataclass
class GameplayPattern:
    """游戏设计模式，基于通用游戏设计理论"""
    name: str
    description: str
    engagement_value: float  # 玩家参与度贡献
    content_types: List[ContentType]
    spatial_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def satisfies(self, position: Tuple[int, int], world_context: Dict) -> float:
        """评估给定位置对该模式的满足程度"""
        if not self.spatial_requirements:
            return 1.0
            
        satisfaction = 1.0
        x, y = position
        
        # 检查生物群系适应性
        if "biomes" in self.spatial_requirements and world_context.get("biome_map"):
            biome = world_context["biome_map"][y][x]["name"]
            preferred_biomes = self.spatial_requirements["biomes"]
            if biome not in preferred_biomes:
                satisfaction *= 0.5
        
        # 检查高度要求
        if "height_range" in self.spatial_requirements and world_context.get("height_map"):
            h_min, h_max = self.spatial_requirements["height_range"]
            height = world_context["height_map"][y][x]
            if not (h_min <= height <= h_max):
                satisfaction *= 0.3
                
        # 检查与其他对象的关系
        if "proximity" in self.spatial_requirements:
            for obj_type, (min_dist, max_dist, importance) in self.spatial_requirements["proximity"].items():
                closest_dist = world_context.get(f"{obj_type}_distance_map", {}).get((x, y), float('inf'))
                if not (min_dist <= closest_dist <= max_dist):
                    satisfaction *= (1.0 - importance)
        
        return satisfaction

@dataclass
class PlayerModel:
    """玩家模型，用于预测玩家偏好和行为"""
    exploration_bias: float = 0.5     # 探索倾向
    challenge_seeking: float = 0.5    # 挑战追求
    collection_focus: float = 0.5     # 收集专注
    story_engagement: float = 0.5     # 故事参与
    social_preference: float = 0.5    # 社交偏好
    completion_drive: float = 0.5     # 完成驱动力
    novelty_seeking: float = 0.5      # 新奇追求
    
    def predict_engagement(self, content: Dict, position: Tuple[int, int], world_context: Dict) -> float:
        """预测玩家对特定内容在特定位置的参与度"""
        engagement = 0.5  # 基础参与度
        
        # 根据内容类型调整参与度
        if content.get("type") in ["treasure", "resource"]:
            engagement += self.collection_focus * 0.3
            
        if content.get("type") in ["enemy", "boss", "trap"]:
            engagement += self.challenge_seeking * 0.4
            
        if content.get("type") in ["npc", "story_event"]:
            engagement += self.story_engagement * 0.4
            
        # 根据位置调整参与度
        if "height_map" in world_context:
            # 高海拔地区对探索型玩家更有吸引力
            height = world_context["height_map"][position[1]][position[0]]
            height_norm = min(1.0, height / 100.0)  # 假设高度范围0-100
            engagement += self.exploration_bias * height_norm * 0.2
            
        # 考虑新奇性
        nearby_similar = 0
        for obj in world_context.get("placed_objects", []):
            if obj.get("type") == content.get("type"):
                dist = math.sqrt((obj["x"] - position[0])**2 + (obj["y"] - position[1])**2)
                if dist < 20:  # 考虑20单位范围内的相似对象
                    nearby_similar += 1
                    
        # 如果附近有相似对象，降低新奇性
        if nearby_similar > 0:
            novelty_penalty = min(0.5, nearby_similar * 0.1)
            engagement -= self.novelty_seeking * novelty_penalty
            
        return max(0.1, min(1.0, engagement))  # 限制在0.1-1.0范围内

# 定义游戏设计模式库
GAMEPLAY_PATTERNS = [
    GameplayPattern(
        name="探索奖励",
        description="在难以到达的地点放置有价值的资源",
        engagement_value=0.8,
        content_types=[ContentType.RESOURCE, ContentType.MYSTERY],
        spatial_requirements={
            "height_range": (70, 100),
            "biomes": ["Mountain", "Highland", "Cave"],
            "proximity": {
                "building": (20, float('inf'), 0.7),  # 远离建筑
                "road": (10, float('inf'), 0.5),      # 远离道路
            }
        }
    ),
    GameplayPattern(
        name="引导路径",
        description="使用视觉指引引导玩家探索",
        engagement_value=0.6,
        content_types=[ContentType.GUIDANCE, ContentType.AESTHETIC],
        spatial_requirements={
            "height_range": (20, 80),
            "proximity": {
                "story_event": (5, 30, 0.8),  # 靠近故事事件但不要太近
            }
        }
    ),
    GameplayPattern(
        name="危险与回报",
        description="高风险区域提供高价值奖励",
        engagement_value=0.9,
        content_types=[ContentType.CHALLENGE, ContentType.RESOURCE],
        spatial_requirements={
            "biomes": ["Desert", "Volcano", "DeepCave", "Jungle"],
            "proximity": {
                "enemy": (0, 15, 0.9),  # 靠近敌人
            }
        }
    ),
    GameplayPattern(
        name="社交中心",
        description="创建玩家自然聚集的地点",
        engagement_value=0.7,
        content_types=[ContentType.SOCIAL, ContentType.STORY],
        spatial_requirements={
            "biomes": ["Plains", "Grassland", "Village"],
            "proximity": {
                "building": (0, 20, 0.8),  # 靠近建筑
                "road": (0, 10, 0.6),      # 靠近道路
            }
        }
    ),
    GameplayPattern(
        name="资源节点",
        description="提供特定资源的集中区域，鼓励玩家回访",
        engagement_value=0.7,
        content_types=[ContentType.RESOURCE, ContentType.GUIDANCE],
        spatial_requirements={
            "biomes": ["Forest", "Highland", "Mountain", "Swamp", "River"],
            "proximity": {
                "road": (5, 40, 0.6),       # 不太远的道路
                "building": (15, 60, 0.5),  # 适度远离建筑
                "resource": (40, float('inf'), 0.8)  # 远离其他资源节点
            }
        }
    ),
    GameplayPattern(
        name="风景点",
        description="提供壮观视野的位置，通常带有成就或收藏品",
        engagement_value=0.6,
        content_types=[ContentType.AESTHETIC, ContentType.MYSTERY],
        spatial_requirements={
            "height_range": (60, 100),  # 较高位置
            "biomes": ["Mountain", "Highland", "Coast", "Cliff"],
            "proximity": {
                "landmark": (0, 15, 0.9),   # 靠近地标
                "path": (0, 20, 0.7)        # 靠近小径
            }
        }
    ),
    GameplayPattern(
        name="挑战序列",
        description="一系列递进难度的挑战，玩家需要按顺序完成",
        engagement_value=0.85,
        content_types=[ContentType.CHALLENGE, ContentType.GUIDANCE],
        spatial_requirements={
            "biomes": ["Ruins", "Dungeon", "DeepCave", "Temple"],
            "proximity": {
                "challenge": (10, 25, 0.9),  # 距离其他挑战适当
                "reward": (5, 10, 0.7)       # 每个挑战后有奖励
            }
        }
    ),
    GameplayPattern(
        name="隐藏区域",
        description="需要特殊条件才能发现或进入的秘密区域",
        engagement_value=0.9,
        content_types=[ContentType.MYSTERY, ContentType.RESOURCE],
        spatial_requirements={
            "height_range": (10, 40),  # 通常在较低位置
            "biomes": ["Cave", "Forest", "Ruins", "UndergroundRiver"],
            "proximity": {
                "road": (30, float('inf'), 0.8),  # 远离主要道路
                "landmark": (15, 50, 0.6)         # 不太远的地标作提示
            }
        }
    ),
    GameplayPattern(
        name="安全避风港",
        description="玩家可以休息、交易和恢复的安全区域",
        engagement_value=0.7,
        content_types=[ContentType.SOCIAL, ContentType.AESTHETIC],
        spatial_requirements={
            "height_range": (20, 60),
            "biomes": ["Village", "Plains", "Forest", "Coast"],
            "proximity": {
                "enemy": (40, float('inf'), 0.9),  # 远离敌人
                "road": (0, 15, 0.8),              # 靠近道路
                "water": (5, 30, 0.6)              # 不远处有水源
            }
        }
    ),
    GameplayPattern(
        name="空间导航挑战",
        description="需要特殊移动能力或解谜才能通过的区域",
        engagement_value=0.8,
        content_types=[ContentType.CHALLENGE, ContentType.GUIDANCE],
        spatial_requirements={
            "biomes": ["Mountain", "Cave", "Ruins", "Cliff"],
            "proximity": {
                "reward": (5, 20, 0.8),  # 通过后有奖励
                "path": (0, 10, 0.7)     # 有明确的路径指引
            }
        }
    ),
    GameplayPattern(
        name="故事讲述点",
        description="展现游戏世界历史和背景的区域",
        engagement_value=0.7,
        content_types=[ContentType.STORY, ContentType.AESTHETIC],
        spatial_requirements={
            "biomes": ["Ruins", "Temple", "Village", "Monument"],
            "proximity": {
                "landmark": (0, 15, 0.9),    # 靠近地标
                "story_event": (20, 50, 0.8) # 与其他故事点保持一定距离
            }
        }
    ),
    GameplayPattern(
        name="战略高地",
        description="提供战略优势的位置，通常是为了战斗或观察",
        engagement_value=0.75,
        content_types=[ContentType.CHALLENGE, ContentType.RESOURCE],
        spatial_requirements={
            "height_range": (70, 100),  # 高位置
            "biomes": ["Mountain", "Highland", "Fortress"],
            "proximity": {
                "enemy": (10, 40, 0.7),       # 有敌人但不会立即遭遇
                "resource": (5, 20, 0.6)      # 有资源支持战斗
            }
        }
    ),
    GameplayPattern(
        name="环境叙事",
        description="通过环境布置和细节暗示故事，不用直接对话",
        engagement_value=0.65,
        content_types=[ContentType.STORY, ContentType.MYSTERY],
        spatial_requirements={
            "biomes": ["Ruins", "AbandonedVillage", "Battlefield", "Shrine"],
            "proximity": {
                "path": (0, 15, 0.7),      # 靠近路径但不太明显
                "landmark": (5, 25, 0.8)   # 与地标关联
            }
        }
    ),
    GameplayPattern(
        name="生态展示",
        description="展示独特生物和生态系统的区域，适合学习和探索",
        engagement_value=0.6,
        content_types=[ContentType.AESTHETIC, ContentType.MYSTERY],
        spatial_requirements={
            "biomes": ["Jungle", "Coral", "MagicForest", "Swamp"],
            "proximity": {
                "water": (0, 25, 0.8),    # 通常靠近水源
                "resource": (5, 15, 0.7)  # 有特殊资源
            }
        }
    ),
    GameplayPattern(
        name="紧张节奏控制",
        description="在高强度区域之后提供放松区域，控制游戏节奏",
        engagement_value=0.7,
        content_types=[ContentType.AESTHETIC, ContentType.SOCIAL],
        spatial_requirements={
            "proximity": {
                "challenge": (15, 40, 0.9),  # 在挑战区域之后
                "enemy": (30, float('inf'), 0.8)  # 远离敌人
            }
        }
    ),
    GameplayPattern(
        name="互动环境",
        description="玩家可以改变或与之互动的环境元素",
        engagement_value=0.8,
        content_types=[ContentType.CHALLENGE, ContentType.MYSTERY],
        spatial_requirements={
            "biomes": ["Temple", "MagicForest", "AncientRuins", "Laboratory"],
            "proximity": {
                "path": (0, 10, 0.6),     # 靠近路径便于发现
                "puzzle": (5, 15, 0.8)    # 与解谜元素关联
            }
        }
    )
]

class ViewState(Enum):
    IDLE = auto()
    GENERATING = auto()
    MAP_READY = auto()
    EVOLVING = auto()
    EVOLUTION_DONE = auto()
    EXPORTING = auto()


class ToolTip:
    """工具提示实现"""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # 创建带边框的工具提示窗口
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(self.tooltip, borderwidth=1, relief="solid")
        frame.pack(fill="both", expand=True)
        
        label = ttk.Label(frame, text=self.text, wraplength=250, 
                          background="#ffffee", justify="left", padding=(5, 3))
        label.pack()
    
    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class MapGeneratorApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("EmoScape Studio")
        
        # 配置和初始化
        self.config_manager = ConfigManager()
        window_width = self.config_manager.get("ui.window_width", 1200)
        window_height = self.config_manager.get("ui.window_height", 800)
        self.master.geometry(f"{window_width}x{window_height}")
        
        # 状态和数据
        self.state = ViewState.IDLE  # 设置应用程序状态
        self.map_params = MapParameters() # 设置地图参数
        self.map_data = MapData(self.map_params.map_width, self.map_params.map_height, False) # 设置地图
        self.population = None # 设置生物种群
        self.evolution_done = False # 设置进化完成标志为None
        self.current_file = "./exports" # 初始化当前文件路径
        
        # 这两行初始化参数管理相关变量
        self.params = {}  # 存储参数值的字典
        self.param_vars = {}  # 存储Tkinter变量对象
        
        # 初始化params字典，从map_params获取初始值
        for attr in dir(self.map_params):
            if not attr.startswith('_') and not callable(getattr(self.map_params, attr)):
                self.params[attr] = getattr(self.map_params, attr)
        
        self.attr_controls = {}  # 属性范围控制
        
        # 初始化变量
        self.target_ratio_var = tk.DoubleVar(value=1.0)  # 默认目标比率为1.0
        
        # 管理
        self.task_manager = TaskManager() # 任务管理器的实例
        self.command_history = CommandHistory(max_history=self.config_manager.get("max_undo", 20)) # 命令历史，撤销重做等功能实例
        
        # 先设置主题基础 - 避免日志初始化后风格不一致
        self._setup_theme_basics()
        
        # 优化的UI启动
        self._setup_ui()
        
        # 确保log_text已经创建后再初始化logger
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.logger = ThreadSafeLogger(self.log_text, max_lines=500, log_file="app.log")
        else:
            # 如果log_text未创建,创建一个临时logger,避免程序崩溃
            import logging
            self.logger = logging.getLogger("temp_logger")
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            self.logger.addHandler(handler)
            # 添加简单的log方法
            self.logger.log = lambda msg, level="INFO": self.logger.info(msg)
            print("警告: 日志文本控件未创建，使用临时日志记录器")
        
        # 完成设置主题
        self._setup_theme()
        
        self.llm = LLMIntegration(self.logger)
        
        # 添加情感管理器
        self.emotion_manager = EmotionManager(logger=self.logger)
        
        # 添加动态音乐生成器
        from core.dynamic_music_generator import DynamicMusicGenerator
        self.music_generator = DynamicMusicGenerator(
            music_resource_path="data/music", 
            logger=self.logger
        )
        
        # 预览
        self.preview_window = None # 预览窗口
        #这个代码会影响预览初始化self.preview_canvas = None # 画布
        
        # 后台任务检查器
        self.master.after(100, self._check_background_tasks)
        
        # 自动保存
        if self.config_manager.get("auto_save", True):
            self._setup_autosave()
        
        # 初始日志
        self.logger.log("欢迎使用EmoScape Studio")
        self.logger.log("系统已就绪，请调整参数并点击'生成地图'按钮")
        
        # 加载参数并绑定关闭事件
        self._load_params_from_config()
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # 添加以下行初始化高级地形参数存储
        self.terrain_advanced_params = {}
        
        # 检查地理数据处理所需的依赖
        self._check_geo_dependencies()
        
        # 绑定窗口大小调整事件
        self.master.bind("<Configure>", self._on_window_resize)
        
        # 初始化按钮状态
        self._update_button_states()

    def _setup_theme_basics(self):
        """设置基本主题元素，在UI建立前调用"""
        # 获取主题名称
        theme_name = self.config_manager.get("ui.theme", "default")
        
        # 设置工具提示基本样式
        self.tooltip_font = ('', 9)
        
        # 设置工具提示颜色
        if theme_name == "dark":
            self.tooltip_bg = "#333333"
            self.tooltip_fg = "#FFFFFF"
        elif theme_name == "blue":
            self.tooltip_bg = "#E8F4FF"
            self.tooltip_fg = "#000033"
        elif theme_name == "light":
            self.tooltip_bg = "#FFFFEA"
            self.tooltip_fg = "#000000"
        else:
            self.tooltip_bg = "#FFFFEA"
            self.tooltip_fg = "#000000"
        
    def _setup_ui(self):
        try:
            # 使用更现代的布局结构
            self.master.configure(bg="#f0f0f0")  # 设置全局背景色
            
            # 创建主框架并应用间距
            main_frame = ttk.Frame(self.master, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # 设置菜单栏
            self._setup_menu()
            
            # 使用可调整的分隔窗格，添加更精细的边框样式
            self.main_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL, style="App.TPanedwindow")
            self.main_paned.pack(fill=tk.BOTH, expand=True)
            
            # 创建左侧控制面板，添加精美边框和标题
            self.control_frame = ttk.LabelFrame(self.main_paned, text="控制面板", padding="5", style="Bold.TLabelframe")
            self.main_paned.add(self.control_frame, weight=1)
            
            # 创建右侧输出面板，统一风格
            self.output_frame = ttk.LabelFrame(self.main_paned, text="输出与预览", padding="5", style="Bold.TLabelframe")
            self.main_paned.add(self.output_frame, weight=3)
            
            # 设置输出面板 - 先创建日志文本控件
            #output_notebook = ttk.Notebook(self.output_frame)
            #output_notebook.pack(fill=tk.BOTH, expand=True)
            
            # 创建日志标签页
            #log_frame = ttk.Frame(output_notebook)
            #output_notebook.add(log_frame, text="日志")
            
            # 确保日志文本控件优先创建，这是关键修复点
            #self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
            #self.log_text.pack(fill=tk.BOTH, expand=True)
            
            # 确保此时self.log_text可用
            #print("日志控件已成功创建")
            
            # 在日志控件创建完成后再设置其他UI元素
            try:
                print("正在设置控制面板...")
                self._setup_control_panel()
                
                print("正在设置输出面板...")
                self._setup_output_panel()
                
                print("正在设置嵌入式编辑器...")
                self._setup_embedded_editors()
            except Exception as e:
                print(f"UI设置过程中出错: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 加载保存的分隔位置并设置，增加默认边距
            paned_position = self.config_manager.get("ui.paned_position", 320)
            self.master.update()
            self.main_paned.sashpos(0, paned_position)
            
            # 创建更现代的状态栏
            self._setup_status_bar()
            
        except Exception as e:
            print(f"UI创建过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 确保在出错的情况下也创建log_text
            if not hasattr(self, 'log_text'):
                # 创建应急日志窗口
                emergency_frame = ttk.Frame(self.master)
                emergency_frame.pack(fill=tk.BOTH, expand=True)
                self.log_text = scrolledtext.ScrolledText(emergency_frame, wrap=tk.WORD, height=10)
                self.log_text.pack(fill=tk.BOTH, expand=True)
                self.log_text.insert(tk.END, f"UI初始化错误: {str(e)}\n")

    def _emergency_debug(self):
        """应急调试功能，在界面出现问题时使用"""
        try:
            debug_window = tk.Toplevel(self.master)
            debug_window.title("调试窗口")
            debug_window.geometry("600x400")
            
            # 创建调试文本框
            debug_text = scrolledtext.ScrolledText(debug_window, wrap=tk.WORD)
            debug_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 显示关键属性状态
            debug_text.insert(tk.END, "===== 调试信息 =====\n\n")
            debug_text.insert(tk.END, f"log_text 是否存在: {hasattr(self, 'log_text')}\n")
            debug_text.insert(tk.END, f"logger 是否存在: {hasattr(self, 'logger')}\n")
            debug_text.insert(tk.END, f"当前Python版本: {sys.version}\n")
            debug_text.insert(tk.END, f"当前Tkinter版本: {tk.TkVersion}\n")
            debug_text.insert(tk.END, f"窗口大小: {self.master.geometry()}\n")
            
            # 添加恢复按钮
            btn_frame = ttk.Frame(debug_window)
            btn_frame.pack(fill=tk.X, pady=10)
            
            ttk.Button(btn_frame, text="尝试修复日志", command=self._fix_logger).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="重新初始化UI", command=self._reinitialize_ui).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="关闭", command=debug_window.destroy).pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            print(f"调试窗口创建失败: {str(e)}")

    def _fix_logger(self):
        """尝试修复日志系统"""
        try:
            if not hasattr(self, 'log_text') or self.log_text is None:
                # 创建应急日志窗口
                emergency_frame = ttk.LabelFrame(self.output_frame, text="应急日志")
                emergency_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                self.log_text = scrolledtext.ScrolledText(emergency_frame, wrap=tk.WORD)
                self.log_text.pack(fill=tk.BOTH, expand=True)
            
            # 重新创建日志记录器
            self.logger = ThreadSafeLogger(self.log_text, max_lines=500, log_file="app.log")
            self.logger.log("日志系统已修复")
            
            messagebox.showinfo("成功", "日志系统已修复")
        except Exception as e:
            messagebox.showerror("修复失败", f"修复日志系统时出错: {str(e)}")

    def _reinitialize_ui(self):
        """重新初始化UI"""
        try:
            # 清除旧UI
            for widget in self.master.winfo_children():
                widget.destroy()
                
            # 重新设置UI
            self._setup_theme_basics()
            self._setup_ui()
            
            # 重新创建日志记录器
            if hasattr(self, 'log_text') and self.log_text is not None:
                self.logger = ThreadSafeLogger(self.log_text, max_lines=500, log_file="app.log")
                self.logger.log("UI已重新初始化")
                
            # 更新显示
            self._setup_theme()
            
            messagebox.showinfo("成功", "UI已重新初始化")
        except Exception as e:
            messagebox.showerror("重初始化失败", f"重新初始化UI时出错: {str(e)}")

    def _setup_status_bar(self):
        # 创建状态栏容器
        status_container = ttk.Frame(self.master)
        status_container.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 设置进度条组件，添加圆角和动画效果
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            status_container, 
            variable=self.progress_var, 
            mode='determinate',
            style="Rounded.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 2))
        
        # 状态栏使用更现代的设计
        status_frame = ttk.Frame(status_container, relief=tk.SUNKEN, style="StatusBar.TFrame")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))
        
        # 左侧状态文本
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var,
            padding=(5, 2),
            anchor=tk.W,
            font=("", 9)
        )
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 右侧添加版本信息
        version_label = ttk.Label(
            status_frame, 
            text=f"EmoScape v{self.config_manager.get('version', '1.0')}",
            padding=(5, 2),
            font=("", 9),
            foreground="#666666"
        )
        version_label.pack(side=tk.RIGHT)

    def _setup_menu(self):
        # 创建主菜单
        self.menu_bar = tk.Menu(self.master)
        
        # 文件菜单
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="新建", command=self._new_project, accelerator="Ctrl+N", 
                            image=self._get_icon("new"), compound=tk.LEFT)
        file_menu.add_command(label="打开...", command=self._open_project, accelerator="Ctrl+O", 
                            image=self._get_icon("open"), compound=tk.LEFT)
        file_menu.add_command(label="保存", command=self._save_project, accelerator="Ctrl+S", 
                            image=self._get_icon("save"), compound=tk.LEFT)
        file_menu.add_command(label="另存为...", command=self._save_project_as, accelerator="Ctrl+Shift+S", 
                            image=self._get_icon("save_as"), compound=tk.LEFT)
        file_menu.add_separator()
        file_menu.add_command(label="导出地图...", command=self._export_map, 
                            image=self._get_icon("export"), compound=tk.LEFT)
        file_menu.add_command(label="一键导出并部署...", command=self._export_and_deploy, 
                            image=self._get_icon("deploy"), compound=tk.LEFT)
        file_menu.add_separator()
        
        # 添加最近文件子菜单
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="最近文件", menu=self.recent_menu)
        self._update_recent_files_menu()  # 这里调用会发生错误，因为方法未定义
        
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_close, 
                            image=self._get_icon("exit"), compound=tk.LEFT)
        self.menu_bar.add_cascade(label="文件", menu=file_menu)
        
        # 编辑菜单
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        edit_menu.add_command(label="撤销", command=self._undo, accelerator="Ctrl+Z", 
                            image=self._get_icon("undo"), compound=tk.LEFT, state=tk.DISABLED)
        edit_menu.add_command(label="重做", command=self._redo, accelerator="Ctrl+Y", 
                            image=self._get_icon("redo"), compound=tk.LEFT, state=tk.DISABLED)
        self.undo_menu = edit_menu.entryconfigure(0, state="disabled")  # 存储菜单项配置引用
        self.redo_menu = edit_menu.entryconfigure(1, state="disabled")  # 存储菜单项配置引用
        
        edit_menu.add_separator()
        edit_menu.add_command(label="复制地图截图", command=self._copy_map_image, 
                            image=self._get_icon("copy"), compound=tk.LEFT)
        edit_menu.add_command(label="导出参数", command=self._export_parameters, 
                            image=self._get_icon("export_params"), compound=tk.LEFT)
        edit_menu.add_command(label="导入参数", command=self._import_parameters, 
                            image=self._get_icon("import_params"), compound=tk.LEFT)
        self.menu_bar.add_cascade(label="编辑", menu=edit_menu)
        
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_command(label="地图预览",command=lambda: preview_map(self.map_data, self.master))
        view_menu.add_command(label="3D地图预览", command=self._preview_map_3d)
        view_menu.add_separator()
        view_menu.add_command(label="关卡布局预览", command=self._preview_level_layout)  # 添加关卡预览选项
        view_menu.add_command(label="地表与地下集成视图", command=self._show_integrated_map_preview)
        view_menu.add_separator()
        view_menu.add_command(label="地下系统预览", command=self._preview_underground_layers_one)  # 新增地下系统预览选项
        view_menu.add_command(label="地下系统3D视图", command=self._show_3d_underground_view)  # 添加地下系统3D视图选项
        view_menu.add_separator()
        view_menu.add_command(label="Minecraft风格预览", command=lambda: self._preview_style_transformation("minecraft", {
            "max_height": 255,
            "sea_level": 63,
            "block_resolution": 1
        }))
        view_menu.add_separator()
        view_menu.add_command(label="情感分析", command=self._show_emotion_analysis)
        view_menu.add_command(label="情感热力图", command=self._show_emotion_heatmap)
        view_menu.add_separator()
        view_menu.add_separator()
        view_menu.add_command(label="查看游戏剧情", command=self._show_story_content)
        view_menu.add_command(label="故事情感分析", command=self._analyze_story_emotions)
        view_menu.add_command(label="情感地图可视化", command=self._show_emotion_map)
        view_menu.add_separator()
        view_menu.add_command(label="体验音乐系统", command=self._experience_music_system)
        view_menu.add_command(label="查看音乐区域地图", command=self._view_music_regions)
        view_menu.add_separator()
        view_menu.add_command(label="为故事生成音乐", command=self._generate_music_from_story)
        self.menu_bar.add_cascade(label="查看", menu=view_menu)
        
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        tools_menu.add_command(label="LLM参数建议", command=self._get_llm_suggestions)
        tools_menu.add_command(label="运行生物进化", command=self._run_evolution)
        tools_menu.add_command(label="导入地理数据", command=self._show_geo_import_dialog)
        view_menu.add_separator()
        tools_menu.add_command(label="旋律生成", command=self._show_inspiremusic_music_dialog)
        tools_menu.add_command(label="歌曲生成(快)", command=self._show_diffrhythm_music_dialog)
        tools_menu.add_command(label="歌曲生成(慢)", command=self._show_yue_music_dialog)
        tools_menu.add_command(label="conda环境工具", command=self._show_conda_tool_dialog)  # 新增选项
        tools_menu.add_command(label="自定义训练模型", command=custom_train_model)
        tools_menu.add_separator()
        tools_menu.add_command(label="3D模型生成", command=self._show_hunyuan_3d_dialog)
        tools_menu.add_command(label="3D模型编辑器", command=self._show_3d_model_editor)
        tools_menu.add_separator()
        tools_menu.add_command(label="修复日志", command=self._emergency_debug)
        
        self.menu_bar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="按钮状态调试", command=self._debug_button_states)        
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="关于", command=self._show_about)
        self.menu_bar.add_cascade(label="帮助", menu=help_menu)
        
        self.master.config(menu=self.menu_bar)
        self.master.bind("<Control-z>", lambda e: self._undo())
        self.master.bind("<Control-y>", lambda e: self._redo())
        
        # 设置菜单
        settings_menu = tk.Menu(self.menu_bar, tearoff=0)
        
        # 主题子菜单
        theme_menu = tk.Menu(settings_menu, tearoff=0)
        
        # 主题选项
        self.theme_var = tk.StringVar(value=self.config_manager.get("ui.theme", "default"))
        theme_menu.add_radiobutton(label="默认主题", variable=self.theme_var, 
                                value="default", command=self._change_theme)
        theme_menu.add_radiobutton(label="亮色主题", variable=self.theme_var, 
                                value="light", command=self._change_theme)
        theme_menu.add_radiobutton(label="暗色主题", variable=self.theme_var, 
                                value="dark", command=self._change_theme)
        theme_menu.add_radiobutton(label="蓝色主题", variable=self.theme_var, 
                                value="blue", command=self._change_theme)
        
        settings_menu.add_cascade(label="主题", menu=theme_menu)
        settings_menu.add_separator()
        
        # 自动保存选项
        # 确保自动保存变量已初始化
        if not hasattr(self, 'autosave_var'):
            self.autosave_var = tk.BooleanVar(value=self.config_manager.get("auto_save", True))
            
        settings_menu.add_checkbutton(label="自动保存", variable=self.autosave_var, 
                                    command=self._toggle_autosave)
        settings_menu.add_command(label="首选项...", command=self._show_preferences)
        
        self.menu_bar.add_cascade(label="设置", menu=settings_menu)
        
        # 应用菜单到主窗口
        self.master.config(menu=self.menu_bar)
        
        # 添加键盘快捷键
        self.master.bind("<Control-z>", lambda e: self._undo())
        self.master.bind("<Control-y>", lambda e: self._redo())
        self.master.bind("<Control-n>", lambda e: self._new_project())
        self.master.bind("<Control-o>", lambda e: self._open_project())
        self.master.bind("<Control-s>", lambda e: self._save_project())
        self.master.bind("<Control-Shift-KeyPress-S>", lambda e: self._save_project_as())

    def _change_theme(self):
        """更改应用程序主题"""
        # 获取新选择的主题
        theme_name = self.theme_var.get()
        
        # 保存到配置
        self.config_manager.set("ui.theme", theme_name)
        self.config_manager.save_config()
        
        # 应用主题
        style = ttk.Style()
        
        # 设置工具提示基本样式
        self.tooltip_font = ('', 9)
        
        # 根据主题名称应用不同的样式
        if theme_name == "dark":
            # 暗色主题
            self._apply_dark_theme(style)
            # 暗色主题的工具提示颜色
            self.tooltip_bg = "#333333"
            self.tooltip_fg = "#FFFFFF"
        elif theme_name == "light":
            # 亮色主题
            self._apply_light_theme(style)
            # 亮色主题的工具提示颜色
            self.tooltip_bg = "#FFFFEA"
            self.tooltip_fg = "#000000"
        elif theme_name == "blue":
            # 蓝色主题
            self._apply_blue_theme(style)
            # 蓝色主题的工具提示颜色
            self.tooltip_bg = "#E8F4FF"
            self.tooltip_fg = "#000033"
        else:
            # 默认主题
            self._apply_default_theme(style)
            # 默认主题的工具提示颜色
            self.tooltip_bg = "#FFFFEA"
            self.tooltip_fg = "#000000"
        
        # 提示用户主题已更改
        self.logger.log(f"已切换到{theme_name}主题")
        self._show_notification("主题已更改", f"已切换到{theme_name}主题", type_="info", timeout=2000)

    def _update_recent_files_menu(self):
        """更新最近文件菜单"""
        # 清空现有菜单项
        self.recent_menu.delete(0, tk.END)
        
        # 从配置获取最近文件列表
        recent_files = self.config_manager.get("recent_files", [])
        
        if not recent_files:
            # 如果没有最近文件，添加禁用的占位项
            self.recent_menu.add_command(label="(无最近文件)", state=tk.DISABLED)
        else:
            # 为每个最近文件添加菜单项
            for i, filepath in enumerate(recent_files):
                # 截断过长的路径，只显示最后部分
                display_path = filepath
                if len(display_path) > 50:
                    display_path = "..." + display_path[-47:]
                    
                # 添加菜单项，点击时打开相应文件
                self.recent_menu.add_command(
                    label=f"{i+1}. {display_path}",
                    command=lambda path=filepath: self._open_specific_project(path)
                )
            
            # 添加分隔线和清除选项
            self.recent_menu.add_separator()
            self.recent_menu.add_command(label="清除最近文件列表", command=self._clear_recent_files)

    def _open_specific_project(self, filepath):
        """打开指定路径的项目文件"""
        if not os.path.exists(filepath):
            messagebox.showerror("错误", f"文件不存在:\n{filepath}")
            
            # 从最近文件列表中移除不存在的文件
            recent_files = self.config_manager.get("recent_files", [])
            if filepath in recent_files:
                recent_files.remove(filepath)
                self.config_manager.set("recent_files", recent_files)
                self.config_manager.save_config()
                self._update_recent_files_menu()
            return
        
        # 调用现有的打开项目函数
        self._open_project(filepath)

    def _clear_recent_files(self):
        """清除最近文件列表"""
        if messagebox.askyesno("确认", "确定要清空最近文件列表吗？"):
            self.config_manager.set("recent_files", [])
            self.config_manager.save_config()
            self._update_recent_files_menu()

    def _on_window_resize(self, event=None):
        """处理窗口大小改变事件"""
        if event and (event.width < 10 or event.height < 10):
            return  # 忽略极小的窗口大小
        
        # 保存当前窗口大小到配置
        current_width = self.master.winfo_width()
        current_height = self.master.winfo_height()
        
        if current_width > 100 and current_height > 100:
            self.config_manager.set("ui.window_width", current_width)
            self.config_manager.set("ui.window_height", current_height)
        
        # 优化面板分割位置
        if hasattr(self, 'main_paned') and self.main_paned.winfo_exists():
            # 在小屏幕上，减小控制面板的比例
            if current_width < 800:
                self.main_paned.sashpos(0, int(current_width * 0.3))
            else:
                # 恢复用户的自定义分割
                paned_pos = self.config_manager.get("ui.paned_position", int(current_width * 0.35))
                self.main_paned.sashpos(0, paned_pos)
        
        # 更新地图预览缩放 - 修复None对象调用错误
        if hasattr(self, 'preview_canvas') and self.preview_canvas is not None:
            try:
                if self.preview_canvas.winfo_exists():
                    self._fit_preview_to_window()
            except Exception as e:
                self.logger.log(f"调整预览窗口大小出错: {str(e)}", "WARNING")
            
    def _show_notification(self, title, message, type_="info", timeout=3000):
        """显示临时通知"""
        # 创建通知窗口
        notif = tk.Toplevel(self.master)
        notif.title("")
        notif.geometry("300x80+{}+{}".format(
            self.master.winfo_rootx() + self.master.winfo_width() - 320,
            self.master.winfo_rooty() + 40
        ))
        notif.attributes("-topmost", True)
        notif.overrideredirect(True)
        
        # 设置样式
        if type_ == "error":
            bg_color = "#FFD2D2"
            fg_color = "#8B0000"
            icon = self._get_icon("error")
        elif type_ == "warning":
            bg_color = "#FFEDCC"
            fg_color = "#805700"
            icon = self._get_icon("warning")
        elif type_ == "success":
            bg_color = "#CCFFCC"
            fg_color = "#006400"
            icon = self._get_icon("success")
        else:  # info
            bg_color = "#CCE5FF"
            fg_color = "#00008B"
            icon = self._get_icon("info")
        
        # 背景框架
        frame = ttk.Frame(notif, style="Notification.TFrame")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 通知内容
        content_frame = ttk.Frame(frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        # 图标
        if icon:
            icon_label = ttk.Label(content_frame, image=icon)
            icon_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # 文本
        text_frame = ttk.Frame(content_frame)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(text_frame, text=title, font=("", 10, "bold"), foreground=fg_color).pack(anchor=tk.W)
        ttk.Label(text_frame, text=message, wraplength=250, foreground=fg_color).pack(anchor=tk.W)
        
        # 关闭按钮
        close_button = ttk.Label(content_frame, text="×", font=("", 12), cursor="hand2")
        close_button.pack(side=tk.RIGHT, padx=(10, 0))
        close_button.bind("<Button-1>", lambda e: notif.destroy())
        
        # 设置自动关闭
        if timeout > 0:
            notif.after(timeout, notif.destroy)
        
        # 为通知添加渐变消失动画
        def fade_away(alpha=1.0):
            if alpha <= 0:
                notif.destroy()
                return
            notif.attributes("-alpha", alpha)
            notif.after(50, lambda: fade_away(alpha - 0.1))
        
        # 设置淡出动画
        if timeout > 0:
            notif.after(timeout - 500, lambda: fade_away())
        
        return notif

    def _setup_control_panel(self):
        """设置控制面板"""
        # 创建容器框架
        control_container = ttk.Frame(self.control_frame, padding="3")
        control_container.pack(fill=tk.BOTH, expand=True)
        
        # 创建标签式Notebook并应用现代风格
        style = ttk.Style()
        style.configure("ControlPanel.TNotebook", tabposition='n', padding=[3, 3, 3, 3])
        style.configure("ControlPanel.TNotebook.Tab", padding=[8, 3], font=('', 9))
        
        self.param_notebook = ttk.Notebook(control_container, style="ControlPanel.TNotebook")
        self.param_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建各标签页
        self.core_tab = ttk.Frame(self.param_notebook, padding="8")
        self.terrain_tab = ttk.Frame(self.param_notebook, padding="8")
        self.biome_tab = ttk.Frame(self.param_notebook, padding="8")
        self.advanced_tab = ttk.Frame(self.param_notebook, padding="8")
        self.geo_tab = ttk.Frame(self.param_notebook, padding="8")
        self.level_tab = ttk.Frame(self.param_notebook, padding="8")
        self.underground_tab = ttk.Frame(self.param_notebook, padding="8")  # 新增地下系统标签页
        
        # 添加标签页到Notebook，使用图标标签组合
        self.param_notebook.add(self.core_tab, text="核心参数", image=self._get_icon("core"), compound=tk.LEFT)
        self.param_notebook.add(self.terrain_tab, text="地形设置", image=self._get_icon("terrain"), compound=tk.LEFT)
        self.param_notebook.add(self.biome_tab, text="生物群系", image=self._get_icon("biome"), compound=tk.LEFT)
        self.param_notebook.add(self.advanced_tab, text="高级选项", image=self._get_icon("advanced"), compound=tk.LEFT)
        self.param_notebook.add(self.geo_tab, text="地理数据", image=self._get_icon("geo"), compound=tk.LEFT)
        self.param_notebook.add(self.level_tab, text="关卡生成", image=self._get_icon("level"), compound=tk.LEFT)
        self.param_notebook.add(self.underground_tab, text="地下系统", image=self._get_icon("cave"), compound=tk.LEFT)  # 新增地下系统标签页
        
        # 设置各标签页内容
        self._setup_core_tab()
        self._setup_terrain_tab()
        self._setup_biome_tab()
        self._setup_advanced_tab()
        self._setup_geo_tab()
        self._setup_level_tab()
        self._setup_underground_tab()  # 新增地下系统标签页设置
        
        # 底部操作按钮
        self._setup_control_buttons()

    def _debug_button_states(self):
        """调试按钮状态"""
        has_map_data = self.map_data is not None and self.map_data.is_valid()
        
        # 创建调试窗口
        debug_window = tk.Toplevel(self.master)
        debug_window.title("按钮状态调试")
        debug_window.geometry("400x300")
        
        # 创建文本框显示状态
        debug_text = scrolledtext.ScrolledText(debug_window, wrap=tk.WORD)
        debug_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 添加状态信息
        debug_text.insert(tk.END, f"地图数据存在: {self.map_data is not None}\n")
        if self.map_data is not None:
            debug_text.insert(tk.END, f"地图数据有效: {self.map_data.is_valid()}\n")
            debug_text.insert(tk.END, f"地图尺寸: {self.map_data.width}x{self.map_data.height}\n")
            for layer in ['height', 'biome', 'rivers', 'roads']:
                has_layer = self.map_data.get_layer(layer) is not None
                debug_text.insert(tk.END, f"地图层 '{layer}' 存在: {has_layer}\n")
        
        debug_text.insert(tk.END, "\n按钮状态:\n")
        if hasattr(self, 'underground_generate_btn'):
            try:
                state = str(self.underground_generate_btn.cget('state'))
                debug_text.insert(tk.END, f"地下系统生成按钮状态: {state}\n")
            except Exception as e:
                debug_text.insert(tk.END, f"获取按钮状态出错: {str(e)}\n")
        else:
            debug_text.insert(tk.END, "地下系统生成按钮未创建\n")
        
        # 添加刷新按钮
        ttk.Button(debug_window, text="强制启用按钮", 
                command=lambda: self._force_enable_buttons()).pack(pady=5)
        ttk.Button(debug_window, text="刷新状态", 
                command=lambda: self._update_button_states()).pack(pady=5)
        ttk.Button(debug_window, text="关闭", 
                command=debug_window.destroy).pack(pady=5)

    def _force_enable_buttons(self):
        """强制启用所有按钮"""
        if hasattr(self, 'underground_generate_btn') and self.underground_generate_btn is not None:
            try:
                self.underground_generate_btn.config(state=tk.NORMAL)
                self.logger.log("已强制启用地下系统生成按钮")
            except Exception as e:
                self.logger.log(f"启用按钮失败: {str(e)}", "ERROR")

    def _generate_underground_system(self):
        """生成地下系统并集成到地图数据中"""
        # 检查地图数据是否已初始化
        if not self.map_data or not self.map_data.is_valid():
            messagebox.showerror("错误", "请先生成基础地图")
            return
        
        # 检查是否启用地下系统
        if not self.param_vars["enable_underground"].get():
            if not messagebox.askyesno("提示", "地下系统未启用，是否继续生成？"):
                return
        
        # 收集地下系统的参数
        underground_config = {
            "enable_underground": self.param_vars["enable_underground"].get(),
            "underground_depth": self.param_vars["underground_depth"].get(),
            "underground_water_prevalence": self.param_vars["underground_water_prevalence"].get(),
            "underground_structure_density": self.param_vars["underground_structure_density"].get(),
            "common_minerals_ratio": self.param_vars["common_minerals_ratio"].get(),
            "rare_minerals_ratio": self.param_vars["rare_minerals_ratio"].get(),
            "cave_network_density": self.param_vars["cave_network_density"].get(),
            "special_structures_frequency": self.param_vars["special_structures_frequency"].get(),
            "underground_danger_level": self.param_vars["underground_danger_level"].get(),
            # 使用与地图相同的种子以保持一致性
            "seed": self.map_params.seed
        }
        
        # 更新状态
        self.status_var.set("生成地下系统中...")
        self.progress_var.set(0)
        self.master.update_idletasks()
        
        # 禁用按钮，防止重复操作
        self._update_button_states(is_busy=True)
        
        try:
            # 检查导入模块是否正确
            import importlib.util
            module_path = os.path.join(os.path.dirname(__file__), "core", "generation", "generate_undergroud.py")
            if not os.path.exists(module_path):
                self.logger.log(f"找不到地下系统生成模块: {module_path}", "ERROR")
                messagebox.showerror("文件错误", f"找不到地下系统生成模块文件:\n{module_path}")
                self._update_button_states(is_busy=False)
                return
                
            # 导入地下系统生成函数
            try:
                from core.generation.generate_undergroud import integrate_underground_to_map_data
                self.logger.log("成功导入地下系统生成模块")
            except ImportError as e:
                self.logger.log(f"导入地下系统生成模块失败: {str(e)}", "ERROR")
                messagebox.showerror("导入错误", f"无法导入地下系统生成模块: {str(e)}")
                self._update_button_states(is_busy=False)
                return
            
            # 直接在主线程中执行生成，不使用任务管理器
            try:
                # 添加进度更新回调
                def update_progress(progress, message=None):
                    if hasattr(self, 'progress_var'):
                        self.progress_var.set(progress * 100)
                    if message and hasattr(self, 'status_var'):
                        self.status_var.set(message)
                    self.master.update_idletasks()
                
                self.logger.log("开始生成地下系统...")
                # 直接调用生成函数
                self.map_data = integrate_underground_to_map_data(
                    self.map_data, 
                    underground_config,
                    self.logger,
                    progress_callback=update_progress  # 传递进度回调
                )
                self.logger.log("地下系统生成完成")
                
                # 成功完成，直接调用完成回调
                self._on_underground_generation_complete((True, "地下系统生成完成"), None)
                
            except Exception as e:
                self.logger.log(f"生成地下系统出错: {str(e)}", "ERROR")
                # 记录完整的错误堆栈信息
                import traceback
                self.logger.log(traceback.format_exc(), "ERROR")
                
                # 失败，调用完成回调
                self._on_underground_generation_complete((False, f"生成地下系统出错: {str(e)}"), None)
                
        except Exception as e:
            self.logger.log(f"启动地下系统生成过程时发生错误: {str(e)}", "ERROR")
            # 记录完整的错误堆栈信息
            import traceback
            self.logger.log(traceback.format_exc(), "ERROR")
            messagebox.showerror("错误", f"启动地下系统生成过程时发生错误: {str(e)}")
            self.status_var.set("就绪")
            self._update_button_states(is_busy=False)

    def _on_underground_generation_complete(self, result, message):
        """地下系统生成完成后的回调"""
        success, msg = result
        
        if success:
            self.status_var.set("地下系统生成完成")
            self.logger.log(msg)
            self._show_notification("成功", "地下系统生成完成", type_="success")
            
            # 添加到撤销/重做堆栈
            # 这里可以添加一个Command来支持撤销操作
            
            # 更新地图预览
            if hasattr(self, 'preview_canvas') and self.preview_canvas:
                self._update_map_preview()
        else:
            self.status_var.set("生成失败")
            self.logger.log(msg, "ERROR")
            messagebox.showerror("生成失败", msg)
        
        # 重新启用按钮
        self._update_button_states(is_busy=False)
        
    def _setup_underground_tab(self):
        """设置地下系统标签页"""
        # 创建滚动框架
        underground_frame = self._create_scrollable_frame(self.underground_tab)
        
        # 地下系统启用设置
        enable_frame = ttk.LabelFrame(underground_frame, text="地下系统设置", padding=(5, 5, 5, 5))
        enable_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 启用地下系统复选框 - 使用pack而不是grid
        enable_checkbox_frame = ttk.Frame(enable_frame)
        enable_checkbox_frame.pack(fill=tk.X, padx=5, pady=5)
        
        enable_var = tk.BooleanVar(value=self.params.get("enable_underground", False))
        self.param_vars["enable_underground"] = enable_var
        enable_checkbox = ttk.Checkbutton(
            enable_checkbox_frame, 
            text="启用地下系统", 
            variable=enable_var
        )
        enable_checkbox.pack(side=tk.LEFT, padx=5, pady=2)
        
        # 添加提示信息
        info_text = "启用后将生成多层地下结构，包括洞穴、矿物资源和特殊结构"
        info_label = ttk.Label(enable_frame, text=info_text, wraplength=280, foreground="#666666", font=('', 8))
        info_label.pack(fill=tk.X, padx=25, pady=(0, 5))
        
        # 地下层数设置 - 使用pack布局
        depth_frame = ttk.Frame(enable_frame)
        depth_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(depth_frame, text="地下层数:").pack(side=tk.LEFT, padx=5)
        
        depth_var = tk.IntVar(value=self.params.get("underground_depth", 3))
        self.param_vars["underground_depth"] = depth_var
        
        depth_spinner = ttk.Spinbox(
            depth_frame, 
            from_=1, 
            to=10, 
            textvariable=depth_var, 
            width=5
        )
        depth_spinner.pack(side=tk.LEFT, padx=5)
        
        # 地下系统参数设置
        params_frame = ttk.LabelFrame(underground_frame, text="地下系统参数", padding=(5, 5, 5, 5))
        params_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # 地下水系统丰富度 - 使用pack布局
        water_frame = ttk.Frame(params_frame)
        water_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(water_frame, text="地下水系统丰富度:").pack(side=tk.LEFT, padx=5)
        
        water_var = tk.DoubleVar(value=self.params.get("underground_water_prevalence", 0.5))
        self.param_vars["underground_water_prevalence"] = water_var
        
        water_slider = ttk.Scale(
            water_frame,
            from_=0.0,
            to=1.0,
            variable=water_var,
            orient=tk.HORIZONTAL
        )
        water_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(water_frame, textvariable=water_var, width=4).pack(side=tk.LEFT)
        
        # 地下结构密度
        structure_frame = ttk.Frame(params_frame)
        structure_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(structure_frame, text="地下结构密度:").pack(side=tk.LEFT, padx=5)
        
        structure_var = tk.DoubleVar(value=self.params.get("underground_structure_density", 0.5))
        self.param_vars["underground_structure_density"] = structure_var
        
        structure_slider = ttk.Scale(
            structure_frame,
            from_=0.0,
            to=1.0,
            variable=structure_var,
            orient=tk.HORIZONTAL
        )
        structure_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(structure_frame, textvariable=structure_var, width=4).pack(side=tk.LEFT)
        
        # 矿物资源设置
        mineral_frame = ttk.LabelFrame(underground_frame, text="矿物资源设置", padding=(5, 5, 5, 5))
        mineral_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # 常见矿物比例
        common_frame = ttk.Frame(mineral_frame)
        common_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(common_frame, text="常见矿物比例:").pack(side=tk.LEFT, padx=5)
        
        common_var = tk.DoubleVar(value=self.params.get("common_minerals_ratio", 0.7))
        self.param_vars["common_minerals_ratio"] = common_var
        
        common_slider = ttk.Scale(
            common_frame,
            from_=0.0,
            to=1.0,
            variable=common_var,
            orient=tk.HORIZONTAL
        )
        common_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(common_frame, textvariable=common_var, width=4).pack(side=tk.LEFT)
        
        # 稀有矿物比例
        rare_frame = ttk.Frame(mineral_frame)
        rare_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(rare_frame, text="稀有矿物比例:").pack(side=tk.LEFT, padx=5)
        
        rare_var = tk.DoubleVar(value=self.params.get("rare_minerals_ratio", 0.3))
        self.param_vars["rare_minerals_ratio"] = rare_var
        
        rare_slider = ttk.Scale(
            rare_frame,
            from_=0.0,
            to=1.0,
            variable=rare_var,
            orient=tk.HORIZONTAL
        )
        rare_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(rare_frame, textvariable=rare_var, width=4).pack(side=tk.LEFT)
        
        # 地下结构设置
        structure_type_frame = ttk.LabelFrame(underground_frame, text="地下结构设置", padding=(5, 5, 5, 5))
        structure_type_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # 洞穴网络密集度
        cave_frame = ttk.Frame(structure_type_frame)
        cave_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(cave_frame, text="洞穴网络密集度:").pack(side=tk.LEFT, padx=5)
        
        cave_var = tk.DoubleVar(value=self.params.get("cave_network_density", 0.5))
        self.param_vars["cave_network_density"] = cave_var
        
        cave_slider = ttk.Scale(
            cave_frame,
            from_=0.0,
            to=1.0,
            variable=cave_var,
            orient=tk.HORIZONTAL
        )
        cave_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(cave_frame, textvariable=cave_var, width=4).pack(side=tk.LEFT)
        
        # 特殊结构频率
        special_frame = ttk.Frame(structure_type_frame)
        special_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(special_frame, text="特殊结构频率:").pack(side=tk.LEFT, padx=5)
        
        special_var = tk.DoubleVar(value=self.params.get("special_structures_frequency", 0.5))
        self.param_vars["special_structures_frequency"] = special_var
        
        special_slider = ttk.Scale(
            special_frame,
            from_=0.0,
            to=1.0,
            variable=special_var,
            orient=tk.HORIZONTAL
        )
        special_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(special_frame, textvariable=special_var, width=4).pack(side=tk.LEFT)
        
        # 地下危险等级
        danger_frame = ttk.Frame(structure_type_frame)
        danger_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(danger_frame, text="地下危险等级:").pack(side=tk.LEFT, padx=5)
        
        danger_var = tk.DoubleVar(value=self.params.get("underground_danger_level", 0.5))
        self.param_vars["underground_danger_level"] = danger_var
        
        danger_slider = ttk.Scale(
            danger_frame,
            from_=0.0,
            to=1.0,
            variable=danger_var,
            orient=tk.HORIZONTAL
        )
        danger_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(danger_frame, textvariable=danger_var, width=4).pack(side=tk.LEFT)
        
        # 地下系统操作按钮框架
        buttons_frame = ttk.Frame(underground_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=10)

        # 添加生成地下系统按钮，并保存引用
        self.underground_generate_btn = ttk.Button(
            buttons_frame, 
            text="生成地下系统", 
            command=self._generate_underground_system,
            style="Accent.TButton"  # 使用醒目的样式
        )
        self.underground_generate_btn.pack(side=tk.LEFT, padx=5)

        # 预览按钮
        preview_button = ttk.Button(
            buttons_frame, 
            text="预览地下系统", 
            command=self._preview_underground_layers_two
        )
        preview_button.pack(side=tk.LEFT, padx=5)

        # 添加3D视图按钮
        view_3d_button = ttk.Button(
            buttons_frame, 
            text="3D地下视图", 
            command=self._show_3d_underground_view
        )
        view_3d_button.pack(side=tk.LEFT, padx=5)

        export_button = ttk.Button(
            buttons_frame, 
            text="导出地下数据", 
            command=self._export_underground_data
        )
        export_button.pack(side=tk.LEFT, padx=5)

    def _preview_underground_layers_one(self):
        """预览地下层系统"""
        if not hasattr(self.map_data, 'underground_layers') or not self.map_data.underground_layers:
            messagebox.showwarning("提示", "地下系统尚未生成，请先生成地下系统")
            return
        
        # 创建一个新窗口来显示地下层
        preview_window = tk.Toplevel(self.master)
        preview_window.title("地下系统预览")
        preview_window.geometry("1200x700")
        preview_window.transient(self.master)
        preview_window.grab_set()
        
        # 创建带标签页的显示区域，每个地下层一个标签页
        notebook = ttk.Notebook(preview_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 遍历所有地下层
        depth = len(self.map_data.underground_layers)
        for i in range(depth):
            layer_name = f"underground_{i}"
            if layer_name in self.map_data.underground_layers:
                # 创建标签页
                tab = ttk.Frame(notebook)
                notebook.add(tab, text=f"地下第{i+1}层")
                
                # 创建matplotlib图形
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                
                # 使用核心模块的可视化函数
                from core.generation.generate_undergroud import visualize_underground_layer
                
                # 创建一个Figure对象
                fig = plt.Figure(figsize=(12, 5), dpi=100)
                
                # 将Figure添加到tkinter窗口
                canvas = FigureCanvasTkAgg(fig, tab)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # 创建Axes
                axes = []
                for j in range(3):
                    axes.append(fig.add_subplot(1, 3, j+1))
                
                # 可视化地下层内容 - 重要：这里传入完整的map_data参数
                underground_layer = self.map_data.underground_layers[layer_name]
                
                # 可视化高度图
                height_map = underground_layer["height"]
                im1 = axes[0].imshow(height_map, cmap='terrain')
                axes[0].set_title(f"层{i+1} - 地形高度")
                fig.colorbar(im1, ax=axes[0])
                
                # 可视化内容类型
                from core.generation.generate_undergroud import UndergroundContentType
                content_map = underground_layer["content"]
                im2 = axes[1].imshow(content_map, cmap='tab20')
                axes[1].set_title(f"层{i+1} - 内容类型")
                fig.colorbar(im2, ax=axes[1])
                
                # 可视化矿物分布 - 关键是从map_data而不是underground_layer获取
                from core.generation.generate_undergroud import MineralType
                if hasattr(self.map_data, 'mineral_layers') and layer_name in self.map_data.mineral_layers:
                    mineral_map = self.map_data.mineral_layers[layer_name]
                    im3 = axes[2].imshow(mineral_map, cmap='rainbow')
                    axes[2].set_title(f"层{i+1} - 矿物分布")
                    fig.colorbar(im3, ax=axes[2])
                else:
                    axes[2].text(0.5, 0.5, "矿物数据不可用", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=axes[2].transAxes)
                
                fig.tight_layout()
        
        # 添加一个统计信息标签页
        stats_tab = ttk.Frame(notebook)
        notebook.add(stats_tab, text="统计信息")
        
        # 从核心模块导入统计函数
        from core.generation.generate_undergroud import get_underground_statistics
        
        # 获取统计信息
        stats = get_underground_statistics(self.map_data)
        
        # 在统计标签页中显示信息
        stats_text = scrolledtext.ScrolledText(stats_tab, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 构建统计信息文本
        stats_info = f"地下系统统计信息:\n\n"
        stats_info += f"地下层数量: {stats['layers']}\n"
        stats_info += f"特殊结构数量: {stats['structures']}\n\n"
        
        # 添加水系统信息
        water_features = stats['water_features']
        stats_info += f"水系统特征:\n"
        stats_info += f"  - 地下河流: {water_features['rivers']}\n"
        stats_info += f"  - 地下湖泊: {water_features['lakes']}\n"
        stats_info += f"  - 渗水区域: {water_features['seepage_areas']}\n\n"
        
        # 添加内容类型统计
        for layer_name, content_counts in stats['content_types'].items():
            layer_idx = int(layer_name.split('_')[-1])
            stats_info += f"第{layer_idx + 1}层内容类型统计:\n"
            
            for content_type, count in content_counts.items():
                percentage = count / (self.map_data.width * self.map_data.height) * 100
                stats_info += f"  - {content_type}: {count} ({percentage:.2f}%)\n"
            stats_info += "\n"
        
        # 添加矿物统计
        for layer_name, mineral_counts in stats['minerals'].items():
            layer_idx = int(layer_name.split('_')[-1])
            stats_info += f"第{layer_idx + 1}层矿物类型统计:\n"
            
            for mineral_type, count in mineral_counts.items():
                percentage = count / (self.map_data.width * self.map_data.height) * 100
                stats_info += f"  - {mineral_type}: {count} ({percentage:.2f}%)\n"
            stats_info += "\n"
        
        stats_text.insert(tk.END, stats_info)
        stats_text.config(state=tk.DISABLED)
        
        # 添加3D视图按钮
        button_frame = ttk.Frame(preview_window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(
            button_frame, 
            text="查看3D地下视图", 
            command=lambda: self._show_3d_underground_view()
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="导出预览图像", 
            command=lambda: self._export_underground_preview()
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="关闭", 
            command=preview_window.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        self.logger.log("已显示地下系统预览")

    def _show_integrated_map_preview(self):
            """显示地表与地下系统的集成预览"""
            # 检查地图数据是否有效
            if not self.map_data or not self.map_data.is_valid():
                messagebox.showerror("错误", "请先生成基础地图")
                return
                
            # 检查是否已生成地下系统
            if not hasattr(self.map_data, 'underground_layers') or not self.map_data.underground_layers:
                if messagebox.askyesno("提示", "地下系统尚未生成，是否先生成地下系统？"):
                    self._generate_underground_system()
                    return
                else:
                    return
            
            # 创建一个新窗口
            integrated_window = tk.Toplevel(self.master)
            integrated_window.title("地形与地下系统集成预览")
            integrated_window.geometry("1200x800")
            integrated_window.transient(self.master)
            
            # 主分隔窗格 - 上部放控制面板，下部放预览内容
            main_paned = ttk.PanedWindow(integrated_window, orient=tk.VERTICAL)
            main_paned.pack(fill=tk.BOTH, expand=True)
            
            # 控制面板
            control_frame = ttk.LabelFrame(main_paned, text="预览控制")
            main_paned.add(control_frame, weight=1)
            
            # 控制选项 - 使用Grid布局
            controls = ttk.Frame(control_frame)
            controls.pack(fill=tk.X, padx=10, pady=5)
            
            # 视图选择
            ttk.Label(controls, text="视图模式:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            view_mode_var = tk.StringVar(value="横截面")
            # 添加"3D所有层视图"选项
            view_mode = ttk.Combobox(controls, textvariable=view_mode_var, 
                                values=["横截面", "垂直截面", "3D视图", "所有层预览", "3D所有层视图"], state="readonly", width=10)
            view_mode.grid(row=0, column=1, padx=5, pady=5)
            view_mode.bind("<<ComboboxSelected>>", lambda e: self._update_integrated_view(preview_frame, view_mode_var.get(), 
                                                                                        layer_var.get(), data_var.get(), 
                                                                                        cross_section_var.get()))
            
            # 层选择
            ttk.Label(controls, text="显示层:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
            
            # 获取可用的地下层
            layers = ["地表"] + [f"地下第{i+1}层" for i in range(len(self.map_data.underground_layers))]
            layer_var = tk.StringVar(value="地表")
            layer_combo = ttk.Combobox(controls, textvariable=layer_var, values=layers, state="readonly", width=10)
            layer_combo.grid(row=0, column=3, padx=5, pady=5)
            layer_combo.bind("<<ComboboxSelected>>", lambda e: self._update_integrated_view(preview_frame, view_mode_var.get(), 
                                                                                        layer_var.get(), data_var.get(), 
                                                                                        cross_section_var.get()))
            
            # 数据类型选择
            ttk.Label(controls, text="数据类型:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
            data_var = tk.StringVar(value="高度/地形")
            data_combo = ttk.Combobox(controls, textvariable=data_var, 
                                    values=["高度/地形", "矿物分布", "内容类型", "生物群系"], state="readonly", width=10)
            data_combo.grid(row=0, column=5, padx=5, pady=5)
            data_combo.bind("<<ComboboxSelected>>", lambda e: self._update_integrated_view(preview_frame, view_mode_var.get(), 
                                                                                        layer_var.get(), data_var.get(), 
                                                                                        cross_section_var.get()))
            
            # 截面位置滑块（横/纵截面模式使用）
            ttk.Label(controls, text="截面位置:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            cross_section_var = tk.IntVar(value=self.map_data.height // 2)
            cross_section_slider = ttk.Scale(
                controls, 
                from_=0, 
                to=self.map_data.height-1, 
                variable=cross_section_var, 
                orient=tk.HORIZONTAL,
                length=200
            )
            cross_section_slider.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky=tk.EW)
            cross_section_slider.bind("<ButtonRelease-1>", lambda e: self._update_integrated_view(preview_frame, view_mode_var.get(), 
                                                                                            layer_var.get(), data_var.get(), 
                                                                                            cross_section_var.get()))
            
            ttk.Label(controls, textvariable=cross_section_var).grid(row=1, column=4, padx=5, pady=5, sticky=tk.W)
            
            # 刷新按钮
            refresh_btn = ttk.Button(
                controls, 
                text="刷新视图", 
                command=lambda: self._update_integrated_view(preview_frame, view_mode_var.get(), layer_var.get(), data_var.get(), cross_section_var.get())
            )
            refresh_btn.grid(row=1, column=5, padx=5, pady=5)
            
            # 预览框架
            preview_frame = ttk.Frame(main_paned)
            main_paned.add(preview_frame, weight=5)
            
            # 设置分隔条位置
            main_paned.pack(fill=tk.BOTH, expand=True)
            # 先让控件显示出来，然后更新界面以获取正确的尺寸
            main_paned.update()
            # 设置分隔条位置，第一次分隔条的位置设为100
            main_paned.sashpos(0, 100)
            
            # 初始化视图
            self._update_integrated_view(preview_frame, view_mode_var.get(), layer_var.get(), data_var.get(), cross_section_var.get())
            
            # 添加导出按钮
            button_frame = ttk.Frame(integrated_window)
            button_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Button(
                button_frame, 
                text="导出当前视图", 
                command=lambda: self._export_integrated_view(preview_frame, view_mode_var.get(), layer_var.get(), data_var.get(), cross_section_var.get())
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame, 
                text="切换至3D视图", 
                command=lambda: view_mode_var.set("3D视图") or self._update_integrated_view(preview_frame, "3D视图", layer_var.get(), data_var.get(), cross_section_var.get())
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame, 
                text="切换至所有层预览", 
                command=lambda: view_mode_var.set("所有层预览") or self._update_integrated_view(preview_frame, "所有层预览", layer_var.get(), data_var.get(), cross_section_var.get())
            ).pack(side=tk.LEFT, padx=5)
            
            # 添加切换到3D所有层视图的按钮
            ttk.Button(
                button_frame, 
                text="3D查看所有层", 
                command=lambda: view_mode_var.set("3D所有层视图") or self._update_integrated_view(preview_frame, "3D所有层视图", layer_var.get(), data_var.get(), cross_section_var.get())
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame, 
                text="关闭", 
                command=integrated_window.destroy
            ).pack(side=tk.RIGHT, padx=5)
            
            self.logger.log("已显示地形与地下系统集成预览")

    def _update_integrated_view(self, frame, view_mode, layer_name, data_type, cross_section_pos):
            """更新集成视图预览"""
            # 清除当前frame中的所有控件
            for widget in frame.winfo_children():
                widget.destroy()
            
            # 导入必要的绘图库
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # 准备数据
            surface_height = self.map_data.get_layer("height")
            surface_biome = self.map_data.get_layer("biome") if "biome" in self.map_data.layers else None
            
            # 获取地下层索引
            underground_index = -1
            if layer_name != "地表":
                underground_index = int(layer_name.replace("地下第", "").replace("层", "")) - 1
            
            # 根据视图模式创建不同的可视化
            if view_mode == "所有层预览":
                # 创建一个Figure用于显示所有层
                fig = Figure(figsize=(12, 8), dpi=100)
                
                # 获取地下层数量
                underground_layers_count = len(self.map_data.underground_layers)
                
                # 设置总行数(地表 + 所有地下层)
                total_layers = underground_layers_count + 1
                
                # 计算图表布局，尽量保持合理的宽高比
                cols = min(3, total_layers)  # 最多3列
                rows = (total_layers + cols - 1) // cols  # 向上取整
                
                # 创建子图
                axes = []
                
                # 地表层显示
                ax_surface = fig.add_subplot(rows, cols, 1)
                
                # 根据数据类型选择地表数据
                if data_type == "高度/地形":
                    surface_data = surface_height
                    cmap = 'terrain'
                    title = "地表 - 高度图"
                elif data_type == "生物群系" and surface_biome is not None:
                    surface_data = surface_biome
                    cmap = 'tab20'
                    title = "地表 - 生物群系"
                else:
                    surface_data = surface_height
                    cmap = 'terrain'
                    title = "地表 - 高度图"
                
                # 显示地表层
                im_surface = ax_surface.imshow(surface_data, cmap=cmap)
                ax_surface.set_title(title)
                fig.colorbar(im_surface, ax=ax_surface)
                
                # 显示所有地下层
                for i in range(underground_layers_count):
                    underground_layer_name = f"underground_{i}"
                    if underground_layer_name in self.map_data.underground_layers:
                        ax = fig.add_subplot(rows, cols, i + 2)  # +2 因为地表已经占了第一个位置
                        
                        # 获取地下层数据
                        layer = self.map_data.underground_layers[underground_layer_name]
                        
                        if data_type == "高度/地形":
                            layer_data = layer["height"]
                            cmap = 'terrain'
                            title = f"地下第{i+1}层 - 高度图"
                        elif data_type == "内容类型":
                            layer_data = layer["content"]
                            cmap = 'tab20'
                            title = f"地下第{i+1}层 - 内容类型"
                        elif data_type == "矿物分布" and hasattr(self.map_data, 'mineral_layers') and underground_layer_name in self.map_data.mineral_layers:
                            layer_data = self.map_data.mineral_layers[underground_layer_name]
                            cmap = 'rainbow'
                            title = f"地下第{i+1}层 - 矿物分布"
                        else:
                            # 默认显示高度
                            layer_data = layer["height"]
                            cmap = 'terrain'
                            title = f"地下第{i+1}层 - 高度图"
                        
                        # 显示地下层
                        im = ax.imshow(layer_data, cmap=cmap)
                        ax.set_title(title)
                        fig.colorbar(im, ax=ax)
                
                # 调整子图布局
                fig.tight_layout()
                
                # 添加到tkinter窗口
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # 添加导航工具栏
                from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
                toolbar = NavigationToolbar2Tk(canvas, frame)
                toolbar.update()
                toolbar.pack(side=tk.BOTTOM, fill=tk.X)

            elif view_mode == "3D所有层视图":
                # 创建3D视图
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = Figure(figsize=(12, 8), dpi=100)
                ax = fig.add_subplot(111, projection='3d')
                
                # 设置图表标题
                ax.set_title("地表与地下层3D集成视图")
                
                # 获取地表数据并降采样以提高性能
                sample_rate = 4  # 每4个点采样一次
                x = np.arange(0, self.map_data.width, sample_rate)
                y = np.arange(0, self.map_data.height, sample_rate)
                X, Y = np.meshgrid(x, y)
                
                # 地表高度数据
                Z_surface = surface_height[::sample_rate, ::sample_rate]
                
                # 绘制地表
                surf = ax.plot_surface(X, Y, Z_surface, cmap='terrain', alpha=0.7, 
                                    linewidth=0, antialiased=True, label="地表")
                
                # 获取地下层数量
                underground_layers_count = len(self.map_data.underground_layers)
                
                # 计算每层之间的偏移量
                mean_height = np.mean(Z_surface)
                offset_per_layer = mean_height * 0.3  # 每层偏移地表平均高度的30%
                
                # 绘制每个地下层
                for i in range(underground_layers_count):
                    underground_layer_name = f"underground_{i}"
                    if underground_layer_name in self.map_data.underground_layers:
                        # 获取地下层数据
                        layer = self.map_data.underground_layers[underground_layer_name]
                        
                        # 根据数据类型选择显示内容
                        if data_type == "高度/地形":
                            Z_underground = layer["height"][::sample_rate, ::sample_rate]
                            cmap_name = 'terrain'
                        elif data_type == "内容类型":
                            Z_underground = layer["content"][::sample_rate, ::sample_rate]
                            cmap_name = 'tab20'
                        elif data_type == "矿物分布" and hasattr(self.map_data, 'mineral_layers') and underground_layer_name in self.map_data.mineral_layers:
                            Z_underground = self.map_data.mineral_layers[underground_layer_name][::sample_rate, ::sample_rate]
                            cmap_name = 'rainbow'
                        else:
                            # 默认使用高度数据
                            Z_underground = layer["height"][::sample_rate, ::sample_rate]
                            cmap_name = 'terrain'
                        
                        # 计算当前层的垂直偏移量
                        layer_offset = (i + 1) * offset_per_layer
                        
                        # 设置透明度，较深的层透明度更低
                        alpha = max(0.3, 0.8 - i * 0.1)
                        
                        # 绘制地下层
                        if data_type == "矿物分布" and hasattr(self.map_data, 'mineral_layers') and underground_layer_name in self.map_data.mineral_layers:
                            # 为矿物分布创建更加吸引人的颜色映射
                            from matplotlib import cm
                            # 规范化数据到0-1范围
                            norm_data = (Z_underground - np.min(Z_underground)) / (np.max(Z_underground) - np.min(Z_underground) + 1e-10)
                            colors = cm.get_cmap('rainbow')(norm_data)
                            
                            # 绘制带有矿物颜色的地下层
                            surf_underground = ax.plot_surface(
                                X, Y, Z_underground - layer_offset, 
                                facecolors=colors,
                                alpha=alpha,
                                linewidth=0, 
                                antialiased=True, 
                                label=f"地下第{i+1}层"
                            )
                        else:
                            # 使用常规颜色映射
                            surf_underground = ax.plot_surface(
                                X, Y, Z_underground - layer_offset, 
                                cmap=cmap_name, 
                                alpha=alpha,
                                linewidth=0, 
                                antialiased=True, 
                                label=f"地下第{i+1}层"
                            )
                
                # 设置视角和标签
                ax.view_init(elev=30, azim=45)
                ax.set_xlabel('X坐标')
                ax.set_ylabel('Y坐标')
                ax.set_zlabel('高度')
                
                # 添加图例说明
                # 通过创建一个空的代理艺术家来绕过plot_surface不支持图例的问题
                legend_elements = [
                    plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.7, label='地表'),
                ]
                # 为每个地下层添加图例项
                for i in range(underground_layers_count):
                    color = plt.cm.get_cmap('tab10')(i / 10)  # 使用不同颜色区分
                    legend_elements.append(
                        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.6, label=f'地下第{i+1}层')
                    )
                
                # 添加图例到3D轴
                ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
                
                # 调整布局
                fig.tight_layout()
                
                # 添加到tkinter窗口
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # 添加3D控制工具栏
                from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
                toolbar = NavigationToolbar2Tk(canvas, frame)
                toolbar.update()
                toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                
            elif view_mode == "横截面":
                # 创建一个Figure
                fig = Figure(figsize=(12, 8), dpi=100)
                
                # 单层横截面视图
                if layer_name == "地表" or underground_index >= 0:
                    # 获取当前选定层的数据
                    if layer_name == "地表":
                        layer_data = surface_height
                        if data_type == "生物群系" and surface_biome is not None:
                            layer_data = surface_biome
                    else:
                        # 获取地下层数据
                        underground_layer_name = f"underground_{underground_index}"
                        underground_layer = self.map_data.underground_layers[underground_layer_name]
                        
                        if data_type == "高度/地形":
                            layer_data = underground_layer["height"]
                        elif data_type == "内容类型":
                            layer_data = underground_layer["content"]
                        elif data_type == "矿物分布" and hasattr(self.map_data, 'mineral_layers') and underground_layer_name in self.map_data.mineral_layers:
                            layer_data = self.map_data.mineral_layers[underground_layer_name]
                        else:
                            # 默认显示高度
                            layer_data = underground_layer["height"]
                    
                    # 创建横截面图
                    ax = fig.add_subplot(111)
                    
                    # 获取截面
                    cross_section = layer_data[cross_section_pos, :]
                    
                    # 绘制截面线图
                    ax.plot(cross_section, color='blue', linewidth=2)
                    ax.set_title(f"{layer_name}横截面视图 - {data_type} (Y={cross_section_pos})")
                    ax.set_xlabel("X坐标")
                    ax.set_ylabel("高度/值")
                    ax.grid(True)
                    
                    # 设置Y轴范围
                    if data_type == "高度/地形":
                        ax.set_ylim(0, np.max(layer_data) * 1.1)
                
                # 将图形添加到tkinter窗口
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            elif view_mode == "垂直截面":
                # 创建一个Figure
                fig = Figure(figsize=(12, 8), dpi=100)
                ax = fig.add_subplot(111)
                
                # 获取地表数据
                surface_data = surface_height
                
                # 创建一个表示垂直截面的数组
                depth = len(self.map_data.underground_layers)
                width = self.map_data.width
                vertical_section = np.zeros((depth + 1, width))
                
                # 填充地表数据
                vertical_section[0, :] = surface_data[cross_section_pos, :]
                
                # 填充每个地下层的数据
                for i in range(depth):
                    underground_layer_name = f"underground_{i}"
                    if underground_layer_name in self.map_data.underground_layers:
                        layer = self.map_data.underground_layers[underground_layer_name]
                        
                        if data_type == "高度/地形":
                            vertical_section[i + 1, :] = layer["height"][cross_section_pos, :]
                        elif data_type == "内容类型":
                            vertical_section[i + 1, :] = layer["content"][cross_section_pos, :]
                        elif data_type == "矿物分布" and hasattr(self.map_data, 'mineral_layers') and underground_layer_name in self.map_data.mineral_layers:
                            vertical_section[i + 1, :] = self.map_data.mineral_layers[underground_layer_name][cross_section_pos, :]
                
                # 创建热图
                from matplotlib import cm
                im = ax.imshow(vertical_section, cmap='terrain' if data_type == "高度/地形" else 'viridis', aspect='auto')
                fig.colorbar(im, ax=ax)
                
                # 设置标题和标签
                ax.set_title(f"垂直截面视图 - {data_type} (Y={cross_section_pos})")
                ax.set_xlabel("X坐标")
                ax.set_ylabel("深度 (0=地表)")
                
                # Y轴标签
                y_labels = ["地表"] + [f"地下{i+1}" for i in range(depth)]
                ax.set_yticks(np.arange(depth + 1))
                ax.set_yticklabels(y_labels)
                
                # 将图形添加到tkinter窗口
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            elif view_mode == "3D视图":
                # 创建3D视图
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = Figure(figsize=(12, 8), dpi=100)
                ax = fig.add_subplot(111, projection='3d')
                
                # 获取地表数据
                x = np.arange(0, self.map_data.width, 3)
                y = np.arange(0, self.map_data.height, 3)
                X, Y = np.meshgrid(x, y)
                
                # 设置z值
                Z = surface_height[::3, ::3]
                
                # 绘制地表
                surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.7, 
                                    linewidth=0, antialiased=True)
                
                # 如果选择了地下层，也显示它
                if underground_index >= 0:
                    underground_layer_name = f"underground_{underground_index}"
                    if underground_layer_name in self.map_data.underground_layers:
                        underground_layer = self.map_data.underground_layers[underground_layer_name]
                        
                        # 绘制地下层，偏移Z轴以显示在地表下方
                        Z_underground = underground_layer["height"][::3, ::3]
                        offset = np.mean(Z) - np.mean(Z_underground) - 20  # 添加偏移以便在地表下方可见
                        
                        # 绘制地下层
                        if data_type == "矿物分布" and hasattr(self.map_data, 'mineral_layers') and underground_layer_name in self.map_data.mineral_layers:
                            # 获取矿物数据
                            mineral_data = self.map_data.mineral_layers[underground_layer_name][::3, ::3]
                            
                            # 创建颜色映射
                            from matplotlib import cm
                            colors = cm.rainbow(mineral_data / np.max(mineral_data))
                            
                            # 绘制带有矿物颜色的地下层
                            surf_underground = ax.plot_surface(X, Y, Z_underground - offset, 
                                                            facecolors=colors, alpha=0.8,
                                                            linewidth=0, antialiased=True)
                        else:
                            # 默认绘制地下层
                            surf_underground = ax.plot_surface(X, Y, Z_underground - offset, 
                                                            cmap='plasma', alpha=0.8,
                                                            linewidth=0, antialiased=True)
                
                # 设置视角
                ax.view_init(elev=30, azim=45)
                
                # 添加标签
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('高度')
                ax.set_title(f"3D地形与地下视图 - {layer_name} {data_type}")
                
                # 将图形添加到tkinter窗口
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # 添加3D控制工具栏
                from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
                toolbar = NavigationToolbar2Tk(canvas, frame)
                toolbar.update()
            
            else:
                # 默认情况，显示简单的文本说明
                ttk.Label(frame, text="请选择有效的视图模式").pack(pady=20)

    def _export_integrated_view(self, frame, view_mode, layer_name, data_type, cross_section_pos):
        """导出集成视图为图像文件"""
        # 打开文件选择对话框
        filepath = filedialog.asksaveasfilename(
            title="保存集成视图",
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg"), ("所有文件", "*.*")]
        )
        
        if not filepath:
            return
        
        # 找到当前视图中的图形
        for widget in frame.winfo_children():
            if hasattr(widget, 'figure'):
                # 保存图形为图像
                widget.figure.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.log(f"已导出集成视图至: {filepath}")
                messagebox.showinfo("导出成功", f"集成视图已保存至:\n{filepath}")
                return
        
        # 如果没有找到图形，可能是3D视图工具栏或其他非图形组件
        for widget in frame.winfo_children():
            if hasattr(widget, 'get_tk_widget'):
                # 获取FigureCanvasTkAgg组件的图形
                figure = widget.figure
                figure.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.log(f"已导出集成视图至: {filepath}")
                messagebox.showinfo("导出成功", f"集成视图已保存至:\n{filepath}")
                return
        
        # 如果仍然没找到，报告错误
        self.logger.log("导出失败: 未找到可导出的图形", "ERROR")
        messagebox.showerror("导出失败", "未找到可导出的图形")

    def _show_3d_underground_view(self):
        """显示地下层的3D视图"""
        # 检查地图数据是否有效
        if not self.map_data or not self.map_data.is_valid():
            messagebox.showerror("错误", "请先生成基础地图")
            return
            
        # 检查是否已生成地下系统
        if not hasattr(self.map_data, 'underground_layers') or not self.map_data.underground_layers:
            if messagebox.askyesno("提示", "地下系统尚未生成，是否先生成地下系统？"):
                self._generate_underground_system()
                return
            else:
                return
        
        try:
            # 尝试导入必要的3D绘图库
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
            from core.generation.generate_undergroud import UndergroundContentType, MineralType
        except ImportError as e:
            self.logger.log(f"加载3D视图所需库失败: {e}", "ERROR")
            messagebox.showerror("导入错误", f"无法加载3D视图所需库:\n{e}\n\n请确保已安装matplotlib和numpy")
            return
        
        # 创建一个新窗口
        view_window = tk.Toplevel(self.master)
        view_window.title("地下系统3D视图")
        view_window.geometry("1000x800")
        view_window.transient(self.master)
        
        # 创建控制面板
        control_frame = ttk.LabelFrame(view_window, text="视图控制")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建控制选项
        control_grid = ttk.Frame(control_frame)
        control_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # 视图类型
        ttk.Label(control_grid, text="显示数据:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        view_type_var = tk.StringVar(value="内容类型")
        view_type = ttk.Combobox(control_grid, textvariable=view_type_var, 
                                values=["内容类型", "矿物分布", "高度图"], state="readonly", width=15)
        view_type.grid(row=0, column=1, padx=5, pady=5)
        
        # 采样率
        ttk.Label(control_grid, text="采样率:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        sample_rate_var = tk.IntVar(value=5)
        sample_rate = ttk.Spinbox(control_grid, from_=1, to=20, textvariable=sample_rate_var, width=5)
        sample_rate.grid(row=0, column=3, padx=5, pady=5)
        
        # 透明度
        ttk.Label(control_grid, text="透明度:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        alpha_var = tk.DoubleVar(value=0.7)
        alpha_scale = ttk.Scale(control_grid, from_=0.1, to=1.0, variable=alpha_var, orient=tk.HORIZONTAL)
        alpha_scale.grid(row=0, column=5, padx=5, pady=5, sticky=tk.EW)
        
        # 刷新按钮
        refresh_btn = ttk.Button(control_grid, text="刷新视图", command=lambda: update_3d_view())
        refresh_btn.grid(row=0, column=6, padx=15, pady=5)
        
        # 创建matplotlib图形
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # 添加到tkinter窗口
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        canvas = FigureCanvasTkAgg(fig, master=view_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 添加导航工具栏
        toolbar_frame = ttk.Frame(view_window)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # 内容类型颜色映射
        content_colors = {
            UndergroundContentType.EMPTY: 'white',
            UndergroundContentType.SOIL: 'brown',
            UndergroundContentType.ROCK: 'grey',
            UndergroundContentType.CAVE: 'darkgrey',
            UndergroundContentType.TUNNEL: 'black',
            UndergroundContentType.WATER: 'blue',
            UndergroundContentType.LAVA: 'red',
            UndergroundContentType.UNDERGROUND_RIVER: 'lightblue',
            UndergroundContentType.CRYSTAL_CAVE: 'purple',
            UndergroundContentType.FUNGAL_CAVE: 'green',
            UndergroundContentType.ANCIENT_RUINS: 'gold',
            UndergroundContentType.ABANDONED_MINE: 'saddlebrown'
        }
        
        # 矿物类型颜色映射
        mineral_colors = {
            MineralType.NONE: 'white',
            MineralType.COAL: '#333333',  # 黑色
            MineralType.IRON: '#A52A2A',  # 棕色
            MineralType.COPPER: '#B87333',  # 铜色
            MineralType.GOLD: '#FFD700',   # 金色
            MineralType.SILVER: '#C0C0C0',  # 银色
            MineralType.DIAMOND: '#B9F2FF', # 钻石蓝
            MineralType.EMERALD: '#50C878', # 绿色
            MineralType.RUBY: '#E0115F',    # 深红色
            MineralType.SAPPHIRE: '#0F52BA', # 蓝色
            MineralType.CRYSTAL: '#FFB6C1',  # 粉色
            MineralType.OBSIDIAN: '#2F4F4F', # 墨绿色
            MineralType.MYTHRIL: '#7DF9FF',  # 亮蓝色
            MineralType.ADAMANTINE: '#800080' # 紫色
        }
        
        def update_3d_view():
            """更新3D视图显示"""
            # 清除当前图形
            ax.clear()
            
            # 获取参数
            view = view_type_var.get()
            sample = sample_rate_var.get()
            alpha = alpha_var.get()
            
            # 获取地下层数据
            depth = len(self.map_data.underground_layers)
            width = self.map_data.width
            height = self.map_data.height
            
            # 设置坐标轴标签
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('深度')
            
            # 遍历所有地下层
            for i in range(depth):
                layer_name = f"underground_{i}"
                if layer_name not in self.map_data.underground_layers:
                    continue
                    
                # 基于视图类型选择数据和颜色映射
                if view == "内容类型":
                    data = self.map_data.underground_layers[layer_name]["content"]
                    color_map = content_colors
                    title_text = "地下内容类型3D视图"
                elif view == "矿物分布":
                    if hasattr(self.map_data, 'mineral_layers') and layer_name in self.map_data.mineral_layers:
                        data = self.map_data.mineral_layers[layer_name]
                        color_map = mineral_colors
                        title_text = "地下矿物分布3D视图"
                    else:
                        self.logger.log(f"矿物数据不可用: {layer_name}", "WARNING")
                        continue
                else:  # 高度图
                    data = self.map_data.underground_layers[layer_name]["height"]
                    # 使用连续颜色映射
                    from matplotlib import cm
                    title_text = "地下高度3D视图"
                
                # 使用采样率减少点数量
                for y in range(0, height, sample):
                    for x in range(0, width, sample):
                        if view == "高度图":
                            # 高度数据使用不同的显示方法
                            z_value = i
                            color = cm.get_cmap('terrain')(data[y, x])  # 使用地形色标
                        else:
                            # 内容类型和矿物使用枚举映射颜色
                            z_value = i
                            value = data[y, x]
                            if value in color_map:
                                color = color_map[value]
                            else:
                                color = 'lightgrey'  # 默认颜色
                        
                        # 绘制点
                        ax.scatter(x, y, z_value, c=color, marker='s', s=30, alpha=alpha)
            
            # 设置标题和视图参数
            ax.set_title(title_text)
            ax.invert_zaxis()  # 反转Z轴，使深度增加时值增加
            fig.tight_layout()
            canvas.draw()
        
        # 初始更新视图
        update_3d_view()
        
        self.logger.log("已显示地下系统3D视图")

    def _export_underground_preview(self):
        """导出地下层预览图像"""
        if not hasattr(self.map_data, 'underground_layers') or not self.map_data.underground_layers:
            messagebox.showwarning("提示", "地下系统尚未生成，无法导出")
            return
        
        # 弹出目录选择对话框
        export_dir = filedialog.askdirectory(title="选择导出目录")
        if not export_dir:
            return
        
        try:
            import matplotlib.pyplot as plt
            from core.generation.generate_undergroud import visualize_underground_layer
            
            # 导出每一层的预览
            for i in range(len(self.map_data.underground_layers)):
                layer_name = f"underground_{i}"
                if layer_name in self.map_data.underground_layers:
                    # 设置保存路径
                    save_path = os.path.join(export_dir, f"地下层_{i+1}.png")
                    
                    # 导出图像
                    visualize_underground_layer(
                        self.map_data.underground_layers[layer_name], 
                        i, 
                        map_data=self.map_data,  # 重要: 传入完整map_data以显示矿物
                        title=f"地下第{i+1}层", 
                        save_path=save_path
                    )
                    
                    self.logger.log(f"已导出: {save_path}")
            
            # 显示成功消息
            messagebox.showinfo("导出成功", f"已将地下层预览导出至:\n{export_dir}")
            
        except Exception as e:
            self.logger.log(f"导出地下层预览失败: {e}", "ERROR")
            messagebox.showerror("导出失败", f"导出地下层预览时发生错误:\n{e}")

    def _create_scrollable_frame(self, parent):
        """创建带滚动条的标签页容器框架"""
        # 创建画布作为滚动视图容器
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        
        # 创建可滚动的框架
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # 在画布上创建窗口
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # 根据画布宽度调整框架宽度
        def resize_frame(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", resize_frame)
        
        # 配置画布滚动
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 添加鼠标滚轮支持
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 使用grid布局以便同时放置滚动条
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # 确保画布扩展填充
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        return scrollable_frame
    
    def _show_notification(self, title, message, type_="info", timeout=3000):
        """显示临时通知"""
        # 创建通知窗口
        notif = tk.Toplevel(self.master)
        notif.title("")
        notif.geometry("300x80+{}+{}".format(
            self.master.winfo_rootx() + self.master.winfo_width() - 320,
            self.master.winfo_rooty() + 40
        ))
        notif.attributes("-topmost", True)
        notif.overrideredirect(True)
        
        # 设置样式
        if type_ == "error":
            bg_color = "#FFD2D2"
            fg_color = "#8B0000"
            icon = self._get_icon("error", size=24)
        elif type_ == "warning":
            bg_color = "#FFEDCC"
            fg_color = "#805700"
            icon = self._get_icon("warning", size=24)
        elif type_ == "success":
            bg_color = "#CCFFCC"
            fg_color = "#006400"
            icon = self._get_icon("success", size=24)
        else:  # info
            bg_color = "#CCE5FF"
            fg_color = "#00008B"
            icon = self._get_icon("info", size=24)
        
        # 创建通知框架
        notif_frame = tk.Frame(notif, bg=bg_color, bd=1, relief=tk.SOLID)
        notif_frame.pack(fill=tk.BOTH, expand=True)
        
        # 内容布局
        content_frame = tk.Frame(notif_frame, bg=bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        # 图标
        if icon:
            icon_label = tk.Label(content_frame, image=icon, bg=bg_color)
            icon_label.pack(side=tk.LEFT, padx=(0, 10))
            # 保持引用，防止垃圾回收
            icon_label.image = icon
        
        # 文本
        text_frame = tk.Frame(content_frame, bg=bg_color)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(text_frame, text=title, font=("", 10, "bold"), 
            bg=bg_color, fg=fg_color).pack(anchor=tk.W)
        
        tk.Label(text_frame, text=message, wraplength=250, 
            bg=bg_color, fg=fg_color).pack(anchor=tk.W)
        
        # 关闭按钮
        close_button = tk.Label(content_frame, text="×", font=("", 16), 
                            bg=bg_color, fg=fg_color, cursor="hand2")
        close_button.pack(side=tk.RIGHT, padx=(10, 0))
        close_button.bind("<Button-1>", lambda e: notif.destroy())
        
        # 设置自动关闭和淡出效果
        if timeout > 0:
            def fade_away(alpha=1.0):
                if alpha <= 0:
                    notif.destroy()
                    return
                notif.attributes("-alpha", alpha)
                notif.after(50, lambda: fade_away(alpha - 0.1))
                
            notif.after(timeout - 500, lambda: fade_away())
        
        return notif

    def _show_yue_music_dialog(self):
        """打开音乐生成器对话框"""
        try:
            # 导入音乐生成器对话框
            from plugin.yue.yue_gui import MusicGeneratorDialog
            
            # 创建对话框实例
            music_dialog = MusicGeneratorDialog(self.master, logger=self.logger)
            
            # 记录日志
            self.logger.log("已打开音乐生成器对话框")
        except Exception as e:
            error_msg = f"打开音乐生成器时出错: {str(e)}"
            self.logger.log(error_msg, level="ERROR")
            messagebox.showerror("错误", error_msg)

    def _show_3d_model_editor(self):
        """显示3D模型编辑器"""
        from utils.model_editor import Model3DEditor
        
        # 创建编辑器实例
        editor = Model3DEditor(self.master, map_data=self.map_data)
        
        # 将模型编辑器设置为模态窗口
        editor.dialog.transient(self.master)
        editor.dialog.grab_set()
        
        # 等待编辑器窗口关闭
        self.master.wait_window(editor.dialog)
        
        # 如果地图数据被更新，更新显示
        if hasattr(editor, 'map_data') and editor.map_data:
            self._update_map_preview()
            self._show_notification("模型更新", "3D模型已更新到地图", type_="info")

    def _update_map_preview(self):
        """更新地图预览显示"""
        # 检查是否有预览窗口
        if hasattr(self, 'preview_canvas') and self.preview_canvas is not None:
            try:
                # 重新渲染地图
                self._fit_preview_to_window()
            except Exception as e:
                self.logger.log(f"更新地图预览失败: {str(e)}", "WARNING")

    def _show_diffrhythm_music_dialog(self):
        """显示歌曲生成器对话框"""
        try:
            from plugin.diffrhythm.diffrhythm_client import DiffRhythmDialog
            dialog = DiffRhythmDialog(self.master, logger=self.logger)
        except ImportError as e:
            self.logger.log(f"导入DiffRhythmDialog失败: {str(e)}")
            messagebox.showerror("导入错误", f"无法导入音乐生成器对话框:\n{str(e)}")
        except Exception as e:
            self.logger.log(f"打开音乐生成对话框出错: {str(e)}")
            messagebox.showerror("错误", f"打开音乐生成对话框出错:\n{str(e)}")

    def _show_inspiremusic_music_dialog(self):
        """显示旋律生成器对话框"""
        try:
            from plugin.inspiremusic.inspiremusic_client import InspireMusicDialog
            self.logger.log("打开旋律生成器对话框")
            dialog = InspireMusicDialog(self.master, logger=self.logger)
            dialog.dialog.grab_set()
        except ImportError:
            self.logger.log("无法加载旋律生成器插件，请检查安装", "ERROR")
            messagebox.showerror("插件错误", "无法加载旋律生成器插件，请确保插件已正确安装。")
        except Exception as e:
            self.logger.log(f"启动旋律生成器时出错: {str(e)}", "ERROR")
            messagebox.showerror("错误", f"启动旋律生成器时发生错误:\n{str(e)}")

    def _show_hunyuan_3d_dialog(self):
        """显示Hunyuan 3D模型生成对话框"""
        self.logger.log("打开Hunyuan 3D模型生成对话框")
        dialog = Hunyuan3DDialog(self.master, logger=self.logger)
        dialog.grab_set()  # 使对话框成为模态

    def _generate_music_from_story(self):
        """从故事内容生成音乐文件"""
        if not hasattr(self.map_data, 'story_content') or not self.map_data.story_content:
            messagebox.showwarning("缺少故事内容", "请先生成或导入故事内容")
            return
        
        # 创建进度对话框
        progress_dialog = tk.Toplevel(self.master)
        progress_dialog.title("生成音乐")
        progress_dialog.geometry("400x200")
        progress_dialog.transient(self.master)
        progress_dialog.grab_set()
        
        # 创建进度信息
        progress_frame = ttk.Frame(progress_dialog, padding=20)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(progress_frame, text="从故事内容生成音乐文件...").pack(pady=10)
        
        progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, 
                                    mode='determinate', length=300)
        progress_bar.pack(pady=10)
        
        status_var = tk.StringVar(value="准备中...")
        status_label = ttk.Label(progress_frame, textvariable=status_var)
        status_label.pack(pady=5)
        
        # 生成音乐的线程函数
        def generate_music_thread():
            if not hasattr(self, 'music_generator'):
                # 创建音乐生成器
                from core.dynamic_music_generator import DynamicMusicGenerator
                self.music_generator = DynamicMusicGenerator(
                    map_width=self.map_data.width,
                    map_height=self.map_data.height,
                    music_resource_path="data/music",
                    logger=self.logger
                )
            
            # 获取故事章节或事件
            story_events = []
            try:
                story_content = self.map_data.story_content
                if isinstance(story_content, dict):
                    if "chapters" in story_content:
                        story_events = story_content["chapters"]
                    elif "events" in story_content:
                        story_events = story_content["events"]
                    elif "sections" in story_content:
                        story_events = story_content["sections"]
                    else:
                        # 尝试找到第一个列表类型的值
                        for key, value in story_content.items():
                            if isinstance(value, list) and len(value) > 0:
                                story_events = value
                                break
                elif isinstance(story_content, list):
                    story_events = story_content
                
                if not story_events:
                    # 如果没有找到事件列表，尝试将整个内容作为一个事件
                    story_events = [{"content": str(story_content)}]
            except Exception as e:
                self.logger.log(f"解析故事内容错误: {str(e)}")
                status_var.set(f"解析故事内容错误: {str(e)}")
                return
            
            total_events = len(story_events)
            if total_events == 0:
                status_var.set("未找到任何故事事件")
                return
            
            # 更新状态
            status_var.set(f"找到 {total_events} 个故事事件，开始生成音乐...")
            
            # 生成音乐
            for i, event in enumerate(story_events):
                # 提取事件内容
                event_text = ""
                if isinstance(event, dict):
                    if "content" in event:
                        event_text = event["content"]
                    elif "description" in event:
                        event_text = event["description"]
                    elif "text" in event:
                        event_text = event["text"]
                    else:
                        event_text = str(event)
                else:
                    event_text = str(event)
                
                # 更新进度
                progress_var.set((i / total_events) * 100)
                status_var.set(f"正在为事件 {i+1}/{total_events} 生成音乐...")
                progress_dialog.update()
                
                # 为文本生成音乐
                try:
                    success = self.music_generator.generate_music_from_text(event_text)
                    if not success:
                        status_var.set(f"生成事件 {i+1} 的音乐失败")
                except Exception as e:
                    self.logger.log(f"生成音乐错误: {str(e)}")
                    status_var.set(f"生成音乐错误: {str(e)}")
            
            # 完成生成
            progress_var.set(100)
            status_var.set("音乐生成完成!")
            
            # 加载音乐资源
            try:
                self.music_generator.load_music_resources()
                self.logger.log("音乐资源加载成功")
            except Exception as e:
                self.logger.log(f"加载音乐资源错误: {str(e)}")
            
            # 三秒后关闭对话框
            progress_dialog.after(3000, progress_dialog.destroy)
        
        # 启动线程
        thread = Thread(target=generate_music_thread)
        thread.daemon = True
        thread.start()
        
    def _experience_music_system(self):
        """打开增强型音乐体验系统窗口"""
        if not self.map_data or not self.map_data.is_valid():
            messagebox.showerror("错误", "请先生成地图数据")
            return
        
        # 检查是否已加载情感数据
        if not hasattr(self, 'emotion_manager') or not self.emotion_manager.emotion_data:
            if messagebox.askyesno("提示", "未检测到情感数据，是否立即分析地图情感?"):
                self.status_var.set("正在分析地图情感...")
                success = self.emotion_manager.analyze_map_emotions(self.map_data)
                if not success:
                    messagebox.showerror("错误", "情感分析失败")
                    return
            else:
                return
        
        # 加载音乐数据
        self.status_var.set("正在加载音乐区域数据...")
        self.master.update_idletasks()
        self.music_generator.load_emotion_data(emotion_manager=self.emotion_manager, 
                                            story_emotion_map=self.map_data.emotion_map)
        
        # 初始化音乐过渡管理器
        from core.music.emotion_music_transition import MusicTransitionManager  # 导入过渡管理器
        self.music_transition_manager = MusicTransitionManager(
            music_generator=self.music_generator, 
            logger=self.logger
        )
        
        # 创建音乐体验窗口
        music_window = tk.Toplevel(self.master)
        music_window.title("增强型动态音乐体验系统")
        music_window.geometry("1000x700")
        music_window.transient(self.master)
        
        # 创建分隔窗格
        paned = ttk.PanedWindow(music_window, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.Frame(paned)
        paned.add(control_frame, weight=1)
        
        # 控制面板使用Notebook (标签页)
        control_notebook = ttk.Notebook(control_frame)
        control_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 基本控制标签页
        basic_tab = ttk.Frame(control_notebook)
        control_notebook.add(basic_tab, text="基本控制")
        
        # 协作模式标签页
        collab_tab = ttk.Frame(control_notebook)
        control_notebook.add(collab_tab, text="模型协作")
        
        # 参数调整标签页
        params_tab = ttk.Frame(control_notebook)
        control_notebook.add(params_tab, text="音乐参数")
        
        # 自动移动标签页 - 新增
        auto_move_tab = ttk.Frame(control_notebook)
        control_notebook.add(auto_move_tab, text="自动移动")
        
        # 过渡控制标签页 - 新增
        transition_tab = ttk.Frame(control_notebook)
        control_notebook.add(transition_tab, text="过渡控制")
        
        # 可视化标签页
        visual_tab = ttk.Frame(control_notebook)
        control_notebook.add(visual_tab, text="音乐可视化")
        
        # 右侧地图预览区域
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        # 右侧使用垂直分割
        right_paned = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)
        
        # 地图预览
        map_frame = ttk.LabelFrame(right_paned, text="地图与情感区域")
        right_paned.add(map_frame, weight=2)
        
        # 音频可视化
        audio_visual_frame = ttk.LabelFrame(right_paned, text="音频可视化")
        right_paned.add(audio_visual_frame, weight=1)
        
        # ==== 设置基本控制标签页内容 ====
        self._setup_basic_music_controls(basic_tab, audio_visual_frame)
        
        # ==== 设置协作模式标签页内容 ====
        self._setup_collaboration_mode_controls(collab_tab)
        
        # ==== 设置参数调整标签页内容 ====
        self._setup_music_parameter_controls(params_tab)
        
        # ==== 设置自动移动标签页内容 ====
        self._setup_auto_move_controls(auto_move_tab, map_frame)
        
        # ==== 设置过渡控制标签页内容 - 新增 ====
        self._setup_transition_controls(transition_tab)
        
        # ==== 设置可视化标签页内容 ====
        self._setup_music_visualization_controls(visual_tab, audio_visual_frame)
        
        # ==== 设置地图预览区域，添加拖拽支持 ====
        self._setup_map_preview_for_music_with_drag(map_frame)
        
        # 初始化音频可视化
        self._setup_audio_visualization(audio_visual_frame)
        
        self.status_var.set("音乐系统已准备就绪")
        
        # 窗口关闭时停止播放
        music_window.protocol("WM_DELETE_WINDOW", lambda: [self._stop_music_playback(), music_window.destroy()])

    # 添加过渡控制标签页设置方法
    def _setup_transition_controls(self, parent_frame):
        """设置音乐过渡控制界面"""
        # 过渡参数框架
        params_frame = ttk.LabelFrame(parent_frame, text="过渡参数")
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 过渡速度控制
        speed_frame = ttk.Frame(params_frame)
        speed_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(speed_frame, text="过渡速度:").pack(side=tk.LEFT, padx=5)
        
        self.transition_speed_var = tk.DoubleVar(value=1.0)
        speed_slider = ttk.Scale(
            speed_frame, 
            from_=0.1, 
            to=2.0, 
            variable=self.transition_speed_var,
            orient=tk.HORIZONTAL,
            command=self._update_transition_speed
        )
        speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(speed_frame, textvariable=self.transition_speed_var, width=4).pack(side=tk.LEFT, padx=5)
        
        # 预测半径控制
        prediction_frame = ttk.Frame(params_frame)
        prediction_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(prediction_frame, text="预测半径:").pack(side=tk.LEFT, padx=5)
        
        self.prediction_radius_var = tk.IntVar(value=20)
        radius_slider = ttk.Scale(
            prediction_frame, 
            from_=5, 
            to=50, 
            variable=self.prediction_radius_var,
            orient=tk.HORIZONTAL
        )
        radius_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(prediction_frame, textvariable=self.prediction_radius_var, width=4).pack(side=tk.LEFT, padx=5)
        
        # 过渡类型选择
        type_frame = ttk.Frame(params_frame)
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(type_frame, text="过渡类型:").pack(side=tk.LEFT, padx=5)
        
        self.transition_type_var = tk.StringVar(value="平滑")
        transition_types = ["平滑", "交叉淡变", "瞬时", "分层"]
        type_combo = ttk.Combobox(
            type_frame, 
            textvariable=self.transition_type_var, 
            values=transition_types, 
            state="readonly",
            width=15
        )
        type_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 过渡状态信息框架
        info_frame = ttk.LabelFrame(parent_frame, text="当前过渡状态")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 过渡信息显示
        self.transition_info_text = scrolledtext.ScrolledText(info_frame, height=10, wrap=tk.WORD)
        self.transition_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.transition_info_text.insert(tk.END, "未开始音乐过渡。\n\n"
                                        "当进入新的情感区域时，这里会显示过渡状态信息。")
        self.transition_info_text.config(state=tk.DISABLED)
        
        # 预览过渡按钮
        ttk.Button(
            parent_frame,
            text="预览过渡效果",
            command=self._preview_transition_effect
        ).pack(fill=tk.X, padx=10, pady=5)
        
        # 重置过渡按钮
        ttk.Button(
            parent_frame,
            text="重置过渡状态",
            command=self._reset_transition_state
        ).pack(fill=tk.X, padx=10, pady=5)

    def _update_transition_speed(self, *args):
        """更新过渡速度"""
        if hasattr(self, 'music_transition_manager'):
            speed = self.transition_speed_var.get()
            self.music_transition_manager.set_transition_speed(speed)
            self._update_transition_info_display()

    def _update_transition_info_display(self):
        """更新过渡信息显示"""
        if not hasattr(self, 'transition_info_text') or not hasattr(self, 'music_transition_manager'):
            return
        
        try:
            # 获取当前过渡信息
            transition_info = self.music_transition_manager.get_transition_info()
            
            # 更新显示
            self.transition_info_text.config(state=tk.NORMAL)
            self.transition_info_text.delete(1.0, tk.END)
            
            if transition_info["in_transition"]:
                self.transition_info_text.insert(tk.END, f"正在进行音乐过渡 ({transition_info['progress']:.2f})\n\n")
                self.transition_info_text.insert(tk.END, f"从: {transition_info['from_emotion']} ({transition_info['from_region']})\n")
                self.transition_info_text.insert(tk.END, f"到: {transition_info['to_emotion']} ({transition_info['to_region']})\n\n")
                self.transition_info_text.insert(tk.END, f"过渡速度倍率: {transition_info['speed_multiplier']:.1f}x")
                
                # 高亮进度条
                progress_bar = "▓" * int(transition_info["progress"] * 20) + "░" * int((1 - transition_info["progress"]) * 20)
                self.transition_info_text.insert(tk.END, f"\n\n进度: {progress_bar}")
            else:
                if transition_info["to_region"] is None:
                    self.transition_info_text.insert(tk.END, "未在任何情感区域内\n")
                else:
                    self.transition_info_text.insert(tk.END, f"当前区域: {transition_info['to_emotion']} ({transition_info['to_region']})\n")
                    self.transition_info_text.insert(tk.END, "\n等待进入新的情感区域...")
            
            self.transition_info_text.config(state=tk.DISABLED)
        except Exception as e:
            self.logger.log(f"更新过渡信息出错: {e}", "ERROR")

    def _preview_transition_effect(self):
        """预览过渡效果 - 在当前区域和附近区域间创建一个模拟过渡"""
        if not hasattr(self, 'music_transition_manager'):
            messagebox.showinfo("提示", "请先开始音乐播放")
            return
        
        # 获取当前区域
        current_region = self.music_transition_manager.target_region
        if not current_region:
            messagebox.showinfo("提示", "请先移动到一个情感区域内")
            return
        
        # 查找一个不同的情感区域
        available_regions = []
        for region_id, region_data in self.music_generator.emotion_regions.items():
            if region_id != current_region:
                emotion = region_data.get("emotion", "")
                available_regions.append((region_id, emotion))
        
        if not available_regions:
            messagebox.showinfo("提示", "地图上没有其他情感区域")
            return
        
        # 创建选择对话框
        transition_dialog = tk.Toplevel(self.master)
        transition_dialog.title("选择过渡目标")
        transition_dialog.transient(self.master)
        transition_dialog.grab_set()
        
        ttk.Label(transition_dialog, text="选择要过渡到的情感区域:").pack(padx=10, pady=10)
        
        # 创建列表框
        region_listbox = tk.Listbox(transition_dialog, height=10, width=40)
        region_listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # 添加区域到列表
        for i, (region_id, emotion) in enumerate(available_regions):
            region_listbox.insert(tk.END, f"{emotion} ({region_id})")
        
        # 选择按钮
        def on_select():
            selection = region_listbox.curselection()
            if selection:
                idx = selection[0]
                target_region, target_emotion = available_regions[idx]
                
                # 创建模拟过渡
                self._simulate_transition(current_region, target_region)
                transition_dialog.destroy()
            else:
                messagebox.showinfo("提示", "请选择一个目标区域")
        
        ttk.Button(transition_dialog, text="开始过渡", command=on_select).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(transition_dialog, text="取消", command=transition_dialog.destroy).pack(side=tk.RIGHT, padx=10, pady=10)

    def _simulate_transition(self, from_region, to_region):
        """模拟从一个区域到另一个区域的过渡"""
        if not hasattr(self, 'music_transition_manager'):
            return
        
        # 保存当前状态
        self.music_transition_manager.previous_region = from_region
        self.music_transition_manager.target_region = to_region
        self.music_transition_manager.in_transition = True
        self.music_transition_manager.transition_progress = 0.0
        
        # 显示消息
        self.logger.log(f"开始模拟过渡效果: {from_region} → {to_region}")
        self._update_transition_info_display()
        
        # 创建过渡动画
        def animate_transition():
            if not hasattr(self, 'music_transition_manager') or not self.music_transition_manager.in_transition:
                return
            
            # 逐步更新过渡进度
            progress = self.music_transition_manager.transition_progress
            progress += 0.05  # 每步增加5%
            
            if progress >= 1.0:
                # 过渡完成
                self.music_transition_manager.transition_progress = 1.0
                self.music_transition_manager.in_transition = False
                self._update_transition_info_display()
                return
            
            # 更新进度
            self.music_transition_manager.transition_progress = progress
            
            # 应用过渡效果
            self.music_transition_manager._apply_transition_interpolation()
            
            # 更新显示
            self._update_transition_info_display()
            
            # 安排下一次更新
            self.master.after(100, animate_transition)
        
        # 开始动画
        self.master.after(100, animate_transition)

    def _reset_transition_state(self):
        """重置过渡状态"""
        if hasattr(self, 'music_transition_manager'):
            self.music_transition_manager.in_transition = False
            self.music_transition_manager.transition_progress = 1.0
            self._update_transition_info_display()
            self.logger.log("过渡状态已重置")
        
    def _setup_auto_move_controls(self, parent_frame, map_frame):
        """设置自动移动控制界面"""
        # 自动移动模式框架
        auto_move_frame = ttk.LabelFrame(parent_frame, text="自动移动设置")
        auto_move_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 启用自动移动选项
        enable_frame = ttk.Frame(auto_move_frame)
        enable_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_move_var = tk.BooleanVar(value=False)
        auto_move_check = ttk.Checkbutton(
            enable_frame, 
            text="启用自动移动", 
            variable=self.auto_move_var,
            command=self._toggle_auto_move
        )
        auto_move_check.pack(side=tk.LEFT, padx=5)
        
        # 移动速度控制
        speed_frame = ttk.Frame(auto_move_frame)
        speed_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(speed_frame, text="移动速度:").pack(side=tk.LEFT, padx=5)
        
        self.move_speed_var = tk.DoubleVar(value=0.5)
        speed_slider = ttk.Scale(
            speed_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.move_speed_var,
            orient=tk.HORIZONTAL
        )
        speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(speed_frame, textvariable=self.move_speed_var, width=4).pack(side=tk.LEFT, padx=5)
        
        # 移动路径模式
        path_frame = ttk.Frame(auto_move_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(path_frame, text="移动路径:").pack(side=tk.LEFT, padx=5)
        
        self.move_path_var = tk.StringVar(value="随机漫步")
        path_combo = ttk.Combobox(
            path_frame, 
            textvariable=self.move_path_var, 
            values=["随机漫步", "沿河流", "沿道路", "全图探索"], 
            state="readonly",
            width=15
        )
        path_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 移动范围控制
        range_frame = ttk.Frame(auto_move_frame)
        range_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(range_frame, text="移动范围:").pack(side=tk.LEFT, padx=5)
        
        self.move_range_var = tk.DoubleVar(value=0.7)
        range_slider = ttk.Scale(
            range_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.move_range_var,
            orient=tk.HORIZONTAL
        )
        range_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(range_frame, textvariable=self.move_range_var, width=4).pack(side=tk.LEFT, padx=5)
        
        # 情感区域吸引控制
        attraction_frame = ttk.Frame(auto_move_frame)
        attraction_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(attraction_frame, text="情感区域吸引力:").pack(side=tk.LEFT, padx=5)
        
        self.emotion_attraction_var = tk.DoubleVar(value=0.5)
        attraction_slider = ttk.Scale(
            attraction_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.emotion_attraction_var,
            orient=tk.HORIZONTAL
        )
        attraction_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(attraction_frame, textvariable=self.emotion_attraction_var, width=4).pack(side=tk.LEFT, padx=5)
        
        # 向特定情感区域移动
        target_frame = ttk.LabelFrame(parent_frame, text="目标情感区域")
        target_frame.pack(fill=tk.X, padx=10, pady=10)
        
        emotions_frame = ttk.Frame(target_frame)
        emotions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(emotions_frame, text="目标情感:").pack(side=tk.LEFT, padx=5)
        
        self.target_emotion_var = tk.StringVar(value="自动")
        emotions = ["自动", "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
        emotion_combo = ttk.Combobox(
            emotions_frame, 
            textvariable=self.target_emotion_var, 
            values=emotions, 
            state="readonly",
            width=15
        )
        emotion_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 移动到目标情感区域按钮
        ttk.Button(
            target_frame, 
            text="移动到所选情感区域", 
            command=self._move_to_emotion_area
        ).pack(fill=tk.X, padx=5, pady=5)
        
        # 移动路径信息显示
        info_frame = ttk.LabelFrame(parent_frame, text="移动信息")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.move_info_text = scrolledtext.ScrolledText(info_frame, height=6, wrap=tk.WORD)
        self.move_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.move_info_text.insert(tk.END, "自动移动未启动。\n\n启用自动移动后，这里将显示移动路径的相关信息。")
        self.move_info_text.config(state=tk.DISABLED)

    def _setup_map_preview_for_music_with_drag(self, parent_frame):
        """设置音乐系统的地图预览，支持拖拽和点击移动"""
        # 创建地图画布
        self.music_map_canvas = tk.Canvas(parent_frame, bg="white", cursor="hand2")
        self.music_map_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 获取地图数据
        height_map = self.map_data.get_layer("height")
        biome_map = self.map_data.get_layer("biome")
        
        if height_map is None or biome_map is None:
            ttk.Label(self.music_map_canvas, text="无法获取地图数据").pack(expand=True)
            return
        
        # 计算缩放因子
        map_height, map_width = height_map.shape
        canvas_height = self.music_map_canvas.winfo_height() or 600
        canvas_width = self.music_map_canvas.winfo_width() or 600
        scale_y = canvas_height / map_height
        scale_x = canvas_width / map_width
        
        self.music_map_scale = min(scale_x, scale_y) * 0.9  # 留出边距
        
        # 创建玩家位置标记和图像初始化
        self.music_map_image = None
        self.player_marker = None
        self.current_music_position = (map_width // 2, map_height // 2)
        
        # 延迟加载地图，等canvas准备好
        self.music_map_canvas.update()
        self.music_map_canvas.after(100, lambda: self._draw_music_map_and_regions())
        
        # 绑定鼠标事件 - 点击和拖动
        self.music_map_canvas.bind("<Button-1>", self._on_music_map_click)
        self.music_map_canvas.bind("<B1-Motion>", self._on_music_map_drag)
        
        # 绑定窗口大小改变事件
        self.music_map_canvas.bind("<Configure>", lambda e: self._on_music_map_resize())

    def _draw_music_map_and_regions(self):
        """绘制音乐地图和情感区域"""
        height_map = self.map_data.get_layer("height")
        if height_map is None:
            return
        
        # 地图尺寸
        map_height, map_width = height_map.shape
        
        # 获取画布大小
        canvas_width = self.music_map_canvas.winfo_width()
        canvas_height = self.music_map_canvas.winfo_height()
        
        # 重新计算比例
        scale_y = canvas_height / map_height
        scale_x = canvas_width / map_width
        self.music_map_scale = min(scale_x, scale_y) * 0.9
        
        try:
            # 使用matplotlib绘制地图
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # 使用自定义配色方案
            cmap = plt.cm.get_cmap('terrain')
            norm = plt.Normalize(np.min(height_map), np.max(height_map))
            
            # 应用光照效果
            ls = LightSource(azdeg=315, altdeg=45)
            rgb = ls.shade(height_map, cmap=cmap, norm=norm, blend_mode='soft')
            
            # 绘制图像 - 添加 origin='upper' 参数修复坐标系问题
            # origin='upper' 参数告诉 Matplotlib 将数组的原点 (0,0) 放在图像的左上角
            # 这样 y 坐标就会从上往下增长，与大多数图像和游戏地图中使用的坐标系一致 
            # 点击和拖动后，玩家位置会按照直观预期的方向移动，而不是镜像方向
            ax.imshow(rgb, origin='upper')
            
            # 隐藏坐标轴
            ax.set_axis_off()
            
            # 创建玩家标记
            x, y = self.current_music_position
            self.player_marker, = ax.plot([x], [y], 'ro', markersize=8, markeredgecolor='white')
            
            # 绘制情感区域
            self._draw_emotion_regions_on_plot(ax)
            
            # 嵌入图表到tkinter
            self.map_figure = fig
            self.map_ax = ax
            canvas = FigureCanvasTkAgg(fig, master=self.music_map_canvas)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.map_canvas = canvas
            
            # 将鼠标事件绑定到matplotlib canvas上
            canvas.get_tk_widget().bind("<Button-1>", self._on_music_map_click)
            canvas.get_tk_widget().bind("<B1-Motion>", self._on_music_map_drag)
            
            # 更新音乐生成器位置
            self.music_generator.update_position(x, y)
            
        except Exception as e:
            self.logger.log(f"绘制音乐地图错误: {e}", "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")
            ttk.Label(self.music_map_canvas, text=f"无法显示地图: {e}").pack(expand=True)

    def _draw_emotion_regions_on_plot(self, ax):
        """在matplotlib绘图上绘制情感区域"""
        if not hasattr(self.music_generator, 'emotion_regions') or not self.music_generator.emotion_regions:
            return
        
        # 情感区域颜色映射
        emotion_colors = {
            "joy": "#90EE90",        # 淡绿色
            "trust": "#ADD8E6",      # 淡蓝色
            "fear": "#FFB6C1",       # 淡红色
            "surprise": "#FFFF99",   # 淡黄色
            "sadness": "#D3D3D3",    # 淡灰色
            "disgust": "#E6E6FA",    # 淡紫色
            "anger": "#FFA07A",      # 浅珊瑚色
            "anticipation": "#FFDAB9" # 桃色
        }
        
        # 绘制每个情感区域
        for region_id, region_data in self.music_generator.emotion_regions.items():
            bb = region_data.get("bounding_box", (0, 0, 0, 0))
            x_start, x_end, y_start, y_end = bb
            
            # 获取情感和强度
            emotion = region_data.get("emotion", "").lower()
            intensity = region_data.get("intensity", 0.5)
            
            # 创建矩形区域
            color = emotion_colors.get(emotion, "#CCCCCC")
            alpha = min(0.5, intensity * 0.7)  # 根据强度设置透明度
            
            rect = plt.Rectangle((x_start, y_start), 
                            x_end - x_start, 
                            y_end - y_start,
                            color=color, 
                            alpha=alpha,
                            fill=True)
            ax.add_patch(rect)
            
            """
            # 计算背景色的亮度以决定文本颜色
            # 简单RGB转亮度公式: 0.299*R + 0.587*G + 0.114*B
            r, g, b = [int(color[i:i+2], 16)/255.0 for i in (1, 3, 5)]
            brightness = 0.299*r + 0.587*g + 0.114*b
            text_color = 'black' if brightness > 0.5 else 'white'
            
            # 添加带轮廓的标签，提高可读性
            text = ax.text((x_start + x_end) / 2, 
                    (y_start + y_end) / 2,
                    emotion.capitalize(),
                    color=text_color,
                    fontsize=8,
                    fontweight='bold',
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
            """

    def _on_music_map_click(self, event):
        """处理地图点击事件"""
        if not hasattr(self, 'map_figure') or not hasattr(self, 'map_canvas'):
            return
        
        # 获取matplotlib的坐标
        try:
            ax = self.map_ax
            
            # 获取canvas的widget
            canvas_widget = self.map_canvas.get_tk_widget()
            
            # 使用相对于canvas的坐标，而不是屏幕坐标
            x, y = event.x, event.y
            
            # 将点击坐标转换为数据坐标
            inv = ax.transData.inverted()
            data_coord = inv.transform((x, y))
            
            map_x, map_y = int(data_coord[0]), int(data_coord[1])
            
            # 确保坐标在有效范围内
            map_x = max(0, min(map_x, self.map_data.width - 1))
            map_y = max(0, min(map_y, self.map_data.height - 1))
            
            # 更新位置信息
            self.logger.log(f"点击地图位置: ({map_x}, {map_y})")
            
            # 只保留这一行，删除错误的那一行调用
            self._update_music_player_position(map_x, map_y)
        except Exception as e:
            self.logger.log(f"处理地图点击出错: {e}", "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")

    def _on_music_map_drag(self, event):
        """处理地图拖拽事件"""
        # 如果没有必要的组件，直接返回
        if not hasattr(self, 'map_figure') or not hasattr(self, 'map_canvas'):
            return
        
        # 使用节流机制控制更新频率，避免频繁更新导致性能问题
        current_time = time.time()
        if hasattr(self, '_last_drag_update') and current_time - self._last_drag_update < 0.05:  # 限制为每50ms更新一次
            return
        
        # 更新时间戳
        self._last_drag_update = current_time
        
        # 使用相同的点击处理逻辑处理坐标转换和位置更新
        self._on_music_map_click(event)
        
        # 可以在这里添加拖拽特有的视觉反馈
        if hasattr(self, 'player_marker') and self.player_marker is not None:
            # 在拖拽时提供视觉反馈，例如临时改变标记颜色或大小
            self.player_marker.set_markeredgewidth(3)  # 加粗边框
            # 在拖拽结束后恢复正常状态
            self.map_canvas.draw_idle()
            
            # 使用定时器在拖拽后恢复正常状态
            def reset_marker():
                if hasattr(self, 'player_marker') and self.player_marker is not None:
                    self.player_marker.set_markeredgewidth(2)  # 恢复正常边框宽度
                    self.map_canvas.draw_idle()
            
            # 清除可能存在的先前计时器
            if hasattr(self, '_marker_reset_timer'):
                self.master.after_cancel(self._marker_reset_timer)
            
            # 设置新的恢复计时器
            self._marker_reset_timer = self.master.after(200, reset_marker)

    def _update_music_player_position(self, x, y):
        """更新音乐播放器位置，使用平滑过渡系统"""
        if not hasattr(self, 'player_marker') or self.player_marker is None:
            return
        
        # 更新标记位置
        self.player_marker.set_data([x], [y])
        self.map_canvas.draw_idle()  # 使用draw_idle比draw()更高效
        
        # 保存当前位置
        self.current_music_position = (x, y)
        
        # 使用过渡管理器更新位置
        if hasattr(self, 'music_transition_manager'):
            self.music_transition_manager.update_position(x, y)
        else:
            # 如果没有过渡管理器，直接更新位置
            self.music_generator.update_position(x, y)
        
        # 更新位置显示
        if hasattr(self, 'music_position_var'):
            self.music_position_var.set(f"X: {x}, Y: {y}")
        
        # 如果有自动移动线程，且手动点击，则禁用它
        if hasattr(self, 'auto_move_var') and self.auto_move_var.get():
            self.auto_move_var.set(False)
            
        # 更新音乐信息显示
        self._update_music_info_display()
        
        # 如果有过渡信息显示，更新它
        self._update_transition_info_display()

    def _on_music_map_resize(self):
        """处理音乐地图窗口大小调整"""
        # 需要重绘地图
        if hasattr(self, 'music_map_canvas') and self.music_map_canvas.winfo_exists():
            # 使用延迟重绘避免频繁刷新
            self.music_map_canvas.after_cancel(self._redraw_timer) if hasattr(self, '_redraw_timer') else None
            self._redraw_timer = self.music_map_canvas.after(200, self._redraw_music_map)

    def _redraw_music_map(self):
        """重绘音乐地图"""
        if hasattr(self, 'map_canvas'):
            for widget in self.music_map_canvas.winfo_children():
                widget.destroy()
            self._draw_music_map_and_regions()

    def _toggle_auto_move(self):
        """切换自动移动状态"""
        if not hasattr(self, 'auto_move_var'):
            return
        
        if self.auto_move_var.get():
            # 启动自动移动
            self._start_auto_move()
            # 更新移动信息
            self._update_move_info("自动移动已启动")
        else:
            # 停止自动移动
            self._stop_auto_move()
            # 更新移动信息
            self._update_move_info("自动移动已停止")

    def _start_auto_move(self):
        """启动自动移动功能"""
        if not hasattr(self, 'auto_move_task_id'):
            self.auto_move_task_id = self.master.after(100, self._auto_move_step)
            self.auto_move_target = None  # 初始化目标点

    def _stop_auto_move(self):
        """停止自动移动功能"""
        if hasattr(self, 'auto_move_task_id'):
            self.master.after_cancel(self.auto_move_task_id)
            delattr(self, 'auto_move_task_id')

    def _auto_move_step(self):
        """执行一步自动移动"""
        if not hasattr(self, 'player_marker') or self.player_marker is None:
            self._stop_auto_move()
            return
        
        if not hasattr(self, 'move_path_var') or not hasattr(self, 'move_speed_var'):
            self._stop_auto_move()
            return
        
        # 获取当前位置
        x, y = self.current_music_position
        
        # 获取移动速度和路径类型
        speed = self.move_speed_var.get() * 5  # 缩放速度
        path_mode = self.move_path_var.get()
        
        # 地图尺寸
        map_width, map_height = self.map_data.width, self.map_data.height
        
        # 根据路径模式计算下一个位置
        if path_mode == "随机漫步":
            # 随机移动
            dx = random.uniform(-1, 1) * speed
            dy = random.uniform(-1, 1) * speed
            new_x = max(0, min(map_width-1, x + int(dx)))
            new_y = max(0, min(map_height-1, y + int(dy)))
            
            self._update_move_info(f"随机漫步移动: ({x},{y}) -> ({new_x},{new_y})")
            
        elif path_mode == "沿河流":
            # 尝试沿着河流移动
            river_map = self.map_data.get_layer("river")
            
            if river_map is not None:
                # 在周围寻找河流
                search_radius = int(max(5, speed * 2))
                found_river = False
                
                for search_x in range(max(0, x - search_radius), min(map_width, x + search_radius + 1)):
                    for search_y in range(max(0, y - search_radius), min(map_height, y + search_radius + 1)):
                        if river_map[search_y, search_x] > 0:
                            # 找到河流点，向其移动
                            dx = search_x - x
                            dy = search_y - y
                            # 归一化方向向量
                            mag = (dx**2 + dy**2)**0.5
                            if mag > 0:
                                dx = dx / mag * speed
                                dy = dy / mag * speed
                            
                            new_x = max(0, min(map_width-1, x + int(dx)))
                            new_y = max(0, min(map_height-1, y + int(dy)))
                            found_river = True
                            
                            self._update_move_info(f"沿河流移动: ({x},{y}) -> ({new_x},{new_y})")
                            break
                    if found_river:
                        break
                
                if not found_river:
                    # 没找到河流，随机移动
                    dx = random.uniform(-1, 1) * speed
                    dy = random.uniform(-1, 1) * speed
                    new_x = max(0, min(map_width-1, x + int(dx)))
                    new_y = max(0, min(map_height-1, y + int(dy)))
                    
                    self._update_move_info(f"未找到河流，随机移动: ({x},{y}) -> ({new_x},{new_y})")
            else:
                # 没有河流层，随机移动
                dx = random.uniform(-1, 1) * speed
                dy = random.uniform(-1, 1) * speed
                new_x = max(0, min(map_width-1, x + int(dx)))
                new_y = max(0, min(map_height-1, y + int(dy)))
                
                self._update_move_info(f"无河流数据，随机移动: ({x},{y}) -> ({new_x},{new_y})")
                
        elif path_mode == "沿道路":
            # 尝试沿着道路移动
            road_map = self.map_data.get_layer("road")
            
            if road_map is not None:
                # 寻找附近的道路
                search_radius = int(max(5, speed * 2))
                found_road = False
                
                for search_x in range(max(0, x - search_radius), min(map_width, x + search_radius + 1)):
                    for search_y in range(max(0, y - search_radius), min(map_height, y + search_radius + 1)):
                        if road_map[search_y, search_x] > 0:
                            # 找到道路点，向其移动
                            dx = search_x - x
                            dy = search_y - y
                            # 归一化方向向量
                            mag = (dx**2 + dy**2)**0.5
                            if mag > 0:
                                dx = dx / mag * speed
                                dy = dy / mag * speed
                            
                            new_x = max(0, min(map_width-1, x + int(dx)))
                            new_y = max(0, min(map_height-1, y + int(dy)))
                            found_road = True
                            
                            self._update_move_info(f"沿道路移动: ({x},{y}) -> ({new_x},{new_y})")
                            break
                    if found_road:
                        break
                
                if not found_road:
                    # 没找到道路，随机移动
                    dx = random.uniform(-1, 1) * speed
                    dy = random.uniform(-1, 1) * speed
                    new_x = max(0, min(map_width-1, x + int(dx)))
                    new_y = max(0, min(map_height-1, y + int(dy)))
                    
                    self._update_move_info(f"未找到道路，随机移动: ({x},{y}) -> ({new_x},{new_y})")
            else:
                # 没有道路层，随机移动
                dx = random.uniform(-1, 1) * speed
                dy = random.uniform(-1, 1) * speed
                new_x = max(0, min(map_width-1, x + int(dx)))
                new_y = max(0, min(map_height-1, y + int(dy)))
                
                self._update_move_info(f"无道路数据，随机移动: ({x},{y}) -> ({new_x},{new_y})")
                
        else:  # 全图探索
            # 选择或更新目标点
            if not hasattr(self, 'auto_move_target') or self.auto_move_target is None:
                # 选择新目标
                if hasattr(self, 'target_emotion_var') and self.target_emotion_var.get() != "自动":
                    # 移动到指定情感区域
                    target_emotion = self.target_emotion_var.get()
                    target_point = self._find_emotion_area_center(target_emotion)
                    if target_point:
                        self.auto_move_target = target_point
                        self._update_move_info(f"向情感区域移动: {target_emotion} 位置: {target_point}")
                    else:
                        # 如果找不到指定情感区域，随机选择
                        self.auto_move_target = (random.randint(0, map_width-1), random.randint(0, map_height-1))
                        self._update_move_info(f"未找到情感区域 {target_emotion}，随机选择目标: {self.auto_move_target}")
                else:
                    # 随机选择目标
                    self.auto_move_target = (random.randint(0, map_width-1), random.randint(0, map_height-1))
                    self._update_move_info(f"随机选择目标点: {self.auto_move_target}")
                    
            # 向目标移动
            target_x, target_y = self.auto_move_target
            
            # 检查是否接近目标
            dist_to_target = ((x - target_x)**2 + (y - target_y)**2)**0.5
            if dist_to_target < speed:
                # 到达目标，选择新目标
                self.auto_move_target = None
                self._update_move_info(f"已到达目标点 ({target_x},{target_y})，将选择新目标")
                
                # 递归调用自己继续移动
                self._auto_move_step()
                return
            
            # 计算移动方向
            dx = target_x - x
            dy = target_y - y
            
            # 归一化方向向量
            mag = (dx**2 + dy**2)**0.5
            if mag > 0:
                dx = dx / mag * speed
                dy = dy / mag * speed
            
            new_x = max(0, min(map_width-1, x + int(dx)))
            new_y = max(0, min(map_height-1, y + int(dy)))
            
            self._update_move_info(f"向目标点移动: ({x},{y}) -> ({new_x},{new_y}) [目标:{self.auto_move_target}]")
        
        # 更新玩家位置
        self._update_music_player_position(new_x, new_y)
        
        # 设置下一次移动
        self.auto_move_task_id = self.master.after(100, self._auto_move_step)

    def _find_emotion_area_center(self, emotion):
        """查找指定情感区域的中心点"""
        if not hasattr(self.music_generator, 'emotion_regions') or not self.music_generator.emotion_regions:
            return None
        
        matching_regions = []
        
        # 寻找所有匹配情感的区域
        for region_id, region_data in self.music_generator.emotion_regions.items():
            if region_data.get("emotion", "").lower() == emotion.lower():
                bb = region_data.get("bounding_box", (0, 0, 0, 0))
                x_start, x_end, y_start, y_end = bb
                intensity = region_data.get("intensity", 0.5)
                
                # 计算中心点
                center_x = (x_start + x_end) // 2
                center_y = (y_start + y_end) // 2
                
                matching_regions.append((center_x, center_y, intensity))
        
        if not matching_regions:
            return None
        
        # 根据情感强度选择最强的区域
        matching_regions.sort(key=lambda x: x[2], reverse=True)
        return (matching_regions[0][0], matching_regions[0][1])

    def _update_move_info(self, message):
        """更新移动信息显示"""
        if hasattr(self, 'move_info_text') and self.move_info_text.winfo_exists():
            try:
                self.move_info_text.config(state=tk.NORMAL)
                
                # 保持最多显示10行
                text_content = self.move_info_text.get(1.0, tk.END).strip().split('\n')
                if len(text_content) > 9:
                    text_content = text_content[-9:]
                
                # 更新内容
                self.move_info_text.delete(1.0, tk.END)
                for line in text_content:
                    if line.strip():
                        self.move_info_text.insert(tk.END, line + '\n')
                
                # 添加新信息
                self.move_info_text.insert(tk.END, message + '\n')
                self.move_info_text.see(tk.END)
                
                self.move_info_text.config(state=tk.DISABLED)
            except Exception as e:
                self.logger.log(f"更新移动信息错误: {e}", "ERROR")

    def _move_to_emotion_area(self):
        """移动到选择的情感区域"""
        if not hasattr(self, 'target_emotion_var'):
            return
        
        # 获取选择的情感
        target_emotion = self.target_emotion_var.get()
        if target_emotion == "自动":
            self._update_move_info("请选择一个具体的情感区域")
            return
        
        # 查找情感区域中心
        target_point = self._find_emotion_area_center(target_emotion)
        if not target_point:
            self._update_move_info(f"未找到情感区域: {target_emotion}")
            return
        
        # 更新位置
        x, y = target_point
        self._update_music_player_position(x, y)
        self._update_move_info(f"已移动到情感区域: {target_emotion} 位置: {(x, y)}")

    def _update_music_info_display(self):
        """更新音乐信息显示"""
        if not hasattr(self, 'music_generator') or not self.music_generator.is_playing:
            return
        
        # 获取当前播放的区域信息
        playing_regions = self.music_generator.get_currently_playing_regions()
        
        # 更新活跃区域列表
        if hasattr(self, 'regions_listbox'):
            self.regions_listbox.delete(0, tk.END)
            for region in playing_regions:
                self.regions_listbox.insert(tk.END, f"{region['emotion']} ({region['weight']:.2f})")
        
        # 如果有区域在播放，更新主导情感
        if playing_regions and hasattr(self, 'music_emotion_var'):
            main_region = playing_regions[0]
            self.music_emotion_var.set(f"{main_region['emotion']} ({main_region['weight']:.2f})")
        
        # 更新音乐参数
        if hasattr(self, 'music_params_var'):
            music_params = self.music_generator.get_current_music_params()
            if music_params:
                param_text = f"速度: {music_params.get('tempo', 0):.1f}BPM, "
                param_text += f"调式: {music_params.get('scale', 'major')}, "
                param_text += f"混响: {music_params.get('reverb', 0):.2f}"
                self.music_params_var.set(param_text)

    def _setup_basic_music_controls(self, parent_frame, visualization_frame):
        """设置基本音乐控制界面"""
        # 音乐信息框架
        info_frame = ttk.LabelFrame(parent_frame, text="当前音乐信息")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(info_frame, text="当前位置:").pack(anchor=tk.W, padx=5, pady=2)
        self.music_position_var = tk.StringVar(value="X: 0, Y: 0")
        ttk.Label(info_frame, textvariable=self.music_position_var).pack(anchor=tk.W, padx=20, pady=2)
        
        ttk.Label(info_frame, text="主导情感:").pack(anchor=tk.W, padx=5, pady=2)
        self.music_emotion_var = tk.StringVar(value="无")
        ttk.Label(info_frame, textvariable=self.music_emotion_var).pack(anchor=tk.W, padx=20, pady=2)
        
        ttk.Label(info_frame, text="音乐参数:").pack(anchor=tk.W, padx=5, pady=2)
        self.music_params_var = tk.StringVar(value="未播放")
        ttk.Label(info_frame, textvariable=self.music_params_var, wraplength=300).pack(anchor=tk.W, padx=20, pady=2)
        
        # 当前播放区域
        regions_frame = ttk.LabelFrame(parent_frame, text="活跃音乐区域")
        regions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 创建活跃区域列表
        self.regions_listbox = tk.Listbox(regions_frame, height=5)
        self.regions_listbox.pack(fill=tk.X, padx=5, pady=5)
        
        # 播放控制
        control_frame = ttk.LabelFrame(parent_frame, text="播放控制")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 控制按钮框架
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 加载资源按钮
        load_btn = ttk.Button(buttons_frame, text="加载音乐资源", 
                            command=lambda: self._load_music_resources())
        load_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # 播放/暂停按钮
        self.play_var = tk.StringVar(value="开始播放")
        play_btn = ttk.Button(buttons_frame, textvariable=self.play_var,
                            command=lambda: self._toggle_music_playback(self.play_var))
        play_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # 停止按钮
        stop_btn = ttk.Button(buttons_frame, text="停止播放",
                            command=self._stop_music_playback)
        stop_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        # 音量控制
        volume_frame = ttk.Frame(control_frame)
        volume_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(volume_frame, text="主音量:").pack(side=tk.LEFT, padx=5)
        self.master_volume_var = tk.DoubleVar(value=1.0)
        volume_slider = ttk.Scale(volume_frame, from_=0.0, to=1.0, 
                                variable=self.master_volume_var, orient=tk.HORIZONTAL,
                                command=self._update_master_volume)
        volume_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 添加轨道混音器
        mixer_frame = ttk.LabelFrame(parent_frame, text="音轨混音器")
        mixer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建轨道控制
        self.track_sliders = {}
        tracks = ["旋律", "和声", "节奏", "氛围", "效果"]
        
        for i, track in enumerate(tracks):
            track_frame = ttk.Frame(mixer_frame)
            track_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(track_frame, text=f"{track}:", width=8).pack(side=tk.LEFT, padx=5)
            
            # 音量滑块
            self.track_sliders[track] = tk.DoubleVar(value=0.8)
            track_slider = ttk.Scale(track_frame, from_=0.0, to=1.0, 
                                variable=self.track_sliders[track], orient=tk.HORIZONTAL)
            track_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # 静音按钮
            mute_var = tk.BooleanVar(value=False)
            mute_btn = ttk.Checkbutton(track_frame, text="静音", variable=mute_var)
            mute_btn.pack(side=tk.LEFT, padx=5)
            
            # 绑定音量滑块的更新事件
            self.track_sliders[track].trace("w", lambda *args, t=track: self._update_track_volume(t))
        
        # 轮询更新UI信息
        self._schedule_music_ui_update(visualization_frame)

    def _update_master_volume(self, *args):
        """更新主音量"""
        if hasattr(self, 'music_generator') and hasattr(self.music_generator, 'music_channels'):
            volume = self.master_volume_var.get()
            # 设置所有通道的音量
            for channel_name, channel in self.music_generator.music_channels.items():
                try:
                    channel.set_volume(volume)
                except Exception as e:
                    self.logger.log(f"设置{channel_name}音量时出错: {str(e)}", "ERROR")
            
            self.logger.log(f"主音量已设置为: {volume:.2f}")

    def _setup_collaboration_mode_controls(self, parent_frame):
        """设置音乐模型协作模式控制"""
        # 协作模式选择
        mode_frame = ttk.LabelFrame(parent_frame, text="协作模式")
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.collab_mode_var = tk.StringVar(value="并行生成")
        modes = ["并行生成", "分层生成", "混合生成", "强化学习协作"]
        
        for i, mode in enumerate(modes):
            ttk.Radiobutton(
                mode_frame, 
                text=mode, 
                variable=self.collab_mode_var, 
                value=mode
            ).pack(anchor=tk.W, padx=20, pady=2)
        
        # 模式说明
        desc_frame = ttk.LabelFrame(parent_frame, text="模式说明")
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        mode_descriptions = {
            "并行生成": "多个模型同时生成不同音轨（如旋律、节奏、和声），适合多声部音乐。",
            "分层生成": "按顺序生成音乐结构→旋律→和声→编曲，适合复杂结构音乐。",
            "混合生成": "融合多个模型的输出结果，适合风格创新和用户交互式生成。",
            "强化学习协作": "根据环境反馈动态调整音乐，适合游戏和交互场景。"
        }
        
        self.mode_desc_var = tk.StringVar(value=mode_descriptions["并行生成"])
        desc_label = ttk.Label(desc_frame, textvariable=self.mode_desc_var, wraplength=300, justify="left")
        desc_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 切换模式描述
        def update_mode_desc(*args):
            mode = self.collab_mode_var.get()
            self.mode_desc_var.set(mode_descriptions.get(mode, ""))
            self._update_music_generation_method(mode)
        
        self.collab_mode_var.trace("w", update_mode_desc)
        
        # 模型选择
        models_frame = ttk.LabelFrame(parent_frame, text="可用模型")
        models_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 添加可用模型复选框
        self.model_vars = {}
        models = {
            "MusicVAE": "Google Magenta 的变分自编码器，擅长旋律生成",
            "GrooVAE": "节奏生成模型，适合打击乐和鼓点",
            "Transformer": "基于注意力机制的序列模型，适合长旋律",
            "GANSynth": "生成对抗网络合成器，生成高质量声音",
            "NSynth": "神经网络音频合成，适合音色生成"
        }
        
        for i, (model, desc) in enumerate(models.items()):
            model_frame = ttk.Frame(models_frame)
            model_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.model_vars[model] = tk.BooleanVar(value=model in ["MusicVAE", "GrooVAE"])
            check = ttk.Checkbutton(model_frame, text=model, variable=self.model_vars[model])
            check.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(model_frame, text=desc, wraplength=250).pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)


    def _setup_music_parameter_controls(self, parent_frame):
        """设置音乐参数控制界面"""
        # 音乐参数控制
        canvas = tk.Canvas(parent_frame)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 基础参数
        basic_params_frame = ttk.LabelFrame(scrollable_frame, text="基础参数")
        basic_params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 速度控制
        tempo_frame = ttk.Frame(basic_params_frame)
        tempo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(tempo_frame, text="速度 (BPM):").pack(side=tk.LEFT, padx=5)
        self.tempo_var = tk.IntVar(value=100)
        tempo_slider = ttk.Scale(tempo_frame, from_=60, to=180, 
                            variable=self.tempo_var, orient=tk.HORIZONTAL)
        tempo_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tempo_value = ttk.Label(tempo_frame, textvariable=self.tempo_var, width=4)
        tempo_value.pack(side=tk.LEFT, padx=5)
        
        # 调式控制
        scale_frame = ttk.Frame(basic_params_frame)
        scale_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(scale_frame, text="调式:").pack(side=tk.LEFT, padx=5)
        self.scale_var = tk.StringVar(value="major")
        scales = ["major", "minor", "pentatonic", "blues", "dorian", "phrygian", "lydian", "mixolydian"]
        scale_combo = ttk.Combobox(scale_frame, textvariable=self.scale_var, 
                                values=scales, state="readonly", width=15)
        scale_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 生成参数
        gen_params_frame = ttk.LabelFrame(scrollable_frame, text="生成参数")
        gen_params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 复杂度
        complexity_frame = ttk.Frame(gen_params_frame)
        complexity_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(complexity_frame, text="复杂度:").pack(side=tk.LEFT, padx=5)
        self.complexity_var = tk.DoubleVar(value=0.5)
        complexity_slider = ttk.Scale(complexity_frame, from_=0.0, to=1.0, 
                                variable=self.complexity_var, orient=tk.HORIZONTAL)
        complexity_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 温度
        temperature_frame = ttk.Frame(gen_params_frame)
        temperature_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(temperature_frame, text="随机性:").pack(side=tk.LEFT, padx=5)
        self.temperature_var = tk.DoubleVar(value=0.8)
        temperature_slider = ttk.Scale(temperature_frame, from_=0.1, to=1.5, 
                                variable=self.temperature_var, orient=tk.HORIZONTAL)
        temperature_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 情感响应参数
        emotion_params_frame = ttk.LabelFrame(scrollable_frame, text="情感响应参数")
        emotion_params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 情感敏感度
        sensitivity_frame = ttk.Frame(emotion_params_frame)
        sensitivity_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sensitivity_frame, text="敏感度:").pack(side=tk.LEFT, padx=5)
        self.sensitivity_var = tk.DoubleVar(value=0.7)
        sensitivity_slider = ttk.Scale(sensitivity_frame, from_=0.1, to=1.0, 
                                variable=self.sensitivity_var, orient=tk.HORIZONTAL)
        sensitivity_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 过渡速度
        transition_frame = ttk.Frame(emotion_params_frame)
        transition_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(transition_frame, text="过渡速度:").pack(side=tk.LEFT, padx=5)
        self.transition_var = tk.DoubleVar(value=0.5)
        transition_slider = ttk.Scale(transition_frame, from_=0.1, to=1.0, 
                                variable=self.transition_var, orient=tk.HORIZONTAL)
        transition_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 效果参数
        effect_params_frame = ttk.LabelFrame(scrollable_frame, text="音效参数")
        effect_params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 混响
        reverb_frame = ttk.Frame(effect_params_frame)
        reverb_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(reverb_frame, text="混响:").pack(side=tk.LEFT, padx=5)
        self.reverb_var = tk.DoubleVar(value=0.3)
        reverb_slider = ttk.Scale(reverb_frame, from_=0.0, to=1.0, 
                            variable=self.reverb_var, orient=tk.HORIZONTAL)
        reverb_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 延迟
        delay_frame = ttk.Frame(effect_params_frame)
        delay_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(delay_frame, text="延迟:").pack(side=tk.LEFT, padx=5)
        self.delay_var = tk.DoubleVar(value=0.2)
        delay_slider = ttk.Scale(delay_frame, from_=0.0, to=1.0, 
                            variable=self.delay_var, orient=tk.HORIZONTAL)
        delay_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 应用参数按钮
        ttk.Button(
            scrollable_frame, 
            text="应用参数", 
            command=self._apply_music_parameters
        ).pack(padx=10, pady=10)

    def _apply_music_parameters(self):
        """应用音乐参数到生成器"""
        if not hasattr(self, 'tempo_var'):
            return
        
        # 收集当前参数
        params = {
            "tempo": self.tempo_var.get(),
            "scale": self.scale_var.get(),
            "complexity": self.complexity_var.get(),
            "temperature": self.temperature_var.get(),
            "sensitivity": self.sensitivity_var.get(),
            "transition_speed": self.transition_var.get(),
            "reverb": self.reverb_var.get(),
            "delay": self.delay_var.get()
        }
        
        # 检查音乐生成器是否可用
        if not hasattr(self, 'music_generator'):
            self.logger.log("音乐生成器未初始化，无法应用参数", "ERROR")
            return
        
        try:
            # 转换UI参数为生成器参数格式
            generator_params = {
                "tempo": params["tempo"],
                "scale": params["scale"],
                "density": params["complexity"],
                "variation": params["temperature"],
                "emotion_sensitivity": params["sensitivity"],
                "transition_rate": params["transition_speed"],
                "effects": {
                    "reverb": params["reverb"],
                    "delay": params["delay"]
                }
            }
            
            # 应用参数到生成器
            self.music_generator.set_parameters(generator_params)
            
            # 更新UI显示的参数信息
            self.music_params_var.set(
                f"节奏: {params['tempo']}BPM, "
                f"调式: {params['scale']}, "
                f"混响: {params['reverb']:.2f}"
            )
            
            # 记录操作
            self.logger.log("已应用新的音乐参数")
            
            # 更新音频可视化
            if hasattr(self, 'audio_canvas'):
                self._update_audio_visualization(None)
                
        except AttributeError as e:
            self.logger.log(f"音乐生成器缺少必要方法: {e}", "ERROR")
        except Exception as e:
            self.logger.log(f"应用音乐参数时出错: {e}", "ERROR")

    def _setup_music_visualization_controls(self, parent_frame, visualization_frame):
        """设置音乐可视化控制界面"""
        # 可视化类型选择
        vis_type_frame = ttk.LabelFrame(parent_frame, text="可视化类型")
        vis_type_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.vis_type_var = tk.StringVar(value="频谱图")
        vis_types = ["频谱图", "波形图", "情感映射", "音符流", "3D效果"]
        
        for i, vis_type in enumerate(vis_types):
            ttk.Radiobutton(
                vis_type_frame, 
                text=vis_type, 
                variable=self.vis_type_var, 
                value=vis_type
            ).pack(anchor=tk.W, padx=20, pady=2)
        
        # 可视化参数
        vis_params_frame = ttk.LabelFrame(parent_frame, text="可视化参数")
        vis_params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 颜色方案
        color_frame = ttk.Frame(vis_params_frame)
        color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(color_frame, text="颜色方案:").pack(side=tk.LEFT, padx=5)
        self.color_scheme_var = tk.StringVar(value="彩虹")
        color_schemes = ["彩虹", "热图", "蓝调", "情感映射", "单色"]
        color_combo = ttk.Combobox(color_frame, textvariable=self.color_scheme_var, 
                                values=color_schemes, state="readonly", width=15)
        color_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 更新速率
        update_frame = ttk.Frame(vis_params_frame)
        update_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(update_frame, text="更新速率:").pack(side=tk.LEFT, padx=5)
        self.update_rate_var = tk.IntVar(value=30)
        update_slider = ttk.Scale(update_frame, from_=10, to=60, 
                            variable=self.update_rate_var, orient=tk.HORIZONTAL)
        update_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        update_value = ttk.Label(update_frame, textvariable=self.update_rate_var, width=4)
        update_value.pack(side=tk.LEFT, padx=5)
        
        # 连接可视化类型变更事件
        self.vis_type_var.trace("w", lambda *args: self._update_visualization_type(visualization_frame))
        
        # 截图按钮
        ttk.Button(
            parent_frame,
            text="截图保存",
            command=self._save_visualization_image
        ).pack(padx=10, pady=10)
        
        # 导出录制按钮
        self.record_var = tk.StringVar(value="开始录制")
        self.recording = False
        
        ttk.Button(
            parent_frame,
            textvariable=self.record_var,
            command=self._toggle_visualization_recording
        ).pack(padx=10, pady=5)


    def _setup_map_preview_for_music(self, parent_frame):
        """设置音乐系统的地图预览"""
        # 创建地图画布
        self.music_map_canvas = tk.Canvas(parent_frame, bg="white", cursor="hand2")
        self.music_map_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 获取地图数据
        height_map = self.map_data.get_layer("height")
        biome_map = self.map_data.get_layer("biome")
        
        if height_map is None or biome_map is None:
            ttk.Label(parent_frame, text="没有地图数据可显示").pack(expand=True)
            return
        
        # 计算缩放因子
        map_height, map_width = height_map.shape
        canvas_height = 400
        canvas_width = 600
        scale_y = canvas_height / map_height
        scale_x = canvas_width / map_width
        self.music_map_scale = min(scale_x, scale_y) * 0.9  # 留出边距
        
        # 绘制地图
        try:
            # 绘制地形
            img = get_combined_preview_image(self.map_data, max_size=(canvas_width, canvas_height))
            if img:
                self.music_map_image = ImageTk.PhotoImage(image=img)
                self.music_map_canvas.create_image(
                    canvas_width/2, canvas_height/2, 
                    image=self.music_map_image, 
                    tags="map_image"
                )
                
                # 绘制情感区域
                self._draw_emotion_regions()
                
                # 创建玩家位置标记
                self.player_marker = self.music_map_canvas.create_oval(
                    -10, -10, -5, -5,  # 初始位置在画布外
                    fill="red", outline="black", width=2,
                    tags="player"
                )
                
                # 绑定鼠标事件 - 点击地图移动位置
                self.music_map_canvas.bind("<Button-1>", self._on_map_click)
                
                # 当前位置显示 (初始位置设为地图中心)
                self.current_music_position = (map_width // 2, map_height // 2)
                self._update_player_marker()
                
                # 更新音乐生成器位置
                self.music_generator.update_position(*self.current_music_position)
                
            else:
                ttk.Label(parent_frame, text="地图渲染失败").pack(expand=True)
                
        except Exception as e:
            ttk.Label(parent_frame, text=f"地图渲染错误: {e}").pack(expand=True)


    def _draw_emotion_regions(self):
        """在地图上绘制情感区域"""
        # 获取情感区域数据
        emotion_regions = self.music_generator.emotion_regions
        if not emotion_regions:
            return
        
        # 清除现有区域
        self.music_map_canvas.delete("emotion_region")
        
        # 情感颜色映射
        emotion_colors = {
            "joy": "#90EE90",       # 淡绿色
            "trust": "#ADD8E6",     # 淡蓝色
            "fear": "#FFB6C1",      # 淡红色
            "surprise": "#FFFF99",  # 淡黄色
            "sadness": "#D3D3D3",   # 淡灰色
            "disgust": "#E6E6FA",   # 淡紫色
            "anger": "#FFA07A",     # 浅珊瑚色
            "anticipation": "#FFDAB9"  # 桃色
        }
        
        # 获取画布大小
        canvas_width = self.music_map_canvas.winfo_width()
        canvas_height = self.music_map_canvas.winfo_height()
        
        # 绘制每个情感区域
        for region_id, region_data in emotion_regions.items():
            bb = region_data.get("bounding_box", (0, 0, 0, 0))
            x_start, x_end, y_start, y_end = bb
            
            # 转换坐标到画布空间
            cx1 = int(x_start * self.music_map_scale + canvas_width/2 - self.map_data.width * self.music_map_scale / 2)
            cy1 = int(y_start * self.music_map_scale + canvas_height/2 - self.map_data.height * self.music_map_scale / 2)
            cx2 = int(x_end * self.music_map_scale + canvas_width/2 - self.map_data.width * self.music_map_scale / 2)
            cy2 = int(y_end * self.music_map_scale + canvas_height/2 - self.map_data.height * self.music_map_scale / 2)
            
            # 获取情感类型和强度
            emotion = region_data.get("emotion", "")
            intensity = region_data.get("intensity", 0.5)
            
            # 获取颜色，默认为灰色
            color = emotion_colors.get(emotion.lower(), "#CCCCCC")
            
            # 创建具有透明度的矩形
            self.music_map_canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=color,
                fill=color,
                stipple="gray50",  # 使用点画效果创建半透明
                width=2,
                tags=("emotion_region", region_id)
            )
            
            # 显示区域标签
            self.music_map_canvas.create_text(
                (cx1 + cx2) / 2, (cy1 + cy2) / 2,
                text=emotion.capitalize(),
                font=("Arial", 8),
                fill="black",
                tags=("emotion_region", f"{region_id}_label")
            )

    def _on_map_click(self, event):
        """处理地图点击事件，移动播放位置"""
        canvas_width = self.music_map_canvas.winfo_width()
        canvas_height = self.music_map_canvas.winfo_height()
        
        # 转换画布坐标到地图坐标
        map_x = int((event.x - (canvas_width/2 - self.map_data.width * self.music_map_scale / 2)) / self.music_map_scale)
        map_y = int((event.y - (canvas_height/2 - self.map_data.height * self.music_map_scale / 2)) / self.music_map_scale)
        
        # 范围检查
        map_x = max(0, min(map_x, self.map_data.width - 1))
        map_y = max(0, min(map_y, self.map_data.height - 1))
        
        # 更新当前位置
        self.current_music_position = (map_x, map_y)
        self._update_player_marker()
        
        # 更新音乐生成器位置
        self.music_generator.update_position(map_x, map_y)
        
        # 更新位置信息
        self.music_position_var.set(f"X: {map_x}, Y: {map_y}")


    def _update_player_marker(self):
        """更新玩家位置标记"""
        canvas_width = self.music_map_canvas.winfo_width()
        canvas_height = self.music_map_canvas.winfo_height()
        
        map_x, map_y = self.current_music_position
        
        # 转换地图坐标到画布坐标
        cx = int(map_x * self.music_map_scale + canvas_width/2 - self.map_data.width * self.music_map_scale / 2)
        cy = int(map_y * self.music_map_scale + canvas_height/2 - self.map_data.height * self.music_map_scale / 2)
        
        # 标记大小
        size = 6
        
        # 更新标记位置
        self.music_map_canvas.coords(
            self.player_marker,
            cx - size, cy - size, cx + size, cy + size
        )


    def _setup_audio_visualization(self, parent_frame):
        """设置音频可视化显示"""
        # 创建用于音频可视化的画布
        self.audio_canvas = tk.Canvas(parent_frame, bg="black", height=150)
        self.audio_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 初始化可视化数据
        self.vis_data = {
            "type": "频谱图",
            "color_scheme": "彩虹",
            "bars": 64,
            "update_rate": 30,
            "last_update": time.time(),
            "recording": False,
            "frames": []
        }
        
        # 创建初始可视化效果
        self._create_visualization_placeholder()


    def _create_visualization_placeholder(self):
        """创建可视化占位图形"""
        width = self.audio_canvas.winfo_width() or 400
        height = self.audio_canvas.winfo_height() or 150
        
        # 清除画布
        self.audio_canvas.delete("all")
        
        # 绘制占位文本
        self.audio_canvas.create_text(
            width/2, height/2,
            text="开始播放音乐查看可视化效果",
            fill="white",
            font=("Arial", 12),
            tags="placeholder"
        )


    def _update_visualization_type(self, visualization_frame):
        """更新可视化类型"""
        # 更新可视化类型
        self.vis_data["type"] = self.vis_type_var.get()
        
        # 可视化类型变更时，清除画布重新绘制
        if hasattr(self, 'audio_canvas'):
            self._create_visualization_placeholder()


    def _save_visualization_image(self):
        """保存可视化图像"""
        # 打开保存对话框
        file_path = filedialog.asksaveasfilename(
            title="保存可视化图像",
            filetypes=[("PNG图像", "*.png"), ("所有文件", "*.*")],
            defaultextension=".png"
        )
        
        if not file_path:
            return
        
        try:
            # 将画布转换为图像
            if hasattr(self, 'audio_canvas'):
                width = self.audio_canvas.winfo_width()
                height = self.audio_canvas.winfo_height()
                
                # 创建图像
                image = Image.new("RGB", (width, height), "black")
                
                # 获取画布内容作为PostScript
                ps_data = self.audio_canvas.postscript(colormode="color")
                
                # 转换PostScript为图像
                ps_image = Image.open(io.BytesIO(ps_data.encode("utf-8")))
                image.paste(ps_image)
                
                # 保存图像
                image.save(file_path)
                
                messagebox.showinfo("保存成功", f"可视化图像已保存到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存图像时出错:\n{e}")


    def _toggle_visualization_recording(self):
        """切换可视化录制状态"""
        if not hasattr(self, 'vis_data'):
            return
        
        if not self.recording:
            # 开始录制
            self.recording = True
            self.vis_data["recording"] = True
            self.vis_data["frames"] = []
            self.record_var.set("停止录制")
            messagebox.showinfo("录制开始", "开始录制可视化效果，再次点击按钮停止录制")
        else:
            # 停止录制
            self.recording = False
            self.vis_data["recording"] = False
            
            # 如果没有帧，直接返回
            if not self.vis_data["frames"]:
                self.record_var.set("开始录制")
                messagebox.showinfo("录制取消", "没有捕获到任何帧")
                return
            
            # 询问是否保存
            if messagebox.askyesno("录制完成", "要保存录制的动画吗?"):
                self._save_visualization_animation()
            
            self.record_var.set("开始录制")


    def _save_visualization_animation(self):
        """保存可视化动画"""
        # 打开保存对话框
        file_path = filedialog.asksaveasfilename(
            title="保存可视化动画",
            filetypes=[("GIF动画", "*.gif"), ("所有文件", "*.*")],
            defaultextension=".gif"
        )
        
        if not file_path:
            return
        
        try:
            # 保存帧为GIF
            if self.vis_data["frames"]:
                frames = self.vis_data["frames"]
                
                # 使用第一帧尺寸
                frame_duration = 1000 // self.update_rate_var.get()  # 毫秒
                
                frames[0].save(
                    file_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=frame_duration,
                    loop=0
                )
                
                messagebox.showinfo("保存成功", f"可视化动画已保存到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存动画时出错:\n{e}")


    def _update_track_volume(self, track_name):
        """更新音轨音量"""
        if not hasattr(self, 'track_sliders'):
            return
        
        volume = self.track_sliders[track_name].get()
        
        # 连接到音乐生成器的音轨控制
        if hasattr(self, 'music_generator') and self.music_generator.is_playing:
            # 将界面音轨名称映射到音乐生成器的通道名称
            channel_map = {
                "旋律": "melody",
                "和声": "harmony", 
                "节奏": "percussion",
                "氛围": "ambient",
                "效果": "effects"
            }
            
            if track_name in channel_map and hasattr(self.music_generator, 'music_channels'):
                channel_name = channel_map[track_name]
                if channel_name in self.music_generator.music_channels:
                    channel = self.music_generator.music_channels[channel_name]
                    channel.set_volume(volume)
                    self.logger.log(f"音轨 '{track_name}' 音量已设置为 {volume:.2f}")
            else:
                self.logger.log(f"无法找到对应的音频通道: {track_name}", "WARNING")


    def _update_music_generation_method(self, mode):
        """更新音乐生成方法"""
        # 根据不同的协作模式调整生成策略
        self.logger.log(f"切换到音乐生成模式: {mode}")
        
        if not hasattr(self, 'music_generator'):
            self.logger.log("音乐生成器未初始化，无法更新生成模式", "ERROR")
            return
        
        try:
            # 为不同模式配置生成器
            if mode == "并行生成":
                # 并行模式 - 多个模型同时生成不同音轨
                self.music_generator.set_generation_mode("parallel")
                
                # 更新激活的模型
                active_models = [model for model, var in self.model_vars.items() 
                                if var.get()]
                if active_models:
                    self.music_generator.set_active_models(active_models)
                
            elif mode == "分层生成":
                # 分层模式 - 按顺序生成音乐结构
                self.music_generator.set_generation_mode("layered")
                
                # 配置生成顺序
                generation_order = ["structure", "melody", "harmony", "arrangement"]
                self.music_generator.set_generation_order(generation_order)
                
            elif mode == "混合生成":
                # 混合模式 - 融合多模型输出
                self.music_generator.set_generation_mode("hybrid")
                
                # 配置混合参数
                if hasattr(self, 'complexity_var'):
                    blend_factor = self.complexity_var.get()
                    self.music_generator.set_blend_factor(blend_factor)
                    
            elif mode == "强化学习协作":
                # RL协作模式 - 动态调整
                self.music_generator.set_generation_mode("reinforcement")
                
                # 启用环境反馈机制
                self.music_generator.enable_feedback(True)
            
            # 如果当前正在播放，应用新的生成模式
            if self.music_generator.is_playing:
                # 记录更新状态
                self.status_var.set(f"已更新音乐生成模式: {mode}")
                
                # 更新参数后刷新UI
                self._update_music_info_display()
                
        except AttributeError as e:
            self.logger.log(f"音乐生成器缺少必要方法: {e}", "ERROR")
        except Exception as e:
            self.logger.log(f"更新音乐生成模式时出错: {e}", "ERROR")

    def _schedule_music_ui_update(self, visualization_frame):
        """定时更新音乐UI信息"""
        def update_ui():
            # 如果音乐正在播放，更新信息
            if hasattr(self, 'music_generator') and self.music_generator.is_playing:
                # 获取当前播放区域信息
                playing_regions = self.music_generator.get_currently_playing_regions()
                
                # 更新区域列表
                if hasattr(self, 'regions_listbox'):
                    self.regions_listbox.delete(0, tk.END)
                    for region in playing_regions:
                        self.regions_listbox.insert(tk.END, f"{region['emotion']} ({region['weight']:.2f})")
                
                # 如果有区域在播放，更新主导情感
                if playing_regions:
                    main_region = playing_regions[0]
                    self.music_emotion_var.set(f"{main_region['emotion']} ({main_region['weight']:.2f})")
                else:
                    self.music_emotion_var.set("无")
                
                # 更新音乐参数
                music_params = self.music_generator.get_current_music_params()
                if music_params:
                    param_text = f"节奏: {music_params.get('tempo', 0):.1f}BPM, "
                    param_text += f"调式: {music_params.get('scale', 'major')}, "
                    param_text += f"混响: {music_params.get('reverb', 0):.2f}"
                    self.music_params_var.set(param_text)
                
                # 更新可视化
                self._update_audio_visualization(visualization_frame)
            
            # 每100ms更新一次
            self.master.after(100, update_ui)
        
        # 开始更新循环
        self.master.after(100, update_ui)


    def _update_audio_visualization(self, visualization_frame):
        """更新音频可视化显示"""
        if not hasattr(self, 'audio_canvas') or not hasattr(self, 'vis_data'):
            return
        
        # 检查是否到了更新时间
        current_time = time.time()
        if current_time - self.vis_data["last_update"] < 1.0 / self.vis_data["update_rate"]:
            return
        
        self.vis_data["last_update"] = current_time
        
        # 获取画布尺寸
        width = self.audio_canvas.winfo_width()
        height = self.audio_canvas.winfo_height()
        
        # 如果尺寸太小或未初始化，不进行更新
        if width < 10 or height < 10:
            return
        
        # 清除画布上的旧内容
        self.audio_canvas.delete("visualization")
        
        # 音频数据处理
        num_bars = self.vis_data["bars"]
        spectrum_data = []
        
        # 检查播放状态，添加日志记录帮助调试
        is_playing = False
        if hasattr(self, 'music_generator'):
            is_playing = self.music_generator.is_playing
            # 记录播放状态，帮助调试
            if hasattr(self, 'logger'):
                self.logger.log(f"音频播放状态: {is_playing}", "DEBUG")
        
        # 始终生成更有活力的频谱数据以显示音频正在播放
        if is_playing:
            # 获取当前播放区域信息
            playing_regions = []
            main_emotion = "默认"
            weight = 0.5
            
            if hasattr(self, 'music_generator'):
                playing_regions = self.music_generator.get_currently_playing_regions()
                if playing_regions:
                    main_region = playing_regions[0]
                    main_emotion = main_region.get('emotion', '默认')
                    weight = main_region.get('weight', 0.5)
                    
                    if hasattr(self, 'logger'):
                        self.logger.log(f"活跃区域: {main_emotion}, 权重: {weight}", "DEBUG")
            
            # 即使没有活跃区域，也显示音频正在播放
            base_height = 0.3 + (weight * 0.5)
            
            # 根据情感类型生成不同的频谱形状
            if main_emotion in ["joy", "anticipation"]:
                # 高能量、高频
                spectrum_data = [random.uniform(0.1, base_height) * 
                            (0.5 + abs(math.sin(i/num_bars * math.pi * 3)) + random.uniform(0.1, 0.3)) 
                            for i in range(num_bars)]
            elif main_emotion in ["sadness", "fear"]:
                # 低能量、低频
                spectrum_data = [random.uniform(0.1, base_height * 0.7) * 
                            (0.8 - abs(math.sin(i/num_bars * math.pi)) + random.uniform(0.05, 0.2)) 
                            for i in range(num_bars)]
            elif main_emotion in ["anger"]:
                # 高能量、全频段
                spectrum_data = [random.uniform(0.3, base_height * 1.2) * 
                            (0.7 + random.uniform(0, 0.5)) 
                            for i in range(num_bars)]
            else:
                # 默认频谱 - 确保显示活跃状态
                spectrum_data = [random.uniform(0.15, base_height) * (0.7 + random.uniform(0, 0.3)) 
                                for i in range(num_bars)]
        else:
            # 未播放时显示静音状态
            spectrum_data = [0.02 * random.random() for _ in range(num_bars)]
        
        # 渲染可视化效果
        if self.vis_data["type"] == "频谱图":
            # 绘制频谱图代码保持不变
            bar_width = width / num_bars
            for i, value in enumerate(spectrum_data):
                bar_height = value * height
                x1 = i * bar_width
                y1 = height - bar_height
                x2 = (i + 1) * bar_width - 1
                y2 = height
                
                # 选择颜色 - 彩虹渐变
                if self.vis_data["color_scheme"] == "彩虹":
                    hue = i / num_bars * 360
                    r, g, b = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
                    color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                elif self.vis_data["color_scheme"] == "热图":
                    intensity = value / max(spectrum_data) if max(spectrum_data) > 0 else 0
                    r = min(255, int(intensity * 255))
                    g = min(255, int(intensity * 200))
                    b = 50
                    color = f"#{r:02x}{g:02x}{b:02x}"
                else:
                    color = "#3498db"  # 默认蓝色
                
                self.audio_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="",
                    tags="visualization"
                )
        
        # 其他可视化类型代码保持不变
        elif self.vis_data["type"] == "波形图":
            # 绘制频谱图
            bar_width = width / num_bars
            for i, value in enumerate(spectrum_data):
                bar_height = value * height
                x1 = i * bar_width
                y1 = height - bar_height
                x2 = (i + 1) * bar_width - 1
                y2 = height
                
                # 选择颜色 - 彩虹渐变
                if self.vis_data["color_scheme"] == "彩虹":
                    hue = i / num_bars * 360
                    r, g, b = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
                    color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                elif self.vis_data["color_scheme"] == "热图":
                    intensity = value / max(spectrum_data) if max(spectrum_data) > 0 else 0
                    r = min(255, int(intensity * 255))
                    g = min(255, int(intensity * 200))
                    b = 50
                    color = f"#{r:02x}{g:02x}{b:02x}"
                else:
                    color = "#3498db"  # 默认蓝色
                
                self.audio_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="",
                    tags="visualization"
                )
        
        elif self.vis_data["type"] == "波形图":
            # 绘制波形图
            points = []
            step = width / (num_bars - 1)
            mid_height = height / 2
            
            for i, value in enumerate(spectrum_data):
                x = i * step
                y = mid_height - value * mid_height * 0.8
                points.append(x)
                points.append(y)
            
            if points:
                self.audio_canvas.create_line(
                    points,
                    fill="#2ecc71",
                    width=2,
                    smooth=True,
                    tags="visualization"
                )
        
        elif self.vis_data["type"] == "情感映射":
            # 情感映射可视化
            # 这是一个更复杂的可视化，仅做示例
            playing_regions = self.music_generator.get_currently_playing_regions()
            
            if playing_regions:
                # 清除画布
                self.audio_canvas.delete("all")
                
                # 绘制背景
                self.audio_canvas.create_rectangle(
                    0, 0, width, height,
                    fill="black",
                    tags="visualization"
                )
                
                # 绘制情感圆圈
                for i, region in enumerate(playing_regions[:5]):  # 最多显示5个情感
                    emotion = region['emotion']
                    weight = region['weight']
                    valence = region.get('valence', 0.5)
                    arousal = region.get('arousal', 0.5)
                    
                    # 确定情感颜色
                    emotion_colors = {
                        "joy": "#90EE90",       # 淡绿色
                        "trust": "#ADD8E6",     # 淡蓝色
                        "fear": "#FFB6C1",      # 淡红色
                        "surprise": "#FFFF99",  # 淡黄色
                        "sadness": "#D3D3D3",   # 淡灰色
                        "disgust": "#E6E6FA",   # 淡紫色
                        "anger": "#FFA07A",     # 浅珊瑚色
                        "anticipation": "#FFDAB9"  # 桃色
                    }
                    color = emotion_colors.get(emotion.lower(), "#CCCCCC")
                    
                    # 计算位置 (基于valence和arousal)
                    center_x = width * (0.3 + valence * 0.6)
                    center_y = height * (0.9 - arousal * 0.7)
                    
                    # 计算大小 (基于weight)
                    radius = 20 + weight * 60
                    
                    # 绘制圆圈
                    self.audio_canvas.create_oval(
                        center_x - radius, center_y - radius,
                        center_x + radius, center_y + radius,
                        fill=color,
                        stipple="gray50",  # 半透明效果
                        outline=color,
                        width=2,
                        tags="visualization"
                    )
                    
                    # 绘制情感标签
                    self.audio_canvas.create_text(
                        center_x, center_y,
                        text=emotion.capitalize(),
                        fill="white",
                        font=("Arial", 9 + int(weight * 4)),
                        tags="visualization"
                    )
        
        # 检查是否在录制动画
        if self.vis_data["recording"]:
            # 截图当前帧
            try:
                width = self.audio_canvas.winfo_width()
                height = self.audio_canvas.winfo_height()
                
                # 创建图像
                image = Image.new("RGB", (width, height), "black")
                
                # 获取画布内容作为PostScript
                ps_data = self.audio_canvas.postscript(colormode="color")
                
                # 转换PostScript为图像
                ps_image = Image.open(io.BytesIO(ps_data.encode("utf-8")))
                image.paste(ps_image)
                
                # 添加到帧列表
                self.vis_data["frames"].append(image)
            except Exception as e:
                print(f"录制帧出错: {e}")

    def _load_music_resources(self, parent_window=None):
        """加载音乐资源"""
        self.status_var.set("正在加载音乐资源...")
        if parent_window:
            parent_window.update_idletasks()
        
        success = self.music_generator.load_music_resources()
        
        if success:
            self.status_var.set("音乐资源加载完成")
            if parent_window:
                messagebox.showinfo("加载成功", "音乐资源加载完成，可以开始播放", parent=parent_window)
        else:
            self.status_var.set("音乐资源加载失败")
            if parent_window:
                messagebox.showerror("加载失败", "无法加载音乐资源，请检查资源文件夹", parent=parent_window)

    def _toggle_music_playback(self, button_var=None):
        """切换音乐播放状态"""
        if not self.music_generator.is_playing:
            # 开始播放
            success = self.music_generator.start_playback()
            if success:
                self.logger.log("正在检查音频通道状态...", "INFO")
                # 检查各通道状态
                if hasattr(self.music_generator, 'music_channels'):
                    for channel_name, channel in self.music_generator.music_channels.items():
                        self.logger.log(f"通道 {channel_name} 状态: 活跃={channel.get_busy()}, 音量={channel.get_volume()}", "DEBUG")
                
                self.status_var.set("音乐播放中...")
                if button_var:
                    button_var.set("暂停播放")
                
                # 确认播放状态已设置
                if hasattr(self, 'logger'):
                    self.logger.log(f"音乐播放已启动，状态: {self.music_generator.is_playing}", "INFO")
                
                # 立即更新可视化
                if hasattr(self, 'audio_canvas'):
                    self._update_audio_visualization(None)
            else:
                self.status_var.set("无法启动音乐播放")
        else:
            # 停止播放
            self.music_generator.stop_playback()
            self.status_var.set("音乐播放已停止")
            if button_var:
                button_var.set("开始播放")
            
            # 确认播放状态已更新
            if hasattr(self, 'logger'):
                self.logger.log(f"音乐播放已停止，状态: {self.music_generator.is_playing}", "INFO")
            
    def _stop_music_playback(self):
        """停止音乐播放"""
        self.music_generator.stop_playback()
        self.status_var.set("音乐播放已停止")

    def _view_music_regions(self):
        """查看音乐区域地图"""
        if not self.map_data or not self.map_data.is_valid():
            self.show_warning_dialog("请先生成地图")
            return
        
        # 检查是否已加载情感数据
        if not hasattr(self, 'emotion_manager') or not self.emotion_manager.emotion_data:
            # 如果没有情感数据，先分析情感
            self.status_var.set("正在分析地图情感...")
            self.master.update_idletasks()
            success = self.emotion_manager.analyze_map_emotions(self.map_data)
            
            if not success:
                self.show_error_dialog("无法分析地图情感，请重试")
                return
        
        # 加载音乐数据
        self.status_var.set("正在准备音乐区域地图...")
        self.master.update_idletasks()
        self.music_generator.load_emotion_data(emotion_manager=self.emotion_manager, 
                                            story_emotion_map=self.map_data.emotion_map)
        
        # 创建区域可视化图像
        fig = self.music_generator.create_region_map_visualization()
        
        # 显示可视化窗口
        preview_window = tk.Toplevel(self.master)
        preview_window.title("音乐情感区域地图")
        preview_window.geometry("800x600")
        
        # 创建图表容器
        canvas_frame = ttk.Frame(preview_window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 嵌入图表
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加导航工具栏
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()
        
        # 底部按钮
        button_frame = ttk.Frame(preview_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame, 
            text="导出地图图像", 
            command=lambda: self._export_music_region_map()
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="关闭", 
            command=preview_window.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        self.status_var.set("音乐区域地图已显示")

    def _export_music_region_map(self):
        """导出音乐区域地图"""
        filepath = filedialog.asksaveasfilename(
            title="导出音乐区域地图",
            filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg"), ("所有文件", "*.*")],
            defaultextension=".png"
        )
        
        if not filepath:
            return
        
        success = self.music_generator.save_region_map_visualization(filepath)
        
        if success:
            self.status_var.set(f"音乐区域地图已保存到 {filepath}")
            messagebox.showinfo("导出成功", f"音乐区域地图已保存到:\n{filepath}")
        else:
            self.status_var.set("保存音乐区域地图失败")
            messagebox.showerror("导出失败", "无法保存音乐区域地图，请检查文件路径和权限")

    def _show_conda_tool_dialog(self):
        """显示外部conda环境工具对话框"""
        conda_dialog = tk.Toplevel(self.master)
        conda_dialog.title("外部Conda环境工具")
        conda_dialog.geometry("650x500")
        conda_dialog.transient(self.master)
        conda_dialog.grab_set()
        
        # 创建主框架
        main_frame = ttk.Frame(conda_dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 环境名称和检查状态
        env_frame = ttk.LabelFrame(main_frame, text="Conda环境")
        env_frame.pack(fill=tk.X, padx=5, pady=5)
        
        env_top_frame = ttk.Frame(env_frame)
        env_top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(env_top_frame, text="环境名称:").pack(side=tk.LEFT, padx=5, pady=5)
        
        # 获取上次使用的环境或默认值
        last_env = self.config_manager.get("conda_tool.last_env", "gamemap_update")
        env_var = tk.StringVar(value=last_env)
        
        # 环境下拉列表框（将在获取环境列表后填充）
        env_combo = ttk.Combobox(env_top_frame, textvariable=env_var, width=20)
        env_combo.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # 环境状态标签
        env_status_var = tk.StringVar(value="检测中...")
        env_status_label = ttk.Label(env_top_frame, textvariable=env_status_var, width=15)
        env_status_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 刷新按钮
        refresh_btn = ttk.Button(env_top_frame, text="刷新环境", width=10)
        refresh_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Shell选择框架
        shell_frame = ttk.Frame(env_frame)
        shell_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(shell_frame, text="执行Shell:").pack(side=tk.LEFT, padx=5, pady=2)
        
        shell_var = tk.StringVar(value=self.config_manager.get("conda_tool.shell", "cmd"))
        cmd_radio = ttk.Radiobutton(shell_frame, text="CMD", variable=shell_var, value="cmd")
        cmd_radio.pack(side=tk.LEFT, padx=10, pady=2)
        
        ps_radio = ttk.Radiobutton(shell_frame, text="PowerShell", variable=shell_var, value="powershell")
        ps_radio.pack(side=tk.LEFT, padx=10, pady=2)
        
        # 命令选择
        cmd_frame = ttk.LabelFrame(main_frame, text="命令选择")
        cmd_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 预设命令选项
        preset_frame = ttk.Frame(cmd_frame)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(preset_frame, text="预设命令:").pack(side=tk.LEFT, padx=5)
        
        # 获取自定义预设
        custom_presets = self.config_manager.get("conda_tool.custom_presets", {})
        
        # 预设命令列表
        default_presets = {
            "生成3D模型":"cd 'C:\Program Files\EmoScape Studio\EmoScape Studio\plugin\hunyuan';python api_server.py --host 127.0.0.1 --port 8080",
            "生成旋律":"cd 'C:\Program Files\EmoScape Studio\EmoScape Studio\plugin\inspiremusic';启动.exe",
            "生成歌曲(快)":"cd 'C:\Program Files\EmoScape Studio\EmoScape Studio\plugin\diffrhythm';启动.exe",
            "生成歌曲(慢)":"cd 'C:\Program Files\EmoScape Studio\EmoScape Studio\plugin\yue',启动.exe"
        }
        
        # 合并默认和自定义预设
        all_presets = {**default_presets, **custom_presets}
        preset_names = list(all_presets.keys()) + ["自定义命令..."]
        
        preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(preset_frame, textvariable=preset_var, 
                                values=preset_names, state="readonly", width=30)
        preset_combo.current(0)
        preset_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 管理预设按钮
        manage_btn = ttk.Button(preset_frame, text="管理预设", width=10)
        manage_btn.pack(side=tk.LEFT, padx=5)
        
        # 命令详情
        command_frame = ttk.LabelFrame(cmd_frame, text="命令详情")
        command_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 获取上次使用的命令或默认值
        last_cmd = self.config_manager.get("conda_tool.last_cmd", "python -m pygamemap.pygamemap")
        cmd_var = tk.StringVar(value=last_cmd)
        
        # 自定义命令输入框
        cmd_text = scrolledtext.ScrolledText(command_frame, height=8, width=50)
        cmd_text.insert(tk.END, cmd_var.get())
        cmd_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 输出框架
        output_frame = ttk.LabelFrame(main_frame, text="命令输出")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 输出文本框
        output_text = scrolledtext.ScrolledText(output_frame, height=8, width=50, state=tk.DISABLED)
        output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 获取系统conda环境列表
        def get_conda_environments():
            try:
                # 执行conda env list命令获取所有环境
                process = subprocess.Popen(
                    ["conda", "env", "list"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=True  # 在Windows可能需要shell=True
                )
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    return []
                
                # 解析环境列表
                env_list = []
                for line in stdout.splitlines():
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if parts:
                            env_list.append(parts[0])
                return env_list
            except Exception as e:
                raise e  # 将异常抛给线程捕获
        
        # 刷新环境列表并更新下拉框
        # 修改refresh_environments方法，使用线程执行
        def refresh_environments():
            def thread_target():
                try:
                    # 在子线程中获取环境列表
                    environments = get_conda_environments()
                    if environments:
                        # 在主线程中更新UI
                        conda_dialog.after(0, lambda: update_env_combobox(environments))
                except Exception as e:
                    conda_dialog.after(0, lambda: show_refresh_error(e))
            
            # 启动后台线程
            threading.Thread(target=thread_target, daemon=True).start()

        def update_env_combobox(environments):
            # 记住当前选择的环境
            current_env = env_var.get()
            
            # 更新下拉框
            env_combo['values'] = environments
            
            # 尝试恢复之前的选择
            if current_env in environments:
                env_var.set(current_env)
            else:
                if environments:
                    env_var.set(environments[0])
                else:
                    env_var.set("")
            
            # 显示成功消息
            output_text.config(state=tk.NORMAL)
            output_text.delete(1.0, tk.END)
            if environments:
                output_text.insert(tk.END, f"发现{len(environments)}个conda环境:\n" + "\n".join(environments))
            else:
                output_text.insert(tk.END, "未找到任何conda环境")
            output_text.config(state=tk.DISABLED)

        def show_refresh_error(e):
            env_status_var.set("刷新失败")
            env_status_label.configure(foreground="red")
            output_text.config(state=tk.NORMAL)
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, f"刷新环境列表时出错:\n{str(e)}")
            output_text.config(state=tk.DISABLED)
        
        # 检查conda环境是否存在
        def check_conda_env():
            env_name = env_var.get().strip()
            if not env_name:
                messagebox.showerror("错误", "请输入Conda环境名称")
                return
            
            env_status_var.set("正在检查...")
            env_status_label.configure(foreground="black")
            conda_dialog.update_idletasks()
            
            environments = get_conda_environments()
            
            # 检查指定环境是否存在
            if env_name in environments:
                env_status_var.set("环境存在")
                env_status_label.configure(foreground="green")
            else:
                env_status_var.set("环境不存在")
                env_status_label.configure(foreground="red")
        
        # 绑定环境选择框事件
        env_combo.bind("<<ComboboxSelected>>", lambda e: check_conda_env())
        
        # 保存预设对话框
        def save_preset_dialog():
            # 获取当前命令
            command = cmd_text.get(1.0, tk.END).strip()
            if not command:
                messagebox.showerror("错误", "请先输入要保存的命令")
                return
            
            # 显示保存预设对话框
            preset_dialog = tk.Toplevel(conda_dialog)
            preset_dialog.title("保存预设命令")
            preset_dialog.geometry("350x120")
            preset_dialog.transient(conda_dialog)
            preset_dialog.grab_set()
            
            ttk.Label(preset_dialog, text="预设名称:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
            name_var = tk.StringVar()
            name_entry = ttk.Entry(preset_dialog, textvariable=name_var, width=30)
            name_entry.grid(row=0, column=1, padx=10, pady=10)
            name_entry.focus_set()
            
            btn_frame = ttk.Frame(preset_dialog)
            btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
            
            def save_preset():
                name = name_var.get().strip()
                if not name:
                    messagebox.showerror("错误", "请输入预设名称")
                    return
                
                # 检查是否覆盖默认预设
                if name in default_presets:
                    if not messagebox.askyesno("确认覆盖", f"'{name}'是内置预设，确定要覆盖吗？"):
                        return
                
                # 保存预设
                custom_presets[name] = command
                self.config_manager.set("conda_tool.custom_presets", custom_presets)
                self.config_manager.save_config()
                
                # 更新预设下拉列表
                nonlocal all_presets, preset_names
                all_presets = {**default_presets, **custom_presets}
                preset_names = list(all_presets.keys()) + ["自定义命令..."]
                preset_combo['values'] = preset_names
                
                # 选择新添加的预设
                try:
                    preset_var.set(name)
                except:
                    pass
                    
                preset_dialog.destroy()
                messagebox.showinfo("成功", f"预设命令'{name}'已保存")
            
            ttk.Button(btn_frame, text="保存", command=save_preset).pack(side=tk.LEFT, padx=10)
            ttk.Button(btn_frame, text="取消", command=preset_dialog.destroy).pack(side=tk.LEFT, padx=10)
        
        # 管理预设对话框
        def manage_presets_dialog():
            manage_dialog = tk.Toplevel(conda_dialog)
            manage_dialog.title("管理预设命令")
            manage_dialog.geometry("500x300")
            manage_dialog.transient(conda_dialog)
            manage_dialog.grab_set()
            
            ttk.Label(manage_dialog, text="选择要管理的预设:").pack(anchor=tk.W, padx=10, pady=5)
            
            # 创建预设列表
            preset_list = tk.Listbox(manage_dialog, height=10, width=50)
            preset_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # 填充列表
            for preset in all_presets.keys():
                preset_list.insert(tk.END, preset)
            
            # 预设命令预览框架
            preview_frame = ttk.LabelFrame(manage_dialog, text="预设命令")
            preview_frame.pack(fill=tk.X, padx=10, pady=5)
            
            preview_text = tk.Text(preview_frame, height=3, width=50, wrap=tk.WORD)
            preview_text.pack(fill=tk.X, padx=5, pady=5)
            preview_text.config(state=tk.DISABLED)
            
            # 预设选择变更时更新预览
            def on_preset_select(event):
                try:
                    selection = preset_list.curselection()
                    if selection:
                        preset_name = preset_list.get(selection[0])
                        if preset_name in all_presets:
                            preview_text.config(state=tk.NORMAL)
                            preview_text.delete(1.0, tk.END)
                            preview_text.insert(tk.END, all_presets[preset_name])
                            preview_text.config(state=tk.DISABLED)
                except Exception as e:
                    preview_text.config(state=tk.NORMAL)
                    preview_text.delete(1.0, tk.END)
                    preview_text.insert(tk.END, f"错误: {str(e)}")
                    preview_text.config(state=tk.DISABLED)
            
            preset_list.bind('<<ListboxSelect>>', on_preset_select)
            
            btn_frame = ttk.Frame(manage_dialog)
            btn_frame.pack(fill=tk.X, padx=10, pady=10)
            
            def delete_preset():
                selection = preset_list.curselection()
                if not selection:
                    messagebox.showinfo("提示", "请先选择一个预设")
                    return
                    
                preset_name = preset_list.get(selection[0])
                
                # 不允许删除默认预设
                if preset_name in default_presets:
                    messagebox.showinfo("提示", "内置预设无法删除")
                    return
                    
                if messagebox.askyesno("确认删除", f"确定要删除预设'{preset_name}'吗？"):
                    if preset_name in custom_presets:
                        del custom_presets[preset_name]
                        self.config_manager.set("conda_tool.custom_presets", custom_presets)
                        self.config_manager.save_config()
                        
                        # 更新预设列表
                        preset_list.delete(selection[0])
                        
                        # 更新预设下拉列表
                        nonlocal all_presets, preset_names
                        all_presets = {**default_presets, **custom_presets}
                        preset_names = list(all_presets.keys()) + ["自定义命令..."]
                        preset_combo['values'] = preset_names
                        preset_combo.current(0)
                        
                        # 清空预览
                        preview_text.config(state=tk.NORMAL)
                        preview_text.delete(1.0, tk.END)
                        preview_text.config(state=tk.DISABLED)
            
            def edit_preset():
                selection = preset_list.curselection()
                if not selection:
                    messagebox.showinfo("提示", "请先选择一个预设")
                    return
                    
                preset_name = preset_list.get(selection[0])
                
                # 不允许编辑默认预设
                if preset_name in default_presets:
                    messagebox.showinfo("提示", "内置预设无法直接编辑")
                    return
                
                # 获取预设命令
                preset_cmd = all_presets.get(preset_name, "")
                
                # 显示编辑对话框
                edit_dialog = tk.Toplevel(manage_dialog)
                edit_dialog.title(f"编辑预设 - {preset_name}")
                edit_dialog.geometry("500x200")
                edit_dialog.transient(manage_dialog)
                edit_dialog.grab_set()
                
                ttk.Label(edit_dialog, text="编辑命令:").pack(anchor=tk.W, padx=10, pady=5)
                
                edit_text = tk.Text(edit_dialog, height=5, width=60, wrap=tk.WORD)
                edit_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                edit_text.insert(tk.END, preset_cmd)
                
                btn_frame = ttk.Frame(edit_dialog)
                btn_frame.pack(fill=tk.X, padx=10, pady=10)
                
                def save_edit():
                    new_cmd = edit_text.get(1.0, tk.END).strip()
                    if not new_cmd:
                        messagebox.showerror("错误", "命令不能为空")
                        return
                    
                    # 更新预设
                    custom_presets[preset_name] = new_cmd
                    self.config_manager.set("conda_tool.custom_presets", custom_presets)
                    self.config_manager.save_config()
                    
                    # 更新预览
                    preview_text.config(state=tk.NORMAL)
                    preview_text.delete(1.0, tk.END)
                    preview_text.insert(tk.END, new_cmd)
                    preview_text.config(state=tk.DISABLED)
                    
                    # 更新全局预设
                    nonlocal all_presets
                    all_presets = {**default_presets, **custom_presets}
                    
                    edit_dialog.destroy()
                    messagebox.showinfo("成功", f"预设命令'{preset_name}'已更新")
                
                ttk.Button(btn_frame, text="保存", command=save_edit).pack(side=tk.RIGHT, padx=5)
                ttk.Button(btn_frame, text="取消", command=edit_dialog.destroy).pack(side=tk.RIGHT, padx=5)
            
            ttk.Button(btn_frame, text="编辑预设", command=edit_preset).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="删除预设", command=delete_preset).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="关闭", command=manage_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # 绑定管理预设按钮事件
        manage_btn.config(command=manage_presets_dialog)
        
        # 更新命令的函数
        def update_command(*args):
            selected = preset_var.get()
            
            if selected in all_presets:
                cmd_text.delete(1.0, tk.END)
                cmd_text.insert(tk.END, all_presets[selected])
        
        # 绑定预设选择事件
        preset_combo.bind("<<ComboboxSelected>>", lambda e: update_command())
        
        # 底部按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # 执行函数
        def run_command():
            env_name = env_var.get().strip()
            command = cmd_text.get(1.0, tk.END).strip()
            shell_type = shell_var.get()
            
            if not env_name:
                messagebox.showerror("错误", "请输入Conda环境名称")
                return
            
            if not command:
                messagebox.showerror("错误", "请输入要执行的命令")
                return
            
            # 保存最后使用的环境、命令和shell类型
            self.config_manager.set("conda_tool.last_env", env_name)
            self.config_manager.set("conda_tool.last_cmd", command)
            self.config_manager.set("conda_tool.shell", shell_type)
            self.config_manager.save_config()
            
            # 清空输出框
            output_text.config(state=tk.NORMAL)
            output_text.delete(1.0, tk.END)
            output_text.config(state=tk.DISABLED)
            
            # 禁用按钮，显示执行中状态
            run_btn.config(state=tk.DISABLED)
            self.status_var.set("正在执行Conda命令...")
            
            # 创建重定向的输出处理函数
            def process_output(process):
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # 更新GUI输出框
                        output_text.config(state=tk.NORMAL)
                        output_text.insert(tk.END, output)
                        output_text.see(tk.END)
                        output_text.config(state=tk.DISABLED)
                        # 更新界面
                        conda_dialog.update_idletasks()
                
                # 命令完成后恢复按钮状态
                run_btn.config(state=tk.NORMAL)
                self.status_var.set("Conda命令执行完成")
            
            # 启动后台线程执行命令
            def run_in_thread():
                try:
                    # 根据选择的shell构建命令
                    if shell_type == "powershell":
                        # PowerShell执行方式 - 修复命令语法
                        # 使用分号替代&&或直接使用conda run命令
                        ps_cmd = f"conda activate {env_name}; {command}"
                        cmd = ['powershell', '-Command', ps_cmd]
                    else:
                        # CMD执行方式 (默认)
                        command_parts = shlex.split(command)
                        cmd = ['conda', 'run', '-n', env_name] + command_parts
                    
                    # 执行命令并实时输出
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True
                    )
                    
                    # 处理输出
                    process_output(process)
                    
                    # 检查返回码
                    return_code = process.wait()
                    if return_code != 0:
                        output_text.config(state=tk.NORMAL)
                        output_text.insert(tk.END, f"\n命令执行失败，退出码: {return_code}\n")
                        output_text.see(tk.END)
                        output_text.config(state=tk.DISABLED)
                    
                except Exception as e:
                    output_text.config(state=tk.NORMAL)
                    output_text.insert(tk.END, f"\n错误: {str(e)}\n")
                    output_text.see(tk.END)
                    output_text.config(state=tk.DISABLED)
                    
                    # 恢复按钮状态
                    run_btn.config(state=tk.NORMAL)
                    self.status_var.set("Conda命令执行出错")
            
            # 启动线程
            threading.Thread(target=run_in_thread, daemon=True).start()
        
        # 添加保存预设按钮
        save_preset_btn = ttk.Button(button_frame, text="保存为预设", command=save_preset_dialog)
        save_preset_btn.pack(side=tk.LEFT, padx=5)
        
        # 执行按钮
        run_btn = ttk.Button(button_frame, text="执行命令", command=run_command)
        run_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="关闭", command=conda_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # 初始化显示
        update_command()
        
        # 自动获取环境列表
        conda_dialog.after(100, refresh_environments)

    def _show_emotion_map(self):
        """显示情感地图可视化"""
        # 检查是否有地图数据
        if not hasattr(self.map_data, 'story_content') or not self.map_data.story_content:
            messagebox.showinfo("提示", "当前地图没有游戏剧情内容。请先生成游戏剧情。")
            return
            
        # 如果没有情感地图，创建一个
        if not hasattr(self.map_data, 'emotion_map') or not self.map_data.emotion_map:
            # 显示等待消息
            self.status_var.set("正在创建情感地图...")
            self.master.update_idletasks()
            
            # 创建情感地图
            from core.emotional.story_emotion_map import StoryEmotionMap
            self.map_data.emotion_map = StoryEmotionMap(
                width=self.map_data.width,
                height=self.map_data.height,
                logger=self.logger
            )
            
            # 分析故事情感
            self.map_data.emotion_map.analyze_story_content(
                self.map_data.story_content,
                self.map_data
            )
            
            self.status_var.set("情感地图创建完成")
        
        # 创建情感地图对话框
        self._show_emotion_map_dialog(self.map_data.emotion_map)

    def _show_emotion_map_dialog(self, emotion_map):
        """显示情感地图对话框"""
        # 创建对话框
        dialog = tk.Toplevel(self.master)
        dialog.title("情感地图可视化")
        dialog.geometry("900x700")
        dialog.transient(self.master)
        dialog.grab_set()
        
        # 创建标签页
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 情感热力图标签页
        heatmap_frame = ttk.Frame(notebook)
        notebook.add(heatmap_frame, text="情感热力图")
        
        # 创建情感选择控件
        control_frame = ttk.Frame(heatmap_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="选择情感:").pack(side=tk.LEFT, padx=5)
        
        emotions = ["valence", "arousal", "joy", "trust", "fear", "surprise", 
                    "sadness", "disgust", "anger", "anticipation"]
        emotion_var = tk.StringVar(value="valence")
        emotion_combo = ttk.Combobox(control_frame, textvariable=emotion_var, 
                                    values=emotions, state="readonly", width=20)
        emotion_combo.pack(side=tk.LEFT, padx=5)
        
        # 热力图显示区域
        heatmap_display = ttk.Frame(heatmap_frame)
        heatmap_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建初始热力图
        fig = emotion_map.create_emotion_heatmap_figure("valence")
        canvas = FigureCanvasTkAgg(fig, heatmap_display)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 更新热力图的函数
        def update_heatmap(*args):
            emotion = emotion_var.get()
            new_fig = emotion_map.create_emotion_heatmap_figure(emotion)
            canvas.figure = new_fig
            canvas.draw()
        
        # 绑定更新函数
        emotion_var.trace("w", update_heatmap)
        
        # 情感对比标签页
        comparison_frame = ttk.Frame(notebook)
        notebook.add(comparison_frame, text="情感对比")
        
        # 创建情感对比图
        fig = emotion_map.create_emotion_comparison_chart()
        canvas = FigureCanvasTkAgg(fig, comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 情感弧线标签页
        arc_frame = ttk.Frame(notebook)
        notebook.add(arc_frame, text="情感弧线")
        
        # 创建情感弧线图
        fig = emotion_map.create_emotional_arc_chart()
        canvas = FigureCanvasTkAgg(fig, arc_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 情感洞见标签页
        insights_frame = ttk.Frame(notebook)
        notebook.add(insights_frame, text="情感洞见")
        
        # 创建洞见显示区域
        insights_text = scrolledtext.ScrolledText(insights_frame, wrap=tk.WORD, font=("TkDefaultFont", 11))
        insights_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 显示洞见
        insights = emotion_map.generate_emotion_insights()
        for i, insight in enumerate(insights):
            insights_text.insert(tk.END, f"{i+1}. {insight}\n\n")
        insights_text.config(state=tk.DISABLED)  # 设为只读
        
        # 底部按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame, 
            text="导出分析报告", 
            command=lambda: self._export_emotion_map_report(emotion_map)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="关闭", 
            command=dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)

    def _export_emotion_map_report(self, emotion_map):
        """导出情感地图分析报告"""
        # 打开文件对话框
        filepath = filedialog.asksaveasfilename(
            title="导出情感地图分析报告",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            defaultextension=".json"
        )
        
        if not filepath:
            return
        
        # 导出报告
        success = emotion_map.export_emotion_analysis(filepath)
        
        if success:
            messagebox.showinfo("导出成功", f"情感地图分析报告已导出到: {filepath}")
        else:
            messagebox.showerror("导出失败", "导出情感地图分析报告时发生错误")

    def _analyze_story_emotions(self):
        """分析故事情感并显示分析结果"""
        # 检查是否有故事内容
        if not hasattr(self.map_data, 'story_content') or not self.map_data.story_content:
            messagebox.showinfo("提示", "当前地图没有游戏剧情内容。请先生成游戏剧情。")
            return
            
        # 显示等待消息
        self.status_var.set("正在分析故事情感...")
        self.master.update_idletasks()
        
        # 检查是否已有情感地图
        if not hasattr(self.map_data, 'emotion_map') or not self.map_data.emotion_map:
            from core.emotional.story_emotion_map import StoryEmotionMap
            self.map_data.emotion_map = StoryEmotionMap(
                width=self.map_data.width,
                height=self.map_data.height,
                logger=self.logger
            )
            
        # 分析故事情感
        analysis_success = self.map_data.emotion_map.analyze_story_content(
            self.map_data.story_content,
            self.map_data
        )
        
        if not analysis_success:
            messagebox.showerror("分析失败", "故事情感分析失败，请检查故事内容。")
            self.status_var.set("故事情感分析失败")
            return
        
        # 显示分析结果对话框
        self._show_emotion_map_dialog(self.map_data.emotion_map)
        
        self.status_var.set("故事情感分析完成")

    def _show_story_emotion_analysis(self, analyzer):
        """显示故事情感分析结果对话框"""
        # 创建对话框
        dialog = tk.Toplevel(self.master)
        dialog.title("故事情感分析")
        dialog.geometry("900x700")
        dialog.transient(self.master)
        dialog.grab_set()
        
        # 创建标签页
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 概览标签页
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="概览")
        
        # 情感曲线图表
        arc_chart_frame = ttk.LabelFrame(overview_frame, text="故事情感弧线")
        arc_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        arc_fig = analyzer.create_emotional_arc_chart()
        arc_canvas = FigureCanvasTkAgg(arc_fig, arc_chart_frame)
        arc_canvas.draw()
        arc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 洞见和建议
        insights_frame = ttk.LabelFrame(overview_frame, text="情感洞见")
        insights_frame.pack(fill=tk.X, padx=10, pady=5)
        
        insights = analyzer.generate_emotional_insights()
        for i, insight in enumerate(insights):
            ttk.Label(insights_frame, text=f"• {insight}", wraplength=850).pack(
                anchor=tk.W, padx=10, pady=2)
        
        # 事件对比标签页
        events_frame = ttk.Frame(notebook)
        notebook.add(events_frame, text="事件情感对比")
        
        # 事件情感对比图表
        events_chart_frame = ttk.LabelFrame(events_frame, text="事件情感对比")
        events_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        events_fig = analyzer.create_emotion_comparison_chart()
        events_canvas = FigureCanvasTkAgg(events_fig, events_chart_frame)
        events_canvas.draw()
        events_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 事件详情标签页
        details_frame = ttk.Frame(notebook)
        notebook.add(details_frame, text="事件详情")
        
        # 创建事件选择控件
        selection_frame = ttk.Frame(details_frame)
        selection_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(selection_frame, text="选择事件:").pack(side=tk.LEFT, padx=5)
        
        events = analyzer.analysis_results.get("events", [])
        event_names = []
        for event in events:
            index = event.get("index", 0)
            event_type = event.get("type", "未知")
            original_event = event.get("event", {})
            desc = original_event.get("description", "未知事件")
            short_desc = desc[:20] + "..." if len(desc) > 20 else desc
            event_names.append(f"事件 {index+1} [{event_type}]: {short_desc}")
        
        event_var = tk.StringVar()
        event_combo = ttk.Combobox(selection_frame, textvariable=event_var, values=event_names, state="readonly", width=60)
        event_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        if event_names:
            event_combo.current(0)
        
        # 事件详情显示区域
        event_detail_frame = ttk.LabelFrame(details_frame, text="事件情感详情")
        event_detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建详情显示区
        detail_canvas = tk.Canvas(event_detail_frame)
        detail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        detail_scrollbar = ttk.Scrollbar(event_detail_frame, orient=tk.VERTICAL, command=detail_canvas.yview)
        detail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        detail_canvas.configure(yscrollcommand=detail_scrollbar.set)
        
        detail_frame = ttk.Frame(detail_canvas)
        detail_canvas_window = detail_canvas.create_window((0, 0), window=detail_frame, anchor=tk.NW)
        
        def on_detail_frame_configure(event):
            detail_canvas.configure(scrollregion=detail_canvas.bbox("all"))
        
        detail_frame.bind("<Configure>", on_detail_frame_configure)
        
        # 情感数据显示
        emotion_data_frame = ttk.Frame(detail_frame)
        emotion_data_frame.pack(fill=tk.X, pady=5)
        
        # 更新事件详情的函数
        def update_event_detail(*args):
            # 清空之前的内容
            for widget in emotion_data_frame.winfo_children():
                widget.destroy()
            
            # 获取选中的事件
            idx = event_combo.current()
            if idx >= 0 and idx < len(events):
                event = events[idx]
                analysis = event.get("analysis")
                
                if analysis:
                    # 显示事件基本信息
                    ttk.Label(emotion_data_frame, text=f"事件类型: {event.get('type', '未知')}", font=("", 11, "bold")).pack(anchor=tk.W, padx=10, pady=2)
                    
                    original_event = event.get("event", {})
                    ttk.Label(emotion_data_frame, text=f"事件描述: {original_event.get('description', '无描述')}", wraplength=850).pack(anchor=tk.W, padx=10, pady=2)
                    
                    ttk.Label(emotion_data_frame, text=f"事件坐标: ({original_event.get('x', '?')}, {original_event.get('y', '?')})", font=("", 10)).pack(anchor=tk.W, padx=10, pady=2)
                    
                    # 显示情感数据
                    valence, arousal = analysis.global_emotion.to_valence_arousal()
                    ttk.Label(emotion_data_frame, text=f"情感效价 (Valence): {valence:.2f}", font=("", 10)).pack(anchor=tk.W, padx=10, pady=2)
                    ttk.Label(emotion_data_frame, text=f"情感唤醒度 (Arousal): {arousal:.2f}", font=("", 10)).pack(anchor=tk.W, padx=10, pady=2)
                    ttk.Label(emotion_data_frame, text=f"情感复杂度: {analysis.emotional_complexity:.2f}", font=("", 10)).pack(anchor=tk.W, padx=10, pady=2)
                    ttk.Label(emotion_data_frame, text=f"情感变异度: {analysis.emotional_variance:.2f}", font=("", 10)).pack(anchor=tk.W, padx=10, pady=2)
                    ttk.Label(emotion_data_frame, text=f"语言强度: {analysis.language_intensity:.2f}", font=("", 10)).pack(anchor=tk.W, padx=10, pady=2)
                    
                    # 显示主要情感类别
                    emotion_categories = {}
                    for e in analysis.segment_emotions:
                        for category, value in e.emotion_categories.items():
                            if category in emotion_categories:
                                emotion_categories[category] += value
                            else:
                                emotion_categories[category] = value
                    
                    # 计算平均值
                    if analysis.segment_emotions:
                        for category in emotion_categories:
                            emotion_categories[category] /= len(analysis.segment_emotions)
                    
                    # 只显示值大于0.2的情感类别
                    significant_emotions = {k: v for k, v in emotion_categories.items() if v > 0.2}
                    if significant_emotions:
                        ttk.Label(emotion_data_frame, text="主要情感类别:", font=("", 11, "bold")).pack(anchor=tk.W, padx=10, pady=5)
                        for emotion, value in sorted(significant_emotions.items(), key=lambda x: x[1], reverse=True):
                            ttk.Label(emotion_data_frame, text=f"{emotion}: {value:.2f}", font=("", 10)).pack(anchor=tk.W, padx=20, pady=1)
                    
                    # 显示关键情感点
                    if analysis.key_moments:
                        ttk.Label(emotion_data_frame, text="关键情感时刻:", font=("", 11, "bold")).pack(anchor=tk.W, padx=10, pady=5)
                        for i, moment in enumerate(analysis.key_moments[:5]):  # 显示前5个关键点
                            moment_type = moment.get("type", "").replace("_", " ").title()
                            moment_score = moment.get("score", 0)
                            ttk.Label(emotion_data_frame, text=f"{i+1}. {moment_type}: {moment_score:.2f}", font=("", 10)).pack(anchor=tk.W, padx=20, pady=1)
                else:
                    ttk.Label(emotion_data_frame, text="没有可用的情感分析数据", font=("", 11)).pack(anchor=tk.W, padx=10, pady=10)
        
        # 绑定事件
        event_var.trace("w", update_event_detail)
        
        # 初始化显示
        if event_names:
            update_event_detail()
        
        # 底部按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame, 
            text="导出分析报告", 
            command=lambda: self._export_emotion_analysis_report(analyzer)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="关闭", 
            command=dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)

    def _export_emotion_analysis_report(self, analyzer):
        """导出情感分析报告"""
        # 打开文件对话框
        filepath = filedialog.asksaveasfilename(
            title="导出情感分析报告",
            filetypes=[("Markdown文件", "*.md"), ("文本文件", "*.txt"), ("所有文件", "*.*")],
            defaultextension=".md"
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入标题
                f.write("# 游戏剧情情感分析报告\n\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 写入整体情感分析
                f.write("## 整体故事情感分析\n\n")
                
                overall = analyzer.analysis_results.get("overall")
                if overall:
                    valence, arousal = overall.global_emotion.to_valence_arousal()
                    f.write(f"- 情感效价 (Valence): {valence:.2f}\n")
                    f.write(f"- 情感唤醒度 (Arousal): {arousal:.2f}\n")
                    f.write(f"- 情感复杂度: {overall.emotional_complexity:.2f}\n")
                    f.write(f"- 情感变异度: {overall.emotional_variance:.2f}\n")
                    f.write(f"- 语言强度: {overall.language_intensity:.2f}\n\n")
                else:
                    f.write("没有整体故事情感分析数据\n\n")
                
                # 写入洞见和建议
                f.write("## 情感洞见与建议\n\n")
                
                insights = analyzer.generate_emotional_insights()
                for insight in insights:
                    f.write(f"- {insight}\n")
                f.write("\n")
                
                # 写入事件情感分析
                f.write("## 事件情感分析\n\n")
                
                events = analyzer.analysis_results.get("events", [])
                for event in events:
                    index = event.get("index", 0)
                    event_type = event.get("type", "未知")
                    original_event = event.get("event", {})
                    desc = original_event.get("description", "未知事件")
                    
                    f.write(f"### 事件 {index+1}: {desc}\n\n")
                    f.write(f"- 类型: {event_type}\n")
                    f.write(f"- 位置: ({original_event.get('x', '?')}, {original_event.get('y', '?')})\n\n")
                    
                    analysis = event.get("analysis")
                    if analysis:
                        valence, arousal = analysis.global_emotion.to_valence_arousal()
                        f.write(f"#### 情感数据\n\n")
                        f.write(f"- 情感效价 (Valence): {valence:.2f}\n")
                        f.write(f"- 情感唤醒度 (Arousal): {arousal:.2f}\n")
                        f.write(f"- 情感复杂度: {analysis.emotional_complexity:.2f}\n")
                        f.write(f"- 情感变异度: {analysis.emotional_variance:.2f}\n")
                        f.write(f"- 语言强度: {analysis.language_intensity:.2f}\n\n")
                        
                        # 写入主要情感类别
                        emotion_categories = {}
                        for e in analysis.segment_emotions:
                            for category, value in e.emotion_categories.items():
                                if category in emotion_categories:
                                    emotion_categories[category] += value
                                else:
                                    emotion_categories[category] = value
                        
                        # 计算平均值
                        if analysis.segment_emotions:
                            for category in emotion_categories:
                                emotion_categories[category] /= len(analysis.segment_emotions)
                        
                        # 只显示值大于0.2的情感类别
                        significant_emotions = {k: v for k, v in emotion_categories.items() if v > 0.2}
                        if significant_emotions:
                            f.write("#### 主要情感类别\n\n")
                            for emotion, value in sorted(significant_emotions.items(), key=lambda x: x[1], reverse=True):
                                f.write(f"- {emotion}: {value:.2f}\n")
                            f.write("\n")
                    else:
                        f.write("没有情感分析数据\n\n")
                    
                    f.write("---\n\n")
            
            self.logger.log(f"情感分析报告已导出到: {filepath}")
            messagebox.showinfo("导出成功", f"情感分析报告已导出到: {filepath}")
        except Exception as e:
            self.logger.log(f"导出情感分析报告失败: {e}", "ERROR")
            messagebox.showerror("导出失败", f"导出过程中发生错误: {str(e)}")

    def _show_story_content(self):
            """显示生成的游戏剧情内容"""
            # 检查是否有游戏剧情内容
            if not hasattr(self.map_data, 'story_content') or not self.map_data.story_content:
                messagebox.showinfo("提示", "当前地图没有生成游戏剧情内容。请先生成地图，确保包含故事事件点。")
                return
            
            # 创建剧情查看对话框
            story_dialog = tk.Toplevel(self.master)
            story_dialog.title("游戏剧情内容")
            story_dialog.geometry("800x600")
            story_dialog.transient(self.master)
            story_dialog.grab_set()
            
            # 创建标签页
            notebook = ttk.Notebook(story_dialog)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 主线剧情标签页
            main_frame = ttk.Frame(notebook)
            notebook.add(main_frame, text="主线剧情")
            
            # 创建滚动文本框显示主线剧情
            main_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=("TkDefaultFont", 10))
            main_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            main_text.insert(tk.END, self.map_data.story_content.get("overall_story", "未生成主线剧情"))
            main_text.config(state=tk.DISABLED)  # 只读模式
            
            # 事件任务标签页
            event_frame = ttk.Frame(notebook)
            notebook.add(event_frame, text="事件任务")
            
            # 创建事件选择框架
            selection_frame = ttk.Frame(event_frame)
            selection_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(selection_frame, text="选择事件:").pack(side=tk.LEFT, padx=5)
            
            expanded_stories = self.map_data.story_content.get("expanded_stories", [])
            event_names = []
            for i, story in enumerate(expanded_stories):
                event = story.get("original_event", {})
                desc = event.get("description", "未知事件")
                # 显示完整事件描述，不再截断
                event_names.append(f"事件 {i+1}: {desc}")
            
            event_var = tk.StringVar()
            event_combo = ttk.Combobox(selection_frame, textvariable=event_var, values=event_names, state="readonly", width=40)
            event_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            if event_names:
                event_combo.current(0)
            
            # 创建事件内容显示框架
            content_frame = ttk.LabelFrame(event_frame, text="事件详情")
            content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            event_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, font=("TkDefaultFont", 10))
            event_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 更新事件内容的函数
            def update_event_content(*args):
                try:
                    idx = event_combo.current()
                    if idx >= 0 and idx < len(expanded_stories):
                        event_text.config(state=tk.NORMAL)
                        event_text.delete(1.0, tk.END)
                        event_text.insert(tk.END, expanded_stories[idx].get("expanded_content", "未找到事件内容"))
                        event_text.config(state=tk.DISABLED)
                        self.logger.log(f"已更新事件内容：事件 {idx+1}")
                except Exception as e:
                    self.logger.log(f"更新事件内容出错: {e}", "ERROR")
            
            # 绑定事件选择变化事件
            event_combo.bind("<<ComboboxSelected>>", update_event_content)
            # 同时保留原有的trace绑定作为备份触发方式
            event_var.trace("w", update_event_content)
            
            # 初始化事件内容
            if expanded_stories:
                update_event_content()
            
            # 编辑和导出按钮
            button_frame = ttk.Frame(story_dialog)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Button(
                button_frame, 
                text="编辑剧情", 
                command=lambda: self._edit_story_content(self.map_data.story_content)
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame, 
                text="导出剧情", 
                command=self._export_story_content
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame, 
                text="关闭", 
                command=story_dialog.destroy
            ).pack(side=tk.RIGHT, padx=5)

    def _edit_story_content(self, story_content):
            """编辑游戏剧情内容"""
            # 创建编辑对话框
            edit_dialog = tk.Toplevel(self.master)
            edit_dialog.title("编辑剧情内容")
            edit_dialog.geometry("800x600")
            edit_dialog.transient(self.master)
            edit_dialog.grab_set()
            
            # 创建标签页
            notebook = ttk.Notebook(edit_dialog)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 主线剧情编辑标签页
            main_frame = ttk.Frame(notebook)
            notebook.add(main_frame, text="主线剧情")
            
            main_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=("TkDefaultFont", 10))
            main_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            main_text.insert(tk.END, story_content.get("overall_story", ""))
            
            # 事件任务编辑标签页
            event_frame = ttk.Frame(notebook)
            notebook.add(event_frame, text="事件任务")
            
            # 创建事件选择框架
            selection_frame = ttk.Frame(event_frame)
            selection_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(selection_frame, text="选择事件:").pack(side=tk.LEFT, padx=5)
            
            expanded_stories = story_content.get("expanded_stories", [])
            event_names = []
            for i, story in enumerate(expanded_stories):
                event = story.get("original_event", {})
                desc = event.get("description", "未知事件")
                # 显示完整事件描述，不再截断
                event_names.append(f"事件 {i+1}: {desc}")
            
            event_var = tk.StringVar()
            event_combo = ttk.Combobox(selection_frame, textvariable=event_var, values=event_names, state="readonly", width=40)
            event_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            if event_names:
                event_combo.current(0)
            
            # 创建事件编辑框架
            content_frame = ttk.LabelFrame(event_frame, text="编辑事件")
            content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            event_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, font=("TkDefaultFont", 10))
            event_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 当前编辑的事件索引
            current_idx = [0]  # 使用列表以便能在嵌套函数中修改
            
            # 保存当前编辑的事件内容
            def save_current_event():
                try:
                    idx = current_idx[0]
                    if idx >= 0 and idx < len(expanded_stories):
                        expanded_stories[idx]["expanded_content"] = event_text.get(1.0, tk.END).strip()
                except Exception as e:
                    self.logger.log(f"保存事件内容出错: {e}", "ERROR")
            
            # 更新事件内容的函数
            def update_event_content(*args):
                try:
                    save_current_event()
                    
                    idx = event_combo.current()
                    if idx >= 0 and idx < len(expanded_stories):
                        event_text.delete(1.0, tk.END)
                        event_text.insert(tk.END, expanded_stories[idx].get("expanded_content", ""))
                        current_idx[0] = idx
                        self.logger.log(f"已切换到编辑事件 {idx+1}")
                except Exception as e:
                    self.logger.log(f"更新事件内容出错: {e}", "ERROR")
            
            # 绑定事件选择变化事件
            event_combo.bind("<<ComboboxSelected>>", update_event_content)
            # 同时保留原有的trace绑定作为备份触发方式
            event_var.trace("w", update_event_content)
            
            # 初始化事件内容
            if expanded_stories:
                update_event_content()
            
            # 保存和取消按钮
            button_frame = ttk.Frame(edit_dialog)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            def save_changes():
                # 保存当前选中的事件
                save_current_event()
                
                # 更新主线剧情
                story_content["overall_story"] = main_text.get(1.0, tk.END).strip()
                
                # 更新地图数据中的故事内容
                self.map_data.story_content = story_content
                
                self.logger.log("已更新游戏剧情内容")
                edit_dialog.destroy()
            
            ttk.Button(
                button_frame, 
                text="保存更改", 
                command=save_changes
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                button_frame, 
                text="取消", 
                command=edit_dialog.destroy
            ).pack(side=tk.RIGHT, padx=5)

    def _export_story_content(self):
        """导出游戏剧情内容"""
        # 检查是否有故事内容
        if not hasattr(self.map_data, 'story_content') or not self.map_data.story_content:
            messagebox.showinfo("提示", "没有可导出的游戏剧情内容")
            return
        
        # 打开文件对话框
        filepath = filedialog.asksaveasfilename(
            title="导出游戏剧情",
            filetypes=[("Markdown文件", "*.md"), ("文本文件", "*.txt"), ("所有文件", "*.*")],
            defaultextension=".md"
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # 写入主线剧情
                f.write("# 游戏主线剧情\n\n")
                f.write(self.map_data.story_content.get("overall_story", "未生成主线剧情"))
                f.write("\n\n")
                
                # 写入各个事件任务
                f.write("# 事件任务详情\n\n")
                
                for i, story in enumerate(self.map_data.story_content.get("expanded_stories", [])):
                    event = story.get("original_event", {})
                    desc = event.get("description", "未知事件")
                    x, y = event.get("x", "?"), event.get("y", "?")
                    
                    f.write(f"## 事件 {i+1}: {desc}\n\n")
                    f.write(f"位置: ({x}, {y})\n\n")
                    f.write(story.get("expanded_content", "未找到事件内容"))
                    f.write("\n\n---\n\n")
            
            self.logger.log(f"游戏剧情已导出到: {filepath}")
            messagebox.showinfo("导出成功", f"游戏剧情已导出到: {filepath}")
        except Exception as e:
            self.logger.log(f"导出剧情失败: {e}", "ERROR")
            messagebox.showerror("导出失败", f"导出过程中发生错误: {str(e)}")

    def _show_emotion_analysis(self):
        """显示地图情感分析对话框"""
        if not self.map_data or not self.map_data.is_valid():
            self.show_warning_dialog("请先生成地图再进行情感分析")
            return
        
        # 执行情感分析
        self.status_var.set("正在分析地图情感...")
        success = self.emotion_manager.analyze_map_emotions(self.map_data)
        
        if not success:
            self.show_error_dialog("情感分析失败，请查看日志获取详细信息")
            return
        
        # 创建情感分析结果对话框
        emotion_dialog = tk.Toplevel(self.master)
        emotion_dialog.title("地图情感分析")
        emotion_dialog.geometry("600x500")
        emotion_dialog.transient(self.master)
        emotion_dialog.grab_set()
        
        # 创建标签页
        notebook = ttk.Notebook(emotion_dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 摘要标签页
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="情感摘要")
        
        # 显示主导情感
        dominant_emotions = self.emotion_manager.get_dominant_emotions(top_n=3)
        ttk.Label(summary_frame, text="主导情感:", font=("", 12, "bold")).pack(anchor=tk.W, padx=10, pady=5)
        
        for emotion, weight in dominant_emotions:
            weight_percent = weight * 100 if weight <= 1 else weight
            ttk.Label(summary_frame, text=f"{emotion.capitalize()}: {weight_percent:.1f}%").pack(anchor=tk.W, padx=20)
        
        # 展示情感分布柱状图
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        stats = self.emotion_manager.get_emotion_stats()
        emotions = list(stats.keys())
        counts = [stats[e]["count"] for e in emotions]
        intensities = [stats[e]["avg_intensity"] for e in emotions]
        
        x = range(len(emotions))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], counts, width, label='数量')
        ax.bar([i + width/2 for i in x], intensities, width, label='平均强度')
        
        ax.set_ylabel('值')
        ax.set_title('情感分布')
        ax.set_xticks(x)
        ax.set_xticklabels([e.capitalize() for e in emotions], rotation=45)
        ax.legend()
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, summary_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 详细数据标签页
        details_frame = ttk.Frame(notebook)
        notebook.add(details_frame, text="详细数据")
        
        # 创建树状视图显示详细数据
        columns = ("情感", "出现次数", "平均强度", "最大强度")
        tree = ttk.Treeview(details_frame, columns=columns, show="headings")
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=tk.CENTER)
        
        for emotion in stats:
            tree.insert("", tk.END, values=(
                emotion.capitalize(),
                stats[emotion]["count"],
                f"{stats[emotion]['avg_intensity']:.2f}",
                f"{stats[emotion]['max_intensity']:.2f}"
            ))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 热力图标签页
        heatmap_frame = ttk.Frame(notebook)
        notebook.add(heatmap_frame, text="情感热力图")
        
        # 选择情感下拉框
        selection_frame = ttk.Frame(heatmap_frame)
        selection_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(selection_frame, text="选择情感:").pack(side=tk.LEFT, padx=5)
        
        emotion_var = tk.StringVar(value="combined")
        emotion_options = ["combined"] + self.emotion_manager.primary_emotions
        emotion_dropdown = ttk.Combobox(
            selection_frame, 
            textvariable=emotion_var,
            values=[e.capitalize() for e in emotion_options],
            state="readonly",
            width=15
        )
        emotion_dropdown.pack(side=tk.LEFT, padx=5)
        
        # 显示热力图
        def update_heatmap(*args):
            for widget in heatmap_container.winfo_children():
                widget.destroy()
            
            selected = emotion_var.get().lower()
            
            if selected == "combined":
                heatmap = self.emotion_manager.get_combined_emotion_heatmap()
                title = "组合情感热力图"
            else:
                heatmap = self.emotion_manager.get_emotion_heatmap(selected)
                title = f"{selected.capitalize()} 情感热力图"
            
            if heatmap is None:
                ttk.Label(heatmap_container, text="无法生成所选情感的热力图").pack(padx=10, pady=10)
                return
            
            fig = Figure(figsize=(6, 5), dpi=100)
            ax = fig.add_subplot(111)
            
            # 获取情感对应的颜色映射
            cmap_name = "viridis"
            if hasattr(self.emotion_manager, "_get_emotion_colormap"):
                cmap = self.emotion_manager._get_emotion_colormap(selected)
            else:
                cmap = plt.get_cmap(cmap_name)
            
            im = ax.imshow(heatmap, cmap=cmap, interpolation='bilinear')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="情感强度")
            
            canvas = FigureCanvasTkAgg(fig, heatmap_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 热力图容器
        heatmap_container = ttk.Frame(heatmap_frame)
        heatmap_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 绑定下拉框事件
        emotion_var.trace("w", update_heatmap)
        
        # 初始显示热力图
        update_heatmap()
        
        # 添加按钮
        button_frame = ttk.Frame(emotion_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame, 
            text="导出热力图", 
            command=lambda: self._export_emotion_heatmap(emotion_var.get().lower())
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="关闭", 
            command=emotion_dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        self.status_var.set("情感分析完成")

    def _show_emotion_heatmap(self):
        """显示情感热力图查看器"""
        # 检查是否存在情感数据
        if not hasattr(self, 'emotion_manager') or not self.emotion_manager.emotion_data:
            # 尝试执行情感分析
            self._show_emotion_analysis()
            return
        
        # 创建热力图预览对话框
        heatmap_dialog = tk.Toplevel(self.master)
        heatmap_dialog.title("情感热力图查看器")
        heatmap_dialog.geometry("800x600")
        
        # 创建控制面板
        control_frame = ttk.Frame(heatmap_dialog)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 情感选择
        emotions_frame = ttk.LabelFrame(control_frame, text="情感选择")
        emotions_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 创建情感复选框
        emotion_vars = {}
        for emotion in self.emotion_manager.primary_emotions:
            var = tk.BooleanVar(value=False)
            emotion_vars[emotion] = var
            ttk.Checkbutton(
                emotions_frame, 
                text=emotion.capitalize(), 
                variable=var,
                command=lambda: update_weights()
            ).pack(anchor=tk.W, padx=5, pady=2)
        
        # 权重调整
        weights_frame = ttk.LabelFrame(control_frame, text="情感权重")
        weights_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 储存权重滑块的引用
        weight_sliders = {}
        
        def update_weights():
            # 清空当前所有权重滑块
            for widget in weights_frame.winfo_children():
                widget.destroy()
            
            # 为选中的情感创建权重滑块
            row = 0
            for emotion in self.emotion_manager.primary_emotions:
                if emotion_vars[emotion].get():
                    frame = ttk.Frame(weights_frame)
                    frame.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
                    
                    ttk.Label(frame, text=f"{emotion.capitalize()}:").pack(side=tk.LEFT, padx=5)
                    
                    var = tk.DoubleVar(value=1.0)
                    slider = ttk.Scale(
                        frame, 
                        from_=0.1, 
                        to=5.0, 
                        variable=var,
                        orient="horizontal",
                        command=lambda v, e=emotion: update_preview()
                    )
                    slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                    
                    ttk.Label(frame, textvariable=var).pack(side=tk.LEFT, padx=5)
                    
                    weight_sliders[emotion] = var
                    row += 1
            
            # 如果没有选择任何情感，显示提示
            if not weight_sliders:
                ttk.Label(weights_frame, text="请选择至少一种情感").grid(padx=10, pady=10)
            
            # 更新预览
            update_preview()
        
        # 创建预览区域
        preview_frame = ttk.LabelFrame(heatmap_dialog, text="热力图预览")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 预览画布
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        def update_preview():
            # 清空当前预览
            for widget in canvas_frame.winfo_children():
                widget.destroy()
            
            # 获取选择的情感和权重
            selected_emotions = []
            weights = []
            
            for emotion in self.emotion_manager.primary_emotions:
                if emotion in weight_sliders:
                    selected_emotions.append(emotion)
                    weights.append(weight_sliders[emotion].get())
            
            # 如果没有选择情感，显示提示
            if not selected_emotions:
                ttk.Label(canvas_frame, text="请至少选择一种情感以生成热力图").pack(padx=20, pady=20)
                return
            
            # 获取热力图
            heatmap = self.emotion_manager.get_combined_emotion_heatmap(
                emotions=selected_emotions,
                weights=weights
            )
            
            if heatmap is None:
                ttk.Label(canvas_frame, text="无法生成热力图，请检查情感数据").pack(padx=20, pady=20)
                return
            
            # 创建图形
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # 创建自定义颜色映射
            colors = []
            for emotion in selected_emotions:
                if hasattr(self.emotion_manager, "_get_emotion_colormap"):
                    cmap = self.emotion_manager._get_emotion_colormap(emotion)
                    colors.append(cmap(0.8))  # 使用颜色映射的较亮部分
            
            if not colors:
                cmap = plt.get_cmap("viridis")
            else:
                cmap = LinearSegmentedColormap.from_list("custom", colors)
            
            # 显示热力图
            im = ax.imshow(heatmap, cmap=cmap, interpolation='bilinear')
            ax.set_title("自定义情感热力图")
            fig.colorbar(im, ax=ax, label="情感强度")
            
            # 添加到画布
            canvas = FigureCanvasTkAgg(fig, canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 添加工具栏
            toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加底部按钮
        button_frame = ttk.Frame(heatmap_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame, 
            text="导出热力图", 
            command=self._export_custom_emotion_heatmap
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="预设: 积极情感",
            command=lambda: self._apply_emotion_preset({
                "joy": 1.0, 
                "trust": 1.0, 
                "anticipation": 0.8
            }, emotion_vars, update_weights)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="预设: 消极情感",
            command=lambda: self._apply_emotion_preset({
                "fear": 1.0, 
                "anger": 1.0, 
                "sadness": 0.8, 
                "disgust": 0.6
            }, emotion_vars, update_weights)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="关闭", 
            command=heatmap_dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        # 初始化UI
        update_weights()

    def _apply_emotion_preset(self, preset, emotion_vars, update_callback):
        """应用情感预设"""
        # 重置所有复选框
        for emotion in emotion_vars:
            emotion_vars[emotion].set(emotion in preset)
        
        # 更新UI
        update_callback()

    def _export_emotion_heatmap(self, emotion_name):
        """导出情感热力图"""
        # 检查是否有情感数据
        if not hasattr(self, 'emotion_manager') or not self.emotion_manager.emotion_data:
            self.show_warning_dialog("没有可用的情感数据用于导出")
            return
        
        # 打开文件对话框
        filepath = filedialog.asksaveasfilename(
            title="导出情感热力图",
            filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg"), ("所有文件", "*.*")],
            defaultextension=".png"
        )
        
        if not filepath:
            return
        
        # 保存热力图
        success = self.emotion_manager.save_emotion_heatmap(emotion_name, filepath)
        
        if success:
            self.status_var.set(f"情感热力图已保存到 {filepath}")
        else:
            self.show_error_dialog("保存情感热力图失败")

    def _export_custom_emotion_heatmap(self):
        """导出自定义情感热力图"""
        # 由于此函数需要访问热力图对话框中的UI元素，需要在热力图对话框内部实现
        self.show_warning_dialog("请使用热力图查看器中的导出按钮")

    def _setup_level_tab(self):
        """设置关卡生成标签页"""
        # 创建带滚动条的框架
        canvas = tk.Canvas(self.level_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.level_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 关卡类型选择
        level_type_frame = ttk.LabelFrame(scrollable_frame, text="关卡类型")
        level_type_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.level_type_var = tk.StringVar(value="balanced")
        
        ttk.Radiobutton(level_type_frame, text="平衡型", 
                    variable=self.level_type_var, value="balanced").pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(level_type_frame, text="探索型", 
                    variable=self.level_type_var, value="exploration").pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(level_type_frame, text="战斗型", 
                    variable=self.level_type_var, value="combat").pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(level_type_frame, text="解谜型", 
                    variable=self.level_type_var, value="puzzle").pack(anchor=tk.W, padx=20, pady=2)
        
        # 关卡参数
        params_frame = ttk.LabelFrame(scrollable_frame, text="关卡参数")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 难度滑块
        difficulty_frame = ttk.Frame(params_frame)
        difficulty_frame.pack(fill=tk.X, pady=2)
        ttk.Label(difficulty_frame, text="难度:").pack(side=tk.LEFT, padx=5)
        self.difficulty_var = tk.DoubleVar(value=0.5)
        difficulty_slider = ttk.Scale(
            difficulty_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.difficulty_var,
            orient="horizontal"
        )
        difficulty_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(difficulty_frame, textvariable=self.difficulty_var).pack(side=tk.LEFT, padx=5)
        
        # 敌人密度滑块
        enemy_frame = ttk.Frame(params_frame)
        enemy_frame.pack(fill=tk.X, pady=2)
        ttk.Label(enemy_frame, text="敌人密度:").pack(side=tk.LEFT, padx=5)
        self.enemy_density_var = tk.DoubleVar(value=0.5)
        enemy_slider = ttk.Scale(
            enemy_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.enemy_density_var,
            orient="horizontal"
        )
        enemy_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(enemy_frame, textvariable=self.enemy_density_var).pack(side=tk.LEFT, padx=5)
        
        # 奖励多样性滑块
        reward_frame = ttk.Frame(params_frame)
        reward_frame.pack(fill=tk.X, pady=2)
        ttk.Label(reward_frame, text="奖励多样性:").pack(side=tk.LEFT, padx=5)
        self.reward_variety_var = tk.DoubleVar(value=0.7)
        reward_slider = ttk.Scale(
            reward_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.reward_variety_var,
            orient="horizontal"
        )
        reward_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(reward_frame, textvariable=self.reward_variety_var).pack(side=tk.LEFT, padx=5)
        
        # 路径复杂度滑块
        path_frame = ttk.Frame(params_frame)
        path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame, text="路径复杂度:").pack(side=tk.LEFT, padx=5)
        self.path_complexity_var = tk.DoubleVar(value=0.6)
        path_slider = ttk.Scale(
            path_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.path_complexity_var,
            orient="horizontal"
        )
        path_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(path_frame, textvariable=self.path_complexity_var).pack(side=tk.LEFT, padx=5)
        
        # 高级选项
        advanced_frame = ttk.LabelFrame(scrollable_frame, text="高级选项")
        advanced_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 生成关卡起点终点
        point_frame = ttk.Frame(advanced_frame)
        point_frame.pack(fill=tk.X, pady=2)
        self.auto_endpoints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(point_frame, text="自动确定起点和终点", 
                    variable=self.auto_endpoints_var).pack(side=tk.LEFT, padx=5)
        
        # 多路径选项
        path_option_frame = ttk.Frame(advanced_frame)
        path_option_frame.pack(fill=tk.X, pady=2)
        self.multi_path_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(path_option_frame, text="允许多条路径", 
                    variable=self.multi_path_var).pack(side=tk.LEFT, padx=5)
        
        # 分支因子
        branch_frame = ttk.Frame(advanced_frame)
        branch_frame.pack(fill=tk.X, pady=2)
        ttk.Label(branch_frame, text="分支因子:").pack(side=tk.LEFT, padx=5)
        self.branch_factor_var = tk.DoubleVar(value=0.4)
        branch_slider = ttk.Scale(
            branch_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.branch_factor_var,
            orient="horizontal"
        )
        branch_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(branch_frame, textvariable=self.branch_factor_var).pack(side=tk.LEFT, padx=5)
        
        # 结点类型配置
        nodes_frame = ttk.LabelFrame(scrollable_frame, text="节点类型分布")
        nodes_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 战斗节点比例
        combat_frame = ttk.Frame(nodes_frame)
        combat_frame.pack(fill=tk.X, pady=2)
        ttk.Label(combat_frame, text="战斗节点:").pack(side=tk.LEFT, padx=5)
        self.combat_ratio_var = tk.DoubleVar(value=0.3)
        combat_slider = ttk.Scale(
            combat_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.combat_ratio_var,
            orient="horizontal"
        )
        combat_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(combat_frame, textvariable=self.combat_ratio_var).pack(side=tk.LEFT, padx=5)
        
        # 解谜节点比例
        puzzle_frame = ttk.Frame(nodes_frame)
        puzzle_frame.pack(fill=tk.X, pady=2)
        ttk.Label(puzzle_frame, text="解谜节点:").pack(side=tk.LEFT, padx=5)
        self.puzzle_ratio_var = tk.DoubleVar(value=0.2)
        puzzle_slider = ttk.Scale(
            puzzle_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.puzzle_ratio_var,
            orient="horizontal"
        )
        puzzle_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(puzzle_frame, textvariable=self.puzzle_ratio_var).pack(side=tk.LEFT, padx=5)
        
        # 宝藏节点比例
        treasure_frame = ttk.Frame(nodes_frame)
        treasure_frame.pack(fill=tk.X, pady=2)
        ttk.Label(treasure_frame, text="宝藏节点:").pack(side=tk.LEFT, padx=5)
        self.treasure_ratio_var = tk.DoubleVar(value=0.15)
        treasure_slider = ttk.Scale(
            treasure_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.treasure_ratio_var,
            orient="horizontal"
        )
        treasure_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(treasure_frame, textvariable=self.treasure_ratio_var).pack(side=tk.LEFT, padx=5)
        
        # 添加关卡预览按钮
        preview_frame = ttk.Frame(scrollable_frame)
        preview_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(preview_frame, text="预览关卡", 
                command=self._preview_level_layout).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(preview_frame, text="导出关卡数据", 
                command=self._export_level_data).pack(side=tk.LEFT, padx=5)
        
    def _preview_level_layout(self):
        """预览关卡布局"""
        # 检查地图是否已生成
        if not self.map_data or not self.map_data.is_valid():
            messagebox.showinfo("提示", "请先生成地图，再预览关卡")
            return
            
        # 创建关卡数据
        level_params = {
            "level_type": self.level_type_var.get(),
            "difficulty": self.difficulty_var.get(),
            "enemy_density": self.enemy_density_var.get(),
            "reward_variety": self.reward_variety_var.get(),
            "path_complexity": self.path_complexity_var.get(),
            "auto_endpoints": self.auto_endpoints_var.get(),
            "multi_path": self.multi_path_var.get(),
            "branch_factor": self.branch_factor_var.get(),
            "node_ratios": {
                "combat": self.combat_ratio_var.get(),
                "puzzle": self.puzzle_ratio_var.get(),
                "treasure": self.treasure_ratio_var.get()
            }
        }
        
        try:
            # 创建关卡生成器
            level_generator = LevelGenerator(self.map_data, level_params, self.logger)
            level_data = level_generator.generate_level(
                level_type=level_params["level_type"],
                difficulty=level_params["difficulty"]
            )
            
            # 保存关卡数据到地图
            self.map_data.level_data = level_data
            
            # 打开预览窗口
            self._show_level_preview(level_data)
            
        except Exception as e:
            self.logger.error(f"关卡生成失败: {str(e)}")
            messagebox.showerror("错误", f"关卡生成失败: {str(e)}")

    def _show_level_preview(self, level_data):
        """显示关卡预览窗口"""
        preview_window = tk.Toplevel(self.master)
        preview_window.title("关卡预览")
        preview_window.geometry("800x600")
        
        # 创建画布
        canvas_frame = ttk.Frame(preview_window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(canvas_frame, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # 获取地图数据
        height_map = self.map_data.get_layer("height")
        biome_map = self.map_data.get_layer("biome")
        
        if height_map is None or biome_map is None:
            messagebox.showinfo("提示", "地图数据不完整，无法显示预览")
            return
        
        # 计算缩放因子 - 修改为更好地填充画布
        map_height, map_width = height_map.shape
        canvas_height = 500
        canvas_width = 700
        scale_y = canvas_height / map_height
        scale_x = canvas_width / map_width
        scale = min(scale_x, scale_y) * 0.9  # 留出一点边距
        
        # 绘制地形底图
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # 创建图形 - 调整图形大小以匹配画布比例
            fig = Figure(figsize=(canvas_width/100, canvas_height/100), dpi=100)
            
            # 创建子图并绘制高度图 - 删除多余边距
            ax = fig.add_subplot(111)
            cax = ax.imshow(height_map, cmap='terrain', interpolation='bilinear', 
                        extent=[0, map_width, 0, map_height])
            fig.colorbar(cax, ax=ax, label='高度')
            ax.set_title('关卡地形图')
            ax.axis('off')
            fig.tight_layout(pad=0.5)  # 减少边距
            
            # 在canvas中嵌入matplotlib图形
            canvas_matplotlib = FigureCanvasTkAgg(fig, master=canvas)
            canvas_matplotlib.draw()
            canvas_matplotlib.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
            # 绘制关卡节点和路径
            def draw_level_on_plot():
                # 绘制节点
                nodes = []
                if "start_point" in level_data:
                    nodes.append(level_data["start_point"])
                if "end_point" in level_data:
                    nodes.append(level_data["end_point"])
                if "nodes" in level_data:
                    nodes.extend(level_data["nodes"])
                    
                for node in nodes:
                    x, y = node.get("x", 0), node.get("y", 0)
                    node_type = node.get("type", "generic")
                    
                    # 根据节点类型选择颜色
                    if node_type == "start":
                        color = "lime"
                        marker = "o"
                        size = 100
                    elif node_type == "end":
                        color = "red"
                        marker = "o"
                        size = 100
                    elif node_type == "combat":
                        color = "orange"
                        marker = "s"  # 方形
                        size = 50
                    elif node_type == "elite":
                        color = "darkred"
                        marker = "P"  # 五角星
                        size = 80
                    elif node_type == "puzzle":
                        color = "cyan"
                        marker = "D"  # 菱形
                        size = 50
                    elif node_type == "treasure":
                        color = "gold"
                        marker = "*"  # 星形
                        size = 80
                    elif node_type == "rest":
                        color = "green"
                        marker = "h"  # 六边形
                        size = 50
                    elif node_type == "discovery":
                        color = "purple"
                        marker = "X"
                        size = 50
                    else:
                        color = "gray"
                        marker = "o"
                        size = 30
                    
                    ax.scatter(x, y, c=color, marker=marker, s=size, edgecolors='black', zorder=10)
                    
                # 绘制路径
                if "paths" in level_data:
                    for path in level_data["paths"]:
                        points = path.get("points", [])
                        xs = [p["x"] for p in points]
                        ys = [p["y"] for p in points]
                        
                        # 根据路径难度选择线条样式
                        difficulty = path.get("difficulty", 0.5)
                        if difficulty < 0.3:
                            linestyle = "-"
                            color = "green"
                            width = 1
                        elif difficulty < 0.7:
                            linestyle = "--"
                            color = "blue"
                            width = 1.5
                        else:
                            linestyle = ":"
                            color = "red"
                            width = 2
                            
                        # 绘制路径
                        if xs and ys:
                            ax.plot(xs, ys, linestyle=linestyle, color=color, 
                                linewidth=width, alpha=0.7, zorder=5)
            
            # 调用绘制函数
            draw_level_on_plot()
            canvas_matplotlib.draw()
            
        except ImportError:
            # 如果matplotlib不可用，使用基本绘图
            self.logger.warning("matplotlib不可用，使用基本绘图预览")
            
            # 创建缩放后的图像
            scaled_width = int(map_width * scale)
            scaled_height = int(map_height * scale)
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            offset_x = canvas_center_x - scaled_width // 2
            offset_y = canvas_center_y - scaled_height // 2
            
            # 绘制地图底图 - 创建缩放版本
            image = tk.PhotoImage(width=scaled_width, height=scaled_height)
            for y in range(scaled_height):
                for x in range(scaled_width):
                    # 计算对应的原始坐标
                    orig_y = min(int(y / scale), map_height - 1)
                    orig_x = min(int(x / scale), map_width - 1)
                    
                    # 根据高度值生成颜色
                    h = height_map[orig_y, orig_x]
                    # 简单的高度映射到灰度值 (0-255)
                    color_val = min(255, max(0, int(h * 2.55)))
                    # 生成颜色字符串
                    color = f'#{color_val:02x}{color_val:02x}{color_val:02x}'
                    image.put(color, (x, y))
            
            # 将图像放在画布中心
            canvas.create_image(canvas_center_x, canvas_center_y, image=image)
            canvas.image = image  # 保持引用，防止垃圾回收
            
            # 绘制节点
            nodes = []
            if "start_point" in level_data:
                nodes.append(level_data["start_point"])
            if "end_point" in level_data:
                nodes.append(level_data["end_point"])
            if "nodes" in level_data:
                nodes.extend(level_data["nodes"])
                
            for node in nodes:
                x, y = node.get("x", 0), node.get("y", 0)
                node_type = node.get("type", "generic")
                
                # 计算画布上的坐标
                canvas_x = offset_x + x * scale
                canvas_y = offset_y + y * scale
                
                # 根据节点类型选择颜色
                if node_type == "start":
                    color = "green"
                    size = 10
                elif node_type == "end":
                    color = "red"
                    size = 10
                elif node_type == "combat":
                    color = "orange"
                    size = 5
                elif node_type == "elite":
                    color = "darkred"
                    size = 7
                elif node_type == "puzzle":
                    color = "blue"
                    size = 5
                elif node_type == "treasure":
                    color = "gold"
                    size = 6
                elif node_type == "rest":
                    color = "green"
                    size = 5
                else:
                    color = "gray"
                    size = 4
                
                # 绘制节点（椭圆）
                canvas.create_oval(
                    canvas_x - size, canvas_y - size,
                    canvas_x + size, canvas_y + size,
                    fill=color, outline="black"
                )
                
                # 添加标签
                label_text = node_type
                canvas.create_text(
                    canvas_x, canvas_y + size + 5,
                    text=label_text,
                    fill="black",
                    font=("Arial", 7)
                )
            
            # 绘制路径
            if "paths" in level_data:
                for path in level_data["paths"]:
                    points = path.get("points", [])
                    
                    # 根据路径难度选择线条样式
                    difficulty = path.get("difficulty", 0.5)
                    if difficulty < 0.3:
                        color = "green"
                        width = 1
                    elif difficulty < 0.7:
                        color = "blue"
                        width = 1.5
                    else:
                        color = "red"
                        width = 2
                    
                    # 至少需要两个点才能绘制路径
                    if len(points) >= 2:
                        # 创建坐标列表
                        coords = []
                        for p in points:
                            coords.extend([offset_x + p["x"] * scale, offset_y + p["y"] * scale])
                        
                        # 绘制路径线条
                        canvas.create_line(
                            coords,
                            fill=color,
                            width=width,
                            smooth=True
                        )
        
        # 添加缩放控件
        control_frame = ttk.Frame(preview_window)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 显示关卡信息
        info_frame = ttk.LabelFrame(control_frame, text="关卡信息")
        info_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # 获取关卡类型和难度
        level_type = level_data.get("level_type", "未知")
        difficulty = level_data.get("difficulty", 0.5)
        node_count = len(level_data.get("nodes", [])) + 2  # 加上起点和终点
        
        info_text = f"关卡类型: {level_type}\n"
        info_text += f"难度: {difficulty:.2f}\n"
        info_text += f"节点数量: {node_count}\n"
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(padx=5, pady=5)
        
        # 添加按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 刷新按钮
        ttk.Button(button_frame, text="刷新预览", 
                command=lambda: self._preview_level_layout()).pack(side=tk.LEFT, padx=5)
        
        # 导出按钮
        ttk.Button(button_frame, text="导出关卡", 
                command=lambda: self._export_level_data()).pack(side=tk.LEFT, padx=5)
        
        # 关闭按钮
        ttk.Button(button_frame, text="关闭", 
                command=preview_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # 为窗口添加快捷键
        preview_window.bind("<Escape>", lambda e: preview_window.destroy())

    def _export_level_data(self):
        """导出关卡数据到文件"""
        if not hasattr(self.map_data, 'level_data') or not self.map_data.level_data:
            messagebox.showinfo("提示", "没有可用的关卡数据可导出")
            return
        
        # 弹出保存文件对话框
        filepath = filedialog.asksaveasfilename(
            title="导出关卡数据",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            defaultextension=".json"
        )
        
        if not filepath:
            return
        
        try:
            # 准备导出数据
            import json
            import numpy as np
            
            # 创建可序列化的数据结构
            export_data = {
                "level_type": self.map_data.level_data.get("level_type", ""),
                "difficulty": self.map_data.level_data.get("difficulty", 0.5),
                "width": self.map_data.width,
                "height": self.map_data.height,
                "start_point": self.map_data.level_data.get("start_point", {}),
                "end_point": self.map_data.level_data.get("end_point", {}),
                "nodes": self.map_data.level_data.get("nodes", []),
                "paths": self.map_data.level_data.get("paths", []),
                "metadata": {
                    "creation_time": self.map_data.level_data.get("creation_time", ""),
                    "generator_version": self.map_data.level_data.get("generator_version", "1.0")
                }
            }
            
            # 自定义JSON编码器，处理NumPy类型
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            self.logger.log(f"关卡数据已导出到 {filepath}")
            messagebox.showinfo("成功", f"关卡数据已导出到 {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出关卡数据失败: {str(e)}")
            messagebox.showerror("错误", f"导出关卡数据失败: {str(e)}")

    def _preview_style_transformation(self, style_name, style_options):
        """预览指定样式的地图转换"""
        if not self.map_data or not self.map_data.is_valid():
            self.logger.error("无地图数据可预览，请先生成地图")
            return
        
        from utils.style_transformers import get_available_styles
        
        available_styles = get_available_styles()
        transformer_class = available_styles.get(style_name)
        
        if style_name == "default" or transformer_class is None:
            preview_map(self.map_data, self.master)
            return
        
        transformer = transformer_class(self.map_data, config=style_options)
        
        preview_window = tk.Toplevel(self.master)
        preview_window.title(f"{style_name.capitalize()} 风格预览")
        preview_window.geometry("800x600")
        
        try:
            preview_img = transformer.get_preview_image(width=780, height=520)
            
            from PIL import ImageTk
            tk_img = ImageTk.PhotoImage(preview_img)
            
            canvas = tk.Canvas(preview_window, width=780, height=520)
            canvas.pack(pady=10)
            canvas.create_image(390, 260, image=tk_img)
            canvas.image = tk_img
            
            info_text = f"预览 {style_name.capitalize()} 风格转换\n有关完整功能，请导出地图"
            ttk.Label(preview_window, text=info_text, justify=tk.CENTER).pack(pady=5)
            
            ttk.Button(preview_window, text="关闭", command=preview_window.destroy).pack(pady=10)
            
        except Exception as e:
            self.logger.error(f"生成预览失败: {str(e)}")
            import traceback
            self.logger.log(traceback.format_exc(), "ERROR")
            preview_window.destroy()
            messagebox.showerror("预览错误", f"生成预览时出错:\n{str(e)}")
            
    def _setup_core_tab(self):
        self.param_controls = {}
        
        # 创建带滚动条的框架
        canvas = tk.Canvas(self.core_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.core_tab, orient="vertical", command=canvas.yview)
        core_scrollable = ttk.Frame(canvas)
        
        core_scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=core_scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 添加参数预设选择器
        preset_frame = ttk.LabelFrame(core_scrollable, text="参数预设")
        preset_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(preset_frame, text="快速配置:").pack(side=tk.LEFT, padx=5)
        presets = list(self.config_manager.get("presets", {}).keys())
        if not presets:
            presets = ["默认"]  # 如果没有预设，提供默认值
        preset_var = tk.StringVar(value=presets[0] if presets else "默认")
        preset_combo = ttk.Combobox(preset_frame, textvariable=preset_var, values=presets, width=15)
        preset_combo.pack(side=tk.LEFT, padx=5, pady=5)
        self.param_controls["preset_combo"] = preset_combo  # 保存引用以便后续更新

        # 应用预设按钮
        ttk.Button(preset_frame, text="应用预设", 
                command=lambda: self._apply_preset(preset_var.get())).pack(side=tk.LEFT, padx=5)

        # 添加保存预设按钮
        ttk.Button(preset_frame, text="保存为预设", 
                command=self._save_as_preset).pack(side=tk.LEFT, padx=5)
        
        # 玩法参数
        gameplay_frame = ttk.LabelFrame(core_scrollable, text="游戏风格偏好")
        gameplay_frame.pack(fill=tk.X, padx=10, pady=5)
        
        gameplay_params = [
            ("achievement", "成就导向", 0.0, 1.0, "影响地图上的成就点和收集元素密度"),
            ("exploration", "探索要素", 0.0, 1.0, "影响地图的复杂程度和探索的价值"),
            ("social", "社交互动", 0.0, 1.0, "影响城镇、村庄和社交区域的分布"),
            ("combat", "战斗强度", 0.0, 1.0, "影响敌对区域和战斗挑战的难度")
        ]
        
        for i, (param, label, min_val, max_val, tooltip) in enumerate(gameplay_params):
            self._create_enhanced_slider(gameplay_frame, param, label, min_val, max_val, i, tooltip)
        
        # 地图尺寸参数
        size_frame = ttk.LabelFrame(core_scrollable, text="地图尺寸与内容")
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 地图尺寸使用横向布局
        map_size_frame = ttk.Frame(size_frame)
        map_size_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(map_size_frame, text="地图尺寸:").grid(row=0, column=0, sticky="w", padx=5)
        self._create_dimension_control(map_size_frame, "map_width", "map_height", "宽度", "高度", 
                                    10, 200000, 10, 200000)
        
        # 内容密度参数
        content_params = [
            ("vegetation_coverage", "植被覆盖率", 0.0, 1.0, "控制地图上植被的密度"),
            ("river_count", "河流数量", 0, 50, "控制地图上河流的数量"),
            ("city_count", "城市数量", 0, 50, "控制地图上城市和村庄的数量"),
            ("cave_density", "洞穴密度", 0.0, 1.0, "控制地图上洞穴和地下区域的密度")
        ]
        
        for i, (param, label, min_val, max_val, tooltip) in enumerate(content_params):
            control_type = "slider" if isinstance(min_val, float) else "spinbox"
            if control_type == "slider":
                self._create_enhanced_slider(size_frame, param, label, min_val, max_val, i, tooltip)
            else:
                self._create_enhanced_spinbox(size_frame, param, label, min_val, max_val, i, tooltip)

    def _save_as_preset(self):
        """将当前参数保存为新预设"""
        from tkinter import simpledialog
        
        preset_name = simpledialog.askstring("保存预设", "请输入预设名称：")
        if not preset_name:
            return
            
        # 检查预设名称是否已存在
        presets = self.config_manager.get("presets", {})
        if preset_name in presets:
            if not messagebox.askyesno("确认覆盖", f"预设名称'{preset_name}'已存在，是否覆盖？"):
                return
        
        # 收集当前参数
        params = {}
        core_params = ["achievement", "exploration", "social", "combat", 
                    "map_width", "map_height", "vegetation_coverage", 
                    "river_count", "city_count", "cave_density",
                    "scale_factor", "mountain_sharpness", "river_density"]
        
        for param in core_params:
            if hasattr(self.map_params, param):
                params[param] = getattr(self.map_params, param)
        
        # 保存到预设
        presets[preset_name] = params
        self.config_manager.set("presets", presets)
        
        # 更新预设下拉列表
        preset_combo = self.param_controls.get("preset_combo")
        if preset_combo:
            preset_combo["values"] = list(presets.keys())
            preset_combo.set(preset_name)
        
        self.logger.log(f"已保存新预设: {preset_name}")
        self.status_var.set(f"已保存新预设: {preset_name}")

    def _create_enhanced_spinbox(self, parent, param, label, min_val, max_val, row, tooltip=None):
        """创建带工具提示的增强数字调节框控件"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        # 标签
        label_widget = ttk.Label(frame, text=label, width=12)
        label_widget.pack(side=tk.LEFT)
        
        # 如果有提示，添加工具提示
        if tooltip:
            ToolTip(label_widget, tooltip)
        
        # 创建变量和数字调节框
        var = tk.IntVar(value=getattr(self.map_params, param))
        spinbox = ttk.Spinbox(
            frame, 
            from_=min_val, 
            to=max_val, 
            textvariable=var, 
            command=lambda: self._on_param_change(param, var.get()),
            width=8
        )
        spinbox.pack(side=tk.RIGHT, padx=5)
        
        # 添加回车键和失去焦点事件绑定
        spinbox.bind("<Return>", lambda e, p=param, v=var: self._on_param_change(p, v.get()))
        spinbox.bind("<FocusOut>", lambda e, p=param, v=var: self._on_param_change(p, v.get()))
        
        # 存储控件引用
        self.param_controls[param] = {"var": var, "spinbox": spinbox}
                
    def _setup_terrain_tab(self):
        """设置地形标签页"""
        # 创建带滚动条的框架 - 只保留这一种方式
        canvas = tk.Canvas(self.terrain_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.terrain_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 基础地形参数 - 使用scrollable_frame作为父容器
        basic_frame = ttk.LabelFrame(scrollable_frame, text="基础地形参数", padding=(5, 5, 5, 5))
        basic_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加地形基本参数控件
        terrain_params = [
            ("scale_factor", "地形比例", 0.1, 3.0, "控制整体地形的缩放比例"),
            ("mountain_sharpness", "山地锐度", 0.1, 3.0, "控制山峰的锐利程度"),
            ("erosion_iterations", "侵蚀迭代", 0, 10, "地形侵蚀计算的迭代次数"),
            ("river_density", "河流密度", 0.1, 2.0, "控制河流的密度")
        ]

        # 添加新的大地图设置框架 - 使用scrollable_frame作为父容器
        large_map_frame = ttk.LabelFrame(scrollable_frame, text="大地图设置", padding=(5, 5, 5, 5))
        large_map_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 启用大地图模式复选框
        large_map_var = tk.BooleanVar(value=self.params.get("large_map_mode", False))
        self.param_vars["large_map_mode"] = large_map_var
        large_map_cb = ttk.Checkbutton(
            large_map_frame,
            text="启用大地图模式",
            variable=large_map_var,
            command=lambda: self._on_param_change("large_map_mode", large_map_var.get())
        )
        large_map_cb.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        self._create_tooltip(large_map_cb, "启用后将生成更丰富的大尺度地貌特征，适用于大型地图")
        
        # 地质省份数量设置
        ttk.Label(large_map_frame, text="地质省份数量:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        province_var = tk.IntVar(value=self.params.get("province_count", 3))
        self.param_vars["province_count"] = province_var
        province_spinbox = ttk.Spinbox(
            large_map_frame,
            from_=1,
            to=15,
            textvariable=province_var,
            width=5,
            command=lambda: self._on_param_change("province_count", province_var.get())
        )
        province_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        province_spinbox.bind("<Return>", lambda e: self._on_param_change("province_count", province_var.get()))
        province_spinbox.bind("<FocusOut>", lambda e: self._on_param_change("province_count", province_var.get()))
        self._create_tooltip(province_spinbox, "控制地图中大尺度地质区域的数量，影响地形多样性")
        
        # 宏观特征缩放因子
        ttk.Label(large_map_frame, text="宏观特征缩放:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        macro_scale_var = tk.DoubleVar(value=self.params.get("macro_feature_scale", 2.0))
        self.param_vars["macro_feature_scale"] = macro_scale_var
        macro_scale = ttk.Scale(
            large_map_frame,
            from_=0.5,
            to=4.0,
            variable=macro_scale_var,
            orient=tk.HORIZONTAL,
            command=lambda v: self._on_param_change("macro_feature_scale", macro_scale_var.get())
        )
        macro_scale.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(large_map_frame, textvariable=macro_scale_var, width=4).grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        self._create_tooltip(macro_scale, "控制大地形特征的强度，值越大特征越明显")
        
        # 添加说明标签
        info_label = ttk.Label(
            large_map_frame, 
            text="大地图模式会在大尺度上生成更丰富多样的地质特征，包括山脉链、高原区域和裂谷等。建议在地图尺寸大于1024×1024时启用。",
            wraplength=350,
            foreground="#666666",
            font=('', 8),
            justify=tk.LEFT
        )
        info_label.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky=tk.W)
        
        # 自适应大地图提示
        adapt_frame = ttk.Frame(large_map_frame)
        adapt_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)
        
        auto_detect_var = tk.BooleanVar(value=True)
        self.param_vars["auto_detect_large_map"] = auto_detect_var
        auto_detect_cb = ttk.Checkbutton(
            adapt_frame,
            text="自动检测大地图模式",
            variable=auto_detect_var
        )
        auto_detect_cb.pack(side=tk.LEFT, padx=5)
        self._create_tooltip(auto_detect_cb, "启用后将根据地图尺寸自动决定是否使用大地图模式")
        
        for i, (param, label, min_val, max_val, tooltip) in enumerate(terrain_params):
            control_type = "slider" if isinstance(min_val, float) else "spinbox"
            if control_type == "slider":
                self._create_enhanced_slider(basic_frame, param, label, min_val, max_val, i, tooltip)
            else:
                self._create_spinbox(basic_frame, param, label, min_val, max_val, i)
        
        # 添加提示信息
        ttk.Label(scrollable_frame, text="更多高级地形设置请使用顶部的「地形高级设置」按钮").pack(pady=10)

    def _setup_biome_tab(self):
        """设置生物群系标签页"""
        # 创建带滚动条的框架
        canvas = tk.Canvas(self.biome_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.biome_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 生物群系分布框架
        biome_frame = ttk.LabelFrame(scrollable_frame, text="生物群系分布")
        biome_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 添加生物群系参数控件
        biome_params = [
            ("forest_coverage", "森林覆盖", 0.0, 1.0, "控制森林生物群系的比例"),
            ("desert_coverage", "沙漠覆盖", 0.0, 1.0, "控制沙漠生物群系的比例"),
            ("mountain_coverage", "山地覆盖", 0.0, 1.0, "控制山地生物群系的比例"),
            ("water_coverage", "水域覆盖", 0.0, 1.0, "控制水域生物群系的比例")
        ]
        
        for i, (param, label, min_val, max_val, tooltip) in enumerate(biome_params):
            self._create_enhanced_slider(biome_frame, param, label, min_val, max_val, i, tooltip)
        
        # 气候参数框架
        climate_frame = ttk.LabelFrame(scrollable_frame, text="气候设置")
        climate_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 气候参数
        climate_params = [
            ("temperature_range", "温度范围", 0.0, 1.0, "控制区域间温度变化的剧烈程度"),
            ("humidity_range", "湿度范围", 0.0, 1.0, "控制区域间湿度变化的剧烈程度")
        ]
        
        for i, (param, label, min_val, max_val, tooltip) in enumerate(climate_params):
            self._create_enhanced_slider(climate_frame, param, label, min_val, max_val, i, tooltip)

    def _setup_attribute_ranges(self, parent_frame):
        """设置属性范围控制"""
        # 创建属性范围框架
        attr_frame = ttk.LabelFrame(parent_frame, text="属性范围设置")
        attr_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 添加说明标签
        ttk.Label(attr_frame, 
                text="这些设置控制地图上各元素的属性值范围",
                wraplength=350).pack(fill=tk.X, padx=5, pady=5)
        
        # 添加属性范围控制
        attributes = [
            ("strength", "力量", 1, 100),
            ("agility", "敏捷", 1, 100),
            ("intelligence", "智力", 1, 100),
            ("vitality", "生命", 10, 200),
            ("armor", "防御", 0, 50)
        ]
        
        for i, (attr_name, label, min_val, max_val) in enumerate(attributes):
            self._create_attr_range_control(attr_frame, attr_name, min_val, max_val, i)
        
        # 添加目标比率控制
        ratio_frame = ttk.Frame(attr_frame)
        ratio_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ratio_frame, text="目标生态比率:").pack(side=tk.LEFT, padx=5)
        
        global TARGET_RATIO
        if "TARGET_RATIO" in globals():
            self.target_ratio_var.set(TARGET_RATIO)
        
        ratio_slider = ttk.Scale(
            ratio_frame, 
            from_=0.1, 
            to=10.0, 
            variable=self.target_ratio_var,
            command=lambda v: self._update_target_ratio()
        )
        ratio_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ratio_entry = ttk.Entry(
            ratio_frame, 
            width=6, 
            textvariable=self.target_ratio_var
        )
        ratio_entry.pack(side=tk.RIGHT, padx=5)
        
        # 添加回车键和失去焦点事件绑定
        ratio_entry.bind("<Return>", lambda e: self._update_target_ratio())
        ratio_entry.bind("<FocusOut>", lambda e: self._update_target_ratio())

    def _setup_advanced_tab(self):
        """设置高级选项标签页"""
        # 创建带滚动条的框架
        canvas = tk.Canvas(self.advanced_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.advanced_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 添加可视化控制框架
        vis_frame = ttk.LabelFrame(scrollable_frame, text="可视化设置")
        vis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 可视化开关
        self.visualize_var = tk.BooleanVar(value=True)  # 默认启用
        ttk.Checkbutton(
            vis_frame, 
            text="启用生成过程可视化", 
            variable=self.visualize_var
        ).pack(anchor=tk.W, padx=10, pady=2)
        
        # 嵌入预览选项
        self.embed_preview_var = tk.BooleanVar(value=True)  # 默认嵌入
        ttk.Checkbutton(
            vis_frame, 
            text="将可视化嵌入到预览窗口", 
            variable=self.embed_preview_var
        ).pack(anchor=tk.W, padx=10, pady=2)
        
        # 保存可视化选项
        save_vis_frame = ttk.Frame(vis_frame)
        save_vis_frame.pack(fill=tk.X, padx=10, pady=2)
        
        self.save_visualization_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            save_vis_frame, 
            text="保存可视化图像", 
            variable=self.save_visualization_var,
            command=self._toggle_vis_path
        ).pack(side=tk.LEFT)
        
        # 可视化路径 
        self.visualization_path = tk.StringVar(value="./visualization")
        vis_path_entry = ttk.Entry(save_vis_frame, textvariable=self.visualization_path, width=20)
        vis_path_entry.pack(side=tk.LEFT, padx=5)
        vis_path_entry.config(state="disabled" if not self.save_visualization_var.get() else "normal")
        
        ttk.Button(
            save_vis_frame, 
            text="浏览...", 
            command=self._browse_vis_path
        ).pack(side=tk.LEFT)

        # 进化参数框架
        evolution_frame = ttk.LabelFrame(scrollable_frame, text="进化参数设置")
        evolution_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 添加进化参数控件
        self._create_spinbox(evolution_frame, "evolution_generations", "进化代数", 1, 100, 0)
        
        # 性能设置框架
        perf_frame = ttk.LabelFrame(scrollable_frame, text="性能设置")
        perf_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 添加地貌特征控制框架
        landform_frame = ttk.LabelFrame(scrollable_frame, text="真实地貌特征")
        landform_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 启用真实地貌特征选项
        self.enable_landforms_var = tk.BooleanVar(value=getattr(self.map_params, "enable_realistic_landforms", True))
        landforms_frame = ttk.Frame(landform_frame)
        landforms_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(landforms_frame, text="启用真实地貌特征").pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(landforms_frame, variable=self.enable_landforms_var, 
                    command=lambda: self._on_param_change("enable_realistic_landforms", 
                                                        self.enable_landforms_var.get())).pack(side=tk.RIGHT, padx=5)
        
        # 添加属性范围设置
        self._setup_attribute_ranges(scrollable_frame)
        
        # 主导地貌类型选择
        landform_type_frame = ttk.Frame(landform_frame)
        landform_type_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(landform_type_frame, text="主导地貌类型:").pack(side=tk.LEFT, padx=5)
        
        self.landform_type_var = tk.StringVar(value=getattr(self.map_params, "dominant_landform", "auto"))
        landform_combo = ttk.Combobox(
            landform_type_frame,
            textvariable=self.landform_type_var,
            values=["自动", "褶皱山脉", "断块山", "火山山脉",
            "V形谷", "曲流河", "三角洲平原",
            "U形谷", "冰斗"],
            width=15,
            state="readonly"
        )
        landform_combo.pack(side=tk.RIGHT, padx=5)
        landform_combo.bind("<<ComboboxSelected>>", 
                        lambda e: self._on_param_change("dominant_landform", 
                                                        self.landform_type_var.get()))
        
        # 添加说明标签
        ttk.Label(landform_frame, 
                text="提示: 启用真实地貌特征将根据地形学科学标准增强地形形态，使其更接近自然界的真实地貌",
                wraplength=350).pack(fill=tk.X, padx=5, pady=5)
        
        # 地形生成性能设置框架
        terrain_perf_frame = ttk.LabelFrame(scrollable_frame, text="地形生成性能")
        terrain_perf_frame.pack(fill=tk.X, padx=10, pady=5)
        perf_ref_frame = ttk.Frame(terrain_perf_frame)
        perf_ref_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(perf_ref_frame, text="性能参考:", font=("", 9, "bold")).pack(anchor=tk.W)
        perf_text = tk.Text(perf_ref_frame, height=4, width=40, font=("", 9))
        perf_text.pack(fill=tk.X, pady=2)
        perf_text.insert("1.0", "高性能模式: 关闭微细节, 关闭异常检测, 优化级别0 (5-10倍速)\n")
        perf_text.insert("2.0", "平衡模式: 开启微细节, 开启异常检测, 优化级别1 (推荐)\n")
        perf_text.insert("3.0", "高质量模式: 开启微细节, 开启异常检测, 优化级别2 (0.5倍速)")
        perf_text.config(state="disabled")
        
        # 添加微细节控制选项
        self.micro_detail_var = tk.BooleanVar(value=getattr(self.map_params, "enable_micro_detail", True))
        micro_frame = ttk.Frame(terrain_perf_frame)
        micro_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(micro_frame, text="启用微细节生成").pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(micro_frame, variable=self.micro_detail_var, 
                        command=lambda: self._on_param_change("enable_micro_detail", 
                                                        self.micro_detail_var.get())).pack(side=tk.RIGHT, padx=5)
        
        # 添加异常检测选项
        self.extreme_detection_var = tk.BooleanVar(value=getattr(self.map_params, "enable_extreme_detection", True))
        extreme_frame = ttk.Frame(terrain_perf_frame)
        extreme_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(extreme_frame, text="启用异常点检测").pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(extreme_frame, variable=self.extreme_detection_var,
                    command=lambda: self._on_param_change("enable_extreme_detection", 
                                                        self.extreme_detection_var.get())).pack(side=tk.RIGHT, padx=5)
        
        # 添加优化级别选项
        optimization_frame = ttk.Frame(terrain_perf_frame)
        optimization_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(optimization_frame, text="优化级别:").pack(side=tk.LEFT, padx=5)
        
        self.optimization_level_var = tk.IntVar(value=getattr(self.map_params, "optimization_level", 1))
        optimization_combo = ttk.Combobox(
            optimization_frame,
            textvariable=self.optimization_level_var,
            values=["0 - 低(快速)", "1 - 中(均衡)", "2 - 高(高质量)"],
            width=15,
            state="readonly"
        )
        optimization_combo.pack(side=tk.RIGHT, padx=5)
        optimization_combo.bind("<<ComboboxSelected>>", 
                            lambda e: self._on_param_change("optimization_level", 
                                                            int(self.optimization_level_var.get()[0])))
        
        # 添加一个说明标签
        ttk.Label(terrain_perf_frame, 
                text="提示: 降低优化级别和关闭微细节可显著提高生成速度，但可能影响地形质量",
                wraplength=350).pack(fill=tk.X, padx=5, pady=5)
        
        # 添加GPU和多线程选项
        self.gpu_var = tk.BooleanVar(value=False)
        self.multithreaded_var = tk.BooleanVar(value=True)
        
        gpu_frame = ttk.Frame(perf_frame)
        gpu_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(gpu_frame, text="使用GPU加速").pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(gpu_frame, variable=self.gpu_var, command=self._toggle_gpu).pack(side=tk.RIGHT, padx=5)
        
        mt_frame = ttk.Frame(perf_frame)
        mt_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(mt_frame, text="启用多线程").pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(mt_frame, variable=self.multithreaded_var, command=self._toggle_multithreaded).pack(side=tk.RIGHT, padx=5)
        
        # 调试选项
        debug_frame = ttk.LabelFrame(scrollable_frame, text="调试选项")
        debug_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 添加种子设置
        seed_frame = ttk.Frame(debug_frame)
        seed_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(seed_frame, text="随机种子").pack(side=tk.LEFT, padx=5)
        seed_var = tk.IntVar(value=0)
        seed_spinbox = ttk.Spinbox(seed_frame, from_=-1000000, to=1000000, textvariable=seed_var, width=10)
        seed_spinbox.pack(side=tk.RIGHT, padx=5)
        
        ToolTip(micro_frame, "启用微细节生成可增加山地、丘陵和高原区域的微观纹理细节。禁用此选项可显著提高生成速度，但地形会略显平滑。")
        ToolTip(extreme_frame, "启用异常点检测可检测并修复地形中的异常高度变化，增强地形的自然感。禁用可提高速度，但可能产生不自然的尖锐特征。")
        ToolTip(optimization_combo, "低优化级别适合原型设计和测试，生成最快；中级适合大多数应用场景；高级提供最佳质量但生成最慢。")

    def _toggle_vis_path(self):
        """切换可视化路径输入状态"""
        # 找到路径输入框
        for widget in self.master.winfo_children():
            if isinstance(widget, ttk.Entry) and widget.winfo_parent().endswith('save_vis_frame'):
                widget.config(state="normal" if self.save_visualization_var.get() else "disabled")
                break

    def _browse_vis_path(self):
        """选择可视化保存路径"""
        # 如果未启用保存可视化，则不操作
        if not self.save_visualization_var.get():
            return
            
        # 打开文件夹选择对话框
        folder = filedialog.askdirectory(
            title="选择可视化图像保存文件夹",
            initialdir=self.visualization_path.get()
        )
        
        # 如果选择了有效文件夹
        if folder:
            self.visualization_path.set(folder)

    def _setup_geo_tab(self):
        """设置地理数据标签页"""
        # 创建带滚动条的框架
        canvas = tk.Canvas(self.geo_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.geo_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 启用真实地理数据选项
        geo_enable_frame = ttk.Frame(scrollable_frame)
        geo_enable_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.use_real_geo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            geo_enable_frame, 
            text="使用真实地理数据", 
            variable=self.use_real_geo_var,
            command=self._toggle_geo_data_ui
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # 地理数据选项框架
        self.geo_options_frame = ttk.LabelFrame(scrollable_frame, text="地理数据选项")
        
        # 默认隐藏地理选项框架(会通过_toggle_geo_data_ui方法控制显示)
        # self.geo_options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 数据源选择
        source_frame = ttk.Frame(self.geo_options_frame)
        source_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(source_frame, text="数据源:").pack(side=tk.LEFT, padx=5)
        self.geo_source_var = tk.StringVar(value="file")
        source_combo = ttk.Combobox(
            source_frame, 
            textvariable=self.geo_source_var,
            values=["file", "srtm"],
            width=10,
            state="readonly"
        )
        source_combo.pack(side=tk.LEFT, padx=5)
        source_combo.bind("<<ComboboxSelected>>", lambda e: self._update_geo_source_ui())
        
        # 文件选择框架
        self.file_frame = ttk.Frame(self.geo_options_frame)
        self.file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.file_frame, text="GeoTIFF文件:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.geo_file_var = tk.StringVar()
        ttk.Entry(self.file_frame, textvariable=self.geo_file_var, width=30).grid(row=0, column=1, padx=5)
        ttk.Button(self.file_frame, text="浏览...", command=self._browse_geo_file).grid(row=0, column=2, padx=5)
        
        # SRTM区域选择框架
        self.srtm_frame = ttk.Frame(self.geo_options_frame)
        
        # 默认隐藏SRTM框架
        # self.srtm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 预设区域
        ttk.Label(self.srtm_frame, text="预设区域:").grid(row=0, column=0, sticky=tk.W, padx=5)
        preset_regions = ["自定义", "北京", "上海", "黄山", "泰山", "张家界"]
        region_combo = ttk.Combobox(
            self.srtm_frame, 
            values=preset_regions,
            width=12,
            state="readonly"
        )
        region_combo.current(0)
        region_combo.grid(row=0, column=1, padx=5, pady=2)
        region_combo.bind("<<ComboboxSelected>>", self._select_preset_region)
        
        # 坐标范围
        ttk.Label(self.srtm_frame, text="经度范围:").grid(row=1, column=0, sticky=tk.W, padx=5)
        coord_frame1 = ttk.Frame(self.srtm_frame)
        coord_frame1.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.lng_min_var = tk.StringVar(value="116.3")
        self.lng_max_var = tk.StringVar(value="116.5")
        ttk.Entry(coord_frame1, textvariable=self.lng_min_var, width=7).pack(side=tk.LEFT, padx=2)
        ttk.Label(coord_frame1, text="至").pack(side=tk.LEFT, padx=2)
        ttk.Entry(coord_frame1, textvariable=self.lng_max_var, width=7).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(self.srtm_frame, text="纬度范围:").grid(row=2, column=0, sticky=tk.W, padx=5)
        coord_frame2 = ttk.Frame(self.srtm_frame)
        coord_frame2.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.lat_min_var = tk.StringVar(value="39.9")
        self.lat_max_var = tk.StringVar(value="40.0")
        ttk.Entry(coord_frame2, textvariable=self.lat_min_var, width=7).pack(side=tk.LEFT, padx=2)
        ttk.Label(coord_frame2, text="至").pack(side=tk.LEFT, padx=2)
        ttk.Entry(coord_frame2, textvariable=self.lat_max_var, width=7).pack(side=tk.LEFT, padx=2)
        
        # 初始隐藏地理选项
        self._toggle_geo_data_ui()
        # 初始更新数据源UI
        self._update_geo_source_ui()

    def _create_enhanced_slider(self, parent, param, label, min_val, max_val, row, tooltip=None):
        """创建带工具提示的增强滑块控件"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        # 标签
        label_widget = ttk.Label(frame, text=label, width=12)
        label_widget.pack(side=tk.LEFT)
        
        # 如果有提示，添加工具提示
        if tooltip:
            ToolTip(label_widget, tooltip)
        
        # 滑块和数值 - 添加默认值处理
        # 尝试从map_params获取值，如果不存在则使用默认值
        default_value = (min_val + max_val) / 2  # 默认值设为范围中点
        try:
            current_value = getattr(self.map_params, param)
        except AttributeError:
            # 如果属性不存在，则设置属性并使用默认值
            setattr(self.map_params, param, default_value)
            current_value = default_value
        
        var = tk.DoubleVar(value=current_value)
        slider = ttk.Scale(
            frame, 
            from_=min_val, 
            to=max_val, 
            variable=var, 
            command=lambda v, p=param: self._on_param_change(p, float(v))
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        entry = ttk.Entry(
            frame, 
            width=6, 
            textvariable=var, 
            validate="key", 
            validatecommand=(self.master.register(self._validate_float), "%P")
        )
        entry.pack(side=tk.RIGHT)
        
        # 添加回车键和失去焦点事件绑定
        entry.bind("<Return>", lambda e, p=param, v=var: self._on_param_change(p, float(v.get())))
        entry.bind("<FocusOut>", lambda e, p=param, v=var: self._on_param_change(p, float(v.get())))
        
        self.param_controls[param] = {"var": var, "slider": slider, "entry": entry}

    def _create_dimension_control(self, parent, width_param, height_param, width_label, height_label, 
                                min_width, max_width, min_height, max_height, row=0, col=1):
        """创建尺寸控制组合控件（宽度和高度）"""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=5, pady=2, sticky="w")
        
        # 宽度控件
        ttk.Label(frame, text=width_label).grid(row=0, column=0, padx=(0, 5))
        width_var = tk.IntVar(value=getattr(self.map_params, width_param))
        width_spinbox = ttk.Spinbox(
            frame, 
            from_=min_width, 
            to=max_width, 
            textvariable=width_var,
            width=8,
            command=lambda: self._on_param_change(width_param, width_var.get())
        )
        width_spinbox.grid(row=0, column=1, padx=2)
        width_spinbox.bind("<Return>", lambda e: self._on_param_change(width_param, width_var.get()))
        width_spinbox.bind("<FocusOut>", lambda e: self._on_param_change(width_param, width_var.get()))
        
        # 高度控件
        ttk.Label(frame, text=height_label).grid(row=0, column=2, padx=(10, 5))
        height_var = tk.IntVar(value=getattr(self.map_params, height_param))
        height_spinbox = ttk.Spinbox(
            frame, 
            from_=min_height, 
            to=max_height, 
            textvariable=height_var,
            width=8,
            command=lambda: self._on_param_change(height_param, height_var.get())
        )
        height_spinbox.grid(row=0, column=3, padx=2)
        height_spinbox.bind("<Return>", lambda e: self._on_param_change(height_param, height_var.get()))
        height_spinbox.bind("<FocusOut>", lambda e: self._on_param_change(height_param, height_var.get()))
        
        # 存储控件引用
        self.param_controls[width_param] = {"var": width_var, "spinbox": width_spinbox}
        self.param_controls[height_param] = {"var": height_var, "spinbox": height_spinbox}
        
    def _setup_control_buttons(self):
        """设置控制面板底部按钮"""
        # 创建底部按钮容器
        button_container = ttk.Frame(self.control_frame, padding="5")
        button_container.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # 设置按钮样式
        style = ttk.Style()
        style.configure("Action.TButton", font=("", 10, "bold"), padding=5)
        style.configure("Secondary.TButton", padding=5)
        
        # 添加进度条 - 在控制面板底部显示当前操作进度
        self.control_progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(
            button_container,
            variable=self.control_progress_var,
            style="Horizontal.TProgressbar",
            mode="determinate"
        )
        progress_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        
        # 添加操作按钮组
        button_frame = ttk.Frame(button_container)
        button_frame.pack(fill=tk.X)
        
        # 生成地图主按钮
        self.generate_btn = ttk.Button(
            button_frame, 
            text="生成地图", 
            command=self._generate_map,
            style="Action.TButton",
            width=15
        )
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 5), pady=5, fill=tk.X, expand=True)
        
        # 添加工具提示
        if hasattr(self, '_create_tooltip'):
            self._create_tooltip(self.generate_btn, "根据当前参数生成新的地图")
        
        # 重置按钮
        reset_btn = ttk.Button(
            button_frame, 
            text="重置参数", 
            command=self._reset_parameters,
            style="Secondary.TButton",
            width=10
        )
        reset_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 添加工具提示
        if hasattr(self, '_create_tooltip'):
            self._create_tooltip(reset_btn, "将所有参数重置为默认值")
        
        # 导出按钮
        export_btn = ttk.Button(
            button_frame, 
            text="导出地图", 
            command=self._export_map,
            style="Secondary.TButton",
            width=10
        )
        export_btn.pack(side=tk.LEFT, padx=(5, 0), pady=5)
        
        # 添加工具提示
        if hasattr(self, '_create_tooltip'):
            self._create_tooltip(export_btn, "将当前地图导出为文件")
            
    def _update_button_states(self, is_busy=False):
        """更新按钮状态，根据当前应用状态启用或禁用按钮
        
        Args:
            is_busy: 如果为True，表示应用程序正在执行耗时操作，大多数按钮应被禁用
        """
        # 获取当前状态
        has_map_data = self.map_data is not None and self.map_data.is_valid()
        has_commands = self.command_history.can_undo()
        can_redo = self.command_history.can_redo()
        
        # 如果应用程序忙碌，禁用大多数按钮
        if is_busy:
            # 更新菜单
            if hasattr(self, 'undo_menu') and self.undo_menu is not None:
                try:
                    self.undo_menu["state"] = tk.DISABLED
                except (TypeError, tk.TclError):
                    pass
            if hasattr(self, 'redo_menu') and self.redo_menu is not None:
                try:
                    self.redo_menu["state"] = tk.DISABLED
                except (TypeError, tk.TclError):
                    pass
            
            # 更新工具栏按钮
            if hasattr(self, 'generate_btn') and self.generate_btn is not None:
                try:
                    self.generate_btn.config(state=tk.DISABLED)
                except (AttributeError, tk.TclError):
                    pass
            if hasattr(self, 'export_btn') and self.export_btn is not None:
                try:
                    self.export_btn.config(state=tk.DISABLED)
                except (AttributeError, tk.TclError):
                    pass
            if hasattr(self, 'save_btn') and self.save_btn is not None:
                try:
                    self.save_btn.config(state=tk.DISABLED)
                except (AttributeError, tk.TclError):
                    pass
            if hasattr(self, 'preview_btn') and self.preview_btn is not None:
                try:
                    self.preview_btn.config(state=tk.DISABLED)
                except (AttributeError, tk.TclError):
                    pass
                
            # 禁用其他功能按钮
            if hasattr(self, 'underground_generate_btn') and self.underground_generate_btn is not None:
                try:
                    self.underground_generate_btn.config(state=tk.DISABLED)
                except (AttributeError, tk.TclError):
                    pass
            if hasattr(self, 'level_generate_btn') and self.level_generate_btn is not None:
                try:
                    self.level_generate_btn.config(state=tk.DISABLED)
                except (AttributeError, tk.TclError):
                    pass
                
            return
        
        # 根据状态更新菜单项
        if hasattr(self, 'undo_menu') and self.undo_menu is not None:
            try:
                self.undo_menu["state"] = tk.NORMAL if has_commands else tk.DISABLED
            except (TypeError, tk.TclError):
                pass
        if hasattr(self, 'redo_menu') and self.redo_menu is not None:
            try:
                self.redo_menu["state"] = tk.NORMAL if can_redo else tk.DISABLED
            except (TypeError, tk.TclError):
                pass
        
        # 更新工具栏按钮状态
        if hasattr(self, 'generate_btn') and self.generate_btn is not None:
            try:
                self.generate_btn.config(state=tk.NORMAL)  # 生成按钮总是启用
            except (AttributeError, tk.TclError):
                pass
        if hasattr(self, 'export_btn') and self.export_btn is not None:
            try:
                self.export_btn.config(state=tk.NORMAL if has_map_data else tk.DISABLED)
            except (AttributeError, tk.TclError):
                pass
        if hasattr(self, 'save_btn') and self.save_btn is not None:
            try:
                self.save_btn.config(state=tk.NORMAL if has_map_data else tk.DISABLED)
            except (AttributeError, tk.TclError):
                pass
        if hasattr(self, 'preview_btn') and self.preview_btn is not None:
            try:
                self.preview_btn.config(state=tk.NORMAL if has_map_data else tk.DISABLED)
            except (AttributeError, tk.TclError):
                pass
        
        # 更新特殊功能按钮
        if hasattr(self, 'underground_generate_btn') and self.underground_generate_btn is not None:
            try:
                self.underground_generate_btn.config(state=tk.NORMAL if has_map_data else tk.DISABLED)
            except (AttributeError, tk.TclError):
                pass
        if hasattr(self, 'level_generate_btn') and self.level_generate_btn is not None:
            try:
                self.level_generate_btn.config(state=tk.NORMAL if has_map_data else tk.DISABLED)
            except (AttributeError, tk.TclError):
                pass

    def _reset_parameters(self):
        """重置所有参数为默认值"""
        # 弹出确认对话框
        if messagebox.askyesno("确认重置", "确定要将所有参数重置为默认值吗？此操作无法撤销。"):
            # 重新初始化地图参数对象
            self.map_params = MapParameters()
            
            # 更新params字典，从map_params获取初始值
            for attr in dir(self.map_params):
                if not attr.startswith('_') and not callable(getattr(self.map_params, attr)):
                    self.params[attr] = getattr(self.map_params, attr)
            
            # 更新界面控件以反映默认值
            self._update_ui_from_params()
            
            # 显示提示信息
            self.logger.log("已重置所有参数为默认值")
            self._show_notification("重置完成", "所有参数已恢复为默认值", type_="info")

    def _update_ui_from_params(self):
        """根据参数更新UI控件"""
        # 更新所有参数控件
        for param, controls in self.param_controls.items():
            if param in self.params:
                # 跳过非控件字典项
                if not isinstance(controls, dict):
                    continue
                    
                var = controls.get('var')
                if var:
                    # 获取参数值
                    value = self.params.get(param, getattr(self.map_params, param, 0))
                    
                    # 更新变量值
                    try:
                        var.set(value)
                    except Exception:
                        # 如果设置失败，可能是类型不匹配
                        pass
 
    def _get_icon(self, name, size=16):
        """获取图标资源，支持缓存和回退机制"""
        # 图标路径映射
        icon_map = {
            "new": "icons/new.png",
            "open": "icons/open.png",
            "save": "icons/save.png",
            "save_as": "icons/save_as.png",
            "export": "icons/export.png",
            "deploy": "icons/deploy.png",
            "exit": "icons/exit.png",
            "undo": "icons/undo.png",
            "redo": "icons/redo.png",
            "copy": "icons/copy.png",
            "export_params": "icons/export_params.png",
            "import_params": "icons/import_params.png",
            "core": "icons/core.png",
            "terrain": "icons/terrain.png",
            "biome": "icons/biome.png",
            "advanced": "icons/advanced.png",
            "geo": "icons/geo.png",
            "level": "icons/level.png",
            "error": "icons/error.png",
            "warning": "icons/warning.png",
            "info": "icons/info.png",
            "success": "icons/success.png"
        }
        
        # 缓存已加载的图标
        if not hasattr(self, 'icon_cache'):
            self.icon_cache = {}
        
        # 如果图标存在于缓存中，直接返回
        cache_key = f"{name}_{size}"
        if cache_key in self.icon_cache:
            return self.icon_cache[cache_key]
        
        # 尝试加载图标
        try:
            from PIL import Image, ImageTk
            
            if name in icon_map:
                # 获取应用程序根目录
                app_path = self.config_manager.get("app_path", ".")
                icon_path = os.path.join(app_path, icon_map[name])
                
                # 检查文件是否存在
                if os.path.exists(icon_path):
                    icon = Image.open(icon_path)
                    icon = icon.resize((size, size), Image.Resampling.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
                    tk_icon = ImageTk.PhotoImage(icon)
                    self.icon_cache[cache_key] = tk_icon
                    return tk_icon
            
            # 生成占位图标
            return self._generate_placeholder_icon(name, size)
            
        except Exception as e:
            self.logger.log(f"加载图标 {name} 失败: {e}", "WARNING") if hasattr(self, 'logger') else None
            return None
        
    def _generate_placeholder_icon(self, name, size=16):
        """生成简单的占位图标"""
        try:
            from PIL import Image, ImageDraw, ImageTk
            
            # 创建空白图像
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # 根据图标名称选择颜色
            colors = {
                "error": "#FF5555",
                "warning": "#FFAA00",
                "info": "#55AAFF",
                "success": "#55DD55",
                "new": "#88CC88",
                "open": "#8888FF",
                "save": "#AA88FF",
                "export": "#FFAA88"
            }
            
            color = colors.get(name, "#AAAAAA")
            
            # 绘制简单占位图形
            draw.rectangle([1, 1, size-2, size-2], outline=color, width=1)
            
            # 添加首字母
            first_letter = name[0].upper() if name else "?"
            
            # PIL没有提供方便的获取文本尺寸的方法，所以使用固定位置
            text_position = (size // 2 - 3, size // 2 - 5)
            draw.text(text_position, first_letter, fill=color)
            
            # 转换为Tkinter图像
            tk_icon = ImageTk.PhotoImage(img)
            self.icon_cache[f"{name}_{size}"] = tk_icon
            return tk_icon
            
        except Exception:
            return None

    def _open_pipeline_configurator(self):
        """打开地图生成流水线步骤配置器对话框"""
        if not hasattr(self, 'pipeline_configurator') or not self.pipeline_configurator.winfo_exists():
            self.pipeline_configurator = PipelineConfiguratorDialog(self.master, self)
        else:
            self.pipeline_configurator.lift()
            self.pipeline_configurator.focus_force()

    def get_enabled_pipeline_steps(self):
        """获取当前启用的流水线步骤"""
        if not hasattr(self, 'enabled_pipeline_steps'):
            # 默认启用所有步骤
            self.enabled_pipeline_steps = {step for step in MapPipelineStep}
        return self.enabled_pipeline_steps

    def set_enabled_pipeline_steps(self, enabled_steps):
        """设置启用的流水线步骤"""
        self.enabled_pipeline_steps = set(enabled_steps)
        # 保存到配置 - 修复参数数量不匹配问题
        self.config_manager.set('pipeline.enabled_steps', 
                            [step.name for step in self.enabled_pipeline_steps])
        self.config_manager.save_config()

    def get_pipeline_step_order(self):
        """获取当前流水线步骤顺序"""
        if not hasattr(self, 'pipeline_step_order'):
            # 默认使用枚举定义的顺序
            self.pipeline_step_order = list(MapPipelineStep)
        return self.pipeline_step_order

    def set_pipeline_step_order(self, step_order):
        """设置流水线步骤顺序"""
        self.pipeline_step_order = list(step_order)
        # 保存到配置 - 同样修复参数数量不匹配问题
        self.config_manager.set('pipeline.step_order', 
                            [step.name for step in self.pipeline_step_order])
        self.config_manager.save_config()

    def _generate_map(self):
        """重写生成地图方法，使用流水线处理器"""
        # 获取参数
        width = self.width_var.get()
        height = self.height_var.get()
        seed = self.seed_var.get()
        use_real_geo = self.use_geo_var.get()
        export_model = self.export_model_var.get()
        
        # 启用进度条
        self.progress.start()
        
        # 从各个控件获取用户偏好
        preferences = self._gather_preferences()
        
        # 如果设置了随机种子，更新界面显示
        if preferences.get('seed') is None:
            new_seed = random.randint(1, 100000)
            preferences['seed'] = new_seed
            self.seed_var.set(new_seed)
        
        # 创建新的地图数据对象
        self.map_data = MapData(width, height, 
                        MapGenerationConfig(width=width, height=height, export_model=export_model))
        
        # 如果使用实际地理数据，获取地理边界
        geo_bounds = None
        geo_path = None
        if use_real_geo:
            geo_bounds = self._get_geo_bounds()
            geo_path = self.geo_file_var.get() if self.geo_file_var.get() else None
        
        # 使用线程处理地图生成，避免UI冻结
        threading.Thread(target=self._threaded_map_generation, 
                    args=(preferences, width, height, export_model, use_real_geo, 
                            geo_path, geo_bounds)).start()

    def _threaded_map_generation(self, preferences, width, height, export_model, 
                            use_real_geo, geo_path, geo_bounds):
        """在单独线程中处理地图生成"""
        try:
            # 使用管道处理器生成地图
            pipeline_processor = MapPipelineProcessor(logger=self.logger)
            
            # 设置启用的步骤
            pipeline_processor.set_enabled_steps(self.get_enabled_pipeline_steps())
            
            # 处理地图生成
            result_map_data = pipeline_processor.process_map(
                preferences=preferences,
                width=width,
                height=height,
                export_model=export_model,
                use_real_geo_data=use_real_geo,
                geo_data_path=geo_path,
                geo_bounds=geo_bounds,
                visualize=self.visualize_var.get(),
                visualization_path=self.visualization_path.get() if self.visualize_var.get() and self.save_visualization_var.get() else None,
                parent_frame=self.preview_frame if self.embed_preview_var.get() else None,
                use_gui_editors=True,
                callback=self._handle_pipeline_callback
            )
            
            # 检查结果
            if result_map_data is not None:
                # 处理编辑器挂起状态
                if hasattr(result_map_data, 'pending_editor') and result_map_data.pending_editor:
                    self.map_data = result_map_data
                    # 根据不同编辑器类型打开相应界面
                    if result_map_data.pending_editor == "height_editor":
                        self._queue_action(self._open_height_editor)
                    elif result_map_data.pending_editor == "evolution_scorer":
                        self._queue_action(self._open_evolution_scorer)
                else:
                    # 地图生成完成
                    self.map_data = result_map_data
                    # 更新UI上的预览
                    self._queue_action(self._update_preview)
                    # 停止进度条
                    self._queue_action(lambda: self.progress.stop())
                    # 显示成功消息
                    self._queue_action(lambda: messagebox.showinfo("成功", "地图生成完成！"))
            else:
                # 地图生成失败
                self._queue_action(lambda: self.progress.stop())
                self._queue_action(lambda: messagebox.showerror("错误", "地图生成失败！"))
        except Exception as e:
            # 处理异常
            self._queue_action(lambda: self.progress.stop())
            error_message = f"生成地图时出错: {str(e)}"
            self.logger.log(error_message, "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")
            self._queue_action(lambda: messagebox.showerror("错误", error_message))

    def _handle_pipeline_callback(self, data):
        """处理流水线回调事件"""
        action = data.get('action')
        if action == 'show_editor':
            editor_type = data.get('editor_type')
            if editor_type == 'height_editor':
                self._queue_action(self._open_height_editor)
            elif editor_type == 'evolution_scorer':
                self._queue_action(self._open_evolution_scorer)

    def _queue_action(self, action):
        """将UI操作安排在主线程的事件循环中执行，保证线程安全
        
        Args:
            action: 要在主线程执行的回调函数
        """
        try:
            self.master.after_idle(action)
        except Exception as e:
            self.logger.log(f"队列操作失败: {str(e)}", "ERROR")

    def _setup_embedded_editors(self):
        """设置嵌入式编辑器界面"""
        # 创建一个新的标签页
        self.editors_tab = ttk.Frame(self.param_notebook)
        self.param_notebook.add(self.editors_tab, text="地形编辑器")
        
        # 创建编辑器容器
        self.editor_container = ttk.Frame(self.editors_tab)
        self.editor_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加控制按钮
        buttons_frame = ttk.Frame(self.editors_tab)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="启动高度编辑器", 
                command=self._show_height_editor).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="启动进化评分", 
                command=self._show_evolution_scorer).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 状态标签
        self.editor_status_var = tk.StringVar(value="未加载编辑器")
        ttk.Label(buttons_frame, textvariable=self.editor_status_var).pack(side=tk.RIGHT, padx=5, pady=5)

    def _show_height_editor(self, state=None):
        """显示内嵌的高度编辑器"""
        # 清空当前容器
        for widget in self.editor_container.winfo_children():
            widget.destroy()
        
        self.editor_status_var.set("正在加载高度编辑器...")
        
        # 使用传入的状态或当前地图数据
        height_map = state.get('height_map') if state else self.map_data.get_layer("height")
        temp_map = state.get('temp_map') if state else self.map_data.get_layer("temperature")
        humid_map = state.get('humid_map') if state else self.map_data.get_layer("humidity")
        
        # 创建主容器框架，使用更高效的布局
        main_frame = ttk.Frame(self.editor_container)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 分割顶部工具区和底部编辑区
        tool_area = ttk.Frame(main_frame)
        tool_area.pack(fill=tk.X, padx=5, pady=5)
        
        # 在主区域使用Notebook创建工具标签页，避免过长滚动
        tools_notebook = ttk.Notebook(tool_area)
        tools_notebook.pack(fill=tk.X, expand=True)
        
        # 基础工具标签页
        basic_tools_tab = ttk.Frame(tools_notebook)
        terrain_tools_tab = ttk.Frame(tools_notebook)
        advanced_tools_tab = ttk.Frame(tools_notebook)
        
        tools_notebook.add(basic_tools_tab, text="基础编辑工具")
        tools_notebook.add(terrain_tools_tab, text="地形特征工具")
        tools_notebook.add(advanced_tools_tab, text="高级设置")
        
        # 编辑区域（带滚动条）
        edit_area = ttk.Frame(main_frame)
        edit_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建滚动条
        v_scrollbar = ttk.Scrollbar(edit_area, orient="vertical")
        h_scrollbar = ttk.Scrollbar(edit_area, orient="horizontal")
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 创建Canvas用于滚动
        canvas = tk.Canvas(edit_area, 
                        yscrollcommand=v_scrollbar.set,
                        xscrollcommand=h_scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 配置滚动条
        v_scrollbar.config(command=canvas.yview)
        h_scrollbar.config(command=canvas.xview)
        
        # 创建内部框架放置编辑器内容
        editor_frame = ttk.Frame(canvas)
        
        # 在Canvas上创建窗口
        canvas_window = canvas.create_window((0, 0), window=editor_frame, anchor="nw")
        
        # 初始化编辑状态变量
        self.current_tool = "提升"  # 默认工具
        self.brush_size = 5         # 默认笔刷大小
        self.intensity = 0.5        # 默认强度
        self.drawing = False        # 绘制状态
        self.last_pos = None        # 上次位置
        self.terrain_tool_active = None  # 当前活动地形工具
        self.selection_active = False    # 选择模式状态
        self.selected_region = None      # 选择区域
        self.selection_rect = None       # 选择矩形
        
        # 初始化历史栈
        self.history_stack = []
        self.redo_stack = []
        
        # 创建状态指示器（在底部始终可见）
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.BOTTOM)
        
        current_tool_var = tk.StringVar(value="提升")
        ttk.Label(status_frame, text="当前工具:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, textvariable=current_tool_var, 
                font=("Arial", 9, "bold"),
                foreground="blue").pack(side=tk.LEFT, padx=5)
        
        # ===== 填充基础工具标签页 =====
        # 快捷键提示放在顶部醒目位置
        shortcuts_frame = ttk.LabelFrame(basic_tools_tab, text="编辑器快捷键")
        shortcuts_frame.pack(fill=tk.X, padx=5, pady=5)
        
        shortcuts_text = """
        [Z] - 撤销  [Y] - 重做  [Esc] - 取消工具

        • 按住鼠标左键绘制地形 • 使用鼠标滚轮缩放地图 • 
        """
        ttk.Label(shortcuts_frame, text=shortcuts_text, justify="center", 
                font=("Consolas", 9)).pack(padx=5, pady=5)
        
        # 基础绘制工具
        brush_frame = ttk.LabelFrame(basic_tools_tab, text="笔刷工具")
        brush_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 改进的工具选择UI - 使用图标按钮
        tool_var = tk.StringVar(value="提升")
        tools = ["提升", "降低", "平滑", "随机化"]
        
        # 工具按钮布局使用网格，更整齐
        tools_grid = ttk.Frame(brush_frame)
        tools_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # 定义选择工具的回调
        def on_tool_change():
            tool = tool_var.get()
            self.current_tool = tool
            self.editor_status_var.set(f"当前工具：{tool}")
            # 更新状态显示
            current_tool_var.set(tool)
            
            # 高亮当前工具按钮
            for btn in tool_buttons:
                btn.config(style="TRadiobutton" if btn["text"] != tool else "Accent.TRadiobutton")
        
        # 创建工具按钮
        tool_buttons = []
        for i, tool in enumerate(tools):
            btn = ttk.Radiobutton(
                tools_grid, 
                text=tool, 
                variable=tool_var, 
                value=tool,
                command=on_tool_change,
                width=10,
                style="Accent.TRadiobutton" if tool == "提升" else "TRadiobutton"
            )
            btn.grid(row=0, column=i, padx=5, pady=5)
            tool_buttons.append(btn)
        
        # 笔刷参数控制
        params_frame = ttk.Frame(brush_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 笔刷大小
        size_frame = ttk.Frame(params_frame)
        size_frame.pack(fill=tk.X, pady=2)
        ttk.Label(size_frame, text="笔刷大小:").pack(side=tk.LEFT, padx=5)
        brush_size_var = tk.IntVar(value=5)
        brush_size_slider = ttk.Scale(
            size_frame, 
            from_=1, 
            to=20, 
            variable=brush_size_var,
            orient="horizontal",
            command=lambda x: on_brush_size_change()
        )
        brush_size_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(size_frame, textvariable=brush_size_var).pack(side=tk.LEFT, padx=5)
        
        # 笔刷强度
        intensity_frame = ttk.Frame(params_frame)
        intensity_frame.pack(fill=tk.X, pady=2)
        ttk.Label(intensity_frame, text="效果强度:").pack(side=tk.LEFT, padx=5)
        intensity_var = tk.DoubleVar(value=0.5)
        intensity_slider = ttk.Scale(
            intensity_frame, 
            from_=0.1, 
            to=1.0, 
            variable=intensity_var,
            orient="horizontal",
            command=lambda x: on_intensity_change()
        )
        intensity_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(intensity_frame, textvariable=intensity_var).pack(side=tk.LEFT, padx=5)
        
        # 撤销和重做按钮
        history_frame = ttk.Frame(brush_frame)
        history_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(history_frame, text="撤销 (Z)", command=lambda: undo_action()).pack(side=tk.LEFT, padx=5)
        ttk.Button(history_frame, text="重做 (Y)", command=lambda: redo_action()).pack(side=tk.LEFT, padx=5)
        
        # ===== 填充地形特征标签页 =====
        # 智能地形工具
        terrain_frame = ttk.LabelFrame(terrain_tools_tab, text="地形特征工具")
        terrain_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 地形工具提示
        ttk.Label(terrain_frame, 
                text="使用这些工具可以快速添加大型地形特征。选择工具后点击地图位置应用。",
                wraplength=400, justify="center").pack(pady=5)
        
        # 更整洁的地形工具网格
        terrain_grid = ttk.Frame(terrain_frame)
        terrain_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # 定义工具回调函数生成器
        def create_tool_callback(keysym, keycode):
            def callback():
                self.logger.log(f"触发添加{keysym}工具")
                # 创建键盘事件对象
                event = type('Event', (), {})
                event.keysym = keysym
                event.keycode = keycode
                # 调用键盘事件处理函数
                on_key_press(event)
            return callback
        
        # 按钮数据，每行放3个
        terrain_buttons = [
            ("添加山脉 (M)", create_tool_callback("m", 77)),
            ("添加河谷 (V)", create_tool_callback("v", 86)),
            ("添加高原 (P)", create_tool_callback("p", 80)),
            ("添加平原 (L)", create_tool_callback("l", 76)),
            ("添加丘陵 (H)", create_tool_callback("h", 72)),
            ("添加盆地 (B)", create_tool_callback("b", 66)),
            ("添加沙漠 (D)", create_tool_callback("d", 68)),
            ("区域选择 (S)", create_tool_callback("s", 83)),
        ]
        
        # 创建地形工具按钮网格
        for i, (text, cmd) in enumerate(terrain_buttons):
            row, col = i // 3, i % 3
            ttk.Button(terrain_grid, text=text, command=cmd, width=15).grid(
                row=row, column=col, padx=5, pady=5
            )
        
        # 取消工具按钮（特别突出）
        cancel_frame = ttk.Frame(terrain_frame)
        cancel_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(cancel_frame, text="取消工具 (Esc)", 
                command=lambda: on_key_press(type('Event', (), {'keysym': 'Escape'})),
                style="Accent.TButton").pack(pady=5)
        
        # ===== 填充高级设置标签页 =====
        # 保存和放弃按钮
        save_frame = ttk.LabelFrame(advanced_tools_tab, text="完成编辑")
        save_frame.pack(fill=tk.X, padx=5, pady=5)
        
        save_buttons = ttk.Frame(save_frame)
        save_buttons.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(save_buttons, text="保存修改", 
                command=lambda: on_height_edit_complete(
                    self.map_data.get_layer("height"),
                    self.map_data.get_layer("temperature"),
                    self.map_data.get_layer("humidity")
                ),
                style="Accent.TButton").pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(save_buttons, text="放弃修改", 
                command=lambda: self.param_notebook.select(0)).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 高级气候设置
        climate_frame = ttk.LabelFrame(advanced_tools_tab, text="地形气候")
        climate_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(climate_frame, text="更新气候数据", 
                command=lambda: update_climate()).pack(pady=10)
        
        ttk.Label(climate_frame, 
                text="手动修改地形后，点击此按钮重新计算温度和湿度",
                wraplength=400, justify="center").pack(pady=5)
            
        # 初始化历史栈
        self.history_stack = []
        self.redo_stack = []
        
        # 定义编辑完成回调
        def on_height_edit_complete(new_height_map, new_temp_map, new_humid_map):
            self.editor_status_var.set("高度编辑已完成")
            
            # 保存编辑结果
            if hasattr(self, 'map_data') and self.map_data:
                if 'height' in self.map_data.layers:
                    self.map_data.layers['height'] = new_height_map
                if 'temperature' in self.map_data.layers:
                    self.map_data.layers['temperature'] = new_temp_map
                if 'humidity' in self.map_data.layers:
                    self.map_data.layers['humidity'] = new_humid_map
            
            # 准备恢复状态
            self.resume_state = {
                'height_map': new_height_map,
                'temp_map': new_temp_map,
                'humid_map': new_humid_map,
                'resume_point': 'post_height_edit'
            }
            
            # 触发继续生成
            self._continue_generation()
        
        # 保存历史状态函数
        def save_history():
            if hasattr(self, 'map_data') and self.map_data:
                if 'height' in self.map_data.layers:
                    self.history_stack.append(self.map_data.layers['height'].copy())
                    self.redo_stack.clear()  # 新的编辑动作会清空重做栈
                    if len(self.history_stack) > 20:  # 限制历史记录数量
                        self.history_stack.pop(0)
        
        # 撤销操作函数
        def undo_action():
            if self.history_stack:
                # 保存当前状态到重做栈
                if hasattr(self, 'map_data') and self.map_data and 'height' in self.map_data.layers:
                    self.redo_stack.append(self.map_data.layers['height'].copy())
                    # 恢复上一状态
                    self.map_data.layers['height'] = self.history_stack.pop()
                    # 更新温度和湿度
                    update_climate()
                    self.editor_status_var.set("已撤销上一步操作")
        
        # 重做操作函数
        def redo_action():
            if self.redo_stack:
                # 保存当前状态到历史栈
                if hasattr(self, 'map_data') and self.map_data and 'height' in self.map_data.layers:
                    self.history_stack.append(self.map_data.layers['height'].copy())
                    # 恢复重做状态
                    self.map_data.layers['height'] = self.redo_stack.pop()
                    # 更新温度和湿度
                    update_climate()
                    self.editor_status_var.set("已重做操作")
        
        # 更新气候函数
        def update_climate():
            if hasattr(self, 'map_data') and self.map_data:
                from core.generation.generate_height_temp_humid import biome_temperature, moisture_map
                
                # 获取风向和其他参数
                map_params = self.map_params.to_dict() if hasattr(self.map_params, 'to_dict') else self.map_params
                prevailing_wind_x = getattr(map_params, "prevailing_wind_x", 1.0) if hasattr(map_params, "get") else map_params.get("prevailing_wind_x", 1.0)
                prevailing_wind_y = getattr(map_params, "prevailing_wind_y", 0.0) if hasattr(map_params, "get") else map_params.get("prevailing_wind_y", 0.0)
                latitude_effect = getattr(map_params, "latitude_effect", 0.5) if hasattr(map_params, "get") else map_params.get("latitude_effect", 0.5)
                seed = getattr(map_params, "seed", None) if hasattr(map_params, "get") else map_params.get("seed", None)
                use_optimization = getattr(map_params, "use_frequency_optimization", True) if hasattr(map_params, "get") else map_params.get("use_frequency_optimization", True)
                
                # 重新计算温度
                if 'height' in self.map_data.layers:
                    height_map = self.map_data.layers['height']
                    
                    # 计算新温度
                    new_temp = biome_temperature(
                        height_map,
                        latitude_effect=latitude_effect,
                        seed=seed,
                        use_frequency_optimization=use_optimization
                    )
                    
                    # 计算新湿度
                    new_humid = moisture_map(
                        height_map,
                        new_temp,
                        prevailing_wind=(prevailing_wind_x, prevailing_wind_y),
                        seed=seed,
                        use_frequency_optimization=use_optimization
                    )
                    
                    # 更新地图数据
                    self.map_data.layers['temperature'] = new_temp
                    self.map_data.layers['humidity'] = new_humid
                    
                    self.editor_status_var.set("已更新气候数据")
        
        # 应用笔刷工具函数
        def apply_brush(y, x, tool_type, brush_size, intensity):
            if hasattr(self, 'map_data') and self.map_data and 'height' in self.map_data.layers:
                height_map = self.map_data.layers['height']
                height, width = height_map.shape
                
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
                
                # 应用工具
                if tool_type == '提升':
                    # 提升高度
                    height_map[y_start:y_end, x_start:x_end] += mask * intensity * 5
                    self.editor_status_var.set(f"提升地形：坐标({x},{y})")
                elif tool_type == '降低':
                    # 降低高度
                    height_map[y_start:y_end, x_start:x_end] -= mask * intensity * 5
                    self.editor_status_var.set(f"降低地形：坐标({x},{y})")
                elif tool_type == '平滑':
                    # 平滑高度
                    from scipy.ndimage import gaussian_filter
                    area = height_map[y_start:y_end, x_start:x_end].copy()
                    blurred = gaussian_filter(area, sigma=1)
                    height_map[y_start:y_end, x_start:x_end] = area * (1 - mask * intensity) + blurred * (mask * intensity)
                    self.editor_status_var.set(f"平滑地形：坐标({x},{y})")
                elif tool_type == '随机化':
                    # 添加随机噪声
                    noise = np.random.normal(0, 5, size=mask.shape)
                    height_map[y_start:y_end, x_start:x_end] += noise * mask * intensity
                    self.editor_status_var.set(f"随机化地形：坐标({x},{y})")
                
                # 确保高度在合理范围内
                height_map = np.clip(height_map, 0, 100)
                self.map_data.layers['height'] = height_map
        
        # 处理鼠标点击事件
        def on_click(event):
            if event.widget == canvas:
                try:
                    # 获取画布上的坐标
                    x = canvas.canvasx(event.x)
                    y = canvas.canvasy(event.y)
                    
                    # 记录绘制状态和位置
                    self.drawing = True
                    self.last_pos = (int(y), int(x))
                    
                    # 添加日志记录当前使用的工具
                    self.logger.log(f"使用工具: {self.current_tool} 在坐标 ({x},{y})")
                    
                    # 应用当前工具
                    apply_brush(int(y), int(x), self.current_tool, self.brush_size, self.intensity)
                    
                    # 保存历史
                    save_history()
                except Exception as e:
                    self.logger.error(f"鼠标点击处理错误: {str(e)}")
            
        # 处理鼠标释放事件
        def on_release(event):
            if self.drawing:
                self.drawing = False
                update_climate()  # 编辑完成后更新气候
        
        # 处理鼠标移动事件
        def on_motion(event):
            if self.drawing and event.widget == canvas:
                try:
                    # 获取画布上的坐标
                    x = canvas.canvasx(event.x)
                    y = canvas.canvasy(event.y)
                    
                    curr_pos = (int(y), int(x))
                    
                    # 防止重复处理同一点
                    if self.last_pos == curr_pos:
                        return
                    
                    # 绘制从上一个点到当前点的线
                    if self.last_pos:
                        y0, x0 = self.last_pos
                        y1, x1 = curr_pos
                        
                        # 计算步骤数量
                        steps = max(abs(y1 - y0), abs(x1 - x0)) * 2
                        if steps > 0:
                            # 创建补间点
                            for i in range(steps + 1):
                                t = i / steps
                                yi = int(y0 + (y1 - y0) * t)
                                xi = int(x0 + (x1 - x0) * t)
                                apply_brush(yi, xi, self.current_tool, self.brush_size, self.intensity)
                    
                    self.last_pos = curr_pos
                except Exception as e:
                    self.logger.error(f"鼠标移动处理错误: {str(e)}")
        
        # 工具选择变化处理
        def on_tool_change():
            tool = tool_var.get()
            self.current_tool = tool  # 确保这一行正确执行
            self.editor_status_var.set(f"当前工具：{tool}")
            # 更新状态显示
            current_tool_var.set(tool)
            # 添加调试日志
            self.logger.log(f"已切换工具至: {tool}")
            
            # 强制高亮当前工具按钮
            for btn in tool_buttons:
                btn.config(style="TRadiobutton" if btn["text"] != tool else "Accent.TRadiobutton")
        
        # 笔刷大小变化处理
        def on_brush_size_change(event=None):
            try:
                self.brush_size = int(brush_size_var.get())
                self.editor_status_var.set(f"笔刷大小：{self.brush_size}")
            except ValueError:
                pass
        
        # 强度变化处理
        def on_intensity_change(event=None):
            try:
                self.intensity = float(intensity_var.get())
                self.editor_status_var.set(f"效果强度：{self.intensity:.1f}")
            except ValueError:
                pass
        
        # 配置Canvas的滚动区域
        def on_frame_configure(event):
            # 更新滚动区域以包含整个内容，确保足够大
            canvas.configure(scrollregion=(0, 0, 
                                        max(editor_frame.winfo_reqwidth(), canvas.winfo_width()),
                                        max(editor_frame.winfo_reqheight() + 300, canvas.winfo_height())))
            # 确保canvas_window的宽度填满画布
            width = max(editor_frame.winfo_reqwidth(), canvas.winfo_width())
            canvas.itemconfig(canvas_window, width=width)
        
        # 当编辑器框架大小变化时更新滚动区域
        editor_frame.bind("<Configure>", on_frame_configure)
        
        # 当Canvas大小变化时调整内部窗口宽度
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", on_canvas_configure)
        
        # 添加鼠标滚轮支持
        def on_mouse_wheel(event):
            # 首先检查事件widget是否存在且有效，确保是canvas触发的事件
            try:
                # 检查canvas是否仍然存在
                if not canvas.winfo_exists():
                    return
                    
                # 按住Ctrl键水平滚动，否则垂直滚动
                if event.state & 0x4:  
                    canvas.xview_scroll(int(-1*(event.delta/120)), "units")
                else:  
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except (tk.TclError, AttributeError) as e:
                # 出错时自动解绑以防止进一步错误
                try:
                    canvas.unbind_all("<MouseWheel>")
                except:
                    pass
        
        # 绑定时使用特定的canvas绑定，避免全局绑定
        canvas.bind("<MouseWheel>", on_mouse_wheel)
        # 如果需要全局绑定，确保在合适的时候解绑
        # canvas.bind_all("<MouseWheel>", on_mouse_wheel)
        
        # 键盘事件处理
        def on_key_press(event):
            self.logger.log(f"高度编辑器接收到键盘事件: {event.keysym}")
            
            # 根据按键选择工具
            if event.keysym.lower() == 'm':
                self.editor_status_var.set("已选择山脉工具 - 点击地图选择中心点或使用区域选择")
                self.terrain_tool_active = 'mountain'
                # 更新当前工具显示
                current_tool_var.set("山脉工具")
            elif event.keysym.lower() == 'v':
                self.editor_status_var.set("已选择河谷工具 - 点击地图选择起点或使用区域选择")
                self.terrain_tool_active = 'valley'
                current_tool_var.set("河谷工具")
            elif event.keysym.lower() == 'p':
                self.editor_status_var.set("已选择高原工具 - 点击地图选择中心点或使用区域选择")
                self.terrain_tool_active = 'plateau'
                current_tool_var.set("高原工具")
            elif event.keysym.lower() == 'l':
                self.editor_status_var.set("已选择平原工具 - 点击地图选择中心点或使用区域选择")
                self.terrain_tool_active = 'plains'
                current_tool_var.set("平原工具")
            elif event.keysym.lower() == 'h':
                self.editor_status_var.set("已选择丘陵工具 - 点击地图选择中心点或使用区域选择")
                self.terrain_tool_active = 'hills'
                current_tool_var.set("丘陵工具")
            elif event.keysym.lower() == 'b':
                self.editor_status_var.set("已选择盆地工具 - 点击地图选择中心点或使用区域选择")
                self.terrain_tool_active = 'basin'
                current_tool_var.set("盆地工具")
            elif event.keysym.lower() == 'd':
                self.editor_status_var.set("已选择沙漠工具 - 点击地图选择中心点或使用区域选择")
                self.terrain_tool_active = 'desert'
                current_tool_var.set("沙漠工具")
            elif event.keysym.lower() == 's':
                self.editor_status_var.set("已启用区域选择 - 拖拽鼠标选择区域")
                self.selection_active = True
                current_tool_var.set("区域选择")
            elif event.keysym.lower() == 'z':
                self.editor_status_var.set("撤销上一步操作")
                undo_action()
            elif event.keysym.lower() == 'y':
                self.editor_status_var.set("重做操作")
                redo_action()
            elif event.keysym == 'Escape':
                self.editor_status_var.set("已取消当前工具")
                # 取消当前工具
                self.terrain_tool_active = None
                self.selection_active = False
                current_tool_var.set("无工具选择")
        
        # 绑定键盘事件到Canvas
        canvas.bind("<Key>", on_key_press)
        canvas.focus_set()  # 确保Canvas能接收键盘事件
        
        # 绑定鼠标事件
        canvas.bind("<Button-1>", on_click)
        canvas.bind("<ButtonRelease-1>", on_release)
        canvas.bind("<B1-Motion>", on_motion)
        
        try:
            # 切换到编辑器标签页
            for i, tab_name in enumerate(self.param_notebook.tabs()):
                if 'editors_tab' in str(tab_name):
                    self.param_notebook.select(i)
                    break
            
            # 为地图编辑器加载地图数据
            from core.services.map_tools import manually_adjust_height
            self.logger.log("启动嵌入式高度编辑器...")
            manually_adjust_height(
                self.map_data,  
                self.map_params.to_dict() if hasattr(self.map_params, 'to_dict') else self.map_params, 
                self.logger, 
                seed=getattr(self.map_params, "seed", None),
                parent_frame=editor_frame,
                on_complete=on_height_edit_complete
            )
            
            # 配置Canvas的滚动区域
            def on_frame_configure(event):
                # 更新滚动区域以包含整个内容
                canvas.configure(scrollregion=(0, 0, 
                                            max(editor_frame.winfo_reqwidth(), canvas.winfo_width()),
                                            max(editor_frame.winfo_reqheight() + 300, canvas.winfo_height())))
                # 确保canvas_window的宽度填满画布
                width = max(editor_frame.winfo_reqwidth(), canvas.winfo_width())
                canvas.itemconfig(canvas_window, width=width)
            
            # 当编辑器框架大小变化时更新滚动区域
            editor_frame.bind("<Configure>", on_frame_configure)
            
            # 绑定键盘事件和鼠标事件到Canvas
            canvas.bind("<Key>", on_key_press)
            canvas.bind("<Button-1>", on_click)
            canvas.bind("<ButtonRelease-1>", on_release)
            canvas.bind("<B1-Motion>", on_motion)
            canvas.focus_set()  # 确保Canvas能接收键盘事件
            
            self.editor_status_var.set("高度编辑器已加载，请根据需要切换工具标签页")
            
        except Exception as e:
            self.logger.error(f"加载高度编辑器失败: {str(e)}")
            self.editor_status_var.set("加载编辑器失败")
            self.show_error_dialog(f"加载高度编辑器失败: {str(e)}")

    def _show_evolution_scorer(self, state=None):
        """显示内嵌的进化评分界面"""
        try:
            # 清空当前容器并设置状态
            for widget in self.editor_container.winfo_children():
                widget.destroy()
            
            self.editor_status_var.set("正在加载进化评分器...")
            
            # 使用传入的状态或当前数据
            engine = state.get('engine') if state else getattr(self, 'evolution_engine', None)
            
            # 检查是否有进化引擎
            if engine is None:
                self.show_warning_dialog("找不到进化引擎")
                self.editor_status_var.set("未加载评分器")
                return
            
            # 创建独立的评分器框架
            # 重要：使用pack布局管理器，避免与grid_container混用
            scorer_frame = ttk.Frame(self.editor_container)
            scorer_frame.pack(fill=tk.BOTH, expand=True)
            
            # 添加评分状态标志
            self.scoring_in_progress = True
            
            # 定义评分完成回调
            def on_scoring_complete(scores):
                # 避免直接操作界面，通过事件队列执行
                def complete_action():
                    self.scoring_in_progress = False
                    self.editor_status_var.set("评分已完成")
                    
                    # 重要：保存分数并调用继续生成方法
                    if engine and scores:
                        try:
                            engine.evolve_generation(scores)
                            self.map_data.editor_state['scores_applied'] = True
                            self.logger.log("已应用用户评分，继续生成地图...")
                            
                            # 返回到第一个标签页
                            self.param_notebook.select(0)
                            
                            # 继续生成过程
                            self._continue_generation()
                        except Exception as e:
                            self.logger.log(f"应用评分时出错: {e}", "ERROR")
                
                # 通过事件队列执行，避免直接操作UI
                self.master.after(100, complete_action)
            
            # 切换到编辑器标签页
            self.param_notebook.select(self.editors_tab)
            
            # 直接使用scorer_frame作为parent_frame，避免嵌套布局
            from core.services.map_tools import get_visual_scores
            self.logger.log("启动嵌入式进化评分器...")
            scoring_event = get_visual_scores(
                engine,
                parent_frame=scorer_frame,  # 直接使用顶层框架
                on_complete=on_scoring_complete
            )
            
            # 注册一个轮询检查，不阻塞UI线程
            def check_scoring_complete():
                if not hasattr(self, 'scoring_in_progress') or not self.scoring_in_progress:
                    return
                    
                if scoring_event.is_set():
                    # 评分已完成，不需要再检查
                    return
                else:
                    # 继续轮询
                    self.master.after(100, check_scoring_complete)
            
            # 开始轮询
            self.master.after(100, check_scoring_complete)
            
            self.editor_status_var.set("进化评分器已加载，请完成评分")
            
        except Exception as e:
            self.logger.log(f"加载进化评分器失败: {str(e)}", "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")
            self.editor_status_var.set("加载评分器失败")
            self.show_error_dialog(f"加载进化评分器失败: {str(e)}")
            self.scoring_in_progress = False

    def _continue_generation(self):
        """继续地图生成过程"""
        # 检查地图是否已经完成生成
        if hasattr(self.map_data, 'generation_complete') and self.map_data.generation_complete:
            self.logger.log("地图已标记为生成完成，跳过继续生成")
            # 更新状态
            self.status_var.set("地图生成已完成")
            self.progress_var.set(100)
            # 如果有恢复状态，清除它
            if hasattr(self, 'resume_state'):
                delattr(self, 'resume_state')
            return
            
        # 检查评分是否正在进行
        if hasattr(self, 'scoring_in_progress') and self.scoring_in_progress:
            self.logger.log("评分尚未完成，等待评分...")
            
            # 检查评分等待次数
            if not hasattr(self, '_scoring_wait_count'):
                self._scoring_wait_count = 0
            
            self._scoring_wait_count += 1
            
            # 如果等待超过一定次数，认为评分已经完成或失败
            if self._scoring_wait_count > 20:  # 10秒后超时 (500ms * 20)
                self.logger.log("评分等待超时，强制继续生成", "WARNING")
                self.scoring_in_progress = False
                if hasattr(self, '_scoring_wait_count'):
                    delattr(self, '_scoring_wait_count')
            else:
                # 延迟稍后再次检查
                self.master.after(500, self._continue_generation)
                return
        
        # 重置评分等待计数
        if hasattr(self, '_scoring_wait_count'):
            delattr(self, '_scoring_wait_count')
        
        # 评分已完成，继续生成
        self.status_var.set("继续生成地图...")
        self.logger.log("继续生成地图...")
        
        # 创建一个新的future用于继续生成
        future = self.task_manager.submit_task(
            "continue_generation",
            self._run_continue_generation
        )
        
        # 现在可以添加完成回调
        future.add_done_callback(self._on_generate_complete)

    def _run_continue_generation(self):
        """继续执行地图生成任务"""
        from core.generate_map import generate_map
        try:
            # 检查map_data中是否有恢复点信息
            if not hasattr(self.map_data, 'generation_state') or not self.map_data.generation_state:
                self.logger.log("错误：没有找到恢复点信息，无法继续生成", "ERROR")
                return None
                
            # 获取当前地图参数
            preferences = {}
            for attr in dir(self.map_params):
                if not attr.startswith('_') and not callable(getattr(self.map_params, attr)):
                    preferences[attr] = getattr(self.map_params, attr)
                    
            # 读取地图设置
            width = self.map_data.width
            height = self.map_data.height

            # 获取是否启用可视化设置
            enable_visualize = hasattr(self, 'visualize_var') and self.visualize_var.get() if hasattr(self, 'visualize_var') else True
            
            # 为可视化创建专用框架，避免与编辑器共享框架
            visualization_frame = None
            if enable_visualize:
                # 首先检查是否有预览画布
                if hasattr(self, 'preview_canvas') and self.preview_canvas:
                    self.logger.log("使用预览画布")
                    visualization_frame = self.preview_canvas.master                
                # 然后检查是否有预览框架
                elif hasattr(self, 'preview_frame') and self.preview_frame:
                    self.logger.log("使用预览框架")
                    visualization_frame = self.preview_frame
                else:
                    # 尝试创建一个临时画布
                    try:
                        if hasattr(self, 'output_frame'):
                            print("创建临时画布")
                            preview_frame = ttk.LabelFrame(self.output_frame, text="地图预览")
                            preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                            self.preview_frame = preview_frame
                            self.preview_canvas = tk.Canvas(preview_frame, bg="white", cursor="hand2")
                            self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                            visualization_frame = preview_frame
                            self.logger.log("已动态创建预览画布", "INFO")
                        else:
                            self.logger.log("警告：找不到输出框架，可视化将不会嵌入到GUI中", "WARNING")
                    except Exception as e:
                        self.logger.log(f"创建预览画布失败: {str(e)}", "WARNING")
                        self.logger.log("警告：找不到预览画布，可视化将不会嵌入到GUI中", "WARNING")
            
            # 设置进度条
            self._queue_action(lambda: self.progress_var.set(50))  # 设置为50%表示正在继续处理
            self._queue_action(lambda: self.status_var.set("继续执行地图生成..."))
            
            # 调用generate_map函数继续执行，确保传递现有的map_data对象
            self.logger.log("从保存点继续地图生成...")
            result = generate_map(
                preferences=preferences,
                width=width,
                height=height,
                export_model=False,
                logger=self.logger,
                parent_frame=visualization_frame,  # 使用预览框架作为可视化容器
                parent_frame_edit=self.editor_container, # 地形编辑器
                use_gui_editors=False,  # 已经完成了GUI编辑，设为False避免再次打开编辑器
                map_data=self.map_data
            )
            
            # 更新进度
            self._queue_action(lambda: self.progress_var.set(100))
            self._queue_action(lambda: self.status_var.set("地图生成完成"))
            
            return result
            
        except Exception as e:
            self.logger.log(f"继续生成过程中发生错误: {str(e)}", "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")
            self._queue_action(lambda: self.progress_var.set(0))
            self._queue_action(lambda: self.status_var.set("生成失败"))
            raise

    def _continue_evolution_with_scores(self, scores):
        """使用用户评分继续进化过程"""
        # 在这里添加进化逻辑
        self.logger.log("根据用户评分继续进化...")
        try:
            # 执行进化
            self.evolution_engine.evolve_generation(scores)
            
            # 获取并应用进化后的最佳地图
            evolved_biome_map = self.evolution_engine.best_individual
            
            # 更新地图数据
            if isinstance(evolved_biome_map, np.ndarray) and evolved_biome_map.shape == self.map_data.get_layer("biome").shape:
                self.map_data.layers["biome"][:] = evolved_biome_map
                self.logger.log("已应用进化后的生物群落地图")
                
                # 切换回主控制面板
                self.param_notebook.select(0)
                
                # 显示进化完成消息
                self.show_warning_dialog("进化完成，生物群落地图已更新")
                
                # 更新状态
                self.state = ViewState.EVOLUTION_DONE
            else:
                self.logger.log("生物群落进化结果格式无效，保留原始数据", "WARNING")
        except Exception as e:
            self.logger.error(f"进化处理失败: {str(e)}")
            self.show_error_dialog(f"进化处理失败: {str(e)}")
      
    def _apply_preset(self, preset_name):
        """应用预设配置"""
        # 从配置管理器获取预设
        presets = self.config_manager.get("presets", {})
        
        # 确保presets是字典
        if not isinstance(presets, dict):
            self.logger.warning(f"预设配置类型错误：期望字典，实际为{type(presets)}")
            self.status_var.set("预设配置格式错误，无法应用")
            return
        
        if preset_name in presets:
            params = presets[preset_name]
            if not isinstance(params, dict):
                self.logger.warning(f"预设'{preset_name}'类型错误：期望字典，实际为{type(params)}")
                self.status_var.set(f"预设'{preset_name}'格式错误，无法应用")
                return
            
            # 更新参数并更新UI
            for param, value in params.items():
                if hasattr(self.map_params, param):
                    # 更新参数值
                    setattr(self.map_params, param, value)
                    
                    # 更新UI控件
                    if param in self.param_controls:
                        self.param_controls[param]["var"].set(value)
            
            self.logger.log(f"已应用预设: {preset_name}")
            self.status_var.set(f"预设配置 '{preset_name}' 已应用")
        else:
            self.logger.warning(f"未找到预设: {preset_name}")
            
    def _setup_theme(self):
        """设置应用主题和全局样式"""
        # 获取当前主题设置
        theme_name = self.config_manager.get("ui.theme", "default")
        
        # 创建样式对象
        style = ttk.Style()
        
        # 定义工具提示基本样式
        self.tooltip_font = ('', 9)
        
        # 根据主题名称应用不同的样式
        if theme_name == "dark":
            # 暗色主题
            self._apply_dark_theme(style)
            # 暗色主题的工具提示颜色
            self.tooltip_bg = "#333333"
            self.tooltip_fg = "#FFFFFF"
            self.tooltip_border = "#555555"
        elif theme_name == "light":
            # 亮色主题
            self._apply_light_theme(style)
            # 亮色主题的工具提示颜色
            self.tooltip_bg = "#FFFFEA"
            self.tooltip_fg = "#000000"
            self.tooltip_border = "#DDDDDD"
        elif theme_name == "blue":
            # 蓝色主题
            self._apply_blue_theme(style)
            # 蓝色主题的工具提示颜色
            self.tooltip_bg = "#E8F4FF"
            self.tooltip_fg = "#000033"
            self.tooltip_border = "#A0C0E0"
        else:
            # 默认主题
            self._apply_default_theme(style)
            # 默认主题的工具提示颜色
            self.tooltip_bg = "#FFFFEA"
            self.tooltip_fg = "#000000"
            self.tooltip_border = "#CCCCCC"
        
        # 设置通用控件样式
        style.configure("Bold.TLabelframe.Label", font=('', 10, 'bold'))
        style.configure("Bold.TLabelframe", borderwidth=2)
        style.configure("Rounded.Horizontal.TProgressbar", borderwidth=0, background="#4CAF50")
        
        # 设置Notebook选项卡样式
        style.configure("App.TNotebook.Tab", focuscolor=style.configure(".")["background"])

    def _apply_dark_theme(self, style):
        """应用暗色主题"""
        # 基础颜色
        bg_color = "#2E2E2E"
        fg_color = "#FFFFFF"
        select_bg = "#505050"
        accent_color = "#7289DA"
        
        # 设置基础样式
        style.configure(".", 
                    background=bg_color,
                    foreground=fg_color,
                    troughcolor="#1E1E1E",
                    selectbackground=select_bg,
                    selectforeground=fg_color,
                    fieldbackground=bg_color,
                    borderwidth=1,
                    darkcolor=bg_color,
                    lightcolor=bg_color,
                    relief=tk.FLAT)
        
        # 设置特定控件样式
        style.configure("TButton", background="#3E3E3E", foreground=fg_color)
        style.map("TButton", 
                background=[("active", "#505050"), ("pressed", "#404040")],
                foreground=[("active", fg_color)])
        
        style.configure("Action.TButton", background=accent_color)
        style.map("Action.TButton",
                background=[("active", "#8299EA"), ("pressed", "#6279CA")])
        
        style.configure("TNotebook", background=bg_color)
        style.configure("TNotebook.Tab", background="#3E3E3E", foreground=fg_color)
        style.map("TNotebook.Tab",
                background=[("selected", accent_color), ("active", "#505050")],
                foreground=[("selected", "#FFFFFF")])
        
        style.configure("Treeview", background="#333333", foreground=fg_color, fieldbackground="#333333")
        style.map("Treeview", background=[("selected", accent_color)])
        
        # 设置滚动条样式
        style.configure("TScrollbar", background="#3E3E3E", troughcolor="#2A2A2A")
        style.map("TScrollbar", background=[("active", "#505050")])
        
        # 应用到主窗口
        self.master.configure(background=bg_color)

    def _apply_light_theme(self, style):
        """应用亮色主题"""
        # 基础颜色
        bg_color = "#F5F5F5"
        fg_color = "#000000"
        select_bg = "#CCE8FF"
        accent_color = "#007BFF"
        
        # 设置基础样式
        style.configure(".", 
                    background=bg_color,
                    foreground=fg_color,
                    troughcolor="#E0E0E0",
                    selectbackground=select_bg,
                    selectforeground=fg_color,
                    fieldbackground="white",
                    borderwidth=1,
                    darkcolor="#CCCCCC",
                    lightcolor="white")
        
        # 设置特定控件样式
        style.configure("TButton", background="#E0E0E0")
        style.map("TButton", 
                background=[("active", "#D0D0D0"), ("pressed", "#C0C0C0")])
        
        style.configure("Action.TButton", background=accent_color, foreground="white")
        style.map("Action.TButton",
                background=[("active", "#0069D9"), ("pressed", "#0062CC")],
                foreground=[("active", "white"), ("pressed", "white")])
        
        style.configure("TNotebook", background=bg_color)
        style.configure("TNotebook.Tab", background="#E0E0E0", foreground=fg_color)
        style.map("TNotebook.Tab",
                background=[("selected", accent_color), ("active", "#D0D0D0")],
                foreground=[("selected", "white")])
        
        # 应用到主窗口
        self.master.configure(background=bg_color)
        
    def _apply_blue_theme(self, style):
        """应用蓝色主题"""
        # 基础颜色
        bg_color = "#ECF2F9"
        fg_color = "#0A1A2A"
        select_bg = "#B8D0E8"
        accent_color = "#1E6EB7"
        
        # 设置基础样式
        style.configure(".", 
                    background=bg_color,
                    foreground=fg_color,
                    troughcolor="#D4E4F7",
                    selectbackground=select_bg,
                    selectforeground=fg_color,
                    fieldbackground="white",
                    borderwidth=1,
                    darkcolor="#C2D6F0",
                    lightcolor="white")
        
        # 设置特定控件样式
        style.configure("TButton", background="#D4E4F7")
        style.map("TButton", 
                background=[("active", "#C2D6F0"), ("pressed", "#B0C4DE")])
        
        style.configure("Action.TButton", background=accent_color, foreground="white")
        style.map("Action.TButton",
                background=[("active", "#165791"), ("pressed", "#0E3C5E")],
                foreground=[("active", "white"), ("pressed", "white")])
        
        style.configure("TNotebook", background=bg_color)
        style.configure("TNotebook.Tab", background="#D4E4F7", foreground=fg_color)
        style.map("TNotebook.Tab",
                background=[("selected", accent_color), ("active", "#C2D6F0")],
                foreground=[("selected", "white")])
        
        # 应用到主窗口
        self.master.configure(background=bg_color)
        
    def _apply_default_theme(self, style):
        """应用默认主题"""
        # 基础颜色
        bg_color = "#f0f0f0"
        fg_color = "#000000"
        select_bg = "#0078D7"
        
        # 重置为默认样式
        style.theme_use('default')
        
        # 设置基础样式微调
        style.configure(".", 
                    background=bg_color,
                    foreground=fg_color,
                    selectbackground=select_bg,
                    selectforeground="white")
        
        # 添加圆角和动画效果的进度条样式
        style.configure("Rounded.Horizontal.TProgressbar", 
                    background="#4CAF50",
                    troughcolor="#E0E0E0",
                    borderwidth=0,
                    bordercolor=bg_color)
        
        # 设置主窗口背景
        self.master.configure(background=bg_color)
        
    def _toggle_autosave(self):
        """切换自动保存功能"""
        is_enabled = self.autosave_var.get()
        self.config_manager.set("auto_save", is_enabled)
        self.config_manager.save_config()
        
        if is_enabled:
            self._setup_autosave()
            self.logger.log("已启用自动保存功能")
        else:
            if hasattr(self, 'autosave_id'):
                self.master.after_cancel(self.autosave_id)
                self.logger.log("已禁用自动保存功能")
                
    def _show_preferences(self):
        """显示应用程序首选项对话框"""
        pref_dialog = tk.Toplevel(self.master)
        pref_dialog.title("首选项")
        pref_dialog.geometry("550x450")
        pref_dialog.transient(self.master)
        pref_dialog.grab_set()
        
        # 创建标签页
        notebook = ttk.Notebook(pref_dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 常规设置标签页
        general_tab = ttk.Frame(notebook, padding=10)
        notebook.add(general_tab, text="常规")
        
        # 创建常规设置选项
        autosave_frame = ttk.LabelFrame(general_tab, text="自动保存", padding=10)
        autosave_frame.pack(fill=tk.X, pady=5)
        
        # 自动保存启用选项
        autosave_var = tk.BooleanVar(value=self.config_manager.get("auto_save", True))
        autosave_check = ttk.Checkbutton(
            autosave_frame, 
            text="启用自动保存", 
            variable=autosave_var
        )
        autosave_check.pack(anchor=tk.W, padx=5, pady=5)
        
        # 自动保存间隔
        interval_frame = ttk.Frame(autosave_frame)
        interval_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(interval_frame, text="自动保存间隔 (分钟):").pack(side=tk.LEFT, padx=5)
        
        interval_var = tk.IntVar(value=self.config_manager.get("autosave_interval", 5))
        interval_spinbox = ttk.Spinbox(
            interval_frame, 
            from_=1, 
            to=30, 
            textvariable=interval_var, 
            width=5
        )
        interval_spinbox.pack(side=tk.LEFT, padx=5)
        
        # 性能设置
        performance_frame = ttk.LabelFrame(general_tab, text="性能", padding=10)
        performance_frame.pack(fill=tk.X, pady=10)
        
        # 最大撤销历史记录
        undo_frame = ttk.Frame(performance_frame)
        undo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(undo_frame, text="最大撤销历史记录:").pack(side=tk.LEFT, padx=5)
        
        undo_var = tk.IntVar(value=self.config_manager.get("max_undo", 20))
        undo_spinbox = ttk.Spinbox(
            undo_frame, 
            from_=5, 
            to=100, 
            textvariable=undo_var, 
            width=5
        )
        undo_spinbox.pack(side=tk.LEFT, padx=5)
        
        # 多线程设置
        thread_frame = ttk.Frame(performance_frame)
        thread_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(thread_frame, text="最大线程数:").pack(side=tk.LEFT, padx=5)
        
        thread_var = tk.IntVar(value=self.config_manager.get("max_threads", 4))
        thread_spinbox = ttk.Spinbox(
            thread_frame, 
            from_=1, 
            to=16, 
            textvariable=thread_var, 
            width=5
        )
        thread_spinbox.pack(side=tk.LEFT, padx=5)
        
        # 外观设置标签页
        appearance_tab = ttk.Frame(notebook, padding=10)
        notebook.add(appearance_tab, text="外观")
        
        # 主题设置
        theme_frame = ttk.LabelFrame(appearance_tab, text="主题", padding=10)
        theme_frame.pack(fill=tk.X, pady=5)
        
        theme_var = tk.StringVar(value=self.config_manager.get("ui.theme", "default"))
        ttk.Radiobutton(theme_frame, text="默认主题", variable=theme_var, value="default").pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(theme_frame, text="亮色主题", variable=theme_var, value="light").pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(theme_frame, text="暗色主题", variable=theme_var, value="dark").pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(theme_frame, text="蓝色主题", variable=theme_var, value="blue").pack(anchor=tk.W, padx=20, pady=2)
        
        # 字体大小
        font_frame = ttk.LabelFrame(appearance_tab, text="字体", padding=10)
        font_frame.pack(fill=tk.X, pady=10)
        
        ui_font_frame = ttk.Frame(font_frame)
        ui_font_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ui_font_frame, text="界面字体大小:").pack(side=tk.LEFT, padx=5)
        
        ui_font_var = tk.IntVar(value=self.config_manager.get("ui.font_size", 9))
        ui_font_spinbox = ttk.Spinbox(
            ui_font_frame, 
            from_=7, 
            to=14, 
            textvariable=ui_font_var, 
            width=5
        )
        ui_font_spinbox.pack(side=tk.LEFT, padx=5)
        
        # 添加保存和取消按钮
        button_frame = ttk.Frame(pref_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_preferences():
            # 保存所有配置
            self.config_manager.set("auto_save", autosave_var.get())
            self.config_manager.set("autosave_interval", interval_var.get())
            self.config_manager.set("max_undo", undo_var.get())
            self.config_manager.set("max_threads", thread_var.get())
            self.config_manager.set("ui.theme", theme_var.get())
            self.config_manager.set("ui.font_size", ui_font_var.get())
            
            # 保存配置
            self.config_manager.save_config()
            
            # 应用设置
            self.autosave_var.set(autosave_var.get())
            self._toggle_autosave()  # 更新自动保存状态
            
            # 更新命令历史记录大小
            self.command_history.set_max_history(undo_var.get())
            
            # 如果主题发生变化，更新主题
            if theme_var.get() != self.theme_var.get():
                self.theme_var.set(theme_var.get())
                self._change_theme()
            
            # 关闭对话框
            pref_dialog.destroy()
            
            # 显示通知
            self._show_notification("设置已保存", "所有首选项已更新并应用", type_="success")
        
        ttk.Button(button_frame, text="保存", command=save_preferences).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="取消", command=pref_dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _setup_autosave(self):
        """设置自动保存功能"""
        # 如果已经有自动保存计时器，先取消它
        if hasattr(self, 'autosave_id'):
            self.master.after_cancel(self.autosave_id)
        
        # 获取自动保存间隔（分钟）
        interval = self.config_manager.get("autosave_interval", 5)
        # 转换为毫秒
        interval_ms = interval * 60 * 1000
        
        # 设置自动保存函数
        def do_autosave():
            if self.current_file and self.current_file != "./exports":
                try:
                    # 执行保存
                    self._save_project()
                    self.logger.log(f"已自动保存到 {self.current_file}", "INFO")
                except Exception as e:
                    self.logger.log(f"自动保存失败: {str(e)}", "ERROR")
            
            # 重新安排下一次自动保存
            self.autosave_id = self.master.after(interval_ms, do_autosave)
        
        # 安排第一次自动保存
        self.autosave_id = self.master.after(interval_ms, do_autosave)

    def _create_tooltip(self, widget, text, delay=500):
        """为控件添加增强型工具提示"""
        tip_window = None
        tip_id = None
        
        def enter(event):
            nonlocal tip_id
            # 延迟显示工具提示
            tip_id = widget.after(delay, show_tip)
        
        def leave(event):
            nonlocal tip_id, tip_window
            # 取消延迟显示
            if tip_id:
                widget.after_cancel(tip_id)
                tip_id = None
            # 关闭已显示的提示
            if tip_window:
                tip_window.destroy()
                tip_window = None
        
        def show_tip():
            nonlocal tip_window
            # 获取鼠标位置
            x, y = widget.winfo_pointerxy()
            
            # 创建提示窗口
            tip_window = tk.Toplevel(widget)
            tip_window.wm_overrideredirect(True)  # 无边框窗口
            tip_window.wm_geometry(f"+{x+15}+{y+10}")
            
            # 使用带有边框和阴影效果的标签
            try:
                bg_color = getattr(self, 'tooltip_bg', "#FFFFEA")
                fg_color = getattr(self, 'tooltip_fg', "#000000")
                tooltip_font = getattr(self, 'tooltip_font', ('', 9))
            except AttributeError:
                bg_color = "#FFFFEA"
                fg_color = "#000000"
                tooltip_font = ('', 9)
            
            tip_frame = ttk.Frame(tip_window, padding=1)
            tip_frame.pack(fill=tk.BOTH, expand=True)
            
            label = ttk.Label(
                tip_frame, 
                text=text, 
                background=bg_color,
                foreground=fg_color,
                font=tooltip_font,
                justify=tk.LEFT,
                wraplength=300,
                padding=5
            )
            label.pack()
            
            # 淡入效果
            tip_window.attributes("-alpha", 0.0)
            fade_in()
        
        def fade_in(alpha=0.1):
            nonlocal tip_window
            if not tip_window:
                return
            
            if alpha < 1.0:
                tip_window.attributes("-alpha", alpha)
                tip_window.after(30, lambda: fade_in(alpha + 0.1))
            else:
                tip_window.attributes("-alpha", 1.0)
        
        # 绑定事件
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
        widget.bind("<Button-1>", leave)
        widget.bind("<ButtonRelease-1>", enter)

    def _show_terrain_settings(self):
        """显示地形高级设置对话框"""
        # 收集当前地形参数
        current_params = {}
        
        # 从map_params获取现有参数
        for key, value in vars(self.map_params).items():
            current_params[key] = value
        
        # 从存储的高级参数中获取
        if hasattr(self, "terrain_advanced_params"):
            current_params.update(self.terrain_advanced_params)
        
        # 显示设置对话框
        result = show_terrain_settings(self.master, current_params)
        
        if result:
            self.logger.log("应用地形高级设置")
            
            # 更新基本参数到map_params
            for param in ['seed', 'scale_factor', 'mountain_sharpness', 'erosion_iterations', 
                        'river_density', 'use_tectonic', 'detail_level']:
                if param in result:
                    setattr(self.map_params, param, result[param])
                    if param in self.param_controls and hasattr(self.param_controls[param], "var"):
                        self.param_controls[param]["var"].set(result[param])
            
            # 存储所有高级参数供后续使用
            self.terrain_advanced_params = result
            
            # 提取侵蚀类型和参数
            erosion_params = {}
            for key in ['erosion_type', 'erosion_strength', 'talus_angle', 'sediment_capacity',
                    'rainfall', 'evaporation']:
                if key in result:
                    erosion_params[key] = result[key]
            
            # 提取河流参数
            river_params = {}
            for key in ['min_watershed_size', 'precipitation_factor', 'meander_factor']:
                if key in result:
                    river_params[key] = result[key]
            
            # 提取噪声参数
            noise_params = {}
            for key in ['octaves', 'persistence', 'lacunarity', 'use_frequency_optimization']:
                if key in result:
                    noise_params[key] = result[key]
            
            # 提取地形分布参数
            distribution_params = {}
            for key in ['plain_ratio', 'hill_ratio', 'mountain_ratio', 'plateau_ratio']:
                if key in result:
                    distribution_params[key] = result[key]
            
            # 提取气候参数
            climate_params = {}
            for key in ['latitude_effect', 'prevailing_wind_x', 'prevailing_wind_y']:
                if key in result:
                    climate_params[key] = result[key]
                    
            # 记录到日志
            self.logger.log(f"侵蚀设置: {erosion_params}")
            self.logger.log(f"河流设置: {river_params}")
            self.logger.log(f"噪声设置: {noise_params}")
            self.logger.log(f"地形分布: {distribution_params}")
            self.logger.log(f"气候设置: {climate_params}")
            
            self.status_var.set("地形高级设置已更新")
   
    def _export_and_deploy(self):
        """打开一键导出并部署对话框"""
        if not self.map_data:
            self.logger.error("无地图数据可导出 (map_data为空)")
            return
        
        if not hasattr(self.map_data, 'is_valid'):
            self.logger.error("无地图数据可导出 (map_data对象缺少is_valid方法)")
            return
            
        if not self.map_data.is_valid():
            self.logger.error("无地图数据可导出 (map_data.is_valid()返回False)")
            return
        
        # 创建简化的导出对话框
        deploy_dialog = tk.Toplevel(self.master)
        deploy_dialog.title("一键导出并部署")
        deploy_dialog.grab_set()
        deploy_dialog.geometry("400x300")
        
        frame = ttk.Frame(deploy_dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        ttk.Label(frame, text="选择目标引擎/软件", font=("Arial", 12, "bold")).pack(pady=10)
        
        # 导出配置
        config = MapExportConfig()
        
        # 目标选择
        target_var = tk.StringVar(value="unity")
        
        ttk.Radiobutton(
            frame, text="导出并部署到Unity", 
            variable=target_var, value="unity"
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Radiobutton(
            frame, text="导出并部署到Unreal Engine", 
            variable=target_var, value="unreal"
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Radiobutton(
            frame, text="导出并在Blender中打开", 
            variable=target_var, value="blender"
        ).pack(anchor=tk.W, pady=5)
        
        # 输出目录
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(dir_frame, text="输出目录:").pack(side=tk.LEFT)
        output_dir_var = tk.StringVar(value=config.output_dir)
        ttk.Entry(dir_frame, textvariable=output_dir_var, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="浏览...", 
                command=lambda: self._browse_dir(output_dir_var)).pack(side=tk.RIGHT)
        
        # 按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        def start_export_deploy():
            # 设置导出配置
            self.export_config = MapExportConfig(
                output_dir=output_dir_var.get()
            )
            
            # 保存部署选项
            self.auto_deploy = True
            target = target_var.get()
            
            # 根据目标确定要导出的格式
            export_obj = (target == "blender")
            export_unity = (target == "unity")
            export_unreal = (target == "unreal")
            
            # 关闭对话框
            deploy_dialog.destroy()
            
            # 设置特殊标记
            self.open_obj_in = "在Blender中打开" if target == "blender" else "不自动打开"
            
            # 执行导出
            self._do_export_map(export_obj, export_unity, export_unreal)
        
        ttk.Button(btn_frame, text="开始导出并部署", 
                command=start_export_deploy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="取消", 
                command=deploy_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # 居中对话框
        deploy_dialog.update_idletasks()
        width = deploy_dialog.winfo_width()
        height = deploy_dialog.winfo_height()
        x = (deploy_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (deploy_dialog.winfo_screenheight() // 2) - (height // 2)
        deploy_dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
    def _preview_map_3d(self):
        """打开3D地图预览窗口"""
        if not self.map_data or not self.map_data.is_valid():
            self.logger.error("无地图数据可预览，请先生成地图")
            return
            
        try:
            # 导入3D预览模块
            from utils.preview_map_3d import preview_map_3d
            preview_map_3d(self.map_data, self.master)
        except Exception as e:
            self.logger.error(f"打开3D预览时发生错误: {str(e)}")
        
    def _check_geo_dependencies(self):
        """检查地理数据处理所需的依赖库"""
        try:
            import rasterio
            has_rasterio = True
        except ImportError:
            has_rasterio = False
            
        try:
            import elevation
            has_elevation = True
        except ImportError:
            has_elevation = False
            
        if not has_rasterio or not has_elevation:
            missing = []
            if not has_rasterio:
                missing.append("rasterio")
            if not has_elevation:
                missing.append("elevation")
                
            message = f"缺少地理数据处理所需的依赖库: {', '.join(missing)}\n"
            message += "要使用真实地理数据功能，请安装这些库:\n"
            message += "pip install rasterio elevation"
            
            self.logger.log(message, "WARNING")        
    
    def _show_geo_import_dialog(self):
        """显示导入地理数据的对话框"""
        if not hasattr(self, 'use_real_geo_var'):
            self.logger.error("未初始化地理数据控件")
            return
            
        # 设置地理数据选项并滚动到视图
        self.use_real_geo_var.set(True)
        self._toggle_geo_data_ui()
        
        # 尝试滚动到地理数据设置区域
        try:
            # 确保geo_options_frame已经更新布局
            self.geo_options_frame.update_idletasks()
            self.scrollable_frame.update_idletasks()
            
            # 使用yview方法滚动到地理数据框架位置
            # 计算地理数据框架在画布中的相对位置
            canvas = self.scrollable_frame.master
            if isinstance(canvas, tk.Canvas):
                # 获取geo_options_frame相对于scrollable_frame的y坐标
                y_position = self.geo_options_frame.winfo_y() / self.scrollable_frame.winfo_height()
                # 滚动到该位置
                canvas.yview_moveto(y_position)
                
            self.logger.log("请在控制面板中设置地理数据选项")
        except Exception as e:
            self.logger.log(f"无法滚动到地理数据设置: {str(e)}", "WARNING")
        
    def _toggle_geo_data_ui(self):
        """切换地理数据UI的显示/隐藏"""
        if self.use_real_geo_var.get():
            self.geo_options_frame.pack(fill=tk.X, padx=10, pady=5)  # 使用pack而不是grid
        else:
            self.geo_options_frame.pack_forget()  # 使用pack_forget而不是grid_remove

    def _update_geo_source_ui(self):
        """根据选择的地理数据源更新UI"""
        source = self.geo_source_var.get()
        if source == "file":
            self.file_frame.pack(fill=tk.X, padx=5, pady=5)  # 使用pack而不是grid
            self.srtm_frame.pack_forget()  # 使用pack_forget而不是grid_remove
        else:  # srtm
            self.file_frame.pack_forget()  # 使用pack_forget而不是grid_remove
            self.srtm_frame.pack(fill=tk.X, padx=5, pady=5)  # 使用pack而不是grid

    def _browse_geo_file(self):
        """浏览并选择地理数据文件"""
        filepath = filedialog.askopenfilename(
            title="选择GeoTIFF文件",
            filetypes=[
                ("GeoTIFF文件", "*.tif;*.tiff"),
                ("所有支持的格式", "*.tif;*.tiff;*.dem;*.asc;*.hgt"), 
                ("所有文件", "*.*")
            ]
        )
        if filepath:
            self.geo_file_var.set(filepath)

    def _select_preset_region(self, event):
        """选择预设区域坐标"""
        region = event.widget.get()
        # 预设区域的经纬度坐标
        presets = {
            "北京": ("116.3", "116.5", "39.9", "40.0"),
            "上海": ("121.4", "121.6", "31.1", "31.3"),
            "黄山": ("118.10", "118.20", "30.08", "30.18"),
            "泰山": ("117.05", "117.15", "36.20", "36.30"),
            "张家界": ("110.35", "110.45", "29.12", "29.22")
        }
        
        if region in presets:
            lng_min, lng_max, lat_min, lat_max = presets[region]
            self.lng_min_var.set(lng_min)
            self.lng_max_var.set(lng_max)
            self.lat_min_var.set(lat_min)
            self.lat_max_var.set(lat_max)

    def _update_attr_range(self, attr_name, min_val, max_val):
        """更新属性的范围设置"""
        # 首先确保全局变量ATTR_RANGES存在
        global ATTR_RANGES
        if 'ATTR_RANGES' not in globals():
            ATTR_RANGES = {}
        
        # 获取旧值用于命令历史
        old_min, old_max = ATTR_RANGES.get(attr_name, (min_val, max_val))
        
        # 确保最小值不大于最大值
        if min_val > max_val:
            min_val = max_val - 1
            if attr_name in self.attr_controls:
                self.attr_controls[attr_name]['min_var'].set(min_val)
                self.logger.warning(f"属性 {attr_name} 的最小值已调整为 {min_val}")
        
        # 更新全局属性范围
        ATTR_RANGES[attr_name] = (min_val, max_val)
        
        # 添加到命令历史以支持撤销/重做
        self.command_history.add_command({
            "type": "attr_range_change",
            "attr_name": attr_name,
            "old_values": (old_min, old_max),
            "new_values": (min_val, max_val)
        })
        
        # 更新UI状态
        self.status_var.set(f"属性范围已更新: {attr_name} = {min_val} - {max_val}")

    def _create_attr_range_control(self, parent, attr_name, min_val, max_val, row):
        """为单位属性范围创建控制器"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)  # 使用pack替代grid
        
        # 创建一个内部框架来使用grid布局
        inner_frame = ttk.Frame(frame)
        inner_frame.pack(fill=tk.X)
        
        # 属性名称标签
        ttk.Label(inner_frame, text=attr_name, width=12).grid(row=0, column=0, sticky="w")
        
        # 最小值控制
        ttk.Label(inner_frame, text="最小值:").grid(row=0, column=1, padx=(5, 0))
        min_var = tk.IntVar(value=min_val)
        min_spinbox = ttk.Spinbox(
            inner_frame, 
            from_=0, 
            to=1000, 
            textvariable=min_var,
            width=5,
            command=lambda: self._update_attr_range(attr_name, min_var.get(), max_var.get())
        )
        min_spinbox.grid(row=0, column=2)
        
        # 最大值控制
        ttk.Label(inner_frame, text="最大值:").grid(row=0, column=3, padx=(10, 0))
        max_var = tk.IntVar(value=max_val)
        max_spinbox = ttk.Spinbox(
            inner_frame, 
            from_=1, 
            to=1000, 
            textvariable=max_var,
            width=5,
            command=lambda: self._update_attr_range(attr_name, min_var.get(), max_var.get())
        )
        max_spinbox.grid(row=0, column=4)
        
        # 存储控件引用
        self.attr_controls[attr_name] = {
            "min_var": min_var, 
            "max_var": max_var,
            "min_spinbox": min_spinbox,
            "max_spinbox": max_spinbox
        }
        
    def _update_target_ratio(self):
        """更新生态平衡目标比率"""
        global TARGET_RATIO
        old_value = TARGET_RATIO
        try:
            new_value = float(self.target_ratio_var.get())
            if new_value <= 0:
                self.logger.warning("目标比率必须大于0，已重置为1.0")
                new_value = 1.0
                self.target_ratio_var.set(new_value)
        except ValueError:
            self.logger.warning("无效的目标比率值，已重置为1.0")
            new_value = 1.0
            self.target_ratio_var.set(new_value)
                
        TARGET_RATIO = new_value
        
        self.command_history.add_command({
            "type": "target_ratio_change",
            "old_value": old_value,
            "new_value": new_value
        })
        
        self.status_var.set(f"目标比率已更新: {new_value:.2f}")
        
    def _toggle_gpu(self):
        """切换GPU加速选项"""
        use_gpu = self.gpu_var.get()
        if use_gpu and not MapData.HAS_GPU:
            self.logger.warning("GPU加速不可用，已自动禁用该选项")
            self.gpu_var.set(False)
            use_gpu = False
            
        self.status_var.set(f"GPU加速已{'启用' if use_gpu else '禁用'}")
        
    def _toggle_multithreaded(self):
        """切换多线程选项"""
        use_mt = self.multithreaded_var.get()
        self.status_var.set(f"多线程已{'启用' if use_mt else '禁用'}")
        
    def _on_biome_param_change(self, param, value):
        """处理生物群系参数变化"""
        if not hasattr(self, "biome_params"):
            self.biome_params = {}
            
        old_value = self.biome_params.get(param, 0.5)
        self.biome_params[param] = value
        
        self.command_history.add_command({
            "type": "biome_param_change",
            "param": param,
            "old_value": old_value,
            "new_value": value
        })
        
        self.status_var.set(f"生物群系参数已更新: {param} = {value:.2f}")

    def _open_image_browser(self):
        """打开图片浏览器对话框选择多张图片"""
        file_paths = filedialog.askopenfilenames(
            title="选择图片文件",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
                ("PNG文件", "*.png"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_paths:
            # 清除之前的图片列表和缓存
            self.image_list = list(file_paths)
            self.current_image_index = -1
            self.tk_images.clear()
            
            # 开始显示第一张图片
            self._navigate_images(1)
            
            # 更新状态信息
            self.status_var.set(f"已加载 {len(self.image_list)} 张图片")

    def _navigate_images(self, direction):
        """在图片列表中导航，direction为1表示前进，-1表示后退"""
        if not self.image_list:
            self.status_var.set("没有可浏览的图片")
            return
        
        # 计算新的索引并确保在有效范围内
        new_index = self.current_image_index + direction
        if new_index < 0:
            new_index = len(self.image_list) - 1
        elif new_index >= len(self.image_list):
            new_index = 0
        
        # 更新当前索引
        self.current_image_index = new_index
        
        # 更新计数器显示
        self.image_counter_var.set(f"{self.current_image_index + 1}/{len(self.image_list)}")
        
        # 显示当前图片
        self._display_current_image()

    def _display_current_image(self):
        """显示当前索引的图片"""
        if self.current_image_index < 0 or self.current_image_index >= len(self.image_list):
            return
        
        try:
            from PIL import Image, ImageTk
            
            # 获取当前图片路径
            image_path = self.image_list[self.current_image_index]
            
            # 检查图片是否已在缓存中
            if image_path not in self.tk_images:
                # 加载图片
                pil_image = Image.open(image_path)
                
                # 获取画布尺寸
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                
                # 调整图片大小以适应画布（保持纵横比）
                if canvas_width > 1 and canvas_height > 1:
                    # 计算缩放比例
                    img_width, img_height = pil_image.size
                    scale_w = canvas_width / img_width
                    scale_h = canvas_height / img_height
                    scale = min(scale_w, scale_h)
                    
                    # 如果图片比画布大，进行缩放
                    if scale < 1:
                        new_width = int(img_width * scale)
                        new_height = int(img_height * scale)
                        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 转换为Tkinter可用的图像
                self.tk_images[image_path] = ImageTk.PhotoImage(pil_image)
            
            # 清除画布
            self.preview_canvas.delete("all")
            
            # 在画布上显示图片
            self.preview_canvas.create_image(
                self.preview_offset_x,
                self.preview_offset_y,
                anchor=tk.NW,
                image=self.tk_images[image_path],
                tags="preview_image"
            )
            
            # 更新状态
            filename = os.path.basename(image_path)
            self.status_var.set(f"正在查看: {filename}")
            
        except Exception as e:
            # 显示错误信息
            error_msg = f"图片加载错误: {str(e)}"
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() / 2,
                self.preview_canvas.winfo_height() / 2,
                text=error_msg,
                font=("Arial", 12),
                fill="#FF0000"
            )
            if hasattr(self, 'logger'):
                self.logger.log(error_msg, "ERROR")

    def _setup_output_panel(self):
            """设置输出面板"""
            try:
                # 创建垂直分割的输出面板
                output_paned = ttk.PanedWindow(self.output_frame, orient=tk.VERTICAL)
                output_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                # 上方放预览窗口
                preview_frame = ttk.LabelFrame(output_paned, text="地图预览")
                output_paned.add(preview_frame, weight=3)
                
                # 保存预览框架引用用于可视化嵌入
                self.preview_frame = preview_frame
                
                # 创建预览画布
                self.preview_canvas = tk.Canvas(self.preview_frame, bg="white", cursor="hand2")
                self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                # 初始化预览变量
                self.preview_scale = 1.0
                self.preview_offset_x = 0
                self.preview_offset_y = 0
                self.drag_start_x = 0
                self.drag_start_y = 0
                self.is_dragging = False
                
                # 初始化图片浏览相关变量
                self.image_list = []  # 存储图片路径列表
                self.current_image_index = -1  # 当前显示的图片索引
                self.tk_images = {}  # 缓存已加载的Tkinter图像对象
                
                # 设置预览画布事件绑定
                self._setup_preview_events()
                
                # 添加预览控制面板
                preview_controls = ttk.Frame(preview_frame)
                preview_controls.pack(fill=tk.X, padx=5, pady=5)
                
                # 添加控制按钮
                zoom_controls = ttk.Frame(preview_controls)
                zoom_controls.pack(side=tk.LEFT)
                
                ttk.Button(
                    zoom_controls,
                    text="放大",
                    command=lambda: self._zoom_preview(1.25),
                    width=6
                ).pack(side=tk.LEFT, padx=2)
                
                ttk.Button(
                    zoom_controls,
                    text="缩小",
                    command=lambda: self._zoom_preview(0.8),
                    width=6
                ).pack(side=tk.LEFT, padx=2)
                
                ttk.Button(
                    zoom_controls,
                    text="重置缩放",
                    command=self._reset_preview_zoom,
                    width=10
                ).pack(side=tk.LEFT, padx=2)
                
                ttk.Button(
                    zoom_controls,
                    text="适应窗口",
                    command=self._fit_preview_to_window,
                    width=10
                ).pack(side=tk.LEFT, padx=2)
                
                # 添加图片浏览控制按钮
                image_browse_controls = ttk.Frame(preview_controls)
                image_browse_controls.pack(side=tk.LEFT, padx=10)
                
                ttk.Button(
                    image_browse_controls,
                    text="浏览图片",
                    command=self._open_image_browser,
                    width=10
                ).pack(side=tk.LEFT, padx=2)
                
                ttk.Button(
                    image_browse_controls,
                    text="上一张",
                    command=lambda: self._navigate_images(-1),
                    width=6
                ).pack(side=tk.LEFT, padx=2)
                
                ttk.Button(
                    image_browse_controls,
                    text="下一张",
                    command=lambda: self._navigate_images(1),
                    width=6
                ).pack(side=tk.LEFT, padx=2)
                
                # 图片计数显示
                self.image_counter_var = tk.StringVar(value="0/0")
                ttk.Label(
                    image_browse_controls,
                    textvariable=self.image_counter_var,
                    width=8
                ).pack(side=tk.LEFT, padx=5)
                
                # 右侧添加图层选择
                layer_controls = ttk.Frame(preview_controls)
                layer_controls.pack(side=tk.RIGHT)
                
                ttk.Label(layer_controls, text="显示图层:").pack(side=tk.LEFT, padx=(10, 2))
                
                self.preview_layer_var = tk.StringVar(value="height")
                layer_combo = ttk.Combobox(
                    layer_controls, 
                    textvariable=self.preview_layer_var,
                    values=["height", "biome", "temperature", "humidity", "vegetation", "combined"],
                    width=12,
                    state="readonly"
                )
                layer_combo.pack(side=tk.LEFT)
                layer_combo.bind("<<ComboboxSelected>>", lambda e: self._change_preview_layer())
                
                # 添加视图控制说明
                ttk.Label(
                    preview_controls, 
                    text="鼠标拖拽以平移，鼠标滚轮缩放", 
                    font=("", 8),
                    foreground="#666666"
                ).pack(side=tk.BOTTOM, fill=tk.X, anchor=tk.CENTER, pady=(5, 0))
                
                # 下方放日志窗口
                log_frame = ttk.LabelFrame(output_paned, text="日志输出")
                output_paned.add(log_frame, weight=1)
                
                # 创建日志文本框
                self.log_text = scrolledtext.ScrolledText(
                    log_frame, 
                    wrap=tk.WORD, 
                    height=8,
                    background="#F8F8F8",
                    font=("Consolas", 9)
                )
                self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                # 设置日志文本标签样式
                self.log_text.tag_configure("INFO", foreground="#000000")
                self.log_text.tag_configure("WARNING", foreground="#FF8800")
                self.log_text.tag_configure("ERROR", foreground="#FF0000")
                self.log_text.tag_configure("SUCCESS", foreground="#008800")
                
                # 设置分割位置
                self.master.update()
                output_paned.sashpos(0, int(self.output_frame.winfo_height() * 0.7))
                
                # 尝试显示欢迎信息
                self._show_welcome_preview()

            except Exception as e:
                # 如果初始化失败，确保记录错误但不阻止程序启动
                print(f"输出面板初始化错误: {str(e)}")
                # 创建最小的输出面板结构，确保应用程序能够启动
                self._setup_minimal_output_panel()
            
    def _setup_preview_events(self):
        """为预览画布设置交互事件处理"""
        # 鼠标滚轮缩放
        self.preview_canvas.bind("<MouseWheel>", self._on_preview_mousewheel)
        
        # 鼠标拖动平移
        self.preview_canvas.bind("<ButtonPress-1>", self._on_preview_button_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_preview_button_motion)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_preview_button_release)
        
        # 双击恢复适应窗口
        self.preview_canvas.bind("<Double-Button-1>", lambda e: self._fit_preview_to_window())
        
        # 右键菜单
        self.preview_canvas.bind("<ButtonPress-3>", self._on_preview_right_click)

    def _setup_minimal_output_panel(self):
        """设置最小化输出面板（用于处理初始化错误）"""
        # 创建简单的日志框架
        log_frame = ttk.Frame(self.output_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建简单的日志文本框
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            wrap=tk.WORD, 
            height=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 设置基本日志标签
        self.log_text.tag_configure("INFO", foreground="#000000")
        self.log_text.tag_configure("WARNING", foreground="#FF8800")
        self.log_text.tag_configure("ERROR", foreground="#FF0000")
        
        # 添加错误信息
        self.log_text.insert(tk.END, "输出面板初始化失败，已启用最小化模式。\n", "WARNING")
        self.log_text.insert(tk.END, "请尝试重启应用程序。如果问题持续存在，请检查日志文件。\n", "INFO")

    def _on_preview_mousewheel(self, event):
        """处理预览画布的鼠标滚轮事件"""
        if not hasattr(self, 'preview_canvas') or not self.preview_canvas:
            return
        
        # 计算缩放因子
        zoom_factor = 1.1 if event.delta > 0 else 0.9
        
        # 获取鼠标位置
        mouse_x = self.preview_canvas.canvasx(event.x)
        mouse_y = self.preview_canvas.canvasy(event.y)
        
        # 应用缩放
        self._zoom_preview_at(zoom_factor, mouse_x, mouse_y)

    def _zoom_preview(self, zoom_factor):
        """按指定因子缩放预览"""
        if not hasattr(self, 'preview_canvas') or not self.preview_canvas:
            return
        
        # 获取画布中心点
        center_x = self.preview_canvas.winfo_width() / 2
        center_y = self.preview_canvas.winfo_height() / 2
        
        # 应用缩放
        self._zoom_preview_at(zoom_factor, center_x, center_y)

    def _zoom_preview_at(self, zoom_factor, x, y):
        """在指定位置按指定因子缩放预览"""
        # 限制缩放范围
        new_scale = self.preview_scale * zoom_factor
        if new_scale < 0.1:
            new_scale = 0.1
        elif new_scale > 20.0:
            new_scale = 20.0
        
        # 如果缩放比例没有变化，则不处理
        if abs(new_scale - self.preview_scale) < 0.001:
            return
        
        # 计算鼠标相对于图像的位置
        rel_x = (x - self.preview_offset_x) / self.preview_scale
        rel_y = (y - self.preview_offset_y) / self.preview_scale
        
        # 更新缩放比例
        self.preview_scale = new_scale
        
        # 更新偏移以保持鼠标位置不变
        self.preview_offset_x = x - rel_x * self.preview_scale
        self.preview_offset_y = y - rel_y * self.preview_scale
        
        # 更新预览显示
        self._redraw_preview()
        
        # 更新状态信息
        self.status_var.set(f"预览缩放: {self.preview_scale:.2f}x")

    def _on_preview_button_press(self, event):
        """处理预览画布的鼠标按下事件"""
        if not hasattr(self, 'preview_canvas') or not self.preview_canvas:
            return
        
        # 记录拖动起始位置
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.is_dragging = True
        
        # 更改鼠标光标
        self.preview_canvas.config(cursor="fleur")

    def _on_preview_button_motion(self, event):
        """处理预览画布的鼠标移动事件"""
        if not hasattr(self, 'preview_canvas') or not self.preview_canvas or not self.is_dragging:
            return
        
        # 计算拖动的距离
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        # 更新偏移
        self.preview_offset_x += dx
        self.preview_offset_y += dy
        
        # 更新拖动起始位置
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        # 更新预览显示
        self._redraw_preview()

    def _on_preview_button_release(self, event):
        """处理预览画布的鼠标释放事件"""
        if not hasattr(self, 'preview_canvas') or not self.preview_canvas:
            return
        
        # 重置拖动状态
        self.is_dragging = False
        
        # 恢复鼠标光标
        self.preview_canvas.config(cursor="hand2")

    def _on_preview_right_click(self, event):
        """处理预览画布的右键点击事件，弹出上下文菜单"""
        if not hasattr(self, 'preview_canvas') or not self.preview_canvas:
            return
        
        # 创建右键菜单
        menu = tk.Menu(self.preview_canvas, tearoff=0)
        
        # 添加菜单项
        menu.add_command(label="放大", command=lambda: self._zoom_preview(1.25))
        menu.add_command(label="缩小", command=lambda: self._zoom_preview(0.8))
        menu.add_separator()
        menu.add_command(label="重置缩放", command=self._reset_preview_zoom)
        menu.add_command(label="适应窗口", command=self._fit_preview_to_window)
        menu.add_separator()
        
        # 添加图层切换子菜单
        layer_menu = tk.Menu(menu, tearoff=0)
        layers = ["height", "biome", "temperature", "humidity", "vegetation", "combined"]
        for layer in layers:
            layer_menu.add_radiobutton(
                label=layer,
                variable=self.preview_layer_var,
                value=layer,
                command=self._change_preview_layer
            )
        menu.add_cascade(label="切换图层", menu=layer_menu)
        
        menu.add_separator()
        menu.add_command(label="保存预览图像", command=self._save_preview_image)
        
        # 显示菜单
        menu.tk_popup(event.x_root, event.y_root)

    def _change_preview_layer(self):
        """切换预览显示的图层"""
        if not hasattr(self, 'preview_layer_var'):
            return
        
        layer = self.preview_layer_var.get()
        self.status_var.set(f"预览图层: {layer}")
        
        # 重绘预览
        self._redraw_preview()

    def _save_preview_image(self):
        """保存当前预览图像"""
        if not hasattr(self, 'map_data') or not self.map_data or not self.map_data.is_valid():
            self.status_var.set("无地图数据可保存")
            return
        
        # 打开保存对话框
        filepath = filedialog.asksaveasfilename(
            title="保存预览图像",
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg"), ("所有文件", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # 导入PIL
            from PIL import Image
            
            # 获取当前图层
            layer = self.preview_layer_var.get()
            
            # 根据图层获取对应数据并保存
            if layer == "combined":
                # 生成彩色组合图像
                image = self.map_data.to_image()
            else:
                # 获取单一图层数据
                layer_data = self.map_data.get_layer(layer)
                if layer_data is None:
                    self.status_var.set(f"图层 {layer} 不可用")
                    return
                
                # 归一化数据并创建图像
                if layer == "biome":
                    # 彩色生物群系图
                    image = self.map_data.get_biome_image()
                else:
                    # 灰度图
                    normalized = (layer_data - layer_data.min()) / (layer_data.max() - layer_data.min() + 1e-8) * 255
                    image = Image.fromarray(normalized.astype('uint8'), 'L')
            
            # 保存图像
            image.save(filepath)
            
            # 更新状态
            self.status_var.set(f"预览图像已保存为 {filepath}")
            
        except Exception as e:
            error_msg = f"保存预览图像失败: {str(e)}"
            self.status_var.set(error_msg)
            if hasattr(self, 'logger'):
                self.logger.log(error_msg, "ERROR")

    def _show_welcome_preview(self):
        """安全地显示欢迎预览画面"""
        if not hasattr(self, 'preview_canvas') or self.preview_canvas is None:
            return
        
        try:
            # 获取画布尺寸
            self.preview_canvas.update_idletasks()  # 确保画布已渲染
            width = self.preview_canvas.winfo_width()
            height = self.preview_canvas.winfo_height()
            
            # 如果画布还未完全初始化，延迟显示
            if width <= 1 or height <= 1:
                self.master.after(200, self._show_welcome_preview)
                return
            
            # 清除画布
            self.preview_canvas.delete("all")
            
            # 创建欢迎图形背景
            # 淡蓝色背景渐变
            for i in range(height):
                # 从顶部淡蓝色渐变到底部白色
                r = int(230 + (255-230) * i/height)
                g = int(240 + (255-240) * i/height)
                b = int(250 + (255-250) * i/height)
                color = f"#{r:02x}{g:02x}{b:02x}"
                self.preview_canvas.create_line(0, i, width, i, fill=color)
            
            # 创建欢迎提示文本
            title_text = "EmoScape Studio 每一处风景都谱写着自己的交响乐"
            self.preview_canvas.create_text(
                width / 2,
                height / 2 - 40,
                text=title_text,
                font=("Arial", 24, "bold"),
                fill="#3370FF"
            )
            
            instruction_text = "调整左侧参数并点击「生成地图」按钮开始创建地图"
            self.preview_canvas.create_text(
                width / 2,
                height / 2 + 20,
                text=instruction_text,
                font=("Arial", 12),
                fill="#333333"
            )
            
            # 添加装饰元素
            # 左上角小山形状
            mountain_points = [20, height-60, 80, height-150, 140, height-80, 200, height-120, 250, height-60]
            self.preview_canvas.create_polygon(mountain_points, fill="#E0E8F0", outline="#C0D0E0", width=2)
            
            # 右下角装饰线
            for i in range(5):
                x1 = width - 20 - i * 15
                y1 = height - 20 - i * 5
                x2 = width - 80 - i * 25
                y2 = height - 30 - i * 10
                self.preview_canvas.create_line(x1, y1, x2, y2, fill="#A0C0E0", width=2)
            
            # 添加版本信息
            version = self.config_manager.get("version", "1.0") if hasattr(self, 'config_manager') else "1.0"
            version_text = f"版本 {version}"
            self.preview_canvas.create_text(
                width - 10,
                height - 10,
                text=version_text,
                font=("Arial", 8),
                fill="#666666",
                anchor=tk.SE
            )
        except Exception as e:
            print(f"显示欢迎预览失败: {str(e)}")

    def _reset_preview_zoom(self):
        """重置预览画布的缩放到默认状态"""
        if not hasattr(self, 'preview_canvas') or not self.preview_canvas:
            return
            
        # 重置变换参数
        self.preview_scale = 1.0
        self.preview_offset_x = 0
        self.preview_offset_y = 0
        
        # 更新预览显示
        self._redraw_preview()
        
        # 更新状态信息
        self.status_var.set("预览缩放已重置")

    def _fit_preview_to_window(self):
        """将预览内容缩放以适应窗口大小"""
        if not hasattr(self, 'preview_canvas') or not self.preview_canvas or not self.map_data or not self.map_data.is_valid():
            return
            
        # 获取预览画布尺寸
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # 画布尺寸无效，可能还未完成布局
            self.master.after(100, self._fit_preview_to_window)
            return
        
        # 获取地图尺寸
        map_width = self.map_data.width
        map_height = self.map_data.height
        
        if map_width <= 0 or map_height <= 0:
            return
        
        # 计算适合窗口的缩放比例
        scale_x = (canvas_width - 20) / map_width  # 留出边距
        scale_y = (canvas_height - 20) / map_height
        self.preview_scale = min(scale_x, scale_y)
        
        # 重置偏移以居中
        self.preview_offset_x = (canvas_width - map_width * self.preview_scale) / 2
        self.preview_offset_y = (canvas_height - map_height * self.preview_scale) / 2
        
        # 更新预览显示
        self._redraw_preview()
        
        # 更新状态信息
        self.status_var.set("预览已适应窗口")

    def _redraw_preview(self):
        """重绘预览内容"""
        if not hasattr(self, 'preview_canvas') or not self.preview_canvas or not self.map_data or not self.map_data.is_valid():
            return
            
        # 清除当前画布内容
        self.preview_canvas.delete("all")
        
        # 尝试获取地图图像
        try:
            # 导入PIL
            from PIL import Image, ImageTk
            
            # 获取地图数据
            if hasattr(self.map_data, 'to_image'):
                # 使用内置方法生成图像
                map_image = self.map_data.to_image()
            else:
                # 使用高度图创建简单的灰度图像
                height_map = self.map_data.get_layer("height")
                if height_map is None:
                    self.preview_canvas.create_text(
                        self.preview_canvas.winfo_width() / 2,
                        self.preview_canvas.winfo_height() / 2,
                        text="无可用地图数据",
                        font=("Arial", 14, "bold"),
                        fill="#999999"
                    )
                    return
                    
                # 创建灰度图像
                normalized = (height_map - height_map.min()) / (height_map.max() - height_map.min() + 1e-8) * 255
                map_image = Image.fromarray(normalized.astype('uint8'), 'L')
            
            # 应用缩放
            new_width = int(map_image.width * self.preview_scale)
            new_height = int(map_image.height * self.preview_scale)
            
            if new_width > 0 and new_height > 0:
                resized_image = map_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 转换为Tkinter图像
                self.tk_preview_image = ImageTk.PhotoImage(resized_image)
                
                # 在画布上显示图像
                self.preview_canvas.create_image(
                    self.preview_offset_x, 
                    self.preview_offset_y,
                    anchor=tk.NW, 
                    image=self.tk_preview_image, 
                    tags="preview_image"
                )
            else:
                self.preview_canvas.create_text(
                    self.preview_canvas.winfo_width() / 2,
                    self.preview_canvas.winfo_height() / 2,
                    text="缩放比例过小，无法显示",
                    font=("Arial", 12),
                    fill="#999999"
                )
        except Exception as e:
            # 显示错误信息
            error_msg = f"预览渲染错误: {str(e)}"
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() / 2,
                self.preview_canvas.winfo_height() / 2,
                text=error_msg,
                font=("Arial", 12),
                fill="#FF0000"
            )
            if hasattr(self, 'logger'):
                self.logger.log(error_msg, "ERROR")

    def _preview_underground_layers_two(self):
        """预览地下系统的层次结构"""
        # 检查地图数据是否存在
        if not hasattr(self.map_data, 'underground_layers') or not self.map_data.underground_layers:
            messagebox.showinfo("提示", "没有可用的地下层数据。请先生成包含地下系统的地图。")
            return
        
        # 创建预览窗口
        preview_window = tk.Toplevel(self.master)
        preview_window.title("地下系统预览")
        preview_window.geometry("1200x700")
        preview_window.transient(self.master)
        
        # 创建分割窗格
        paned = ttk.PanedWindow(preview_window, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧控制面板
        control_frame = ttk.Frame(paned)
        paned.add(control_frame, weight=1)
        
        # 创建层级选择框
        layer_frame = ttk.LabelFrame(control_frame, text="层级选择")
        layer_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # 获取地下层数量
        depth = len(self.map_data.underground_layers)
        
        # 创建层级下拉框
        layer_var = tk.StringVar()
        layer_values = [f"地下层 {i+1}" for i in range(depth)]
        if layer_values:
            layer_var.set(layer_values[0])
        
        layer_combo = ttk.Combobox(layer_frame, textvariable=layer_var, values=layer_values, state="readonly")
        layer_combo.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建数据类型选择框
        type_frame = ttk.LabelFrame(control_frame, text="数据类型")
        type_frame.pack(fill=tk.X, padx=5, pady=10)
        
        type_var = tk.StringVar(value="高度图")
        type_values = ["高度图", "内容类型", "矿物分布"]
        
        for val in type_values:
            ttk.Radiobutton(type_frame, text=val, value=val, variable=type_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # 创建统计数据显示框
        stats_frame = ttk.LabelFrame(control_frame, text="统计信息")
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        stats_text.config(state=tk.DISABLED)
        
        # 创建右侧显示面板
        display_frame = ttk.Frame(paned)
        paned.add(display_frame, weight=3)
        
        # 创建matplotlib画布
        fig = Figure(figsize=(10, 8), dpi=100)
        canvas = FigureCanvasTkAgg(fig, display_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, display_frame)
        toolbar.update()
        
        # 函数：更新预览画布
        def update_preview(*args):
            try:
                layer_index = layer_values.index(layer_var.get())
                layer_name = f"underground_{layer_index}"
                
                if layer_name not in self.map_data.underground_layers:
                    return
                
                layer_data = self.map_data.underground_layers[layer_name]
                data_type = type_var.get()
                
                fig.clear()
                ax = fig.add_subplot(111)
                
                title = f"地下层 {layer_index + 1}"
                
                if data_type == "高度图" and "height" in layer_data:
                    im = ax.imshow(layer_data["height"], cmap='terrain')
                    ax.set_title(f"{title} - 高度图")
                    fig.colorbar(im, ax=ax)
                elif data_type == "内容类型" and "content" in layer_data:
                    im = ax.imshow(layer_data["content"], cmap='tab20')
                    ax.set_title(f"{title} - 内容类型")
                    fig.colorbar(im, ax=ax)
                    
                    # 创建图例
                    from core.generation.generate_undergroud import UndergroundContentType
                    content_types = list(UndergroundContentType)
                    legend_elements = []
                    for i, content_type in enumerate(content_types):
                        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=plt.cm.tab20(i / len(content_types)), 
                                            markersize=10, label=content_type.name))
                    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                elif data_type == "矿物分布" and "minerals" in layer_data:
                    mineral_data = layer_data["minerals"]
                    im = ax.imshow(mineral_data, cmap='rainbow')
                    ax.set_title(f"{title} - 矿物分布")
                    fig.colorbar(im, ax=ax)
                    
                    # 创建图例
                    from core.generation.generate_undergroud import MineralType
                    mineral_types = list(MineralType)
                    legend_elements = []
                    for i, mineral_type in enumerate(mineral_types):
                        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=plt.cm.rainbow(i / len(mineral_types)), 
                                            markersize=10, label=mineral_type.name))
                    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                fig.tight_layout()
                canvas.draw()
                
                # 更新统计信息
                update_stats(layer_index)
            except Exception as e:
                messagebox.showerror("预览错误", f"显示预览时出错: {str(e)}")
        
        # 函数：更新统计数据
        def update_stats(layer_index):
            from core.generation.generate_undergroud import get_underground_statistics
            
            stats_data = get_underground_statistics(self.map_data)
            layer_name = f"underground_{layer_index}"
            
            stats_text.config(state=tk.NORMAL)
            stats_text.delete(1.0, tk.END)
            
            stats_text.insert(tk.END, f"总地下层数: {stats_data['layers']}\n\n")
            stats_text.insert(tk.END, f"地下结构总数: {stats_data['structures']}\n\n")
            
            # 当前层的统计信息
            stats_text.insert(tk.END, f"--- 当前层统计 ---\n")
            
            if layer_name in stats_data['content_types']:
                stats_text.insert(tk.END, f"\n内容类型分布:\n")
                for content_type, count in stats_data['content_types'][layer_name].items():
                    stats_text.insert(tk.END, f"  {content_type}: {count}\n")
            
            if layer_name in stats_data['minerals']:
                stats_text.insert(tk.END, f"\n矿物分布:\n")
                for mineral_type, count in stats_data['minerals'][layer_name].items():
                    stats_text.insert(tk.END, f"  {mineral_type}: {count}\n")
            
            # 水系统信息
            stats_text.insert(tk.END, f"\n--- 水系统特征 ---\n")
            stats_text.insert(tk.END, f"地下河流: {stats_data['water_features']['rivers']}\n")
            stats_text.insert(tk.END, f"地下湖泊: {stats_data['water_features']['lakes']}\n")
            stats_text.insert(tk.END, f"渗水区域: {stats_data['water_features']['seepage_areas']}\n")
            
            stats_text.config(state=tk.DISABLED)
        
        # 绑定事件
        layer_var.trace("w", update_preview)
        type_var.trace("w", update_preview)
        
        # 初始更新
        if layer_values:
            update_preview()

    def _export_underground_data(self):
        """导出地下系统数据"""
        if not hasattr(self.map_data, 'underground_layers') or not self.map_data.underground_layers:
            messagebox.showinfo("提示", "没有可用的地下层数据。请先生成包含地下系统的地图。")
            return
        
        # 选择导出目录
        export_dir = filedialog.askdirectory(
            title="选择导出目录", 
            initialdir=self.config_manager.get("default_export_path", "./exports")
        )
        
        if not export_dir:
            return
        
        # 创建进度对话框
        progress_window = tk.Toplevel(self.master)
        progress_window.title("导出地下数据")
        progress_window.geometry("400x150")
        progress_window.transient(self.master)
        progress_window.grab_set()
        
        # 添加进度信息
        ttk.Label(progress_window, text="正在导出地下系统数据...", padding=10).pack()
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=300)
        progress_bar.pack(pady=10, padx=20)
        
        status_var = tk.StringVar(value="准备中...")
        status_label = ttk.Label(progress_window, textvariable=status_var)
        status_label.pack(pady=5)
        
        # 开始导出任务
        def export_task():
            try:
                import os
                import json
                import numpy as np
                import time
                
                # 创建目录
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                export_subdir = os.path.join(export_dir, f"underground_export_{timestamp}")
                os.makedirs(export_subdir, exist_ok=True)
                
                # 获取地下层数量
                depth = len(self.map_data.underground_layers)
                
                # 用于可视化的导入
                from core.generation.generate_undergroud import visualize_underground_layer
                from core.generation.generate_undergroud import get_underground_statistics
                
                # 导出统计数据
                status_var.set("导出统计数据...")
                progress_var.set(10)
                
                stats_data = get_underground_statistics(self.map_data)
                with open(os.path.join(export_subdir, "underground_stats.json"), 'w', encoding='utf-8') as f:
                    json.dump(stats_data, f, indent=2, ensure_ascii=False)
                
                # 导出每一层的可视化图像
                for i in range(depth):
                    layer_name = f"underground_{i}"
                    if layer_name in self.map_data.underground_layers:
                        layer_data = self.map_data.underground_layers[layer_name]
                        
                        # 更新状态
                        status_var.set(f"处理地下层 {i+1}...")
                        progress_var.set(10 + (i / depth) * 80)
                        
                        # 导出可视化图像
                        save_path = os.path.join(export_subdir, f"layer_{i}_visualization.png")
                        visualize_underground_layer(layer_data, i, f"地下层 {i+1}", save_path)
                        
                        # 导出原始数据
                        np.save(os.path.join(export_subdir, f"layer_{i}_height.npy"), layer_data["height"])
                        np.save(os.path.join(export_subdir, f"layer_{i}_content.npy"), layer_data["content"])
                        
                        # 导出矿物数据
                        if "minerals" in layer_data:
                            np.save(os.path.join(export_subdir, f"layer_{i}_minerals.npy"), layer_data["minerals"])
                
                # 导出地下水系统数据
                if hasattr(self.map_data, 'underground_water') and self.map_data.underground_water:
                    status_var.set("导出地下水系统...")
                    progress_var.set(90)
                    
                    with open(os.path.join(export_subdir, "underground_water.json"), 'w', encoding='utf-8') as f:
                        # 需要转换numpy数组为列表以便序列化
                        water_data = {}
                        for key, value in self.map_data.underground_water.items():
                            if isinstance(value, list):
                                water_data[key] = []
                                for item in value:
                                    if isinstance(item, dict):
                                        water_item = {}
                                        for k, v in item.items():
                                            if isinstance(v, np.ndarray):
                                                water_item[k] = v.tolist()
                                            else:
                                                water_item[k] = v
                                        water_data[key].append(water_item)
                                    else:
                                        water_data[key].append(item)
                            else:
                                water_data[key] = value
                        
                        json.dump(water_data, f, indent=2)
                
                # 完成导出
                progress_var.set(100)
                status_var.set("导出完成!")
                
                # 显示成功消息
                messagebox.showinfo("导出成功", f"地下系统数据已成功导出到:\n{export_subdir}")
                
                # 关闭进度窗口
                progress_window.destroy()
                
            except Exception as e:
                status_var.set(f"导出错误: {str(e)}")
                messagebox.showerror("导出错误", f"导出过程中发生错误:\n{str(e)}")
                progress_window.destroy()
        
        # 在新线程中运行导出任务
        import threading
        threading.Thread(target=export_task, daemon=True).start()

    def _create_spinbox(self, parent, param, label, min_val, max_val, row):
        # 创建一个包含标签和微调框的组件，用于调整地图参数
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)  # 使用pack而不是grid
        ttk.Label(frame, text=label, width=12).pack(side=tk.LEFT)
        var = tk.IntVar(value=getattr(self.map_params, param))
        
        # 创建微调框
        spinbox = ttk.Spinbox(
            frame, 
            from_=min_val, 
            to=max_val, 
            textvariable=var, 
            command=lambda: self._on_param_change(param, var.get()),
            width=8
        )
        spinbox.pack(side=tk.RIGHT, padx=5)
        
        # 添加回车键和失去焦点事件绑定
        spinbox.bind("<Return>", lambda e, p=param, v=var: self._on_param_change(p, v.get()))
        spinbox.bind("<FocusOut>", lambda e, p=param, v=var: self._on_param_change(p, v.get()))
        
        # 存储控件引用
        self.param_controls[param] = {"var": var, "spinbox": spinbox}

    def _create_slider(self, parent, param, label, min_val, max_val, row):
        # 创建一个标签、滑块、输入框的组件，用于调整玩家偏好
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew")
        ttk.Label(frame, text=label, width=12).pack(side=tk.LEFT)
        var = tk.DoubleVar(value=getattr(self.map_params, param))
        slider = ttk.Scale(
            frame, 
            from_=min_val, 
            to=max_val, 
            variable=var, 
            command=lambda v, p=param: self._on_param_change(p, float(v))
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        entry = ttk.Entry(
            frame, 
            width=6, 
            textvariable=var, 
            validate="key", 
            validatecommand=(self.master.register(self._validate_float), "%P")
        )
        entry.pack(side=tk.RIGHT)
        
        # 添加回车键和失去焦点事件绑定
        entry.bind("<Return>", lambda e, p=param, v=var: self._on_param_change(p, float(v.get())))
        entry.bind("<FocusOut>", lambda e, p=param, v=var: self._on_param_change(p, float(v.get())))
        
        self.param_controls[param] = {"var": var, "slider": slider, "entry": entry}

    def _on_param_change(self, param, value):
        # 处理参数的变化
        old_value = getattr(self.map_params, param)
        setattr(self.map_params, param, value)
        self.command_history.add_command({
            "type": "param_change",
            "param": param,
            "old_value": old_value,
            "new_value": value
        })
        
        # 格式化显示的值，浮点数保留2位小数
        display_value = value
        if isinstance(value, float):
            display_value = f"{value:.2f}"
        
        self.status_var.set(f"参数已更新: {param} = {display_value}")

    def _validate_float(self, value):
        # 判断是否是浮点数和空字符串
        try:
            if value.strip() == "":
                return True
            float(value)
            return True
        except ValueError:
            return False
        
    def _create_checkbox(self, parent, param, label, row):
        """创建复选框控件"""
        # 使用map_params获取初始值，如果不存在则使用params字典或默认为False
        default_value = False
        if hasattr(self.map_params, param):
            default_value = getattr(self.map_params, param)
        elif param in self.params:
            default_value = self.params[param]
        
        var = tk.BooleanVar(value=default_value)
        self.param_vars[param] = var
        
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="we", padx=5, pady=2)
        
        ttk.Label(frame, text=label).pack(side="left", padx=5)
        cb = ttk.Checkbutton(frame, variable=var, 
                            command=lambda: self._on_param_change(param, var.get()))
        cb.pack(side="right", padx=5)

    def _create_combobox(self, parent, param, label, values, row):
        """创建下拉框控件"""
        # 使用map_params获取初始值，如果不存在则使用params字典或默认值
        default_value = values[0]
        if hasattr(self.map_params, param):
            default_value = getattr(self.map_params, param)
        elif param in self.params:
            default_value = self.params[param]
        
        var = tk.StringVar(value=default_value)
        self.param_vars[param] = var
        
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="we", padx=5, pady=2)
        
        ttk.Label(frame, text=label).pack(side="left", padx=5)
        combo = ttk.Combobox(frame, textvariable=var, values=values, width=10)
        combo.pack(side="right", padx=5)
        combo.bind("<<ComboboxSelected>>", 
                lambda e: self._on_param_change(param, var.get()))

    def _generate_map(self):
        """开始地图生成流程"""
        self.logger.log("准备生成新地图...")
        
        # 重置UI状态
        self.state = ViewState.GENERATING
        self.status_var.set("正在生成地图...")
        self.progress_var.set(0)
        
        # 创建新的map_data对象，而不是在generate_map中创建
        width = self.map_params.map_width
        height = self.map_params.map_height
        use_gpu = self.gpu_var.get() if hasattr(self, 'gpu_var') else False
        
        # 初始化地图数据
        self.map_data = MapData(width, height, use_gpu)
        self.map_data.generation_state = {
            'resume_point': 'start',
            'completed_gens': 0
        }
        
        # 提交任务
        future = self.task_manager.submit_task(
            "map_generation",
            self._run_generate_task,
            use_gui_editors=True
        )
        
        future.add_done_callback(self._on_generate_complete)

    def _check_generation_status(self, future):
        """检查地图生成状态，决定是否完成或等待编辑器"""
        try:
            self.map_data = future.result()
            
            # 检查是否有待处理的编辑器请求
            if hasattr(self.map_data, 'pending_editor') and self.map_data.pending_editor:
                # 有待处理的编辑器，应该显示编辑器而不是完成回调
                self._handle_pending_editor(self.map_data.pending_editor, self.map_data.editor_state)
            else:
                # 没有待处理编辑器，可以直接调用完成回调
                self._on_generate_complete(future)
        except Exception as e:
            self.state = ViewState.IDLE
            self.status_var.set("地图生成失败")
            self.logger.error(f"地图生成错误: {str(e)}")
        
    def _run_generate_task(self, use_gui_editors=True):
        """执行地图生成任务"""
        from core.generate_map import generate_map
        try:
            # 获取是否启用可视化设置
            enable_visualize = hasattr(self, 'visualize_var') and self.visualize_var.get() if hasattr(self, 'visualize_var') else True
            
            # 为可视化创建专用框架，避免与编辑器共享框架
            visualization_frame = None
            if enable_visualize:
                # 首先检查是否有预览画布
                if hasattr(self, 'preview_canvas') and self.preview_canvas:
                    self.logger.log("使用预览画布")
                    self.preview_canvas.delete("all")  # 清除现有内容
                    visualization_frame = self.preview_frame              
                # 然后检查是否有预览框架
                elif hasattr(self, 'preview_frame') and self.preview_frame:
                    self.logger.log("使用预览框架")
                    visualization_frame = self.preview_frame
                else:
                    # 尝试创建一个临时画布
                    try:
                        if hasattr(self, 'output_frame'):
                            print("创建临时画布")
                            preview_frame = ttk.LabelFrame(self.output_frame, text="地图预览")
                            preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                            self.preview_frame = preview_frame
                            self.preview_canvas = tk.Canvas(preview_frame, bg="white", cursor="hand2")
                            self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                            visualization_frame = preview_frame
                            self.logger.log("已动态创建预览画布", "INFO")
                        else:
                            self.logger.log("警告：找不到输出框架，可视化将不会嵌入到GUI中", "WARNING")
                    except Exception as e:
                        self.logger.log(f"创建预览画布失败: {str(e)}", "WARNING")
                        self.logger.log("警告：找不到预览画布，可视化将不会嵌入到GUI中", "WARNING")
                    # 防止在GUI线程外更新Tkinter
                try:
                    self.master.update_idletasks()  # 确保UI已更新
                except Exception:
                    pass
            self.map_data = generate_map(
                preferences=self.map_params.to_dict(),
                width=self.map_params.map_width,
                height=self.map_params.map_height,
                export_model=False,
                logger=self.logger,
                use_real_geo_data=self.use_real_geo_var.get() if hasattr(self, 'use_real_geo_var') else False,
                geo_data_path=self.geo_file_var.get() if hasattr(self, 'geo_file_var') and self.geo_source_var.get() == "file" else None,
                geo_bounds=self._get_geo_bounds() if hasattr(self, 'geo_source_var') and self.geo_source_var.get() == "srtm" else None,
                visualize=enable_visualize,  # 控制是否启用可视化
                parent_frame=visualization_frame,  # 使用预览框架作为可视化容器
                parent_frame_edit=self.editor_container, # 地形编辑器
                use_gui_editors=use_gui_editors,
                map_data=self.map_data  # 传入已经初始化的map_data
            )
            
            return self.map_data
        except Exception as e:
            self.logger.log(f"地图生成失败: {str(e)}", "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")
            raise
        
    def _handle_editor_complete(self, editor_type, result):
        """处理编辑器完成事件"""
        self.logger.log(f"编辑器 {editor_type} 已完成")
        
        if editor_type == "height_editor":
            # 处理高度编辑器结果
            if result and isinstance(result, dict):
                # 更新地图数据
                if 'height_map' in result:
                    self.map_data.layers["height"] = result['height_map']
                if 'temp_map' in result:
                    self.map_data.layers["temperature"] = result['temp_map']
                if 'humid_map' in result:
                    self.map_data.layers["humidity"] = result['humid_map']
        
        elif editor_type == "evolution_scorer":
            # 处理进化评分结果
            if result and isinstance(result, list):
                # 存储评分结果到map_data
                self.map_data.evolution_scores = result
        
        # 清除pending_editor标记，表示编辑已完成
        if hasattr(self.map_data, 'pending_editor'):
            self.map_data.pending_editor = None
        
        # 继续地图生成流程
        self.logger.log("准备继续地图生成流程...")
        self._continue_generation()

    def _handle_pending_editor(self, editor_type, state):
        """处理挂起的编辑器请求"""
        if editor_type == "height_editor":
            self._show_height_editor(state)
        elif editor_type == "evolution_scorer":
            self._show_evolution_scorer(state)
        else:
            self.logger.log(f"未知的编辑器类型: {editor_type}", "WARNING")

    def _get_geo_bounds(self):
        """获取地理边界坐标"""
        if hasattr(self, 'lng_min_var') and hasattr(self, 'lng_max_var') and \
        hasattr(self, 'lat_min_var') and hasattr(self, 'lat_max_var'):
            try:
                return (
                    float(self.lng_min_var.get()),
                    float(self.lat_min_var.get()),
                    float(self.lng_max_var.get()),
                    float(self.lat_max_var.get())
                )
            except (ValueError, TypeError):
                self.logger.log("地理坐标格式无效，使用默认值", "WARNING")
        
        # 默认值 (北京附近)
        return (116.3, 39.9, 116.5, 40.0)

    def _on_generate_complete(self, future):
        """处理地图生成完成事件"""
        try:
            self.map_data = future.result()
            
            # 检查是否需要显示编辑器
            if hasattr(self.map_data, 'pending_editor') and self.map_data.pending_editor:
                editor_type = self.map_data.pending_editor
                editor_state = self.map_data.editor_state if hasattr(self.map_data, 'editor_state') else None
                
                self.logger.log(f"检测到需要编辑器: {editor_type}")
                
                # 根据编辑器类型显示相应的界面
                if editor_type == "height_editor":
                    self._show_height_editor(editor_state)
                    self.param_notebook.select(self.editors_tab)  # 切换到编辑器标签页
                elif editor_type == "evolution_scorer":
                    self._show_evolution_scorer(editor_state)
                    self.param_notebook.select(self.editors_tab)  # 切换到编辑器标签页
                
                # 更新状态
                self.state = ViewState.EVOLVING
                self.status_var.set("等待用户输入...")
                return
            
            # 如果没有编辑器请求或生成已完成
            if hasattr(self.map_data, 'generation_complete') and self.map_data.generation_complete:
                self.logger.log("地图生成完成!")
                self.state = ViewState.MAP_READY
                self.status_var.set("地图生成完成")
                self.progress_var.set(100)
                
                # 显示地图预览
                self._preview_map_3d()
            else:
                # 如果生成未完成但没有编辑器请求，可能是因为出现了错误
                self.logger.log("地图生成未完成，但没有编辑器请求")
                self.state = ViewState.IDLE
                self.status_var.set("地图生成未完成")
        
        except Exception as e:
            self.logger.log(f"生成完成回调出错: {str(e)}", "ERROR")
            self.state = ViewState.IDLE
            self.status_var.set("生成失败")
            messagebox.showerror("错误", f"地图生成失败: {str(e)}")

    def _update_map_based_on_feedback(self):
        if not self.map_data.is_valid():
            self.logger.error("尚未生成初始地图")
            return
        self.state = ViewState.GENERATING
        future = self.task_manager.submit_task("update_map", self._run_update_task)
        future.add_done_callback(self._on_update_complete)

    def _run_update_task(self):
        """运行地图更新任务"""
        try:
            # 直接传递整个MapData对象
            updated_map_data = update_map(self.map_data, self.log_text)
            return updated_map_data
        except Exception as e:
            self.logger.error(f"地图更新异常: {str(e)}")
            return self.map_data  # 发生错误时返回原始地图数据

    def _on_update_complete(self, future):
        try:
            self.map_data = future.result()
            self.state = ViewState.MAP_READY
            self.status_var.set("地图更新完成")
            self.progress_var.set(100)
            self.logger.log("地图更新成功")
            preview_map(self.map_data,self.master)
        except Exception as e:
            self.state = ViewState.IDLE
            self.logger.error(f"地图更新错误: {str(e)}")

    def _run_evolution(self):
        if not self.map_data.is_valid():
            self.logger.error("请先生成地图并初始化生物种群")
            return
        self.state = ViewState.EVOLVING
        future = self.task_manager.submit_task("evolution", self._run_evolution_task)
        future.add_done_callback(self._on_evolution_complete)

    def _run_evolution_task(self):
        """运行生物群系进化任务"""
        try:
            self.logger.log("初始化生物群系进化引擎...")
            
            # 确保地图数据有效
            if not self.map_data.is_valid():
                raise ValueError("地图数据无效，无法启动进化")
            
            # 获取必要的图层
            biome_map = self.map_data.get_layer("biome")
            height_map = self.map_data.get_layer("height")
            temp_map = self.map_data.get_layer("temperature")
            humid_map = self.map_data.get_layer("humidity")
            
            # 初始化进化引擎
            from core.evolution.evolve_generation import BiomeEvolutionEngine
            self.evolution_engine = BiomeEvolutionEngine(
                biome_map=biome_map,
                height_map=height_map,
                temperature_map=temp_map,
                moisture_map=humid_map,
                memory_path="./data/models/evolution_memory.dat"
            )
            
            # 设置待处理的编辑器
            self.map_data.pending_editor = "evolution_scorer"
            
            return self.map_data
        except Exception as e:
            self.logger.error(f"生物群系进化初始化失败: {str(e)}")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")
            raise

    def _setup_export_dialog(self):
        """带样式选项的导出对话框"""
        export_dialog = tk.Toplevel(self.master)
        export_dialog.title("导出选项")
        export_dialog.grab_set()
        export_dialog.geometry("500x650")
        
        frame = ttk.Frame(export_dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # 创建导出配置
        config = MapExportConfig()

        # 基本设置框架
        basic_frame = ttk.LabelFrame(frame, text="基本设置")
        basic_frame.pack(fill=tk.X, pady=5)
        
        # 输出目录
        ttk.Label(basic_frame, text="输出目录:").grid(row=0, column=0, sticky=tk.W, pady=2)
        output_dir_var = tk.StringVar(value=config.output_dir)
        output_dir_entry = ttk.Entry(basic_frame, textvariable=output_dir_var, width=30)
        output_dir_entry.grid(row=0, column=1, sticky=tk.W+tk.E, pady=2)
        ttk.Button(basic_frame, text="浏览...", command=lambda: self._browse_dir(output_dir_var)).grid(row=0, column=2, pady=2)
        
        # 文件名前缀
        ttk.Label(basic_frame, text="文件名前缀:").grid(row=1, column=0, sticky=tk.W, pady=2)
        filename_var = tk.StringVar(value=config.base_filename)
        ttk.Entry(basic_frame, textvariable=filename_var, width=30).grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E, pady=2)
        
        # 样式设置框架
        style_frame = ttk.LabelFrame(frame, text="风格设置")
        style_frame.pack(fill=tk.X, pady=5)
        
        # 地图样式选择
        ttk.Label(style_frame, text="地图风格:").grid(row=0, column=0, sticky=tk.W, pady=2)
        style_var = tk.StringVar(value="default")
        style_combo = ttk.Combobox(style_frame, textvariable=style_var, values=["default", "minecraft"], state="readonly")
        style_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # Minecraft特定设置框架
        minecraft_frame = ttk.Frame(style_frame)
        
        def on_style_change(*args):
            if style_var.get() == "minecraft":
                minecraft_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, pady=5)
            else:
                minecraft_frame.grid_forget()
        
        style_var.trace("w", on_style_change)
        
        # Minecraft设置项
        ttk.Label(minecraft_frame, text="最大高度:").grid(row=0, column=0, sticky=tk.W, pady=2)
        max_height_var = tk.IntVar(value=255)
        ttk.Spinbox(minecraft_frame, from_=64, to=384, textvariable=max_height_var, width=8).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(minecraft_frame, text="海平面高度:").grid(row=1, column=0, sticky=tk.W, pady=2)
        sea_level_var = tk.IntVar(value=63)
        ttk.Spinbox(minecraft_frame, from_=0, to=128, textvariable=sea_level_var, width=8).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(minecraft_frame, text="方块大小:").grid(row=2, column=0, sticky=tk.W, pady=2)
        block_size_var = tk.IntVar(value=1)
        ttk.Spinbox(minecraft_frame, from_=1, to=16, textvariable=block_size_var, width=8).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # 预览按钮
        ttk.Button(
            style_frame, 
            text="预览风格", 
            command=lambda: self._preview_style_transformation(style_var.get(), {
                "max_height": max_height_var.get(),
                "sea_level": sea_level_var.get(),
                "block_resolution": block_size_var.get()
            })
        ).grid(row=2, column=0, columnspan=2, pady=10)
        
        # 纹理设置
        texture_frame = ttk.LabelFrame(frame, text="纹理设置")
        texture_frame.pack(fill=tk.X, pady=5)
        
        # 纹理尺寸
        ttk.Label(texture_frame, text="纹理宽度:").grid(row=0, column=0, sticky=tk.W, pady=2)
        texture_width_var = tk.IntVar(value=config.texture_size[0])
        ttk.Spinbox(texture_frame, from_=256, to=8192, textvariable=texture_width_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(texture_frame, text="纹理高度:").grid(row=1, column=0, sticky=tk.W, pady=2)
        texture_height_var = tk.IntVar(value=config.texture_size[1])
        ttk.Spinbox(texture_frame, from_=256, to=8192, textvariable=texture_height_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # 生成纹理开关
        generate_textures_var = tk.BooleanVar(value=config.generate_textures)
        ttk.Checkbutton(texture_frame, text="生成纹理", variable=generate_textures_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # 导出选项
        export_frame = ttk.LabelFrame(frame, text="导出选项")
        export_frame.pack(fill=tk.X, pady=5)
        
        # 包含元数据
        include_metadata_var = tk.BooleanVar(value=config.include_metadata)
        ttk.Checkbutton(export_frame, text="包含元数据", variable=include_metadata_var).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # 压缩输出
        compress_output_var = tk.BooleanVar(value=config.compress_output)
        ttk.Checkbutton(export_frame, text="压缩输出", variable=compress_output_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # 导出法线
        export_normals_var = tk.BooleanVar(value=config.export_normals)
        ttk.Checkbutton(export_frame, text="导出法线", variable=export_normals_var).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # 导出高度图
        export_heightmap_var = tk.BooleanVar(value=config.export_heightmap)
        ttk.Checkbutton(export_frame, text="导出高度图", variable=export_heightmap_var).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # 性能设置
        perf_frame = ttk.LabelFrame(frame, text="性能设置")
        perf_frame.pack(fill=tk.X, pady=5)
        
        # LOD级别
        ttk.Label(perf_frame, text="细节级别 (LOD):").grid(row=0, column=0, sticky=tk.W, pady=2)
        lod_var = tk.IntVar(value=config.level_of_detail)
        ttk.Spinbox(perf_frame, from_=1, to=8, textvariable=lod_var, width=5).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # 内存高效模式
        memory_efficient_var = tk.BooleanVar(value=config.memory_efficient)
        ttk.Checkbutton(perf_frame, text="内存高效模式", variable=memory_efficient_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # 多线程处理
        multithreaded_var = tk.BooleanVar(value=config.multithreaded)
        ttk.Checkbutton(perf_frame, text="启用多线程", variable=multithreaded_var).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # 工作线程数
        ttk.Label(perf_frame, text="最大线程数:").grid(row=2, column=1, sticky=tk.W, pady=2)
        max_workers_var = tk.IntVar(value=config.max_workers)
        ttk.Spinbox(perf_frame, from_=1, to=16, textvariable=max_workers_var, width=5).grid(row=2, column=2, sticky=tk.W, pady=2)
        
        # 引擎专用设置
        engine_frame = ttk.LabelFrame(frame, text="游戏引擎设置")
        engine_frame.pack(fill=tk.X, pady=5)
        
        # Unity版本
        ttk.Label(engine_frame, text="Unity版本:").grid(row=0, column=0, sticky=tk.W, pady=2)
        unity_version_var = tk.StringVar(value=config.unity_export_version)
        ttk.Combobox(engine_frame, textvariable=unity_version_var, values=["2019.4", "2020.3", "2021.3", "2022.3"]).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # Unreal版本
        ttk.Label(engine_frame, text="Unreal版本:").grid(row=1, column=0, sticky=tk.W, pady=2)
        unreal_version_var = tk.StringVar(value=config.unreal_export_version)
        ttk.Combobox(engine_frame, textvariable=unreal_version_var, values=["4.27", "5.0", "5.1", "5.2", "5.3"]).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # 导出碰撞数据
        export_collision_var = tk.BooleanVar(value=config.export_collision)
        ttk.Checkbutton(engine_frame, text="导出碰撞数据", variable=export_collision_var).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # 生成光照贴图UV
        lightmap_uvs_var = tk.BooleanVar(value=config.lightmap_uvs)
        ttk.Checkbutton(engine_frame, text="生成光照贴图UV", variable=lightmap_uvs_var).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # 导出格式设置
        format_frame = ttk.LabelFrame(frame, text="导出格式")
        format_frame.pack(fill=tk.X, pady=5)
        
        # 在format_frame后添加部署选项框架
        deploy_frame = ttk.LabelFrame(frame, text="一键部署选项")
        deploy_frame.pack(fill=tk.X, pady=5)
        
        # 是否自动部署
        auto_deploy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(deploy_frame, text="导出后自动部署", variable=auto_deploy_var).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # 提示信息
        ttk.Label(deploy_frame, text="提示: 系统将自动检测已安装的引擎，或提示您选择项目路径。").grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # OBJ模型自动打开选择器
        ttk.Label(deploy_frame, text="OBJ导出:").grid(row=2, column=0, sticky=tk.W, pady=2)
        open_in_var = tk.StringVar(value="不自动打开")
        ttk.Combobox(deploy_frame, textvariable=open_in_var, values=["不自动打开", "在Blender中打开"]).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # 各种格式的复选框
        obj_format_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(format_frame, text="OBJ格式", variable=obj_format_var).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        unity_format_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(format_frame, text="Unity格式", variable=unity_format_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        unreal_format_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(format_frame, text="Unreal格式", variable=unreal_format_var).grid(row=0, column=2, sticky=tk.W, pady=2)
        
        # 按钮区域
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def on_export():
            # 创建导出配置
            export_config = MapExportConfig(
                output_dir=output_dir_var.get(),
                base_filename=filename_var.get(),
                texture_size=(texture_width_var.get(), texture_height_var.get()),
                generate_textures=generate_textures_var.get(),
                include_metadata=include_metadata_var.get(),
                compress_output=compress_output_var.get(),
                export_normals=export_normals_var.get(),
                export_heightmap=export_heightmap_var.get(),
                level_of_detail=lod_var.get(),
                memory_efficient=memory_efficient_var.get(),
                multithreaded=multithreaded_var.get(),
                max_workers=max_workers_var.get(),
                unity_export_version=unity_version_var.get(),
                unreal_export_version=unreal_version_var.get(),
                export_collision=export_collision_var.get(),
                lightmap_uvs=lightmap_uvs_var.get()
            )
            
            # 添加样式设置
            export_config.style = style_var.get()
            export_config.style_options = {
                "max_height": max_height_var.get(),
                "sea_level": sea_level_var.get(),
                "block_resolution": block_size_var.get()
            }
            
            # 保存导出配置到当前会话
            self.export_config = export_config
            
            # 保存部署选项
            self.auto_deploy = auto_deploy_var.get()
            self.open_obj_in = open_in_var.get()
            
            # 关闭对话框
            export_dialog.destroy()
            
            # 执行导出
            self._do_export_map(obj_format_var.get(), unity_format_var.get(), unreal_format_var.get())
        
        ttk.Button(button_frame, text="导出", command=on_export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="取消", command=export_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # 居中对话框
        export_dialog.update_idletasks()
        width = export_dialog.winfo_width()
        height = export_dialog.winfo_height()
        x = (export_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (export_dialog.winfo_screenheight() // 2) - (height // 2)
        export_dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def _browse_dir(self, var):
        """浏览目录并更新变量"""
        directory = filedialog.askdirectory(initialdir=var.get())
        if directory:
            var.set(directory)

    def _export_map(self):
        """打开导出对话框"""
        if not self.map_data:
            self.logger.error("无地图数据可导出 (map_data为空)")
            return
        
        if not hasattr(self.map_data, 'is_valid'):
            self.logger.error("无地图数据可导出 (map_data对象缺少is_valid方法)")
            return
            
        if not self.map_data.is_valid():
            self.logger.error("无地图数据可导出 (map_data.is_valid()返回False)")
            # 检查基本图层是否存在
            if hasattr(self.map_data, 'layers'):
                self.logger.error(f"可用图层: {list(self.map_data.layers.keys())}")
            return
            
        self._setup_export_dialog()

    def _do_export_map(self, export_obj=True, export_unity=True, export_unreal=True, export_minecraft=True):
        """执行地图导出（支持选择性导出格式）"""
        self.state = ViewState.EXPORTING
        self.status_var.set("正在导出地图...")
        self.progress_var.set(0)
        
        # 提交导出任务
        future = self.task_manager.submit_task(
            "export", 
            self._export_map_task, 
            export_obj, export_unity, export_unreal, export_minecraft
        )
        future.add_done_callback(self._on_export_complete)

    def _export_map_task(self, export_obj, export_unity, export_unreal, export_minecraft):
        """带样式转换的导出任务"""
        try:
            from utils.export import ObjExporter, UnityExporter, UnrealExporter, MinecraftExporter, MapExportConfig
            from utils.style_transformers import get_available_styles
            
            if not self.map_data or not hasattr(self.map_data, 'is_valid') or not self.map_data.is_valid():
                self.logger.error("无效的地图数据，无法导出")
                return {"success": False, "message": "无效的地图数据"}
            
            config = getattr(self, 'export_config', None)
            if config is None:
                config = MapExportConfig()
                self.logger.warning("使用默认导出配置")
            
            os.makedirs(config.output_dir, exist_ok=True)
            self.logger.info(f"将导出地图到目录: {config.output_dir}")
            
            export_map_data = self.map_data
            if hasattr(config, 'style') and config.style != "default":
                self.logger.info(f"应用样式变换: {config.style}")
                try:
                    style_transformers = get_available_styles()
                    if config.style in style_transformers:
                        transformer = style_transformers[config.style](self.map_data, config.style_options)
                        export_map_data = transformer.transform()
                        self.logger.info(f"样式变换成功: {config.style}")
                    else:
                        self.logger.warning(f"未找到样式变换器: {config.style}")
                except Exception as e:
                    self.logger.error(f"样式变换失败: {str(e)}")
            
            results = {}
            deploy_results = {}
            
            # 导出逻辑（并行或串行导出）
            # 串行导出示例
            if not config.multithreaded or (export_obj + export_unity + export_unreal + export_minecraft) <= 1:
                self.logger.info("使用串行导出模式")
                
                if export_obj:
                    self.logger.info("开始导出OBJ格式...")
                    obj_exporter = ObjExporter(config, self.logger)
                    results["obj"] = obj_exporter.export(export_map_data)
                    self.logger.info(f"OBJ导出完成: {results['obj']}")
                    self.progress_var.set(25)
                
                if export_unity:
                    self.logger.info("开始导出Unity格式...")
                    unity_exporter = UnityExporter(config, self.logger)
                    results["unity"] = unity_exporter.export(export_map_data)
                    self.logger.info(f"Unity导出完成: {results['unity']}")
                    self.progress_var.set(50)
                
                if export_unreal:
                    self.logger.info("开始导出Unreal格式...")
                    unreal_exporter = UnrealExporter(config, self.logger)
                    results["unreal"] = unreal_exporter.export(export_map_data)
                    self.logger.info(f"Unreal导出完成: {results['unreal']}")
                    self.progress_var.set(75)
                    
                if export_minecraft:
                    self.logger.info("开始导出Minecraft格式...")
                    minecraft_exporter = MinecraftExporter(config, self.logger)
                    results["minecraft"] = minecraft_exporter.export(export_map_data)
                    self.logger.info(f"Minecraft导出完成: {results['minecraft']}")
                    self.progress_var.set(100)
            else:
                self.logger.info("使用并行导出模式")
                # 并行导出实现
                # 这里可以使用ThreadPoolExecutor或ProcessPoolExecutor来实现
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                    futures = {}
                    
                    if export_obj:
                        self.logger.info("提交OBJ导出任务...")
                        obj_exporter = ObjExporter(config, self.logger)
                        futures['obj'] = executor.submit(obj_exporter.export, export_map_data)
                    
                    if export_unity:
                        self.logger.info("提交Unity导出任务...")
                        unity_exporter = UnityExporter(config, self.logger)
                        futures['unity'] = executor.submit(unity_exporter.export, export_map_data)
                    
                    if export_unreal:
                        self.logger.info("提交Unreal导出任务...")
                        unreal_exporter = UnrealExporter(config, self.logger)
                        futures['unreal'] = executor.submit(unreal_exporter.export, export_map_data)
                        
                    if export_minecraft:
                        self.logger.info("提交Minecraft导出任务...")
                        minecraft_exporter = MinecraftExporter(config, self.logger)
                        futures['minecraft'] = executor.submit(minecraft_exporter.export, export_map_data)
                    
                    # 等待所有任务完成并收集结果
                    completed = 0
                    total = len(futures)
                    
                    for format_type, future in futures.items():
                        try:
                            results[format_type] = future.result()
                            self.logger.info(f"{format_type.upper()}导出完成: {results[format_type]}")
                        except Exception as e:
                            self.logger.error(f"{format_type.upper()}导出失败: {str(e)}")
                            results[format_type] = None
                        
                        completed += 1
                        self.progress_var.set(int(completed / total * 100))

            # 自动部署
            if hasattr(self, 'auto_deploy') and self.auto_deploy:
                self.logger.info("开始自动部署...")
                try:
                    from utils.export import DeploymentManager
                    deploy_manager = DeploymentManager(self.logger)
                    
                    for format_type, path in results.items():
                        if not path:
                            continue
                            
                        if format_type == "obj" and hasattr(self, "open_obj_in") and self.open_obj_in == "在Blender中打开":
                            self.logger.info("准备在Blender中打开OBJ文件...")
                            deploy_results["blender"] = deploy_manager.deploy_to_blender(path)
                        elif format_type == "unity":
                            self.logger.info("准备部署到Unity项目...")
                            deploy_results["unity"] = deploy_manager.deploy_to_unity(path)
                        elif format_type == "unreal":
                            self.logger.info("准备部署到Unreal项目...")
                            deploy_results["unreal"] = deploy_manager.deploy_to_unreal(path)
                        # 可以添加Minecraft导出的处理
                except Exception as e:
                    self.logger.error(f"自动部署失败: {str(e)}")
            
            return {
                "success": True,
                "results": results,
                "deploy_results": deploy_results
            }

        except Exception as e:
            self.logger.error(f"导出过程中发生错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _on_export_complete(self, future):
        """导出完成的回调处理"""
        try:
            result_data = future.result()
            results = result_data.get("export_results", {})
            deploy_results = result_data.get("deploy_results", {})
            
            # 切换回正常状态
            self.state = ViewState.IDLE
            self.status_var.set("导出完成")
            self.progress_var.set(100)
            
            # 生成导出报告
            report = "=== 导出完成 ===\n\n"
            
            if "obj" in results and results["obj"]:
                report += f"OBJ文件: {results['obj']}\n"
            
            if "unity" in results and results["unity"]:
                path = results["unity"]
                if os.path.isdir(path):
                    report += f"Unity项目包: {path}\n"
                else:
                    report += f"Unity格式: {path}\n"
                
            if "unreal" in results and results["unreal"]:
                path = results["unreal"]
                if os.path.isdir(path):
                    report += f"Unreal项目包: {path}\n"
                else:
                    report += f"Unreal格式: {path}\n"
            
            # 添加部署结果报告
            if deploy_results:
                report += "\n=== 部署结果 ===\n\n"
                for engine, success in deploy_results.items():
                    if success:
                        report += f"{engine}: 部署成功\n"
                    else:
                        report += f"{engine}: 部署失败\n"
            
            # 显示导出结果对话框
            messagebox.showinfo("导出结果", report)
            
            # 询问是否打开导出目录
            if messagebox.askyesno("导出完成", "是否打开导出目录？"):
                output_dir = self.export_config.output_dir
                
                # 根据系统打开文件资源管理器
                if sys.platform == 'win32':
                    normalized_path = os.path.abspath(output_dir)  # 关键转换
                    os.startfile(normalized_path)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.call(['open', output_dir])
                else:  # Linux
                    subprocess.call(['xdg-open', output_dir])
                    
        except Exception as e:
            self.state = ViewState.IDLE
            self.status_var.set("导出失败")
            self.progress_var.set(0)
            self.logger.error(f"导出失败: {str(e)}")
            messagebox.showerror("导出错误", f"导出过程中发生错误:\n{str(e)}")

    def _get_llm_suggestions(self):
        future = self.task_manager.submit_task("llm_suggestions", self._get_llm_suggestions_task)
        future.add_done_callback(self._on_llm_suggestions_complete)

    def _get_llm_suggestions_task(self):
        suggestions = self.llm.get_parameter_suggestions(self.map_params)
        return suggestions

    def _on_llm_suggestions_complete(self, future):
        try:
            suggestions = future.result()
            for param, value in suggestions.items():
                if hasattr(self.map_params, param):
                    setattr(self.map_params, param, value)
                    self.param_controls[param]["var"].set(value)
            self.logger.log(f"已应用LLM建议: {suggestions}")
            self.status_var.set("LLM建议已应用")
        except Exception as e:
            self.logger.error(f"LLM参数更新错误: {str(e)}")
        
    def _undo(self):
        command = self.command_history.undo()
        if not command:
            return
            
        cmd_type = command.get("type", "")
        
        if cmd_type == "param_change":
            param = command["param"]
            setattr(self.map_params, param, command["old_value"])
            self.param_controls[param]["var"].set(command["old_value"])
            self.status_var.set(f"撤销: {param} = {command['old_value']}")
            
        elif cmd_type == "attr_range_change":
            attr_name = command["attr_name"]
            old_min, old_max = command["old_value"]
            ATTR_RANGES[attr_name] = (old_min, old_max)
            self.attr_controls[attr_name]["min_var"].set(old_min)
            self.attr_controls[attr_name]["max_var"].set(old_max)
            self.status_var.set(f"撤销: {attr_name} 范围 = ({old_min}, {old_max})")
            
        elif cmd_type == "target_ratio_change":
            global TARGET_RATIO
            TARGET_RATIO = command["old_value"]
            self.target_ratio_var.set(command["old_value"])
            self.status_var.set(f"撤销: 目标比率 = {command['old_value']:.2f}")
            
        elif cmd_type == "biome_param_change":
            param = command["param"]
            self.biome_params[param] = command["old_value"]
            self.param_controls[param]["var"].set(command["old_value"])
            self.status_var.set(f"撤销: {param} = {command['old_value']:.2f}")

    def _redo(self):
        command = self.command_history.redo()
        if not command:
            return
            
        cmd_type = command.get("type", "")
        
        if cmd_type == "param_change":
            param = command["param"]
            setattr(self.map_params, param, command["new_value"])
            self.param_controls[param]["var"].set(command["new_value"])
            self.status_var.set(f"重做: {param} = {command['new_value']}")
            
        elif cmd_type == "attr_range_change":
            attr_name = command["attr_name"]
            new_min, new_max = command["new_value"]
            ATTR_RANGES[attr_name] = (new_min, new_max)
            self.attr_controls[attr_name]["min_var"].set(new_min)
            self.attr_controls[attr_name]["max_var"].set(new_max)
            self.status_var.set(f"重做: {attr_name} 范围 = ({new_min}, {new_max})")
            
        elif cmd_type == "target_ratio_change":
            global TARGET_RATIO
            TARGET_RATIO = command["new_value"]
            self.target_ratio_var.set(command["new_value"])
            self.status_var.set(f"重做: 目标比率 = {command['new_value']:.2f}")
            
        elif cmd_type == "biome_param_change":
            param = command["param"]
            self.biome_params[param] = command["new_value"]
            self.param_controls[param]["var"].set(command["new_value"])
            self.status_var.set(f"重做: {param} = {command['new_value']:.2f}")
            
    def _check_background_tasks(self):
        for name, future in self.task_manager.get_completed_tasks():
            if hasattr(self, f"_on_{name}_complete"):
                getattr(self, f"_on_{name}_complete")(future)
        self.master.after(100, self._check_background_tasks)

    def _setup_autosave(self):
        interval = self.config_manager.get("auto_save_interval", 5) * 60 * 1000
        self.master.after(interval, self._auto_save)

    def _auto_save(self):
        self._save_project(auto=True)
        interval = self.config_manager.get("auto_save_interval", 5) * 60 * 1000
        self.master.after(interval, self._auto_save)

    def _new_project(self):
        #新建项目
        self.map_params = MapParameters()
        for param, control in self.param_controls.items():
            control["var"].set(getattr(self.map_params, param))
        #self.map_data = MapData(self.map_params.map_width, self.map_params.map_height, False)
        self.current_file = None
        self.state = ViewState.IDLE

    def _save_project(self, auto=False):
        """保存当前项目"""
        # 如果当前文件路径未设置或是目录，则调用另存为对话框
        if not self.current_file or os.path.isdir(self.current_file):
            if auto:
                # 自动保存时，使用默认文件名
                self.current_file = os.path.join(
                    self.config_manager.get("default_export_path", "./exports"),
                    f"autosave_{time.strftime('%Y%m%d_%H%M%S')}.json"
                )
                # 确保目录存在
                os.makedirs(os.path.dirname(self.current_file), exist_ok=True)
            else:
                # 手动保存时，让用户选择文件位置
                self._save_project_as()
                return
        
        # 保存文件
        success = self.map_params.save_to_file(self.current_file)
        if success:
            self.logger.log(f"项目已保存至: {self.current_file}")
        else:
            self.logger.error("项目保存失败")

    def _save_project_as(self):
        """另存为新文件"""
        filepath = filedialog.asksaveasfilename(
            title="保存项目",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if not filepath:
            return
            
        self.current_file = filepath
        self._save_project()

    def _open_project(self):
        """打开项目文件"""
        filepath = filedialog.askopenfilename(
            title="打开项目",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if not filepath:
            return
            
        params = MapParameters.load_from_file(filepath)
        if params:
            self.map_params = params
            # 更新UI控件
            for param, control in self.param_controls.items():
                if hasattr(params, param):
                    control["var"].set(getattr(params, param))
            
            self.current_file = filepath
            self.logger.log(f"项目已加载: {filepath}")
        else:
            self.logger.error(f"无法加载项目: {filepath}")

    def _export_parameters(self):
        """导出当前参数到文件"""
        filepath = filedialog.asksaveasfilename(
            title="导出参数",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filepath:
            success = self.map_params.save_to_file(filepath)
            if success:
                self.logger.log(f"参数已导出至: {filepath}")
            else:
                self.logger.error("参数导出失败")

    def _import_parameters(self):
        """从文件导入参数"""
        filepath = filedialog.askopenfilename(
            title="导入参数",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if not filepath:
            return
            
        params = MapParameters.load_from_file(filepath)
        if params:
            self.map_params = params
            # 更新UI控件
            for param, control in self.param_controls.items():
                if hasattr(params, param):
                    control["var"].set(getattr(params, param))
            
            self.logger.log(f"参数已导入: {filepath}")
        else:
            self.logger.error(f"参数导入失败: {filepath}")

    def _copy_map_image(self):
        pass  # 需要时实施

    def _on_close(self):
        self.config_manager.set("ui.window_width", self.master.winfo_width())
        self.config_manager.set("ui.window_height", self.master.winfo_height())
        self.config_manager.set("ui.paned_position", self.main_paned.sashpos(0))
        self.config_manager.save_config()
        self.task_manager.shutdown()
        self.master.quit()

    def _load_params_from_config(self):
        pass  # 如果config包含默认参数，则实现

    def _show_about(self):
        messagebox.showinfo("关于", "地图生成器 v1.0\n基于LLM的智能参数调整系统")
        
    # 添加弹窗方法
    def show_warning_dialog(self, message):
        """显示警告对话框"""
        messagebox.showwarning("警告", message)
        
    def show_error_dialog(self, message):
        """显示错误对话框"""
        messagebox.showerror("错误", message)
        
    def _on_evolution_complete(self, future):
        try:
            result = future.result()
            self.status_var.set("进化引擎初始化完成")
            
            # 如果有待处理的编辑器，显示它
            if hasattr(result, 'pending_editor') and result.pending_editor == "evolution_scorer":
                # 获取editor_state，如果不存在则创建一个包含evolution_engine的state
                editor_state = getattr(result, 'editor_state', None)
                if not editor_state:
                    editor_state = {'engine': self.evolution_engine}
                
                # 正确传递editor_state参数
                self._handle_pending_editor(result.pending_editor, editor_state)
            else:
                self.state = ViewState.MAP_READY
                self.logger.log("进化引擎初始化完成，但没有待处理的编辑器")
        except Exception as e:
            self.status_var.set("进化失败")
            self.state = ViewState.MAP_READY
            self.logger.error(f"进化过程出错: {str(e)}")
            
    def _setup_gameplay_models(self):
        """初始化游戏设计模型"""
        # 根据当前参数创建玩家模型
        self.player_model = PlayerModel(
            exploration_bias=self.map_params.exploration,
            challenge_seeking=self.map_params.combat,
            social_preference=self.map_params.social,
            completion_drive=self.map_params.achievement,
            # 其他参数使用默认值
            collection_focus=0.5,
            story_engagement=0.5,
            novelty_seeking=0.5
        )
        
        # 初始化默认使用的游戏设计模式
        self.active_patterns = GAMEPLAY_PATTERNS.copy()
        self.logger.log(f"已初始化{len(self.active_patterns)}个游戏设计模式")
        
    def _optimize_content_placement(self):
        """使用游戏设计模式优化地图内容布局"""
        if not hasattr(self, 'player_model') or not hasattr(self, 'active_patterns'):
            self._setup_gameplay_models()
        
        if not self.map_data or not self.map_data.is_valid():
            self.logger.error("无有效地图数据，无法优化内容布局")
            return False
        
        self.logger.log("正在使用游戏设计理论优化地图内容布局...")
        
        # 获取地图基础数据
        height_map = self.map_data.get_layer("height")
        if height_map is None:
            self.logger.error("地图缺少高度图层，无法优化")
            return False
        
        biome_map = self.map_data.get_layer("biome")
        content_layout = self.map_data.content_layout or {}
        
        # 构建世界上下文供模式分析使用
        world_context = {
            "height_map": height_map,
            "biome_map": biome_map,
            "placed_objects": []
        }
        
        # 收集所有已放置对象
        for category in content_layout:
            world_context["placed_objects"].extend(content_layout.get(category, []))
        
        # 计算常用距离图（如与建筑的距离）
        if "buildings" in content_layout and content_layout["buildings"]:
            building_distance_map = {}
            buildings = content_layout["buildings"]
            for y in range(len(height_map)):
                for x in range(len(height_map[0])):
                    min_dist = float('inf')
                    for building in buildings:
                        bx, by = building.get("x", 0), building.get("y", 0)
                        dist = math.sqrt((x - bx)**2 + (y - by)**2)
                        min_dist = min(min_dist, dist)
                    building_distance_map[(x, y)] = min_dist
            world_context["building_distance_map"] = building_distance_map
        
        # 对每种待放置的内容类型应用游戏设计模式
        content_types = ["treasures", "enemies", "story_points", "resources"]
        for content_type in content_types:
            if content_type not in content_layout:
                content_layout[content_type] = []
                
            target_count = self._determine_content_count(content_type)
            current_count = len(content_layout[content_type])
            
            if current_count >= target_count:
                self.logger.log(f"{content_type}内容数量已足够({current_count}/{target_count})")
                continue
                
            self.logger.log(f"为{content_type}寻找最佳放置位置...")
            
            # 为此内容类型选择合适的模式
            applicable_patterns = self._get_applicable_patterns(content_type)
            
            # 评估地图上的候选位置
            candidates = []
            sample_step = max(1, int(len(height_map) / 20))  # 采样步长，避免评估过多位置
            
            for y in range(0, len(height_map), sample_step):
                for x in range(0, len(height_map[0]), sample_step):
                    # 检查位置是否合适（例如，不在水下）
                    if height_map[y][x] < 5:  # 假设高度小于5是水域
                        continue
                        
                    # 评估此位置的适合度
                    satisfaction_score = 0
                    for pattern in applicable_patterns:
                        satisfaction_score += pattern.satisfies((x, y), world_context)
                    
                    # 还要考虑玩家模型的参与度预测
                    content_proto = {"type": content_type}
                    engagement_score = self.player_model.predict_engagement(
                        content_proto, (x, y), world_context
                    )
                    
                    # 结合满意度和参与度得到最终分数
                    final_score = satisfaction_score * engagement_score
                    
                    candidates.append({
                        "x": x, 
                        "y": y, 
                        "score": final_score,
                        "satisfaction": satisfaction_score,
                        "engagement": engagement_score
                    })
            
            # 按分数排序，选择最佳位置
            candidates.sort(key=lambda c: c["score"], reverse=True)
            
            # 选择前N个位置放置内容
            to_place = min(target_count - current_count, len(candidates))
            if to_place > 0:
                for i in range(to_place):
                    if i < len(candidates):
                        pos = candidates[i]
                        # 根据内容类型创建不同的对象
                        new_object = self._create_content_object(content_type, pos["x"], pos["y"])
                        content_layout[content_type].append(new_object)
                        
                        # 更新世界上下文中的已放置对象
                        world_context["placed_objects"].append(new_object)
                
                self.logger.log(f"已放置{to_place}个{content_type}(总计{current_count + to_place})")
        
        # 更新地图数据的内容布局
        self.map_data.content_layout = content_layout
        self.logger.log("地图内容布局优化完成")
        return True
        
    def _determine_content_count(self, content_type):
        """根据地图参数和内容类型确定应该生成多少内容"""
        base_map_area = self.map_params.map_width * self.map_params.map_height
        area_factor = base_map_area / (512 * 512)  # 相对于512x512的面积比例
        
        if content_type == "treasures":
            return int(10 * area_factor * self.map_params.exploration)
        elif content_type == "enemies":
            return int(15 * area_factor * self.map_params.combat)
        elif content_type == "story_points":
            return int(5 * area_factor * self.map_params.social)
        elif content_type == "resources":
            return int(20 * area_factor * self.map_params.achievement)
        else:
            return int(10 * area_factor)  # 默认数量
            
    def _get_applicable_patterns(self, content_type):
        """获取适用于特定内容类型的游戏设计模式"""
        applicable = []
        content_type_mapping = {
            "treasures": [ContentType.RESOURCE, ContentType.MYSTERY],
            "enemies": [ContentType.CHALLENGE, ContentType.COMBAT],
            "story_points": [ContentType.STORY, ContentType.SOCIAL],
            "resources": [ContentType.RESOURCE, ContentType.CHALLENGE]
        }
        
        target_types = content_type_mapping.get(content_type, [])
        
        # 查找包含目标类型的模式
        for pattern in self.active_patterns:
            for ct in pattern.content_types:
                if ct in target_types:
                    applicable.append(pattern)
                    break
        
        # 如果没有找到适合的模式，返回所有模式
        return applicable if applicable else self.active_patterns
        
    def _create_content_object(self, content_type, x, y):
        """根据内容类型创建相应的对象"""
        base_object = {"x": x, "y": y, "type": content_type}
        
        # 根据内容类型添加特定属性
        if content_type == "treasures":
            base_object.update({
                "rarity": random.choice(["common", "uncommon", "rare", "epic", "legendary"]),
                "value": random.randint(10, 100),
                "discovery_threshold": 0.3 + random.random() * 0.5  # 发现难度
            })
        elif content_type == "enemies":
            base_object.update({
                "level": random.randint(1, 10),
                "strength": 0.3 + random.random() * 0.7,
                "aggression": 0.2 + random.random() * 0.8
            })
        elif content_type == "story_points":
            base_object.update({
                "importance": random.random(),
                "dialog_count": random.randint(1, 5),
                "linked_quest": random.random() > 0.5  # 50%几率链接到任务
            })
        elif content_type == "resources":
            base_object.update({
                "resource_type": random.choice(["wood", "stone", "metal", "herb", "crystal"]),
                "quantity": random.randint(1, 10),
                "respawn": random.random() > 0.3  # 70%几率可重生
            })
            
        return base_object
#############################################################################################
class PipelineConfiguratorDialog(tk.Toplevel):
    """地图生成流水线步骤配置器对话框"""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.title("地图生成流水线配置")
        self.geometry("600x500")
        self.minsize(500, 400)
        
        # 设置模态对话框
        self.transient(parent)
        self.grab_set()
        
        # 创建UI
        self._create_widgets()
        
        # 初始化步骤列表
        self._initialize_steps()
        
        # 将窗口置于屏幕中央
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'+{x}+{y}')
    
    def _create_widgets(self):
        """创建配置界面组件"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部说明
        ttk.Label(main_frame, 
                 text="配置地图生成流水线步骤\n勾选您想启用的步骤，并可以拖拽调整执行顺序",
                 justify=tk.CENTER).pack(pady=(0, 10))
        
        # 创建步骤列表框架
        self.steps_frame = ttk.LabelFrame(main_frame, text="生成步骤", padding=10)
        self.steps_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建滚动区域
        self.canvas = tk.Canvas(self.steps_frame)
        self.scrollbar = ttk.Scrollbar(self.steps_frame, orient="vertical", command=self.canvas.yview)
        self.steps_list_frame = ttk.Frame(self.canvas)
        
        # 配置滚动区域
        self.steps_list_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.steps_list_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # 排列滚动区域组件
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # 底部按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="全选", command=self._select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清除", command=self._clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="恢复默认", command=self._reset_default).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="应用", command=self._apply_changes, style="Accent.TButton").pack(side=tk.RIGHT, padx=5)
    
    def _initialize_steps(self):
        """初始化步骤列表"""
        # 获取当前启用的步骤和步骤顺序
        self.enabled_steps = self.app.get_enabled_pipeline_steps()
        self.step_order = self.app.get_pipeline_step_order()
        
        # 为每个步骤创建变量和控件
        self.step_vars = {}
        self.step_frames = {}
        
        # 清除现有控件
        for widget in self.steps_list_frame.winfo_children():
            widget.destroy()
        
        # 创建步骤条目
        for i, step in enumerate(self.step_order):
            frame = ttk.Frame(self.steps_list_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            frame.bind("<Button-1>", lambda e, s=step: self._start_drag(e, s))
            
            # 步骤序号
            order_label = ttk.Label(frame, text=f"{i+1}.", width=3)
            order_label.pack(side=tk.LEFT, padx=(0, 5))
            
            # 复选框
            var = tk.BooleanVar(value=step in self.enabled_steps)
            self.step_vars[step] = var
            check = ttk.Checkbutton(frame, text=step.value, variable=var)
            check.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor=tk.W)
            
            # 拖动提示图标
            drag_icon = ttk.Label(frame, text="≡")
            drag_icon.pack(side=tk.RIGHT, padx=5)
            drag_icon.bind("<Button-1>", lambda e, s=step: self._start_drag(e, s))
            
            # 存储步骤框架引用
            self.step_frames[step] = frame
        
        # 设置拖拽状态变量
        self.drag_step = None
        self.drag_indicator = None
    
    def _start_drag(self, event, step):
        """开始拖拽步骤"""
        if self.drag_step is not None:
            return
        
        self.drag_step = step
        self.step_frames[step].configure(style="Accent.TFrame")
        
        # 创建拖拽指示器
        self.drag_indicator = ttk.Frame(self.steps_list_frame, height=2, style="Accent.TFrame")
        
        # 绑定鼠标移动和释放事件
        self.steps_list_frame.bind("<Motion>", self._drag_motion)
        self.steps_list_frame.bind("<ButtonRelease-1>", self._end_drag)
    
    def _drag_motion(self, event):
        """处理鼠标拖拽移动"""
        if self.drag_step is None:
            return
        
        # 获取当前鼠标位置下的目标位置
        y = event.y_root - self.steps_list_frame.winfo_rooty()
        target_idx = int(y / 30)  # 假设每个步骤高度约为30像素
        
        # 确保目标索引在有效范围内
        target_idx = max(0, min(target_idx, len(self.step_order)-1))
        
        # 显示拖拽指示器
        if self.drag_indicator:
            self.drag_indicator.place_forget()
            target_frame = self.step_frames[self.step_order[target_idx]]
            self.drag_indicator.place(x=0, y=target_frame.winfo_y() - 1, 
                                    width=self.steps_list_frame.winfo_width())
    
    def _end_drag(self, event):
        """结束拖拽操作"""
        if self.drag_step is None:
            return
        
        # 获取目标位置
        y = event.y_root - self.steps_list_frame.winfo_rooty()
        target_idx = int(y / 30)
        target_idx = max(0, min(target_idx, len(self.step_order)-1))
        
        # 获取源步骤位置
        source_idx = self.step_order.index(self.drag_step)
        
        # 如果位置不同，则重新排序
        if source_idx != target_idx:
            # 移除源步骤
            self.step_order.remove(self.drag_step)
            # 在目标位置插入
            self.step_order.insert(target_idx, self.drag_step)
            # 更新UI
            self._initialize_steps()
        
        # 清除拖拽状态
        if self.drag_indicator:
            self.drag_indicator.place_forget()
            self.drag_indicator = None
        
        self.drag_step = None
        self.steps_list_frame.unbind("<Motion>")
        self.steps_list_frame.unbind("<ButtonRelease-1>")
    
    def _select_all(self):
        """选择所有步骤"""
        for var in self.step_vars.values():
            var.set(True)
    
    def _clear_all(self):
        """清除所有步骤选择"""
        for var in self.step_vars.values():
            var.set(False)
    
    def _reset_default(self):
        """重置为默认步骤配置"""
        self.enabled_steps = {step for step in MapPipelineStep}
        self.step_order = list(MapPipelineStep)
        self._initialize_steps()
    
    def _apply_changes(self):
        """应用更改并关闭对话框"""
        # 收集启用的步骤
        enabled_steps = {step for step, var in self.step_vars.items() if var.get()}
        
        # 更新应用中的步骤配置
        self.app.set_enabled_pipeline_steps(enabled_steps)
        self.app.set_pipeline_step_order(self.step_order)
        
        # 显示确认消息
        messagebox.showinfo("成功", "地图生成流水线配置已更新！", parent=self)
        
        # 关闭对话框
        self.destroy()    

class MapPipelineStep(Enum):
    """地图生成流水线步骤枚举"""
    INITIALIZE = "初始化"
    TERRAIN_BASE = "基础地形生成"
    TEMPERATURE = "温度生成"
    HUMIDITY = "湿度生成"
    TERRAIN_EDIT = "地形编辑"
    CAVES_RAVINES = "洞穴与峡谷"
    RIVERS = "河流系统"
    UNDERGROUND_SYSTEM = "地下系统生成"  # 新增步骤
    CLIMATE_UPDATE = "气候更新"
    ECOSYSTEM = "生态系统模拟"
    MICROCLIMATE = "微气候生成"
    BIOMES = "生物群系分类"
    BIOME_TRANSITIONS = "生物群系过渡区"
    VEGETATION_BUILDINGS = "植被与建筑"
    CONTENT_LAYOUT = "内容布局"
    ROAD_NETWORK = "道路网络"
    STORY_GENERATION = "故事生成"
    INTERACTIVE_EVOLUTION = "交互进化"
    EMOTION_ANALYSIS = "情感分析"
    LEVEL_GENERATION = "关卡生成"
    COMPLETE = "完成"

class MapPipelineProcessor:
    """地图生成流水线处理器，实现模块化地图生成"""
    
    def __init__(self, logger=None):
        self.logger = logger or ThreadSafeLogger(max_lines=500, log_file="map_pipeline.log")
        self.perf = PerformanceMonitor()
        self.pipeline_state = {}
        self.config = None
        self.map_data = None
        self.vis = None
        self.current_step = MapPipelineStep.INITIALIZE
        self.completed_steps = set()
        self.enabled_steps = {step for step in MapPipelineStep}  # 默认启用所有步骤
        self.step_handlers = {
            MapPipelineStep.INITIALIZE: self._handle_initialize,
            MapPipelineStep.TERRAIN_BASE: self._handle_terrain_base,
            MapPipelineStep.TEMPERATURE: self._handle_temperature,
            MapPipelineStep.HUMIDITY: self._handle_humidity,
            MapPipelineStep.TERRAIN_EDIT: self._handle_terrain_edit,
            MapPipelineStep.CAVES_RAVINES: self._handle_caves_ravines,
            MapPipelineStep.RIVERS: self._handle_rivers,
            MapPipelineStep.CLIMATE_UPDATE: self._handle_climate_update,
            MapPipelineStep.ECOSYSTEM: self._handle_ecosystem,
            MapPipelineStep.MICROCLIMATE: self._handle_microclimate,
            MapPipelineStep.BIOMES: self._handle_biomes,
            MapPipelineStep.BIOME_TRANSITIONS: self._handle_biome_transitions,
            MapPipelineStep.VEGETATION_BUILDINGS: self._handle_vegetation_buildings,
            MapPipelineStep.CONTENT_LAYOUT: self._handle_content_layout,
            MapPipelineStep.ROAD_NETWORK: self._handle_road_network,
            MapPipelineStep.STORY_GENERATION: self._handle_story_generation,
            MapPipelineStep.INTERACTIVE_EVOLUTION: self._handle_interactive_evolution,
            MapPipelineStep.EMOTION_ANALYSIS: self._handle_emotion_analysis,
            MapPipelineStep.UNDERGROUND_SYSTEM: self._handle_underground_system,
            MapPipelineStep.COMPLETE: self._handle_complete
        }
        
    def set_enabled_steps(self, enabled_steps):
        """设置启用的步骤"""
        self.enabled_steps = set(enabled_steps)
        self.logger.log(f"已启用的地图生成步骤: {[step.value for step in self.enabled_steps]}")
        
    def get_next_step(self):
        """获取下一个要执行的步骤"""
        all_steps = list(MapPipelineStep)
        current_index = all_steps.index(self.current_step)
        
        for i in range(current_index + 1, len(all_steps)):
            next_step = all_steps[i]
            if next_step in self.enabled_steps:
                return next_step
                
        return None  # 没有下一步了
        
    def process_map(self, preferences, width, height, export_model=False,
                   use_real_geo_data=False, geo_data_path=None, geo_bounds=None,
                   visualize=False, visualization_path=None, 
                   parent_frame=None, use_gui_editors=False,
                   resume_state=None, callback=None):
        """启动地图生成流水线处理"""
        try:
            # 初始化pipeline_state
            self.pipeline_state = {
                'preferences': preferences,
                'width': width,
                'height': height,
                'export_model': export_model,
                'use_real_geo_data': use_real_geo_data,
                'geo_data_path': geo_data_path,
                'geo_bounds': geo_bounds,
                'visualize': visualize,
                'visualization_path': visualization_path,
                'parent_frame': parent_frame,
                'use_gui_editors': use_gui_editors,
                'resume_state': resume_state,
                'callback': callback
            }
            
            # 初始化可视化工具
            if visualize:
                from core.services.map_tools import MapGenerationVisualizer
                self.vis = MapGenerationVisualizer(
                    save_path=visualization_path, 
                    show_plots=True, 
                    logger=self.logger, 
                    gui_frame=visualize
                )
                self.logger.log("已启用地图生成可视化功能")
            
            # 创建地图数据容器
            self.config = MapGenerationConfig(width=width, height=height, export_model=export_model)
            self.map_data = MapData(width, height, self.config)
            
            # 处理恢复状态
            if resume_state is not None:
                self._process_resume_state(resume_state)
            
            # 执行流水线
            return self._run_pipeline()
            
        except Exception as e:
            self.logger.log(f"地图生成流水线错误: {e}", "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")
            raise
            
    def _process_resume_state(self, resume_state):
        """处理恢复状态"""
        self.logger.log("从之前的状态恢复地图生成...")
        
        # 从resume_state提取已完成的步骤
        completed_steps = resume_state.get('completed_steps', [])
        if completed_steps:
            self.completed_steps = set(MapPipelineStep[step] for step in completed_steps if step in MapPipelineStep.__members__)
            self.logger.log(f"已完成的步骤: {[step.value for step in self.completed_steps]}")
        
        # 从resume_state提取当前步骤
        current_step = resume_state.get('current_step')
        if current_step and current_step in MapPipelineStep.__members__:
            self.current_step = MapPipelineStep[current_step]
            self.logger.log(f"当前步骤: {self.current_step.value}")
        
        # 从resume_state提取所需变量到pipeline_state
        for key, value in resume_state.items():
            if key not in ('completed_steps', 'current_step'):
                self.pipeline_state[key] = value
                
    def _run_pipeline(self):
        """运行流水线，处理所有启用的步骤"""
        self.logger.log("开始执行地图生成流水线...")
        
        # 循环执行流水线步骤
        while self.current_step != MapPipelineStep.COMPLETE:
            # 检查当前步骤是否启用
            if self.current_step not in self.enabled_steps:
                self.logger.log(f"跳过禁用的步骤: {self.current_step.value}")
                self.current_step = self.get_next_step() or MapPipelineStep.COMPLETE
                continue
                
            # 检查当前步骤是否已完成
            if self.current_step in self.completed_steps:
                self.logger.log(f"跳过已完成的步骤: {self.current_step.value}")
                self.current_step = self.get_next_step() or MapPipelineStep.COMPLETE
                continue
                
            # 执行当前步骤
            self.logger.log(f"执行步骤: {self.current_step.value}")
            try:
                handler = self.step_handlers.get(self.current_step)
                if handler:
                    result = handler()
                    
                    # 如果步骤返回值为MapData，表示步骤需要中断执行
                    if isinstance(result, MapData):
                        self.logger.log(f"步骤 {self.current_step.value} 请求中断执行")
                        return result
                        
                    # 标记步骤完成
                    self.completed_steps.add(self.current_step)
                    
                    # 获取下一个步骤
                    self.current_step = self.get_next_step() or MapPipelineStep.COMPLETE
                else:
                    self.logger.log(f"步骤 {self.current_step.value} 没有对应的处理函数", "ERROR")
                    self.current_step = self.get_next_step() or MapPipelineStep.COMPLETE
            except Exception as e:
                self.logger.log(f"步骤 {self.current_step.value} 执行错误: {e}", "ERROR")
                import traceback
                self.logger.log(traceback.format_exc(), "DEBUG")
                raise
                
        # 流水线完成
        self.logger.log("地图生成流水线执行完成")
        return self.map_data

    def _handle_initialize(self):
        """初始化步骤处理"""
        task_id = self.perf.start("initialize")
        
        # 导入必要的模块和加载数据
        self.logger.log("启动数据加载...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load_objects_db, self.logger)
            data, all_objs = future.result()
            
        self.pipeline_state['data'] = data
        self.pipeline_state['all_objs'] = all_objs
        self.pipeline_state['biome_data'] = load_biome_config(self.logger)
        
        # 映射玩家偏好到参数
        self.logger.log("映射玩家偏好...")
        preferences = self.pipeline_state['preferences']
        self.pipeline_state['map_params'] = map_preferences_to_parameters(preferences)
        
        # 扩展参数字典，添加高级参数支持
        terrain_params = {
            "seed": self.pipeline_state['map_params'].get("seed"),
            "scale_factor": self.pipeline_state['map_params'].get("scale_factor", 2.0),
            "mountain_sharpness": self.pipeline_state['map_params'].get("mountain_sharpness", 1.5),
            "erosion_iterations": self.pipeline_state['map_params'].get("erosion_iterations", 3),
            "river_density": self.pipeline_state['map_params'].get("river_density", 1.0),
            "use_tectonic": self.pipeline_state['map_params'].get("use_tectonic", True)
        }

        # 直接从preferences获取地形参数
        direct_terrain_params = [
            "seed", "scale_factor", "mountain_sharpness", "erosion_iterations",
            "river_density", "use_tectonic", "detail_level", "use_frequency_optimization",
            "erosion_type", "erosion_strength", "talus_angle", "sediment_capacity",
            "rainfall", "evaporation", "min_watershed_size", "precipitation_factor",
            "meander_factor", "octaves", "persistence", "lacunarity", "plain_ratio",
            "hill_ratio", "mountain_ratio", "plateau_ratio", "latitude_effect",
            "prevailing_wind_x", "prevailing_wind_y",
            "enable_micro_detail", "enable_extreme_detection", "optimization_level"
        ]
        
        for param in direct_terrain_params:
            if param in preferences:
                terrain_params[param] = preferences[param]
                
        self.pipeline_state['terrain_params'] = terrain_params
        
        # 初始化LLM集成
        self.pipeline_state['llm'] = LLMIntegration(self.logger)
        
        # 初始化roads_points
        self.pipeline_state['road_points'] = []
        
        self.perf.end(task_id)
        
    def _handle_terrain_base(self):
        """基础地形生成步骤处理"""
        task_id = self.perf.start("terrain_base")
        
        width = self.pipeline_state['width']
        height = self.pipeline_state['height']
        use_real_geo_data = self.pipeline_state['use_real_geo_data']
        geo_data_path = self.pipeline_state['geo_data_path']
        geo_bounds = self.pipeline_state['geo_bounds']
        terrain_params = self.pipeline_state['terrain_params']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 根据用户选择，使用真实地理数据或算法生成地形
        if use_real_geo_data:
            self.logger.log("使用真实地理数据生成地形...")
            
            # 创建地理数据导入器
            geo_importer = GeoDataImporter(self.logger)
            
            # 导入地理数据
            if geo_data_path and os.path.exists(geo_data_path):
                # 从文件导入
                success = geo_importer.from_file(geo_data_path)
            elif geo_bounds and len(geo_bounds) == 4:
                # 下载SRTM数据
                success = geo_importer.download_srtm(geo_bounds)
            else:
                self.logger.log("错误: 使用真实地理数据时必须提供有效的geo_data_path或geo_bounds", "ERROR")
                success = False
            
            if success:
                # 处理并获取高度图
                height_map = geo_importer.get_height_map(width, height, normalize_range=(0.0, 1.0))
                
                # 可视化：原始高度图
                if visualize and height_map is not None and self.vis:
                    self.vis.visualize_height_map(height_map, "原始地理数据高度图", "初始地形")
                
                if height_map is not None:
                    # 确保高度图尺寸正确
                    if height_map.shape != (height, width):
                        self.logger.log(f"调整高度图尺寸从 {height_map.shape} 到 ({height}, {width})", "INFO")
                        from scipy.ndimage import zoom
                        
                        # 计算缩放因子
                        zoom_factors = (height / height_map.shape[0], width / height_map.shape[1])
                        
                        # 使用双三次插值调整高度图
                        try:
                            height_map = zoom(height_map, zoom_factors, order=3)
                            self.logger.log(f"高度图调整成功，当前尺寸: {height_map.shape}", "INFO")
                        except Exception as zoom_error:
                            self.logger.log(f"高度图调整失败: {zoom_error}", "ERROR")
                            height_map = None

                    # 可视化：调整后的高度图
                    if visualize and height_map is not None and self.vis:
                        self.vis.visualize_height_map(height_map, "调整后高度图", "地形处理")
                
                self.logger.log("成功从真实地理数据生成地形")
            else:
                self.logger.log("地理数据导入失败，回退到算法生成", "WARNING")
                height_map = None
        else:
            height_map = None
            
        # 如果没有使用真实地形数据或导入失败，使用算法生成
        if height_map is None:
            # 生成基础高度图
            self.logger.log("使用算法生成基础地形...")
            
            # 过滤参数
            height_gen_params = {k: v for k, v in terrain_params.items() if k in HEIGHT_GEN_PARAMS}
            height_map = generate_height_temp_humid(width, height, **height_gen_params)
            
            # 可视化：基础高度图
            if visualize and self.vis:
                self.vis.visualize_height_map(height_map, "基础高度图", "初始地形")

            # 验证高度图
            if not isinstance(height_map, np.ndarray) or height_map.shape != (height, width):
                raise ValueError(f"地形数据维度错误: 期望({height}, {width}), 实际{height_map.shape if isinstance(height_map, np.ndarray) else type(height_map)}")
        
        # 保存高度图到pipeline_state和map_data
        self.pipeline_state['height_map'] = height_map
        self.map_data.layers["height"] = height_map
        
        self.perf.end(task_id)
        
    def _handle_temperature(self):
        """温度生成步骤处理"""
        task_id = self.perf.start("temperature")
        
        width = self.pipeline_state['width']
        height = self.pipeline_state['height']
        height_map = self.pipeline_state['height_map']
        terrain_params = self.pipeline_state['terrain_params']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 基于高度图生成温度图
        self.logger.log("生成温度分布...")
        temp_map = generate_temperature_map(
            height_map, 
            width, 
            height,
            seed=terrain_params.get("seed"),
            latitude_effect=terrain_params.get("latitude_effect", 0.5),
            use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
        )
        
        # 可视化：温度图
        if visualize and self.vis:
            self.vis.visualize_temperature_map(temp_map, "温度分布", "气候生成")
        
        # 保存温度图到pipeline_state和map_data
        self.pipeline_state['temp_map'] = temp_map
        self.map_data.layers["temperature"] = temp_map
        
        self.perf.end(task_id)
        
    def _handle_humidity(self):
        """湿度生成步骤处理"""
        task_id = self.perf.start("humidity")
        
        width = self.pipeline_state['width']
        height = self.pipeline_state['height']
        height_map = self.pipeline_state['height_map']
        temp_map = self.pipeline_state['temp_map']
        terrain_params = self.pipeline_state['terrain_params']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 基于高度图和温度图生成湿度图
        self.logger.log("生成湿度分布...")
        wind_x = terrain_params.get("prevailing_wind_x", 1.0) if "prevailing_wind_x" in terrain_params else 1.0
        wind_y = terrain_params.get("prevailing_wind_y", 0.0) if "prevailing_wind_y" in terrain_params else 0.0
        
        humid_map = generate_humidity_map(
            height_map,
            width,
            height,
            temp_map=temp_map,
            seed=terrain_params.get("seed"),
            prevailing_wind=(wind_x, wind_y),
            use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
        )
        
        # 可视化：湿度图
        if visualize and self.vis:
            self.vis.visualize_humidity_map(humid_map, "湿度分布", "气候生成")
        
        # 保存湿度图到pipeline_state和map_data
        self.pipeline_state['humid_map'] = humid_map
        self.map_data.layers["humidity"] = humid_map
        
        self.perf.end(task_id)
        
    def _handle_terrain_edit(self):
        """地形编辑步骤处理"""
        # 获取所需参数
        parent_frame = self.pipeline_state.get('parent_frame')
        use_gui_editors = self.pipeline_state.get('use_gui_editors', False)
        callback = self.pipeline_state.get('callback')
        height_map = self.pipeline_state['height_map']
        temp_map = self.pipeline_state['temp_map']
        humid_map = self.pipeline_state['humid_map']
        map_params = self.pipeline_state['map_params']
        terrain_params = self.pipeline_state['terrain_params']
        width = self.pipeline_state['width']
        height = self.pipeline_state['height']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 使用GUI集成编辑器的情况
        if parent_frame and use_gui_editors:
            self.logger.log("等待GUI编辑器完成高度调整...")
            
            # 保存当前流水线状态
            completed_steps_str = [step.name for step in self.completed_steps]
            
            # 保存当前状态
            state_data = {
                'height_map': height_map,
                'temp_map': temp_map,
                'humid_map': humid_map,
                'current_step': self.current_step.name,
                'completed_steps': completed_steps_str,
                'resume_point': 'post_height_edit',  # 明确标记恢复点
                'width': width,
                'height': height,
                'terrain_params': terrain_params,
                'map_params': map_params
            }
            
            # 设置挂起状态
            self.map_data.pending_editor = "height_editor"
            self.map_data.editor_state = state_data
            
            # 如果提供了回调函数，通知上层应用程序
            if callback:
                callback({
                    'action': 'show_editor',
                    'editor_type': 'height_editor',
                    'state': state_data
                })
            
            # 确保标记为未完成
            self.map_data.generation_complete = False
            
            # 返回当前状态，让上层负责显示编辑器
            return self.map_data
        else:
            # 独立窗口模式
            try:
                self.logger.log("启动高度手动调整功能...")
                hand_map_data, hand_height_map, hand_temp_map, hand_humid_map = manually_adjust_height(
                    self.map_data, map_params, self.logger, seed=map_params.get("seed")
                )
                self.logger.log("手动调整完成")
                
                # 更新数据
                self.map_data = hand_map_data
                self.pipeline_state['height_map'] = hand_height_map
                self.pipeline_state['temp_map'] = hand_temp_map
                self.pipeline_state['humid_map'] = hand_humid_map
                
                # 可视化：手动调整后的高度图
                if visualize and self.vis:
                    self.vis.visualize_height_map(hand_height_map, "手动调整后高度图", "高度修正")
                    
            except Exception as e:
                self.logger.log(f"手动调整功能出错: {e}", "WARNING")
    
    def _handle_caves_ravines(self):
        """洞穴和峡谷生成步骤处理"""
        task_id = self.perf.start("caves_ravines")
        
        height_map = self.pipeline_state['height_map']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 生成洞穴和峡谷，并进一步更新高度图
        self.logger.log("处理地理特征：洞穴和峡谷...")
        carved_height_map, cave_data = carve_caves_and_ravines(height_map)
        
        # 可视化：洞穴和峡谷后的高度图
        if visualize and self.vis:
            self.vis.visualize_height_map(carved_height_map, "洞穴和峡谷后高度图", "地形特征")
        
        # 确保洞穴数据结构正确
        if not isinstance(cave_data, dict) or "entrances" not in cave_data:
            self.logger.log("警告: 洞穴数据格式无效，使用空列表", "WARNING")
            cave_entrances = []
        else:
            cave_entrances = cave_data["entrances"]
        
        # 处理洞穴入口点
        formatted_caves = []
        if isinstance(cave_entrances, list):
            for cave in cave_entrances:
                if isinstance(cave, (list, tuple)) and len(cave) >= 2:
                    formatted_caves.append({"x": int(cave[0]), "y": int(cave[1])})
                elif hasattr(cave, 'x') and hasattr(cave, 'y'):
                    formatted_caves.append(cave)
        
        # 更新高度图为包含洞穴的版本
        self.pipeline_state['height_map'] = carved_height_map
        self.pipeline_state['cave_entrances'] = cave_entrances
        self.pipeline_state['formatted_caves'] = formatted_caves
        self.map_data.layers["height"] = carved_height_map
        self.map_data.layers["caves"] = formatted_caves
        
        self.perf.end(task_id)
    
    def _handle_rivers(self):
        """河流系统生成步骤处理"""
        task_id = self.perf.start("rivers")
        
        height_map = self.pipeline_state['height_map']
        width = self.pipeline_state['width']
        height = self.pipeline_state['height']
        terrain_params = self.pipeline_state['terrain_params']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 获取河流参数
        min_watershed_size = terrain_params.get("min_watershed_size", 50)
        precipitation_factor = terrain_params.get("precipitation_factor", 1.0)
        meander_factor = terrain_params.get("meander_factor", 0.3)

        # 确保有一个默认种子值
        river_seed = terrain_params.get("seed")
        if river_seed is None:
            river_seed = np.random.randint(1, 10000)
            self.logger.log(f"为河流生成随机种子: {river_seed}")

        # 生成河流并更新高度图
        self.logger.log("生成河流系统...")
        rivers_map, river_features, new_height_map = generate_rivers_map(
            height_map,
            width,
            height,
            seed=river_seed,
            min_watershed_size=min_watershed_size,
            precipitation_factor=precipitation_factor,
            meander_factor=meander_factor
        )
        
        # 可视化：河流系统和更新后的高度图
        if visualize and self.vis:
            self.vis.visualize_rivers(new_height_map, rivers_map, "河流系统", "地形特征")
            self.vis.visualize_height_map(new_height_map, "河流侵蚀后高度图", "地形特征")
        
        # 转换河流数据结构
        rivers_map_bool = np.zeros((height, width), dtype=bool)
        river_points = []
        
        if river_features:
            for river_path in river_features:
                for y, x in river_path:
                    if 0 <= y < height and 0 <= x < width:
                        rivers_map_bool[y, x] = True
                        river_points.append((x, y))
        
        # 更新数据
        self.pipeline_state['height_map'] = new_height_map
        self.pipeline_state['rivers_map'] = rivers_map_bool
        self.pipeline_state['river_features'] = river_features
        self.pipeline_state['river_points'] = river_points
        self.map_data.layers["height"] = new_height_map
        self.map_data.layers["rivers"] = rivers_map_bool
        self.map_data.layers["river_features"] = river_features
        self.map_data.layers["rivers_points"] = river_points
        
        self.perf.end(task_id)
    
    def _handle_climate_update(self):
        """气候更新步骤处理"""
        task_id = self.perf.start("climate_update")
        
        height_map = self.pipeline_state['height_map']
        width = self.pipeline_state['width']
        height = self.pipeline_state['height']
        terrain_params = self.pipeline_state['terrain_params']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 重新生成温度图
        self.logger.log("更新温度分布...")
        temp_map = generate_temperature_map(
            height_map, 
            width, 
            height,
            seed=terrain_params.get("seed"),
            latitude_effect=terrain_params.get("latitude_effect", 0.5),
            use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
        )
        
        # 可视化：温度图
        if visualize and self.vis:
            self.vis.visualize_temperature_map(temp_map, "更新后温度分布", "气候生成")
        
        # 重新生成湿度图
        self.logger.log("更新湿度分布...")
        wind_x = terrain_params.get("prevailing_wind_x", 1.0) if "prevailing_wind_x" in terrain_params else 1.0
        wind_y = terrain_params.get("prevailing_wind_y", 0.0) if "prevailing_wind_y" in terrain_params else 0.0
        
        humid_map = generate_humidity_map(
            height_map,
            width,
            height,
            temp_map=temp_map,
            seed=terrain_params.get("seed"),
            prevailing_wind=(wind_x, wind_y),
            use_frequency_optimization=terrain_params.get("use_frequency_optimization", True)
        )
        
        # 可视化：湿度图
        if visualize and self.vis:
            self.vis.visualize_humidity_map(humid_map, "更新后湿度分布", "气候生成")
        
        # 更新数据
        self.pipeline_state['temp_map'] = temp_map
        self.pipeline_state['humid_map'] = humid_map
        self.map_data.layers["temperature"] = temp_map
        self.map_data.layers["humidity"] = humid_map
        
        self.perf.end(task_id)
    
    def _handle_ecosystem(self):
        """生态系统模拟步骤处理"""
        task_id = self.perf.start("ecosystem")
        
        height_map = self.pipeline_state['height_map']
        temp_map = self.pipeline_state['temp_map']
        humid_map = self.pipeline_state['humid_map']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 生态系统模拟
        self.logger.log("应用生态系统动态模拟...")
        from core.generation.ecosystem_dynamics import simulate_ecosystem_dynamics
        enhanced_temp_map, enhanced_humid_map = simulate_ecosystem_dynamics(
            height_map, temp_map, humid_map, iterations=3
        )

        # 可视化：生态系统模拟后的气候
        if visualize and self.vis:
            self.vis.visualize_temperature_map(enhanced_temp_map, "生态系统模拟后温度", "生态系统")
            self.vis.visualize_humidity_map(enhanced_humid_map, "生态系统模拟后湿度", "生态系统")
        
        # 更新数据
        self.pipeline_state['temp_map'] = enhanced_temp_map
        self.pipeline_state['humid_map'] = enhanced_humid_map
        self.map_data.layers["temperature"] = enhanced_temp_map
        self.map_data.layers["humidity"] = enhanced_humid_map
        
        self.perf.end(task_id)
    
    def _handle_microclimate(self):
        """微气候生成步骤处理"""
        task_id = self.perf.start("microclimate")
        
        height_map = self.pipeline_state['height_map']
        temp_map = self.pipeline_state['temp_map']
        humid_map = self.pipeline_state['humid_map']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 添加微气候系统
        self.logger.log("生成微气候区域...")
        from core.generation.microclimates import generate_microclimates
        microclimate_map, micro_temp_map, micro_humid_map = generate_microclimates(
            height_map, temp_map, humid_map
        )

        # 可视化：微气候
        if visualize and self.vis:
            self.vis.visualize_microclimate(microclimate_map, height_map, "微气候区域", "生态系统")
        
        # 更新数据
        self.pipeline_state['temp_map'] = micro_temp_map
        self.pipeline_state['humid_map'] = micro_humid_map
        self.pipeline_state['microclimate_map'] = microclimate_map
        self.map_data.layers["temperature"] = micro_temp_map
        self.map_data.layers["humidity"] = micro_humid_map
        self.map_data.layers["microclimate"] = microclimate_map
        
        # 分析地形质量
        quality_metrics, suggestions = analyze_terrain_quality(height_map, micro_temp_map, micro_humid_map)
        for suggestion in suggestions:
            self.logger.log(f"地形建议: {suggestion}", "INFO")
        
        self.perf.end(task_id)
    
    def _handle_biomes(self):
        """生物群系分类步骤处理"""
        task_id = self.perf.start("biomes")
        
        height_map = self.pipeline_state['height_map']
        temp_map = self.pipeline_state['temp_map']
        humid_map = self.pipeline_state['humid_map']
        biome_data = self.pipeline_state['biome_data']
        width = self.pipeline_state['width']
        height = self.pipeline_state['height']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 分类生物群落
        self.logger.log("分配生物群落...")
        biome_map = classify_biome(height_map, temp_map, humid_map, biome_data)

        # 可视化：生物群系分类
        if visualize and self.vis:
            self.vis.visualize_biome_map(biome_map, biome_data, "生物群系分类", "生态分布")

        # 验证生物群系数据
        if not isinstance(biome_map, np.ndarray):
            raise ValueError("生物群系数据必须为NumPy数组")
        if biome_map.shape != (height, width):
            raise ValueError(f"生物群系数据维度错误: 预期({height}, {width}), 实际{biome_map.shape}")

        # 统一创建并填充生物群系图层
        if "biome" not in self.map_data.layers:
            self.map_data.create_layer("biome", dtype=np.int32, fill_value=0)

        # 修改此处以处理CuPy/NumPy兼容性问题
        try:
            # 检查是否是GPU数组
            if hasattr(self.map_data.layers["biome"], 'get') and hasattr(self.map_data.layers["biome"], 'device'):
                # 如果是GPU数组(CuPy)，需要使用CuPy函数进行转换
                self.logger.log("将GPU数组转换成CPU数组")
                import cupy as cp
                cp_biome_map = cp.asarray(biome_map, dtype=cp.int32)
                self.map_data.layers["biome"][:] = cp_biome_map
            else:
                # 如果是CPU数组(NumPy)，直接赋值
                self.logger.log("是CPU数组，直接赋值")
                self.map_data.layers["biome"][:] = biome_map.astype(np.int32)
        except Exception as e:
            self.logger.log(f"生物群系数据赋值失败: {e}", "ERROR")
            # 降级方案：逐个元素复制
            try:
                h, w = biome_map.shape
                for y in range(h):
                    for x in range(w):
                        self.map_data.layers["biome"][y, x] = int(biome_map[y, x])
                self.logger.log("使用逐元素复制方法完成生物群系数据赋值", "WARNING")
            except Exception as e2:
                self.logger.log(f"退化方案也失败: {e2}", "ERROR")
                raise
        
        # 更新数据
        self.pipeline_state['biome_map'] = biome_map
        
        self.perf.end(task_id)
    
    def _handle_biome_transitions(self):
        """生物群系过渡区处理"""
        task_id = self.perf.start("biome_transitions")
        
        biome_map = self.pipeline_state['biome_map']
        height_map = self.pipeline_state['height_map']
        temp_map = self.pipeline_state['temp_map']
        humid_map = self.pipeline_state['humid_map']
        biome_data = self.pipeline_state['biome_data']
        
        # 在生物群系分类后添加过渡区处理
        self.logger.log("创建生物群系过渡区...")
        from utils.biome_transitions import create_biome_transitions
        transition_biome_map = create_biome_transitions(
            biome_map, height_map, temp_map, humid_map, biome_data
        )
                
        # 使用带有过渡区的生物群系图替换原来的图
        try:
            # 检查是否是GPU数组
            if hasattr(self.map_data.layers["biome"], 'get') and hasattr(self.map_data.layers["biome"], 'device'):
                # 如果是GPU数组(CuPy)，需要使用CuPy函数进行转换
                self.logger.log("将GPU数组转换成CPU数组")
                import cupy as cp
                cp_transition_map = cp.asarray(transition_biome_map, dtype=cp.int32)
                self.map_data.layers["biome"][:] = cp_transition_map
            else:
                # 如果是CPU数组(NumPy)，直接赋值
                self.logger.log("是CPU数组，直接赋值")
                self.map_data.layers["biome"][:] = transition_biome_map.astype(np.int32)
        except Exception as e:
            self.logger.log(f"生物群系过渡区数据赋值失败: {e}", "ERROR")
            # 降级方案：逐个元素复制
            try:
                h, w = transition_biome_map.shape
                for y in range(h):
                    for x in range(w):
                        self.map_data.layers["biome"][y, x] = int(transition_biome_map[y, x])
                self.logger.log("使用逐元素复制方法完成生物群系过渡区赋值", "WARNING")
            except Exception as e2:
                self.logger.log(f"退化方案也失败: {e2}", "ERROR")
                raise
        
        # 更新数据
        self.pipeline_state['biome_map'] = transition_biome_map
        
        self.perf.end(task_id)
    
    def _handle_vegetation_buildings(self):
        """植被与建筑生成步骤处理"""
        task_id = self.perf.start("vegetation_buildings")
        
        biome_map = self.pipeline_state['biome_map']
        height_map = self.pipeline_state['height_map']
        rivers = self.pipeline_state['rivers_map']
        preferences = self.pipeline_state['preferences']
        map_params = self.pipeline_state['map_params']
        visualize = self.pipeline_state.get('visualize', False)
        
        # 放置植被和建筑
        self.logger.log("放置植被与建筑...")
        try:
            vegetation, buildings, roads, settlements = place_vegetation_and_buildings(
                biome_map, 
                height_map, 
                rivers, 
                preferences, 
                map_params
            )
        except Exception as e:
            self.logger.log(f"植被和建筑生成错误: {e}", "ERROR")
            # 提供默认值避免后续过程崩溃
            vegetation, buildings, roads, settlements = [], [], [], []
        
        # 存储对象层
        self.map_data.add_object_layer("vegetation", vegetation)
        self.map_data.add_object_layer("buildings", buildings)
        self.map_data.add_object_layer("settlements", settlements)
        self.map_data.add_object_layer("roads", roads)
        
        # 更新数据
        self.pipeline_state['vegetation'] = vegetation
        self.pipeline_state['buildings'] = buildings
        self.pipeline_state['settlements'] = settlements
        self.pipeline_state['roads'] = roads
        
        # 在建筑和植被可视化前添加
        self.logger.log(f"可视化前检查: vegetation={len(vegetation)}, buildings={len(buildings)}, roads={len(roads)}")
        # 可视化：植被和建筑
        if visualize and self.vis:
            self.vis.visualize_objects(height_map, vegetation, buildings, roads, "植被与建筑", "对象放置")
        
        self.perf.end(task_id)
    
    def _handle_content_layout(self):
        """内容布局生成步骤处理"""
        task_id = self.perf.start("content_layout")
        
        buildings = self.pipeline_state['buildings']
        
        # 内容布局生成
        self.logger.log("NEAT进化内容布局...")
        try:
            order, content_layout = run_neat_for_content_layout(buildings)
        except Exception as e:
            self.logger.log(f"内容布局生成错误: {e}", "ERROR")
            order, content_layout = [], {}
        
        # 更新数据
        self.pipeline_state['order'] = order
        self.pipeline_state['content_layout'] = content_layout
        
        self.perf.end(task_id)
    
    def _handle_road_network(self):
        """道路网络生成步骤处理"""
        task_id = self.perf.start("road_network")
        
        height_map = self.pipeline_state['height_map']
        buildings = self.pipeline_state['buildings']
        order = self.pipeline_state['order']
        rivers = self.pipeline_state['rivers_map']
        biome_map = self.pipeline_state['biome_map']
        settlements = self.pipeline_state['settlements']
        width = self.pipeline_state['width']
        height = self.pipeline_state['height']
        visualize = self.pipeline_state.get('visualize', False)
        
        self.logger.log(f"道路生成参数检查: buildings={len(buildings)}, order={len(order)}, rivers类型={type(rivers)}, rivers形状={rivers.shape if hasattr(rivers, 'shape') else 'N/A'}")
        if len(buildings) > 0:
            self.logger.log(f"建筑示例: {buildings[0]}")
        if settlements:
            self.logger.log(f"聚居点数量: {len(settlements)}")
        
        # 构建道路网络
        self.logger.log("构建智能道路网络...")
        try:
            roads_network, roads_types = build_roads(
                height_map, 
                buildings, 
                order,
                rivers,
                biome_map,
                settlements
            )
        except Exception as e:
            self.logger.log(f"道路网络生成错误: {e}", "ERROR")
            # 提供默认数组避免后续过程崩溃
            roads_network = np.zeros((height, width), dtype=np.uint8)
            roads_types = np.zeros((height, width), dtype=np.uint8)
        
        # 确保道路地图是数组
        roads_map = np.array(roads_network, dtype=np.uint8)

        # 创建并填充道路图层
        if "roads_map" not in self.map_data.layers:
            self.map_data.create_layer("roads_map", dtype=np.uint8)

        # 添加类型检查与转换
        try:
            # 检查是否是GPU数组
            if hasattr(self.map_data.layers["roads_map"], 'get') and hasattr(self.map_data.layers["roads_map"], 'device'):
                # 如果是GPU数组(CuPy)，需要使用CuPy函数进行转换
                self.logger.log("道路图层：将NumPy数组转换为CuPy数组")
                import cupy as cp
                cp_roads_map = cp.asarray(roads_map, dtype=cp.uint8)
                self.map_data.layers["roads_map"][:] = cp_roads_map
            else:
                # 如果是CPU数组(NumPy)，直接赋值
                self.logger.log("道路图层：使用NumPy数组直接赋值")
                self.map_data.layers["roads_map"][:] = roads_map
        except Exception as e:
            self.logger.log(f"道路图层数据赋值失败: {e}", "ERROR")
            # 降级方案：逐个元素复制
            try:
                h, w = roads_map.shape
                for y in range(h):
                    for x in range(w):
                        self.map_data.layers["roads_map"][y, x] = int(roads_map[y, x])
                self.logger.log("使用逐元素复制方法完成道路图层赋值", "WARNING")
            except Exception as e2:
                self.logger.log(f"道路图层降级方案也失败: {e2}", "ERROR")
                raise

        # 创建并填充道路类型图层 - 同样需要类型检查
        if "roads_types" not in self.map_data.layers:
            self.map_data.create_layer("roads_types", dtype=np.uint8)

        # 添加类型检查与转换
        try:
            # 检查是否是GPU数组
            if hasattr(self.map_data.layers["roads_types"], 'get') and hasattr(self.map_data.layers["roads_types"], 'device'):
                # 如果是GPU数组(CuPy)，需要使用CuPy函数进行转换
                self.logger.log("道路类型图层：将NumPy数组转换为CuPy数组")
                import cupy as cp
                cp_roads_types = cp.asarray(roads_types, dtype=cp.uint8)
                self.map_data.layers["roads_types"][:] = cp_roads_types
            else:
                # 如果是CPU数组(NumPy)，直接赋值
                self.logger.log("道路类型图层：使用NumPy数组直接赋值")
                self.map_data.layers["roads_types"][:] = np.array(roads_types, dtype=np.uint8)
        except Exception as e:
            self.logger.log(f"道路类型图层数据赋值失败: {e}", "ERROR")
            # 降级方案：逐个元素复制
            try:
                h, w = roads_types.shape
                for y in range(h):
                    for x in range(w):
                        self.map_data.layers["roads_types"][y, x] = int(roads_types[y, x])
                self.logger.log("使用逐元素复制方法完成道路类型图层赋值", "WARNING")
            except Exception as e2:
                self.logger.log(f"道路类型图层降级方案也失败: {e2}", "ERROR")
                raise
        
        # 更新数据
        self.pipeline_state['roads_map'] = roads_map
        self.pipeline_state['roads_types'] = roads_types
        
        # 可视化：道路网络
        if visualize and self.vis:
            road_points = []
            if roads_map is not None:
                for y in range(height):
                    for x in range(width):
                        if roads_map[y, x] > 0:
                            road_points.append({"x": x, "y": y})
            # 添加检查确保有足够数据显示
            if len(road_points) > 0:
                self.vis.visualize_objects(height_map, None, None, road_points, "道路网络", "交通系统")
            else:
                self.logger.log("警告: 道路网络为空，跳过可视化", "WARNING")
        
        self.perf.end(task_id)
    
    def _handle_story_generation(self):
        """故事生成步骤处理"""
        task_id = self.perf.start("story_generation")
        
        content_layout = self.pipeline_state['content_layout']
        biome_map = self.pipeline_state['biome_map']
        preferences = self.pipeline_state['preferences']
        llm = self.pipeline_state['llm']
        
        # 添加LLM内容与故事生成阶段
        self.logger.log("LLM处理内容与故事生成...")
        try:
            if not content_layout:
                content_layout = {"objects": []}
            elif "objects" not in content_layout:
                content_layout["objects"] = []

            # 调用LLM处理任务
            content_layout = llm.process_llm_tasks(
                content_layout,
                self.map_data.layers["biome"],
                preferences
            )

            # 将LLM生成的内容添加到地图数据中
            if "story_events" in content_layout:
                valid_events = []
                for i, event in enumerate(content_layout["story_events"]):
                    if not isinstance(event, dict):
                        self.logger.log(f"故事事件索引 {i} 不是字典类型: {event}", "ERROR")
                        continue
                    if "x" not in event or "y" not in event:
                        self.logger.log(f"故事事件索引 {i} 缺少x或y属性: {event}", "ERROR")
                        continue
                    try:
                        event["x"] = int(event["x"])
                        event["y"] = int(event["y"])
                        valid_events.append(event)
                        self.logger.log(f"验证通过的故事事件: x={event['x']}, y={event['y']}", "DEBUG")
                    except (ValueError, TypeError):
                        self.logger.log(f"故事事件索引 {i} 的坐标无效: {event}", "WARNING")

                if valid_events:
                    # 调用时使用skip_invalid参数
                    self.map_data.add_object_layer("story_events", valid_events, skip_invalid=True)
                    self.logger.log(f"LLM生成了{len(valid_events)}个故事事件点")
                else:
                    self.logger.log("没有有效的故事事件点可添加", "WARNING")

        except Exception as e:
            self.logger.log(f"LLM内容生成错误: {e}", "ERROR")
            if not content_layout:
                content_layout = {"objects": [], "story_events": []}
        
        # 更新数据
        self.pipeline_state['content_layout'] = content_layout
        
        self.perf.end(task_id)
    
    def _handle_interactive_evolution(self):
        """交互进化步骤处理"""
        task_id = self.perf.start("interactive_evolution")
        
        biome_map = self.pipeline_state['biome_map']
        height_map = self.pipeline_state['height_map']
        temp_map = self.pipeline_state['temp_map']
        humid_map = self.pipeline_state['humid_map']
        map_params = self.pipeline_state['map_params']
        parent_frame = self.pipeline_state.get('parent_frame')
        use_gui_editors = self.pipeline_state.get('use_gui_editors', False)
        callback = self.pipeline_state.get('callback')
        width = self.pipeline_state['width']
        height = self.pipeline_state['height']
        vegetation = self.pipeline_state.get('vegetation', [])
        buildings = self.pipeline_state.get('buildings', [])
        rivers = self.pipeline_state.get('rivers_map', None)
        roads = self.pipeline_state.get('roads', [])
        settlements = self.pipeline_state.get('settlements', [])
        content_layout = self.pipeline_state.get('content_layout', {})
        cave_entrances = self.pipeline_state.get('cave_entrances', [])
        roads_map = self.pipeline_state.get('roads_map')
        roads_types = self.pipeline_state.get('roads_types')
        
        self.logger.log("IEC(交互评价)...")
        try:
            # 初始化进化引擎
            engine = BiomeEvolutionEngine(
                biome_map=biome_map,
                height_map=height_map,
                temperature_map=temp_map,
                moisture_map=humid_map,
                memory_path="./data/models/evolution_memory.dat"
            )
            
            # 从map_params获取进化代数，不存在则默认为1
            generations = map_params.get("evolution_generations", 1)
            self.logger.log(f"将进行 {generations} 代进化...")
            
            # 检查是否支持交互式模式
            is_interactive = map_params.get("interactive_evolution", True)
            
            if is_interactive:
                if parent_frame and use_gui_editors:
                    # GUI集成模式 - 保存状态并暂停执行
                    self.logger.log("等待GUI界面完成评分...")
                    
                    # 保存当前流水线状态
                    completed_steps_str = [step.name for step in self.completed_steps]
                    
                    # 保存当前状态
                    state_data = {
                        'biome_map': biome_map,
                        'height_map': height_map,
                        'temp_map': temp_map,
                        'humid_map': humid_map,
                        'engine': engine,
                        'completed_gens': 0,  # 当前代数
                        'generations': generations,
                        'current_step': self.current_step.name,
                        'completed_steps': completed_steps_str,
                        'resume_point': 'post_evolution',  # 明确标记为进化后恢复点
                        'width': width,
                        'height': height,
                        'map_params': map_params,
                        # 保存所有需要的对象和层
                        'vegetation': vegetation,
                        'buildings': buildings,
                        'rivers': rivers,
                        'roads': roads,
                        'settlements': settlements,
                        'content_layout': content_layout,
                        'cave_entrances': cave_entrances,
                        'roads_map': roads_map,
                        'roads_types': roads_types
                    }
                    
                    # 设置挂起状态
                    self.map_data.pending_editor = "evolution_scorer"
                    self.map_data.editor_state = state_data
                    
                    # 通知上层应用程序
                    if callback:
                        callback({
                            'action': 'show_editor',
                            'editor_type': 'evolution_scorer',
                            'state': state_data
                        })
                    # 确保标记为未完成
                    self.map_data.generation_complete = False
                    
                    # 返回当前状态，中断执行
                    return self.map_data
                else:
                    # 独立窗口模式
                    try:
                        # 打开评分界面并等待用户输入
                        self.logger.log("正在打开独立评分界面，请在界面中选择您喜欢的方案...")
                        user_scores = get_visual_scores(engine)
                        self.logger.log(f"用户评分已完成: {len(user_scores)} 个方案被评分")
                    except Exception as score_error:
                        self.logger.log(f"评分界面出错: {score_error}，使用自动评分", "WARNING")
                        # 降级为自动评分
                        user_scores = [np.random.uniform(5, 8) for _ in range(engine.population_size)]
            else:
                # 自动评分
                self.logger.log("使用自动评分模式...")
                user_scores = [np.random.uniform(5, 8) for _ in range(engine.population_size)]
                
            # 执行进化过程
            self.logger.log("开始执行进化...")
            for gen in range(generations):
                self.logger.log(f"执行第 {gen+1}/{generations} 代进化")
                
                # 应用用户评分进行进化
                engine.evolve_generation(user_scores)
                
                # 如果不是最后一代，使用自动评分继续
                if gen < generations - 1:
                    # 自动评分，模拟用户偏好
                    user_scores = engine.auto_score_population()
            
            # 获取最终进化的生物群系地图
            evolved_biome_map = engine.best_individual
            
            # 更新数据
            if evolved_biome_map is not None and isinstance(evolved_biome_map, np.ndarray):
                self.logger.log("应用进化后的生物群系")
                self.pipeline_state['biome_map'] = evolved_biome_map
                
                # 更新map_data中的生物群系
                if "biome" in self.map_data.layers:
                    try:
                        # 检查是否是GPU数组
                        if hasattr(self.map_data.layers["biome"], 'get') and hasattr(self.map_data.layers["biome"], 'device'):
                            # 如果是GPU数组(CuPy)，需要使用CuPy函数进行转换
                            self.logger.log("将进化后的生物群系转换为GPU数组")
                            import cupy as cp
                            cp_evolved_biome = cp.asarray(evolved_biome_map, dtype=cp.int32)
                            self.map_data.layers["biome"][:] = cp_evolved_biome
                        else:
                            # 如果是CPU数组(NumPy)，直接赋值
                            self.logger.log("将进化后的生物群系应用到CPU数组")
                            self.map_data.layers["biome"][:] = evolved_biome_map.astype(np.int32)
                    except Exception as e:
                        self.logger.log(f"进化后生物群系应用失败: {e}", "ERROR")
                        try:
                            h, w = evolved_biome_map.shape
                            for y in range(h):
                                for x in range(w):
                                    self.map_data.layers["biome"][y, x] = int(evolved_biome_map[y, x])
                            self.logger.log("使用逐元素复制方法应用进化后的生物群系", "WARNING")
                        except Exception as e2:
                            self.logger.log(f"退化方案也失败: {e2}", "ERROR")
            else:
                self.logger.log("进化未产生有效的生物群系地图", "WARNING")
            
        except Exception as e:
            self.logger.log(f"交互进化过程出错: {e}", "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")
        
        self.perf.end(task_id)

    def _handle_emotion_analysis(self):
        """情感分析步骤处理"""
        task_id = self.perf.start("emotion_analysis")
        
        biome_map = self.pipeline_state.get('biome_map')
        height_map = self.pipeline_state.get('height_map')
        temp_map = self.pipeline_state.get('temp_map')
        humid_map = self.pipeline_state.get('humid_map')
        content_layout = self.pipeline_state.get('content_layout', {})
        
        self.logger.log("执行地图情感分析...")
        try:
            # 导入情感分析模块
            from core.services.analyze_map_emotions import MapEmotionAnalyzer
            
            # 执行情感分析
            emotion_results = MapEmotionAnalyzer.analyze_map_emotions(
                biome_map=biome_map,
                height_map=height_map,
                temp_map=temp_map,
                humid_map=humid_map,
                content=content_layout
            )
            
            # 记录情感分析结果
            if emotion_results:
                self.logger.log(f"情感分析结果: {emotion_results.get('summary', '无摘要')}")
                
                # 对主要情感进行评估
                if 'primary_emotions' in emotion_results:
                    primary_emotions = emotion_results['primary_emotions']
                    for region, emotion in primary_emotions.items():
                        self.logger.log(f"区域 {region}: 主要情感 - {emotion['name']}, 强度 - {emotion['intensity']:.2f}")
                
                # 保存情感分析结果到地图数据
                self.map_data.emotion_map = emotion_results
                
                # 如果有情感热图数据，保存到地图图层
                if 'emotion_heatmap' in emotion_results and isinstance(emotion_results['emotion_heatmap'], np.ndarray):
                    if "emotion_map" not in self.map_data.layers:
                        h, w = height_map.shape
                        self.map_data.create_layer("emotion_map", dtype=np.float32, shape=(h, w, 3))
                    self.map_data.layers["emotion_map"][:] = emotion_results['emotion_heatmap']
                    self.logger.log("情感热图已保存到地图数据")
            else:
                self.logger.log("情感分析未返回有效结果", "WARNING")
        
        except ImportError:
            self.logger.log("情感分析模块未找到，跳过此步骤", "WARNING")
        except Exception as e:
            self.logger.log(f"情感分析过程出错: {e}", "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")
        
        self.perf.end(task_id)

    def _handle_complete(self):
        """完成步骤处理"""
        task_id = self.perf.start("complete")
        
        self.logger.log("执行最终完成步骤...")
        
        # 标记地图生成已完成
        self.map_data.generation_complete = True
        
        # 计算和记录地图生成的各项统计信息
        stats = {
            'generation_time': self.perf.get_total_time(),
            'map_size': (self.pipeline_state['width'], self.pipeline_state['height']),
            'biome_counts': {},
            'object_counts': {}
        }
        
        # 统计生物群系分布
        if 'biome_map' in self.pipeline_state and self.pipeline_state['biome_map'] is not None:
            biome_map = self.pipeline_state['biome_map']
            unique, counts = np.unique(biome_map, return_counts=True)
            biome_counts = dict(zip(unique, counts))
            stats['biome_counts'] = biome_counts
            
            self.logger.log("生物群系分布统计:")
            biome_data = self.pipeline_state.get('biome_data', {})
            for biome_id, count in biome_counts.items():
                biome_name = biome_data.get(str(biome_id), {}).get('name', f'未知生物群系({biome_id})')
                percentage = (count / (biome_map.shape[0] * biome_map.shape[1])) * 100
                self.logger.log(f"  {biome_name}: {count}像素 ({percentage:.2f}%)")
        
        # 统计对象数量
        object_layers = ['vegetation', 'buildings', 'settlements', 'story_events']
        for layer in object_layers:
            if layer in self.map_data.objects:
                stats['object_counts'][layer] = len(self.map_data.objects[layer])
                self.logger.log(f"{layer}数量: {len(self.map_data.objects[layer])}")
        
        # 保存统计信息到地图数据
        self.map_data.generation_stats = stats
        
        # 清理临时数据，减少内存占用
        for key in list(self.pipeline_state.keys()):
            if key.endswith('_map') and key != 'height_map' and key != 'biome_map':
                del self.pipeline_state[key]
        
        self.logger.log("地图生成完成！")
        self.logger.log(f"总耗时: {stats['generation_time']:.2f}秒")
        
        self.perf.end(task_id)
        
    def _handle_underground_system(self):
        """地下系统生成步骤处理"""
        task_id = self.perf.start("underground_system")
        
        preferences = self.pipeline_state.get('preferences', {})
        callback = self.pipeline_state.get('callback')
        
        self.logger.log("开始生成地下系统...")
        
        # 配置地下系统参数
        underground_config = {
            "enable_underground": preferences.get("enable_underground", False),
            "underground_depth": preferences.get("underground_depth", 3),
            "underground_water_prevalence": preferences.get("underground_water_prevalence", 0.5),
            "underground_structure_density": preferences.get("underground_structure_density", 0.5),
            "common_minerals_ratio": preferences.get("common_minerals_ratio", 0.7),
            "rare_minerals_ratio": preferences.get("rare_minerals_ratio", 0.3),
            "cave_network_density": preferences.get("cave_network_density", 0.5),
            "special_structures_frequency": preferences.get("special_structures_frequency", 0.5),
            "underground_danger_level": preferences.get("underground_danger_level", 0.5),
            "seed": preferences.get("seed", None),
        }
        
        try:
            # 调用地下系统生成函数
            from core.generation.generate_undergroud import integrate_underground_to_map_data
            self.map_data = integrate_underground_to_map_data(self.map_data, underground_config, self.logger)
            
            self.logger.log("地下系统生成完成")
            
            # 更新进度
            if callback:
                callback({
                    "type": "pipeline_progress", 
                    "step": "underground_system", 
                    "message": "地下系统生成完成",
                    "progress": 0.45
                })
        except Exception as e:
            self.logger.log(f"地下系统生成错误: {e}", "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "DEBUG")
        
        self.perf.end(task_id)

    def _handle_level_generation(self):
        """关卡生成步骤处理"""
        task_id = self.perf.start("level_generation")
        
        self.logger.log("开始游戏关卡生成...")
        
        # 获取相关参数
        preferences = self.pipeline_state.get('preferences', {})
        level_type = preferences.get('level_type', 'balanced')
        difficulty = preferences.get('difficulty', 0.5)
        
        # 创建关卡生成器
        level_generator = LevelGenerator(self.map_data, preferences, self.logger)
        
        # 生成关卡
        level_data = level_generator.generate_level(level_type, difficulty)
        
        # 保存关卡数据到map_data
        self.map_data.level_data = level_data
        
        # 保存关卡数据到pipeline_state
        self.pipeline_state['level_data'] = level_data
        
        self.logger.log(f"成功生成{level_type}类型关卡，难度{difficulty}")
        self.perf.end(task_id)

if __name__ == "__main__":
    #matplotlib.font_manager._rebuild()新版已被移除
    shutil.rmtree(matplotlib.get_cachedir())
    root = tk.Tk()
    app = MapGeneratorApp(root)
    root.mainloop()
