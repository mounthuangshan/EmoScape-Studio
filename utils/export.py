#from __future__ import annotations
#标准库
import os
import sys
import time
import shutil
#import logging
import json
import random

#数据处理与科学计算
import numpy as np

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from PIL import Image
from PIL import Image, ImageDraw

#网络与并发

#其他工具
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable
import logging
from abc import ABC, abstractmethod
import colorsys
import gzip
###############
#导出模块
###############
# 设置日志
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#logger = logging.getLogger("MapExporter")

import os
import json
import numpy as np
import trimesh
from PIL import Image, ImageDraw
import math
import shutil
import time
import random
import gzip
import concurrent.futures
import colorsys
import logging
import matplotlib

# 添加3D模型配置字典（从enhanced_3d_export.py）
MODEL_LIBRARY = {
    "tree": {
        "model": "data/models/tree.obj",
        "scale": 2.0,
        "offset_y": 0.0
    },
    "pine": {
        "model": "data/models/pine.obj",
        "scale": 1.8,
        "offset_y": 0.0
    },
    "house": {
        "model": "data/models/house.obj",
        "scale": 3.0,
        "offset_y": 0.0
    },
    "cave": {
        "model": "data/models/cave_entrance.obj",
        "scale": 4.0,
        "offset_y": -1.0
    },
    "Predator": {
        "model": "data/models/predator.obj",
        "scale": 1.5,
        "offset_y": 1.0
    },
    "Prey": {
        "model": "data/models/prey.obj", 
        "scale": 1.0,
        "offset_y": 0.5
    }
}

# 材质配置（从enhanced_3d_export.py）
MATERIAL_LIBRARY = {
    "grass": {
        "diffuse": "data/textures/grass.png",
        "normal": "data/textures/grass_normal.png",
        "roughness": 0.7,
        "metallic": 0.0
    },
    "sand": {
        "diffuse": "data/textures/sand.png",
        "normal": "data/textures/sand_normal.png", 
        "roughness": 0.8,
        "metallic": 0.0
    },
    "rock": {
        "diffuse": "data/textures/rock.png",
        "normal": "data/textures/rock_normal.png",
        "roughness": 0.9, 
        "metallic": 0.1
    },
    "snow": {
        "diffuse": "data/textures/snow.png",
        "normal": "data/textures/snow_normal.png",
        "roughness": 0.3,
        "metallic": 0.0
    },
    "water": {
        "diffuse": "data/textures/water.png",
        "normal": "data/textures/water_normal.png",
        "roughness": 0.1,
        "metallic": 0.0,
        "transparent": True
    }
}

def ensure_asset_dirs(base_dir="./exports"):
    """确保资源目录存在"""
    dirs = [
        base_dir, 
        f"{base_dir}/models", 
        f"{base_dir}/textures", 
        f"{base_dir}/materials", 
        f"{base_dir}/unity", 
        f"{base_dir}/unreal"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def create_heightmap_texture(height_map, filename="exports/textures/heightmap.png"):
    """创建高度图纹理"""
    h = len(height_map)
    w = len(height_map[0])
    
    # 使用numpy处理高度图，便于处理NaN值
    height_array = np.array(height_map, dtype=np.float64)
    
    # 检查并替换NaN值
    mask_nan = np.isnan(height_array)
    if np.any(mask_nan):
        # 查找有效值的平均值作为替换值
        valid_values = height_array[~mask_nan]
        if len(valid_values) > 0:
            replacement_value = np.mean(valid_values)
        else:
            replacement_value = 0.0  # 如果全是NaN，使用0
        
        # 替换NaN值
        height_array[mask_nan] = replacement_value
    
    # 计算最小值和最大值（忽略任何剩余的NaN）
    min_height = np.nanmin(height_array)
    max_height = np.nanmax(height_array)
    range_height = max_height - min_height
    
    img = Image.new('L', (w, h))
    pixels = img.load()
    
    for j in range(h):
        for i in range(w):
            if range_height > 0:
                # 使用安全的计算方式避免NaN
                height_value = height_array[j][i]
                normalized = int(((height_value - min_height) / range_height) * 255)
                pixels[i, j] = normalized
            else:
                pixels[i, j] = 128  # 如果高度范围为0，使用中间灰度值
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    img.save(filename)
    return filename

def create_normal_map(height_map, filename="exports/textures/normal_map.png"):
    """从高度图生成法线贴图"""
    h = len(height_map)
    w = len(height_map[0])
    
    normal_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            # 计算梯度
            dx = height_map[y][x+1] - height_map[y][x-1]
            dy = height_map[y+1][x] - height_map[y-1][x]
            
            # 法向量
            normal = np.array([-dx, -dy, 2.0])
            normal = normal / np.sqrt(np.sum(normal**2) + 1e-10)  # 添加小值避免除零
            
            # 转换到0-255范围
            normal = ((normal + 1.0) * 0.5 * 255).astype(np.uint8)
            normal_map[y, x] = normal
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    img = Image.fromarray(normal_map)
    img.save(filename)
    return filename

def create_splat_map(biome_map, filename="exports/textures/splat_map.png"):
    """创建材质混合贴图"""
    h = len(biome_map)
    w = len(biome_map[0])
    
    # 收集所有生物群落类型
    biome_types = set()
    for row in biome_map:
        for biome in row:
            biome_types.add(biome["name"])
    
    # 为每种生物群落分配通道
    biome_channels = {}
    for i, biome_type in enumerate(biome_types):
        channel = i % 3  # 限制为RGB三个通道
        if channel not in biome_channels:
            biome_channels[channel] = []
        biome_channels[channel].append(biome_type)
    
    # 创建混合贴图
    splat = np.zeros((h, w, 3), dtype=np.uint8)
    
    for j in range(h):
        for i in range(w):
            biome_name = biome_map[j][i]["name"]
            for channel, biomes in biome_channels.items():
                if biome_name in biomes:
                    splat[j, i, channel] = 255
    
    img = Image.fromarray(splat)
    img.save(filename)
    
    # 记录生物群落到通道的映射
    mapping = {biome: channel for channel, biomes in biome_channels.items() for biome in biomes}
    
    return filename, mapping

def create_enhanced_terrain_mesh(height_map, filename="exports/models/terrain.obj"):
    """创建增强的地形网格，包含法线和UV"""
    h = len(height_map)
    w = len(height_map[0])
    
    vertices = []
    normals = []
    uvs = []
    faces = []
    
    # 生成顶点和UV
    for j in range(h):
        for i in range(w):
            vertices.append([i, j, height_map[j][i]])
            uvs.append([i/(w-1), j/(h-1)])
    
    # 计算法线
    for j in range(h):
        for i in range(w):
            nx, ny, nz = 0, 0, 1  # 默认法线朝上
            
            if i > 0 and i < w-1 and j > 0 and j < h-1:
                dx = height_map[j][i+1] - height_map[j][i-1]
                dy = height_map[j+1][i] - height_map[j-1][i]
                magnitude = math.sqrt(dx*dx + dy*dy + 4)
                nx, ny, nz = -dx/magnitude, -dy/magnitude, 2/magnitude
            
            normals.append([nx, ny, nz])
    
    # 生成面
    for j in range(h-1):
        for i in range(w-1):
            v1 = j*w + i
            v2 = j*w + (i+1)
            v3 = (j+1)*w + i
            v4 = (j+1)*w + (i+1)
            
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    # 创建trimesh对象
    mesh = trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        vertex_normals=np.array(normals),
        visual=trimesh.visual.TextureVisuals(uv=np.array(uvs))
    )
    
    # 导出为OBJ
    mesh.export(filename)
    return filename

def create_material_files(biome_mapping, sea_level, filename="exports/materials/terrain_materials.json"):
    """创建材质文件"""
    materials = {}
    
    # 为每种生物群落创建材质
    for biome_name, channel in biome_mapping.items():
        if biome_name == "Ocean":
            materials[biome_name] = MATERIAL_LIBRARY["water"]
        elif biome_name == "Beach" or biome_name == "Desert":
            materials[biome_name] = MATERIAL_LIBRARY["sand"]
        elif biome_name == "Mountain" or biome_name == "Volcano":
            materials[biome_name] = MATERIAL_LIBRARY["rock"]
        elif biome_name == "SnowPeak":
            materials[biome_name] = MATERIAL_LIBRARY["snow"]
        else:
            materials[biome_name] = MATERIAL_LIBRARY["grass"]
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 写入材质文件
    with open(filename, "w") as f:
        json.dump({
            "materials": materials,
            "biome_mapping": biome_mapping,
            "sea_level": sea_level
        }, f, indent=2)
    
    return filename

def place_3d_models(height_map, vegetation, buildings, caves, rivers, content_layout, output_path=None):
    """放置3D模型"""
    models = []
    
    # 添加日志计数器
    stats = {
        "vegetation": {"input": len(vegetation), "processed": 0},
        "buildings": {"input": len(buildings), "processed": 0},
        "caves": {"input": 0, "processed": 0},
        "creatures": {"input": 0, "processed": 0}
    }
    
    # 计算洞穴数量
    if isinstance(caves, list):
        stats["caves"]["input"] = len(caves)
    elif isinstance(caves, dict) and "caves" in caves:
        stats["caves"]["input"] = len(caves.get("caves", []))
        
    # 计算生物数量
    if "creatures" in content_layout:
        stats["creatures"]["input"] = len(content_layout["creatures"])
    
    print(f"开始放置3D模型 - 输入统计: "
          f"植被: {stats['vegetation']['input']}, "
          f"建筑: {stats['buildings']['input']}, "
          f"洞穴: {stats['caves']['input']}, "
          f"生物: {stats['creatures']['input']}")
    
    # 放置植被
    for veg in vegetation:
        # 确保x和y是整数
        if isinstance(veg, (list, tuple)) and len(veg) >= 3:
            x, y, veg_type = int(veg[0]), int(veg[1]), veg[2]
        elif isinstance(veg, dict) and "x" in veg and "y" in veg:
            x, y = int(veg["x"]), int(veg["y"])
            veg_type = veg.get("type", "tree")
        else:
            continue
            
        if veg_type not in MODEL_LIBRARY:
            veg_type = "tree"  # 默认使用树
        
        model_info = MODEL_LIBRARY[veg_type]
        
        print(f"放置模型: 类型={veg_type}, 位置=[{x}, {y}, {height_map[y][x] + model_info['offset_y']}]")
        
        # 确保坐标在有效范围内
        if 0 <= y < len(height_map) and 0 <= x < len(height_map[0]):
            models.append({
                "type": veg_type,
                "position": [x, y, height_map[y][x] + model_info["offset_y"]],
                "rotation": [0, np.random.uniform(0, 360), 0],
                "scale": model_info["scale"] * np.random.uniform(0.8, 1.2),
                "model_path": model_info["model"]
            })
            stats["vegetation"]["processed"] += 1
    
    print(f"植被放置完成 - 成功处理: {stats['vegetation']['processed']}/{stats['vegetation']['input']}")
    
    # 放置建筑
    for bld in buildings:
        if isinstance(bld, (list, tuple)) and len(bld) >= 3:
            x, y, bld_type = int(bld[0]), int(bld[1]), bld[2]
        elif isinstance(bld, dict) and "x" in bld and "y" in bld:
            x, y, bld_type = int(bld["x"]), int(bld["y"]), bld.get("type", "house")
        else:
            continue
            
        if bld_type not in MODEL_LIBRARY:
            bld_type = "house"  # 默认使用房子
        
        model_info = MODEL_LIBRARY[bld_type]
        
        print(f"放置模型: 类型={bld_type}, 位置=[{x}, {y}, {height_map[y][x] + model_info['offset_y']}]")
        
        # 确保坐标在有效范围内
        if 0 <= y < len(height_map) and 0 <= x < len(height_map[0]):
            models.append({
                "type": bld_type,
                "position": [x, y, height_map[y][x] + model_info["offset_y"]],
                "rotation": [0, np.random.uniform(0, 360), 0],
                "scale": model_info["scale"],
                "model_path": model_info["model"]
            })
            stats["buildings"]["processed"] += 1

    print(f"建筑放置完成 - 成功处理: {stats['buildings']['processed']}/{stats['buildings']['input']}")
    
    # 放置洞穴
    for cave in caves:
        if isinstance(cave, (list, tuple)) and len(cave) >= 2:
            cx, cy = int(cave[0]), int(cave[1])
        elif isinstance(cave, dict) and "x" in cave and "y" in cave:
            cx, cy = int(cave["x"]), int(cave["y"])
        else:
            continue
            
        model_info = MODEL_LIBRARY["cave"]
        
        #print(f"放置模型: 类型={cave}, 位置=[{x}, {y}, {height_map[y][x] + model_info['offset_y']}]")
        
        # 确保坐标在有效范围内
        if 0 <= cy < len(height_map) and 0 <= cx < len(height_map[0]):
            models.append({
                "type": "cave",
                "position": [cx, cy, height_map[cy][cx] + model_info["offset_y"]],
                "rotation": [0, np.random.uniform(0, 360), 0],
                "scale": model_info["scale"],
                "model_path": model_info["model"]
            })
            stats["caves"]["processed"] += 1
    
    print(f"洞穴放置完成 - 成功处理: {stats['caves']['processed']}/{stats['caves']['input']}")
    
    # 放置生物
    if "creatures" in content_layout:
        for creature in content_layout["creatures"]:
            try:
                x, y = int(creature["x"]), int(creature["y"])
                role = creature.get("role", "Prey")
                
                model_info = MODEL_LIBRARY.get(role, MODEL_LIBRARY["Prey"])
                
                #print(f"放置模型: 类型={veg_type}, 位置=[{x}, {y}, {height_map[y][x] + model_info['offset_y']}]")
                
                # 确保坐标在有效范围内
                if 0 <= y < len(height_map) and 0 <= x < len(height_map[0]):
                    models.append({
                        "type": role,
                        "position": [x, y, height_map[y][x] + model_info["offset_y"]],
                        "rotation": [0, np.random.uniform(0, 360), 0],
                        "scale": model_info["scale"],
                        "model_path": model_info["model"],
                        "attributes": creature.get("attributes", {})
                    })
                    stats["creatures"]["processed"] += 1
            except (KeyError, ValueError, TypeError, IndexError) as e:
                print(f"处理生物时出错: {e}, 跳过此生物")
                continue
    
    print(f"生物放置完成 - 成功处理: {stats['creatures']['processed']}/{stats['creatures']['input']}")
    
    # 使用传入的输出路径或默认路径
    if output_path is None:
        output_path = "exports/models/placed_models.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存模型信息 - 添加NumPy类型支持
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    # 保存模型数据时使用自定义编码器
    with open(output_path, "w") as f:
        json.dump(models, f, indent=2, cls=NumpyEncoder)
    
    # 打印最终统计信息
    print(f"3D模型放置完成 - 总计: {len(models)} 个模型")
    print(f"详细统计: 植被: {stats['vegetation']['processed']}/{stats['vegetation']['input']}, "
          f"建筑: {stats['buildings']['processed']}/{stats['buildings']['input']}, "
          f"洞穴: {stats['caves']['processed']}/{stats['caves']['input']}, "
          f"生物: {stats['creatures']['processed']}/{stats['creatures']['input']}")
    
    # 记录各类型模型统计
    model_types = {}
    for model in models:
        model_type = model["type"]
        model_types[model_type] = model_types.get(model_type, 0) + 1
    
    print(f"模型类型统计: {model_types}")
    
    return models

@dataclass
class MapExportConfig:
    """导出配置，支持高度定制化的导出参数"""
    output_dir: str = "./exports"
    base_filename: str = "game_map"
    texture_size: Tuple[int, int] = (2048, 2048)
    generate_textures: bool = True
    include_metadata: bool = True
    compress_output: bool = False
    export_normals: bool = True
    export_heightmap: bool = True
    level_of_detail: int = 1  # 1=全分辨率，2=半分辨率，以此类推
    memory_efficient: bool = True  # 使用流式处理减少内存占用
    multithreaded: bool = True
    max_workers: int = 4
    # 新增引擎专用参数
    unity_export_version: str = "2022.3"  # Unity目标版本
    unreal_export_version: str = "5.3"    # Unreal目标版本
    export_collision: bool = True         # 是否导出碰撞数据
    lightmap_uvs: bool = False            # 是否生成光照贴图UV
  
class JSONExporterBase:
    """JSON导出器基类"""
    
    def __init__(self, config: MapExportConfig = None,logger=None):
        self.logger=logger
        self.config = config or MapExportConfig()
        self._cached_data = None  # 用于内存高效模式
    
    def _prepare_export_data(self, map_data: Dict) -> Dict:
        """准备导出数据（模板方法）"""
        # 创建实例再调用方法，而不是直接通过类调用
        normalizer = MapDataNormalizer()
        normalized_data = normalizer.prepare_map_data(map_data)
        
        # 应用LOD
        if self.config.level_of_detail > 1:
            normalized_data = self._apply_lod(normalized_data)
        
        # 流式处理时不清除缓存
        if not self.config.memory_efficient:
            self._cached_data = normalized_data
        
        return normalized_data
    
    def _apply_lod(self, data: Dict) -> Dict:
        """应用细节层次"""
        lod_step = self.config.level_of_detail
        h = data["height"]
        w = data["width"]
        
        # 处理高度图
        data["height_map"] = [
            [row[i] for i in range(0, w, lod_step)]
            for row in data["height_map"][::lod_step]
        ]
        
        # 更新尺寸
        data["width"] = len(data["height_map"][0]) if data["height_map"] else 0
        data["height"] = len(data["height_map"])
        
        # 处理其他图层
        for layer in ["biome_map", "rivers"]:
            data[layer] = [
                [row[i] for i in range(0, w, lod_step)]
                for row in data[layer][::lod_step]
            ]
        
        # 处理实体坐标
        def adjust_coords(coords):
            return [(x//lod_step, y//lod_step) for x, y in coords]
        
        data["road_coords"] = adjust_coords(data["road_coords"])
        data["caves"] = [adjust_coords(cave) for cave in data["caves"]]
        
        return data
    
    def _generate_metadata(self, data: Dict) -> Dict:
        """生成元数据"""
        if not self.config.include_metadata:
            return {}
        
        return {
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": asdict(self.config),
            "stats": {
                "vertex_count": data["width"] * data["height"],
                "entity_counts": {
                    "vegetation": len(data["vegetation"]),
                    "buildings": len(data["buildings"]),
                    "caves": sum(len(c) for c in data["caves"])
                }
            }
        }
        
    def _validate_data(self, data: Dict) -> bool:
        """验证导出数据的有效性"""
        if not data:
            self.logger.error("导出数据为空")
            return False
            
        # 验证必要字段是否存在
        required_fields = ["width", "height", "height_map"]
        for field in required_fields:
            if field not in data:
                self.logger.error(f"导出数据缺少必要字段: {field}")
                return False
                
        # 验证尺寸
        if data["width"] <= 0 or data["height"] <= 0:
            self.logger.error(f"无效的地图尺寸: {data['width']}x{data['height']}")
            return False
            
        return True
    
    def _write_json_file(self, data: Dict, filename: str) -> bool:
        """将数据写入JSON文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 写入JSON文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"成功写入JSON文件: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"写入JSON文件失败: {str(e)}")
            return False
    
    def _generate_texture_path(self) -> str:
        """生成纹理文件路径"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.base_filename}_texture_{timestamp}.png"
        return os.path.join(self.config.output_dir, filename)
    
    def _process_biomes(self, biome_map: List[List[Dict]]) -> Dict:
        """处理生物群系数据"""
        biome_types = set()
        biome_counts = {}
        
        # 统计生物群系类型和数量
        for row in biome_map:
            for biome in row:
                biome_type = biome.get("name", "unknown")
                biome_types.add(biome_type)
                biome_counts[biome_type] = biome_counts.get(biome_type, 0) + 1
        
        return {
            "types": list(biome_types),
            "counts": biome_counts,
            "map": biome_map
        }
    
    def _process_entities(self, entities: List[Dict]) -> List[Dict]:
        """处理实体数据"""
        processed = []
        
        for entity in entities:
            # 创建副本以避免修改原数据
            processed_entity = entity.copy()
            
            # 添加元数据
            if "type" in entity:
                entity_type = entity["type"]
                processed_entity["prefab"] = f"Prefabs/{entity_type}"
                
            processed.append(processed_entity)
            
        return processed
    
    def _process_rivers(self, rivers: List[List[bool]]) -> Dict:
        """处理河流数据"""
        width = len(rivers[0]) if rivers and rivers[0] else 0
        height = len(rivers)
        
        # 提取河流坐标
        river_coords = []
        for y in range(height):
            for x in range(width):
                if rivers[y][x]:
                    river_coords.append([x, y])
        
        # 计算河流网络特性
        river_segments = self._extract_river_segments(river_coords)
        
        return {
            "coordinates": river_coords,
            "segments": river_segments,
            "width": width,
            "height": height
        }
    
    def _extract_river_segments(self, river_coords: List[List[int]]) -> List[Dict]:
        """提取河流分段"""
        # 实际项目中这会更复杂，例如寻找连续的河流段
        # 这里只是一个简化的实现
        return [{"start": [0, 0], "end": [10, 10], "width": 1.0}]
    
    def _process_caves(self, caves: List[Tuple[int, int]]) -> List[Dict]:
        """处理洞穴数据"""
        processed_caves = []
        
        for i, (x, y) in enumerate(caves):
            processed_caves.append({
                "id": i,
                "position": [x, y],
                "type": "cave_entrance",
                "size": random.uniform(1.0, 3.0)
            })
        
        return processed_caves
    
    def _process_roads(self, road_coords: List[Tuple[int, int]]) -> Dict:
        """处理道路数据"""
        road_types = ["dirt", "stone", "paved"]
        processed_roads = []
        
        # 简单处理：为每个坐标分配一个随机道路类型
        for x, y in road_coords:
            road_type = random.choice(road_types)
            processed_roads.append({
                "position": [x, y],
                "type": road_type,
                "width": 1.0 if road_type == "dirt" else (1.5 if road_type == "stone" else 2.0)
            })
        
        return {
            "coordinates": road_coords,
            "processed": processed_roads
        }
        
    def _apply_lod(self, data: Dict) -> Dict:
        """应用细节层次"""
        lod_step = self.config.level_of_detail
        h = len(data["height_map"])
        w = len(data["height_map"][0]) if h > 0 else 0
        
        # 处理高度图
        data["height_map"] = [
            [row[i] for i in range(0, len(row), lod_step)]
            for row in data["height_map"][::lod_step]
        ]
        
        # 更新尺寸
        data["width"] = len(data["height_map"][0]) if data["height_map"] else 0
        data["height"] = len(data["height_map"])
        
        # 处理其他图层
        if "biome_map" in data:
            data["biome_map"] = [
                [row[i] for i in range(0, len(row), lod_step)]
                for row in data["biome_map"][::lod_step]
            ]
        
        if "rivers" in data:
            data["rivers"] = [
                [row[i] for i in range(0, len(row), lod_step)]
                for row in data["rivers"][::lod_step]
            ]
        
        # 处理实体坐标
        def adjust_coords(coords):
            return [(x//lod_step, y//lod_step) for x, y in coords]
        
        if "road_coords" in data:
            data["road_coords"] = adjust_coords(data["road_coords"])
        
        if "caves" in data:
            data["caves"] = adjust_coords(data["caves"])
        
        return data

    def _generate_metadata(self, data: Dict) -> Dict:
        """生成元数据"""
        if not self.config.include_metadata:
            return {}
        
        metadata = {
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": asdict(self.config),
            "stats": {
                "map_size": {"width": data["width"], "height": data["height"]},
                "entity_counts": {}
            }
        }
        
        # 计算各类实体数量
        if "vegetation" in data:
            metadata["stats"]["entity_counts"]["vegetation"] = len(data["vegetation"])
        if "buildings" in data:
            metadata["stats"]["entity_counts"]["buildings"] = len(data["buildings"])
        if "caves" in data:
            metadata["stats"]["entity_counts"]["caves"] = len(data["caves"])
        if "road_coords" in data:
            metadata["stats"]["entity_counts"]["road_segments"] = len(data["road_coords"])
        
        # 添加生物群系统计
        if "biome_map" in data:
            biome_counts = {}
            for row in data["biome_map"]:
                for biome in row:
                    biome_name = biome["name"]
                    biome_counts[biome_name] = biome_counts.get(biome_name, 0) + 1
            metadata["stats"]["biomes"] = biome_counts
        
        return metadata

    def _validate_data(self, data: Dict) -> bool:
        """验证导出数据的有效性"""
        if not data:
            self.logger.error("导出数据为空")
            return False
            
        # 验证必要字段是否存在
        required_fields = ["width", "height", "height_map"]
        for field in required_fields:
            if field not in data:
                self.logger.error(f"导出数据缺少必要字段: {field}")
                return False
                
        # 验证尺寸
        if data["width"] <= 0 or data["height"] <= 0:
            self.logger.error(f"无效的地图尺寸: {data['width']}x{data['height']}")
            return False
            
        # 验证高度图
        if not data["height_map"] or len(data["height_map"]) != data["height"]:
            self.logger.error(f"高度图尺寸不匹配: 期望 {data['height']} 行，实际 {len(data['height_map'])} 行")
            return False
        
        if len(data["height_map"][0]) != data["width"]:
            self.logger.error(f"高度图尺寸不匹配: 期望 {data['width']} 列，实际 {len(data['height_map'][0])} 列")
            return False
            
        return True

    def _write_json_file(self, data: Dict, filename: str) -> bool:
        """将数据写入JSON文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 写入JSON文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"成功写入JSON文件: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"写入JSON文件失败: {str(e)}")
            return False

    def _generate_texture_path(self) -> str:
        """生成纹理文件路径"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.base_filename}_texture_{timestamp}.png"
        return os.path.join(self.config.output_dir, filename)
#########################################
#地图格式标准化
########################################
class MapDataNormalizer:
    """地图数据标准化器 - 处理各种输入格式并转换为统一标准"""
    
    @staticmethod
    def normalize_height_map(height_map: Any) -> List[List[float]]:
        """标准化高度图为二维浮点数组"""
        try:
            # 处理numpy数组
            if isinstance(height_map, np.ndarray):
                return height_map.astype(float).tolist()
            
            # 处理嵌套列表
            if isinstance(height_map, list):
                if not height_map:
                    return []
                
                # 确保每行长度一致
                h = len(height_map)
                w = max(len(row) if isinstance(row, (list, tuple)) else 0 for row in height_map)
                result = []
                
                for j in range(h):
                    row = height_map[j]
                    if not isinstance(row, (list, tuple)):
                        row = [float(row)]
                    
                    # 填充或截断至统一宽度
                    result.append([float(row[i]) if i < len(row) else 0.0 for i in range(w)])
                
                return result
            
            print(f"无法处理的高度图类型: {type(height_map)}")
            return [[0.0]]
            
        except Exception as e:
            print(f"高度图标准化失败: {e}")
            return [[0.0]]
    
    @staticmethod
    def normalize_biome_map(biome_map: Any) -> List[List[Dict[str, Any]]]:
        """标准化生物群系图为二维字典数组"""
        try:
            h = len(biome_map)
            if h == 0:
                return []
            
            # 确定宽度
            w = len(biome_map[0]) if isinstance(biome_map[0], (list, tuple)) else 1
            
            result = []
            for j in range(h):
                row = []
                for i in range(w):
                    try:
                        biome = biome_map[j][i]
                        if isinstance(biome, dict):
                            # 标准化颜色数组
                            if "color" in biome:
                                color = biome["color"]
                                if isinstance(color, np.ndarray):
                                    color = color.tolist()
                                # 确保颜色为RGB三元组
                                if len(color) >= 3:
                                    color = [int(c) for c in color[:3]]
                                else:
                                    color = [128, 128, 128]  # 默认灰色
                                biome["color"] = color
                            
                            # 确保名称为字符串
                            if "name" in biome:
                                biome["name"] = str(biome["name"])
                            else:
                                biome["name"] = "unknown"
                                
                            row.append(biome)
                        else:
                            # 如果不是字典，创建默认值
                            row.append({"name": "unknown", "color": [128, 128, 128]})
                    except IndexError:
                        row.append({"name": "unknown", "color": [128, 128, 128]})
                result.append(row)
            
            return result
        except Exception as e:
            print(f"生物群系标准化失败: {e}")
            return [[{"name": "error", "color": [255, 0, 0]}]]
    
    def normalize_boolean_grid(self, grid, h, w) -> List[List[bool]]:
        """将任意格式的网格数据标准化为布尔值二维数组"""
        try:
            # 处理空值
            if grid is None:
                return [[False for _ in range(w)] for _ in range(h)]
            
            # 处理numpy数组
            if isinstance(grid, np.ndarray):
                # 处理多维数组
                if grid.ndim > 2:
                    # 对于3D+数组，如果任一维度为True则视为True
                    grid = np.any(grid, axis=tuple(range(2, grid.ndim)))
                
                # 确保值是布尔型（有可能是0/1整数）
                grid = grid.astype(bool)
                
                # 调整尺寸
                if grid.shape != (h, w):
                    # 创建新数组，尽可能复制原始数据
                    new_grid = np.zeros((h, w), dtype=bool)
                    h_copy = min(h, grid.shape[0])
                    w_copy = min(w, grid.shape[1])
                    new_grid[:h_copy, :w_copy] = grid[:h_copy, :w_copy]
                    grid = new_grid
                
                # 转为Python列表
                return grid.tolist()
            
            # 处理嵌套列表
            if isinstance(grid, list):
                result = []
                
                # 根据输入列表决定遍历方式
                for j in range(min(h, len(grid))):
                    row = grid[j]
                    if isinstance(row, list):
                        # 处理二维列表
                        result.append([bool(row[i]) if i < len(row) else False 
                                    for i in range(w)])
                    else:
                        # 处理一维列表（视为单行）
                        result.append([bool(grid[j])] + [False] * (w - 1))
                
                # 填充缺失行
                while len(result) < h:
                    result.append([False] * w)
                
                return result
            
            # 处理其他类型（如标量值）
            return [[bool(grid) for _ in range(w)] for _ in range(h)]
            
        except Exception as e:
            print(f"布尔网格标准化失败: {e}")
            return [[False for _ in range(w)] for _ in range(h)]
    
    @staticmethod
    def normalize_entity_data(entities: List, required_fields: List[str] = None) -> List[Dict]:
        """标准化实体数据（如建筑、植被）"""
        if not required_fields:
            required_fields = ["x", "y", "type"]
            
        result = []
        
        if not entities:
            return result
            
        for entity in entities:
            # 跳过无效数据
            if not entity or len(entity) < len(required_fields):
                continue
                
            # 对元组类型进行转换
            if isinstance(entity, (tuple, list)):
                try:
                    entity_dict = {
                        required_fields[i]: int(entity[i]) if i < 2 else str(entity[i])
                        for i in range(min(len(entity), len(required_fields)))
                    }
                    
                    # 确保所有必需字段存在
                    for field in required_fields:
                        if field not in entity_dict:
                            if field in ["x", "y"]:
                                entity_dict[field] = 0
                            else:
                                entity_dict[field] = "unknown"
                    
                    result.append(entity_dict)
                except (TypeError, IndexError) as e:
                    print(f"实体数据转换错误: {e}, 实体: {entity}")
                    continue
            
            # 已经是字典格式
            elif isinstance(entity, dict):
                # 复制字典避免修改原数据
                entity_dict = {}
                
                # 处理必需字段
                for field in required_fields:
                    if field in entity:
                        # 类型转换
                        if field in ["x", "y"]:
                            entity_dict[field] = int(entity[field])
                        else:
                            entity_dict[field] = str(entity[field])
                    else:
                        # 默认值
                        if field in ["x", "y"]:
                            entity_dict[field] = 0
                        else:
                            entity_dict[field] = "unknown"
                            
                # 复制其他字段
                for k, v in entity.items():
                    if k not in required_fields:
                        entity_dict[k] = v
                        
                result.append(entity_dict)
                
        return result
    
    @staticmethod
    def normalize_content_layout(content_layout: Dict) -> Dict:
        """标准化内容布局数据"""
        if not content_layout or not isinstance(content_layout, dict):
            return {}
            
        result = {}
        
        # 标准化故事事件
        if "story_events" in content_layout:
            events = []
            for event in content_layout["story_events"]:
                if isinstance(event, dict):
                    # 处理事件对象
                    event_dict = {
                        "x": int(event.get("x", 0)),
                        "y": int(event.get("y", 0)),
                        "description": str(event.get("description", ""))
                    }
                    
                    # 处理触发器
                    if "trigger" in event and event["trigger"]:
                        try:
                            trigger = event["trigger"]
                            if isinstance(trigger, (list, tuple, np.ndarray)) and len(trigger) >= 2:
                                # 从numpy数组提取值
                                if hasattr(trigger[0], 'item'):
                                    x = int(trigger[0].item())
                                else:
                                    x = int(trigger[0])
                                    
                                if hasattr(trigger[1], 'item'):
                                    y = int(trigger[1].item())
                                else:
                                    y = int(trigger[1])
                                    
                                event_dict["trigger"] = [x, y]
                            else:
                                event_dict["trigger"] = [0, 0]
                        except (IndexError, TypeError, ValueError):
                            event_dict["trigger"] = [0, 0]
                    
                    events.append(event_dict)
                    
            result["story_events"] = events
        
        # 标准化生物数据
        if "creatures" in content_layout:
            creatures = []
            for creature in content_layout["creatures"]:
                if isinstance(creature, dict):
                    # 处理生物属性
                    creature_dict = {
                        "x": int(creature.get("x", 0)),
                        "y": int(creature.get("y", 0)),
                        "role": str(creature.get("role", "unknown")),
                    }
                    
                    # 处理属性字典
                    if "attributes" in creature:
                        attributes = {}
                        for k, v in creature["attributes"].items():
                            # 转换numpy类型
                            if isinstance(v, np.number):
                                attributes[k] = float(v)
                            else:
                                attributes[k] = v
                        creature_dict["attributes"] = attributes
                        
                    creatures.append(creature_dict)
                    
            result["creatures"] = creatures
        
        # 复制其他简单字段
        for k, v in content_layout.items():
            if k not in result and k not in ["story_events", "creatures"]:
                if isinstance(v, (str, int, float, bool)):
                    result[k] = v
                elif isinstance(v, (np.integer, np.floating)):
                    result[k] = v.item()  # 转换numpy标量
                
        return result
    
    @staticmethod
    def extract_road_coords(roads, w, h) -> List[Tuple[int, int]]:
        """从各种格式提取道路坐标"""
        road_coords = []
        
        # 处理坐标列表
        if isinstance(roads, list):
            for point in roads:
                if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                    # 从各种类型提取坐标
                    try:
                        x = int(point[0])
                        y = int(point[1])
                        # 确保坐标在地图范围内
                        if 0 <= x < w and 0 <= y < h:
                            road_coords.append((x, y))
                    except (IndexError, TypeError, ValueError):
                        continue
        # 处理布尔网格
        elif isinstance(roads, np.ndarray) and roads.ndim == 2:
            h_arr, w_arr = roads.shape
            for j in range(min(h_arr, h)):
                for i in range(min(w_arr, w)):
                    if roads[j, i]:
                        road_coords.append((i, j))
        
        return road_coords
    
    def extract_cave_coords(self, caves, w, h) -> List[Tuple[int, int]]:
        """从洞穴数据提取有效坐标"""
        cave_coords = []
        
        if not caves:
            return cave_coords
        
        # 处理新格式：字典格式的洞穴数据 {"caves": [...], "entrances": [...]}
        if isinstance(caves, dict) and "caves" in caves:
            for point in caves.get("caves", []):
                if isinstance(point, dict) and "x" in point and "y" in point:
                    x, y = int(point["x"]), int(point["y"])
                    if 0 <= x < w and 0 <= y < h:
                        cave_coords.append((x, y))
            return cave_coords
                
        # 处理各种可能的洞穴数据格式（旧格式兼容）
        for cave in caves:
            # 处理numpy数组类型
            if isinstance(cave, np.ndarray):
                if cave.ndim == 2 and cave.shape[1] >= 2:
                    # 二维坐标数组
                    for i in range(cave.shape[0]):
                        try:
                            x, y = int(cave[i, 0]), int(cave[i, 1])
                            if 0 <= x < w and 0 <= y < h:
                                cave_coords.append((x, y))
                        except (IndexError, TypeError):
                            continue
                elif cave.ndim == 1 and cave.size >= 2:
                    # 一维坐标数组
                    try:
                        x, y = int(cave[0]), int(cave[1])
                        if 0 <= x < w and 0 <= y < h:
                            cave_coords.append((x, y))
                    except (IndexError, TypeError):
                        continue
            
            # 处理嵌套列表
            elif isinstance(cave, list):
                # 检查是否是点列表
                if all(isinstance(point, (list, tuple, np.ndarray)) for point in cave):
                    for point in cave:
                        try:
                            if len(point) >= 2:
                                x, y = int(point[0]), int(point[1])
                                if 0 <= x < w and 0 <= y < h:
                                    cave_coords.append((x, y))
                        except (IndexError, TypeError):
                            continue
                # 单个坐标点
                elif len(cave) >= 2:
                    try:
                        x, y = int(cave[0]), int(cave[1])
                        if 0 <= x < w and 0 <= y < h:
                            cave_coords.append((x, y))
                    except (IndexError, TypeError):
                        continue
                        
        return cave_coords

    def process_road_types(self, road_network, road_types, w, h):
        """处理道路网络和道路类型数据"""
        # 确保有效的数据
        if road_types is None or not isinstance(road_types, np.ndarray):
            # 如果没有道路类型数据，则创建默认值（全部为1）
            processed_types = np.ones((h, w), dtype=int)
        else:
            # 复制输入，确保不会修改原始数据
            processed_types = road_types.copy()
            
            # 调整尺寸
            if processed_types.shape != (h, w):
                temp_types = np.ones((h, w), dtype=int)
                h_copy = min(h, processed_types.shape[0])
                w_copy = min(w, processed_types.shape[1])
                temp_types[:h_copy, :w_copy] = processed_types[:h_copy, :w_copy]
                processed_types = temp_types
        
        # 将布尔道路网络与道路类型结合
        normalized_network = self.normalize_boolean_grid(road_network, h, w)
        
        # 将道路类型转换为列表
        road_type_list = processed_types.tolist()
        
        # 确保只有在道路存在的地方才有道路类型
        for y in range(h):
            for x in range(w):
                if not normalized_network[y][x]:
                    road_type_list[y][x] = 0
        
        return {
            "network": normalized_network,
            "types": road_type_list
        }

    def prepare_map_data(self, map_data):
        """标准化地图数据包（仅修改此方法）"""
        # 修改点1：解包逻辑兼容旧版和新版元组结构
        if isinstance(map_data, tuple):
            # 处理旧版元组结构（最后一个元素是包含 roads_map 和 roads_types 的元组）
            if len(map_data) == 11:
                (height_map, biome_map, vegetation, buildings, rivers, 
                 content_layout, caves, map_params, biome_data, roads, 
                 roads_data) = map_data
                roads_map, roads_types = roads_data if roads_data else (None, None)
            # 处理新版元组结构（直接包含 roads_map 和 roads_types）
            elif len(map_data) >= 12:
                (height_map, biome_map, vegetation, buildings, rivers, 
                 content_layout, caves, map_params, biome_data, roads, 
                 roads_map, roads_types) = map_data[:12]  # 截取前12个元素
            else:
                raise ValueError("无效的输入元组结构")
        elif hasattr(map_data, 'unpack'):
            # 处理 MapData.unpack() 返回的元组（无论结构如何）
            unpacked = map_data.unpack()
            if len(unpacked) == 11:
                roads_map, roads_types = unpacked[-1]  # 旧版结构，解包嵌套元组
                (height_map, biome_map, vegetation, buildings, rivers, 
                 content_layout, caves, map_params, biome_data, roads) = unpacked[:-1]
            else:
                (height_map, biome_map, vegetation, buildings, rivers, 
                 content_layout, caves, map_params, biome_data, roads, 
                 roads_map, roads_types) = unpacked[:12]  # 新版结构直接解包
        else:
            # 字典格式处理保持不变
            height_map = map_data.get("height_map", [])
            biome_map = map_data.get("biome_map", [])
            vegetation = map_data.get("vegetation", [])
            buildings = map_data.get("buildings", [])
            rivers = map_data.get("rivers", [])
            content_layout = map_data.get("content_layout", {})
            caves = map_data.get("caves", [])
            map_params = map_data.get("params", {})
            biome_data = map_data.get("biome_data", None)
            roads = map_data.get("roads", [])
            roads_map = map_data.get("roads_map", None)
            roads_types = map_data.get("roads_types", None)

        # 计算尺寸
        if isinstance(height_map, np.ndarray):
            h, w = height_map.shape
        elif hasattr(height_map, 'shape'):
            h, w = height_map.shape[0], height_map.shape[1]
        else:
            h = len(height_map)
            w = len(height_map[0]) if h > 0 else 0

        # 添加尺寸验证
        if w <= 0 or h <= 0:
            print(f"无效的地图尺寸: width={w}, height={h}，请检查输入的height_map")
            raise ValueError("地图尺寸无效，width或height为0")

        return {
            "width": w,
            "height": h,
            "height_map": self.normalize_height_map(height_map),
            "biome_map": self.normalize_biome_map(biome_map),
            "vegetation": self.normalize_entity_data(vegetation),
            "buildings": self.normalize_entity_data(buildings),
            "rivers": self.normalize_boolean_grid(rivers, h, w),
            "content_layout": self.normalize_content_layout(content_layout),
            "caves": self.extract_cave_coords(caves, w, h),
            "road_coords": self.extract_road_coords(roads, w, h),
            "roads_data": self.process_road_types(roads_map, roads_types, w, h) if roads_map else None,
            "map_params": map_params,
            "biome_data": biome_data
        }
###########################################
#定义地图导出器接口
###########################################
class MapExporter(ABC):
    """抽象基类，定义地图导出器接口"""
    
    def __init__(self, config: MapExportConfig = None,logger=None):
        self.logger=logger
        self.config = config or MapExportConfig()
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _prepare_map_data_dict(self, map_data):
        """将 MapData 对象转换为字典""" 
        if hasattr(map_data, "layers"):  # 直接检查是否为MapData对象
            width = getattr(map_data, "map_width", None)
            height = getattr(map_data, "map_height", None)
            
            # 如果width或height缺失，尝试从height_map推断
            height_map = map_data.get_layer("height")
            if (width is None or height is None) and height_map is not None:
                if isinstance(height_map, np.ndarray):
                    print("numpy数组")
                    height, width = height_map.shape
                elif isinstance(height_map, list) and len(height_map) > 0:
                    print("普通数组")
                    height = len(height_map)
                    width = len(height_map[0]) if height > 0 else 0
            
            if width is None or height is None or width <= 0 or height <= 0:
                # 添加兼容性检查
                if hasattr(self.logger, 'log') and not hasattr(self.logger, 'info'):
                    # 使用log方法代替info
                    self.logger.log(f"从height_map推断地图尺寸: {w}x{h}")
                else:
                    # 使用标准info方法
                    self.logger.info(f"从height_map推断地图尺寸: {w}x{h}")
                raise ValueError("无法确定地图尺寸")
            print("转换成字典")        
            return {
                "width": width,
                "height": height,
                "height_map": height_map,
                "biome_map": map_data.get_layer("biome"),
                "vegetation": map_data.layers.get("vegetation", []),
                "buildings": map_data.layers.get("buildings", []),
                "rivers": map_data.get_layer("rivers"),
                "content_layout": getattr(map_data, "content_layout", {}),
                "caves": map_data.layers.get("caves", []),
                "map_params": getattr(map_data, "params", {}),
                "biome_data": getattr(map_data, "biome_data", None),
                "road_coords": map_data.layers.get("roads", []),
                "roads_map": (map_data.get_layer("roads_map"), map_data.get_layer("roads_types"))
            }
        print("已经是字典")
        return map_data  # 如果已经是字典，直接返回
    
    @abstractmethod
    def export(self, map_data: Dict, filename: str = None) -> str:
        """导出地图，返回生成的文件路径"""
        pass
    
    def _get_output_path(self, filename_base: str, extension: str) -> str:
        """构造输出文件路径"""
        if not filename_base:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_base = f"{self.config.base_filename}_{timestamp}"
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        return os.path.join(self.config.output_dir, f"{filename_base}.{extension}")
        
    def _generate_texture_image(self, map_data: Dict) -> Image.Image:
        """生成地图纹理贴图，与地图预览显示保持一致的渲染"""
        try:
            # 提取地图数据
            width, height = map_data.get("width", 0), map_data.get("height", 0)
            height_map = map_data.get("height_map", [])
            biome_map = map_data.get("biome_map", [])
            vegetation = map_data.get("vegetation", [])
            buildings = map_data.get("buildings", [])
            rivers = map_data.get("rivers", [])
            caves = map_data.get("caves", [])
            road_coords = map_data.get("road_coords", [])
            content_layout = map_data.get("content_layout", {})
            
            # 获取道路网络数据
            roads_map = map_data.get("roads_data", None)
            road_net, road_types = None, None
            if roads_map:
                road_net = roads_map.get("network", None)
                road_types = roads_map.get("types", None)
            
            # 验证地图尺寸
            if width <= 0 or height <= 0:
                print(f"无效的地图尺寸: {width}x{height}")
                return Image.new('RGB', (256, 256), (255, 0, 0))
            
            # 创建纹理图像尺寸
            texture_w, texture_h = self.config.texture_size
            
            # 计算缩放因子
            scale_x = texture_w / width
            scale_y = texture_h / height
                
            # 初始化布尔掩码
            river_mask = np.zeros((height, width), dtype=bool)
            cave_mask = np.zeros((height, width), dtype=bool)
            veg_mask = np.zeros((height, width), dtype=bool)
            bld_mask = np.zeros((height, width), dtype=bool)
            story_mask = np.zeros((height, width), dtype=bool)
            creature_mask = np.zeros((height, width), dtype=bool)
            
            # 道路系统掩码
            settlement_roads_mask = np.zeros((height, width), dtype=bool)
            building_main_road = np.zeros((height, width), dtype=bool)  
            building_secondary = np.zeros((height, width), dtype=bool)
            building_paths = np.zeros((height, width), dtype=bool)
            
            # 处理河流数据
            if isinstance(rivers, np.ndarray):
                if rivers.shape == (height, width):
                    river_mask = rivers > 0
                elif rivers.shape[:2] == (height, width):
                    river_mask = np.any(rivers > 0, axis=-1) if rivers.ndim > 2 else rivers > 0
            elif isinstance(rivers, list) and len(rivers) > 0:
                for j in range(min(len(rivers), height)):
                    for i in range(min(len(rivers[j]), width)):
                        if isinstance(rivers[j][i], np.ndarray):
                            river_mask[j, i] = np.any(rivers[j][i])
                        else:
                            river_mask[j, i] = bool(rivers[j][i])
            
            # 处理洞穴数据
            if isinstance(caves, dict) and "caves" in caves:
                for point in caves.get("caves", []):
                    if isinstance(point, dict) and "x" in point and "y" in point:
                        x, y = int(point["x"]), int(point["y"])
                        if 0 <= x < width and 0 <= y < height:
                            cave_mask[y, x] = True
            elif isinstance(caves, list):
                for item in caves:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        x, y = int(item[0]), int(item[1])
                        if 0 <= x < width and 0 <= y < height:
                            cave_mask[y, x] = True
                    elif isinstance(item, dict) and "x" in item and "y" in item:
                        x, y = int(item["x"]), int(item["y"])
                        if 0 <= x < width and 0 <= y < height:
                            cave_mask[y, x] = True
            
            # 处理植被数据
            for veg in vegetation:
                if isinstance(veg, (list, tuple)) and len(veg) >= 2:
                    x, y = int(veg[0]), int(veg[1])
                    if 0 <= x < width and 0 <= y < height:
                        veg_mask[y, x] = True
                elif isinstance(veg, dict) and "x" in veg and "y" in veg:
                    x, y = int(veg["x"]), int(veg["y"])
                    if 0 <= x < width and 0 <= y < height:
                        veg_mask[y, x] = True
            
            # 处理建筑数据
            for building in buildings:
                if isinstance(building, (list, tuple)) and len(building) >= 2:
                    x, y = int(building[0]), int(building[1])
                    if 0 <= x < width and 0 <= y < height:
                        bld_mask[y, x] = True
                elif isinstance(building, dict) and "x" in building and "y" in building:
                    x, y = int(building["x"]), int(building["y"])
                    if 0 <= x < width and 0 <= y < height:
                        bld_mask[y, x] = True
            
            # 处理道路数据
            for road in road_coords:
                if isinstance(road, (list, tuple)) and len(road) >= 2:
                    x, y = int(road[0]), int(road[1])
                    if 0 <= x < width and 0 <= y < height:
                        settlement_roads_mask[y, x] = True
                elif isinstance(road, dict) and "x" in road and "y" in road:
                    x, y = int(road["x"]), int(road["y"])
                    if 0 <= x < width and 0 <= y < height:
                        settlement_roads_mask[y, x] = True
            
            # 处理道路网络
            if road_net is not None and road_types is not None:
                if len(road_net) == height and len(road_net[0]) == width and len(road_types) == height and len(road_types[0]) == width:
                    for y in range(height):
                        for x in range(width):
                            if road_net[y][x]:
                                road_type = road_types[y][x]
                                if road_type == 1:
                                    building_main_road[y, x] = True
                                elif road_type == 2:
                                    building_secondary[y, x] = True
                                elif road_type == 3:
                                    building_paths[y, x] = True
            
            # 处理故事点和生物
            if isinstance(content_layout, dict):
                if "story_events" in content_layout:
                    for event in content_layout["story_events"]:
                        if isinstance(event, dict) and "x" in event and "y" in event:
                            x, y = int(event["x"]), int(event["y"])
                            if 0 <= x < width and 0 <= y < height:
                                story_mask[y, x] = True
                
                if "creatures" in content_layout:
                    for creature in content_layout["creatures"]:
                        if isinstance(creature, dict) and "x" in creature and "y" in creature:
                            x, y = int(creature["x"]), int(creature["y"])
                            if 0 <= x < width and 0 <= y < height:
                                creature_mask[y, x] = True
            
            # 尝试加载生物群系配置
            biome_id_to_color = {}
            try:
                import json
                import os
                biome_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                            "data", "configs", "biomes_config.json")
                if os.path.exists(biome_config_path):
                    with open(biome_config_path, "r", encoding="utf-8") as f:
                        biomes_config = json.load(f)
                    
                    # 创建生物群系ID到颜色的映射
                    for idx, biome in enumerate(biomes_config.get("biomes", [])):
                        color = biome.get("color", [0.5, 0.5, 0.5])
                        # 转换到0-255范围
                        if any(c <= 1.0 for c in color[:3]):
                            color = [int(c * 255) for c in color[:3]]
                        biome_id_to_color[idx] = color
                        biome_id_to_color[biome.get("name", f"unknown_{idx}")] = color
            except Exception as e:
                print(f"无法加载生物群系配置: {e}")
                # 使用备用颜色
                biome_id_to_color = {}
            
            # 生成生物群系颜色
            biome_colors = np.ones((height, width, 3), dtype=np.uint8) * 128  # 默认灰色
            
            # 应用生物群系颜色
            for j in range(height):
                for i in range(width):
                    if j < len(biome_map) and i < len(biome_map[j]):
                        biome = biome_map[j][i]
                        color = [128, 128, 128]  # 默认灰色
                        
                        # 处理不同类型的生物群系数据
                        if isinstance(biome, dict):
                            if "color" in biome:
                                color_data = biome["color"]
                                if isinstance(color_data, (list, tuple, np.ndarray)) and len(color_data) >= 3:
                                    color = color_data[:3]
                                    # 确保颜色值在0-255范围内
                                    if any(c <= 1.0 for c in color):
                                        color = [int(c * 255) for c in color]
                            elif "name" in biome and biome["name"] in biome_id_to_color:
                                color = biome_id_to_color[biome["name"]]
                        elif isinstance(biome, (int, np.integer)):
                            # 尝试从配置中获取颜色
                            if biome in biome_id_to_color:
                                color = biome_id_to_color[biome]
                            else:
                                # 为未知ID生成一个颜色
                                backup_colors = [
                                    [204, 51, 51],   # 红色
                                    [51, 204, 51],   # 绿色
                                    [51, 51, 204],   # 蓝色
                                    [204, 204, 51],  # 黄色
                                    [204, 51, 204],  # 紫色
                                    [51, 204, 204],  # 青色
                                ]
                                color = backup_colors[biome % len(backup_colors)]
                        elif isinstance(biome, str) and biome in biome_id_to_color:
                            color = biome_id_to_color[biome]
                        
                        biome_colors[j, i] = color
            
            # 尝试应用地形光照
            try:
                height_np = np.array(height_map, dtype=np.float32)
                from matplotlib.colors import LightSource
                ls = LightSource(azdeg=315, altdeg=45)
                
                # 将颜色转换到0-1范围
                biome_colors_float = biome_colors.astype(np.float32) / 255.0
                
                # 应用光照
                illuminated = ls.shade_rgb(
                    biome_colors_float, 
                    height_np,
                    blend_mode='soft',
                    fraction=0.6,
                    vert_exag=0.3
                )
                
                # 转回0-255范围
                biome_colors = (illuminated * 255).astype(np.uint8)
            except Exception as e:
                print(f"应用地形光照失败: {e}")
                # 继续使用未照明的颜色
            
            # 定义特征颜色 (RGBA格式, 0-255范围)
            COLORS = {
                'river': (0, 128, 255, 255),          # 河流蓝色
                'cave': (40, 40, 60, 255),            # 洞穴深灰色
                'settlement_road': (230, 180, 80, 255),  # 聚落道路橙色
                'main_road': (150, 50, 50, 255),      # 主干道深红色
                'secondary': (180, 150, 130, 255),    # 次干道土黄色
                'path': (150, 150, 150, 255),         # 小路灰色
                'vegetation': (100, 180, 100, 255),   # 植被绿色
                'building': (200, 100, 100, 255),     # 建筑红棕色
                'story': (200, 80, 200, 255),         # 故事点紫色
                'creature': (230, 230, 25, 255)       # 生物黄色
            }
            
            # 创建RGBA图像
            texture_array = np.zeros((texture_h, texture_w, 4), dtype=np.uint8)
            
            # 创建坐标映射
            y_indices, x_indices = np.mgrid[0:texture_h, 0:texture_w]
            map_x = np.floor(x_indices / scale_x).astype(int)
            map_y = np.floor(y_indices / scale_y).astype(int)
            
            # 限制坐标范围
            valid_mask = (map_x < width) & (map_y < height) & (map_x >= 0) & (map_y >= 0)
            
            # 填充基础地形颜色
            for j in range(texture_h):
                for i in range(texture_w):
                    if valid_mask[j, i]:
                        map_j, map_i = map_y[j, i], map_x[j, i]
                        # 设置RGB颜色和不透明度
                        texture_array[j, i, :3] = biome_colors[map_j, map_i]
                        texture_array[j, i, 3] = 255
                        
                        # 按照优先级应用特征颜色
                        if river_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['river']
                        
                        # 依次应用各种道路类型（优先级递增）
                        if building_paths[map_j, map_i]:
                            texture_array[j, i] = COLORS['path']
                        if building_secondary[map_j, map_i]:
                            texture_array[j, i] = COLORS['secondary']
                        if building_main_road[map_j, map_i]:
                            texture_array[j, i] = COLORS['main_road']
                        if settlement_roads_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['settlement_road']
                        
                        # 应用其他特征（优先级递增）
                        if veg_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['vegetation']
                        if cave_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['cave']
                        if bld_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['building']
                        if story_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['story']
                        if creature_mask[map_j, map_i]:
                            texture_array[j, i] = COLORS['creature']
            
            # 创建PIL图像
            img = Image.fromarray(texture_array, 'RGBA')
            
            # 应用轻微的平滑过滤器以改善视觉效果
            from PIL import ImageFilter
            img = img.filter(ImageFilter.SMOOTH)
            
            return img
            
        except Exception as e:
            # 详细的错误日志
            import traceback
            print(f"生成纹理图像失败: {str(e)}")
            print(traceback.format_exc())
            # 返回一个简单的错误图像
            return Image.new('RGB', (256, 256), (255, 0, 0))
#############################################
#导出为obj文件
############################################# 
class ObjExporter(MapExporter):
    def __init__(self, config = None, logger=None):
        super().__init__(config, logger)
        self.logger=logger
    
    def export(self, map_data: Dict, filename: str = None, auto_deploy: bool = False) -> str:
        """
        导出为OBJ文件
        
        参数:
            map_data: 地图数据
            filename: 输出文件名
            auto_deploy: 是否自动部署到3D软件
        """
        map_data_dict = self._prepare_map_data_dict(map_data)
        
        # 如果缺少宽度或高度，尝试从height_map推断
        if ("width" not in map_data_dict or "height" not in map_data_dict) and "height_map" in map_data_dict:
            height_map = map_data_dict["height_map"]
            # 修改这里的检查，避免对NumPy数组进行布尔测试
            if height_map is not None:
                if isinstance(height_map, np.ndarray):
                    h, w = height_map.shape
                else:
                    h = len(height_map)
                    w = len(height_map[0]) if h > 0 else 0
                map_data_dict["height"] = h
                map_data_dict["width"] = w
                self.logger.info(f"从height_map推断地图尺寸: {w}x{h}")

        # 标准化文件名
        if not filename:
            filename = self._get_output_path(None, "obj")
        
        # 确保必要字段存在
        if "width" not in map_data_dict or "height" not in map_data_dict or "height_map" not in map_data_dict:
            self.logger.error("导出数据缺少必要字段")
            return ""
        
        w = map_data_dict["width"]
        h = map_data_dict["height"]
        height_map = map_data_dict["height_map"]  
        
        # 获取模型放置数据
        vegetation = map_data_dict.get("vegetation", [])
        buildings = map_data_dict.get("buildings", [])
        caves = map_data_dict.get("caves", [])
        rivers = map_data_dict.get("rivers", [])
        content_layout = map_data_dict.get("content_layout", {})
        
        # 生成并保存纹理
        texture_path = os.path.splitext(filename)[0] + "_texture.png"
        if self.config.generate_textures:
            try:
                texture_img = self._generate_texture_image(map_data_dict)
                # 检查是否是错误图像
                if texture_img.size == (256, 256) and list(texture_img.getdata())[0] == (255, 0, 0):
                    print("警告：纹理生成返回了错误图像，将尝试使用简单的高度图替代")
                    # 尝试生成一个基于高度图的简单替代纹理
                    texture_img = self._generate_fallback_texture(map_data_dict)
                texture_img.save(texture_path)
            except Exception as e:
                print(f"保存纹理时出错: {e}")
                texture_path = None  # 在mtl文件中不引用纹理
        
        # 生成材质文件 - 为不同类型的模型定义不同的材质
        mtl_path = os.path.splitext(filename)[0] + ".mtl"
        with open(mtl_path, 'w') as mtl_file:
            # 地形材质
            mtl_file.write("newmtl MapTexture\n")
            mtl_file.write("Ka 1.0 1.0 1.0\n")  # 环境光
            mtl_file.write("Kd 1.0 1.0 1.0\n")  # 漫反射
            mtl_file.write("Ks 0.5 0.5 0.5\n")  # 高光
            mtl_file.write(f"map_Kd {os.path.basename(texture_path)}\n\n")
            
            # 植被材质
            mtl_file.write("newmtl Vegetation\n")
            mtl_file.write("Ka 0.1 0.5 0.1\n")  # 绿色环境光
            mtl_file.write("Kd 0.2 0.8 0.2\n")  # 绿色漫反射
            mtl_file.write("Ks 0.1 0.1 0.1\n\n")  # 低高光
            
            # 建筑材质
            mtl_file.write("newmtl Building\n")
            mtl_file.write("Ka 0.5 0.3 0.3\n")  # 棕色环境光
            mtl_file.write("Kd 0.8 0.4 0.4\n")  # 棕色漫反射
            mtl_file.write("Ks 0.2 0.2 0.2\n\n")  # 中等高光
            
            # 洞穴材质
            mtl_file.write("newmtl Cave\n")
            mtl_file.write("Ka 0.1 0.1 0.2\n")  # 暗蓝色环境光
            mtl_file.write("Kd 0.2 0.2 0.4\n")  # 暗蓝色漫反射
            mtl_file.write("Ks 0.1 0.1 0.2\n\n")  # 低高光
            
            # 生物材质 - 捕食者
            mtl_file.write("newmtl Predator\n")
            mtl_file.write("Ka 0.5 0.1 0.1\n")  # 暗红色环境光
            mtl_file.write("Kd 0.9 0.2 0.2\n")  # 红色漫反射
            mtl_file.write("Ks 0.4 0.4 0.4\n\n")  # 高高光
            
            # 生物材质 - 猎物
            mtl_file.write("newmtl Prey\n")
            mtl_file.write("Ka 0.5 0.5 0.1\n")  # 黄色环境光
            mtl_file.write("Kd 0.9 0.9 0.2\n")  # 黄色漫反射
            mtl_file.write("Ks 0.3 0.3 0.3\n\n")  # 中高光
        
        # 计算LOD步长和尺寸
        lod_step = self.config.level_of_detail
        w_lod = (w - 1) // lod_step + 1
        h_lod = (h - 1) // lod_step + 1
        
        # 流式写入OBJ文件
        with open(filename, 'w') as obj_file:
            obj_file.write(f"mtllib {os.path.basename(mtl_path)}\n")
            
            # 地形部分
            obj_file.write("g Terrain\n")
            obj_file.write("usemtl MapTexture\n")
            
            # 记录顶点总数以便后续为模型计算索引
            vertex_count = 0
            
            # 流式写入顶点数据
            for j in range(0, h, lod_step):
                for i in range(0, w, lod_step):
                    # 写入顶点坐标
                    y = height_map[j][i]
                    obj_file.write(f"v {i} {y} {j}\n")
                    vertex_count += 1
            
            # 写入纹理坐标
            for j in range(0, h, lod_step):
                for i in range(0, w, lod_step):
                    u = i / (w - 1) if w > 1 else 0.0
                    v_val = 1.0 - (j / (h - 1)) if h > 1 else 0.0
                    obj_file.write(f"vt {u} {v_val}\n")
            
            # 写入法线
            if self.config.export_normals:
                for j in range(0, h, lod_step):
                    for i in range(0, w, lod_step):
                        # 计算相邻点高度差
                        left = max(0, i - lod_step)
                        right = min(w-1, i + lod_step)
                        down = max(0, j - lod_step)
                        up = min(h-1, j + lod_step)
                        
                        # 计算梯度
                        dx = (height_map[j][right] - height_map[j][left]) / (right - left) if (right != left) else 0.0
                        dz = (height_map[up][i] - height_map[down][i]) / (up - down) if (up != down) else 0.0
                        
                        # 计算法线并归一化
                        normal = (-dx, 1.0, -dz)
                        length = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
                        if length == 0:
                            normal = (0.0, 1.0, 0.0)
                        else:
                            normal = (normal[0]/length, normal[1]/length, normal[2]/length)
                        obj_file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
            
            # 流式写入地形面数据
            for j in range(0, h - lod_step, lod_step):
                for i in range(0, w - lod_step, lod_step):
                    # 计算顶点索引
                    current_row = j // lod_step
                    current_col = i // lod_step
                    next_row = (j + lod_step) // lod_step
                    next_col = (i + lod_step) // lod_step
                    
                    v1 = current_row * w_lod + current_col + 1
                    v2 = current_row * w_lod + next_col + 1
                    v3 = next_row * w_lod + next_col + 1
                    v4 = next_row * w_lod + current_col + 1
                    
                    # 写入面（两个三角形）
                    if self.config.export_normals:
                        obj_file.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")
                        obj_file.write(f"f {v1}/{v1}/{v1} {v3}/{v3}/{v3} {v4}/{v4}/{v4}\n")
                    else:
                        obj_file.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")
                        obj_file.write(f"f {v1}/{v1} {v3}/{v3} {v4}/{v4}\n")
            
            # 获取所有模型放置数据
            models = place_3d_models(height_map, vegetation, buildings, caves, rivers, content_layout)
            
            # 按类型对模型分组
            model_by_types = {}
            for model in models:
                model_type = model["type"]
                if model_type not in model_by_types:
                    model_by_types[model_type] = []
                model_by_types[model_type].append(model)
            
            # 添加一个函数从文件导入模型几何体
            def load_model_geometry(model_type):
                model_info = MODEL_LIBRARY.get(model_type, MODEL_LIBRARY["tree"])
                model_path = model_info.get("model", "")
                model_filename = os.path.basename(model_path)
                
                # 查找模型文件路径
                model_dirs_to_check = [
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models"),
                    os.path.join(self.config.output_dir, "models"),
                    os.path.join(os.path.dirname(filename), "models"),
                ]
                
                for model_dir in model_dirs_to_check:
                    full_path = os.path.join(model_dir, model_filename)
                    if os.path.exists(full_path):
                        try:
                            # 使用trimesh加载模型
                            mesh = trimesh.load(full_path)
                            return mesh
                        except Exception as e:
                            self.logger.warning(f"无法加载模型 {model_filename}: {e}")
                            break
                
                # 返回None表示使用默认几何体
                return None
            
            # 当前顶点索引
            current_vertex_idx = vertex_count + 1
            
            # 依次添加每种类型的模型
            for model_type, model_list in model_by_types.items():
                # 加载模型几何体
                model_mesh = load_model_geometry(model_type)
                
                # 确定材质
                material_name = "Vegetation"  # 默认材质
                if model_type in ["house", "building"]:
                    material_name = "Building"
                elif model_type == "cave":
                    material_name = "Cave"
                elif model_type == "Predator":
                    material_name = "Predator"
                elif model_type == "Prey":
                    material_name = "Prey"
                
                # 为此类型模型创建组
                obj_file.write(f"\ng {model_type}\n")
                obj_file.write(f"usemtl {material_name}\n")
                
                # 添加每个模型实例
                for model in model_list:
                    pos = model["position"]
                    scale = model["scale"]
                    rotation = model["rotation"]
                    
                    # 如果有可用的模型网格，使用实际模型，否则创建简化几何体
                    if model_mesh is not None:
                        try:
                            # 创建模型副本并进行变换
                            instance_mesh = model_mesh.copy()
                            
                            # 应用缩放
                            scale_matrix = np.eye(4)
                            scale_matrix[0, 0] = scale
                            scale_matrix[1, 1] = scale
                            scale_matrix[2, 2] = scale
                            instance_mesh.apply_transform(scale_matrix)
                            
                            # 应用旋转（绕Y轴）
                            rot_rad = np.radians(rotation[1])
                            rot_matrix = np.eye(4)
                            rot_matrix[0, 0] = np.cos(rot_rad)
                            rot_matrix[0, 2] = np.sin(rot_rad)
                            rot_matrix[2, 0] = -np.sin(rot_rad)
                            rot_matrix[2, 2] = np.cos(rot_rad)
                            instance_mesh.apply_transform(rot_matrix)
                            
                            # 修正坐标顺序：在OBJ中x对应x，y对应高度，z对应y
                            translation_matrix = np.eye(4)
                            translation_matrix[0, 3] = pos[0]  # x → x
                            translation_matrix[1, 3] = pos[2]  # 高度 → y
                            translation_matrix[2, 3] = pos[1]  # y → z
                            instance_mesh.apply_transform(translation_matrix)
                            
                            # 添加顶点、法线到OBJ文件
                            vert_offset = current_vertex_idx
                            for vertex in instance_mesh.vertices:
                                obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                            
                            if hasattr(instance_mesh, "vertex_normals") and len(instance_mesh.vertex_normals) > 0:
                                for normal in instance_mesh.vertex_normals:
                                    obj_file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
                            
                            # 添加UV坐标（如果存在）
                            has_uvs = False
                            if hasattr(instance_mesh, "visual") and hasattr(instance_mesh.visual, "uv"):
                                if instance_mesh.visual.uv is not None and len(instance_mesh.visual.uv) > 0:
                                    for uv in instance_mesh.visual.uv:
                                        obj_file.write(f"vt {uv[0]} {uv[1]}\n")
                                    has_uvs = True
                            
                            # 添加面
                            for face in instance_mesh.faces:
                                if has_uvs:
                                    # 带材质和UV的面
                                    face_str = " ".join([f"{vert_offset+v+1}/{vert_offset+v+1}/{vert_offset+v+1}" for v in face])
                                    obj_file.write(f"f {face_str}\n")
                                else:
                                    # 仅有顶点的面
                                    face_str = " ".join([f"{vert_offset+v+1}" for v in face])
                                    obj_file.write(f"f {face_str}\n")
                            
                            # 更新顶点索引
                            current_vertex_idx += len(instance_mesh.vertices)
                            
                        except Exception as e:
                            self.logger.warning(f"处理模型 {model_type} 时出错: {e}, 将使用简化几何体替代")
                            # 如果模型处理失败，回退到简化几何体
                            use_simplified_geometry = True
                            
                            # 简化模型几何体 - 创建一个立方体或者圆柱体表示不同类型的实体
                            # 通过不同的缩放和位置来区分
                            
                            # 这里以立方体为例，生成顶点
                            cube_half_size = scale * 0.5
                            
                            # 为了简化，在 y 轴上稍微偏移一点，确保模型不会完全嵌入地形
                            y_offset = pos[2] + cube_half_size

                            # 立方体的8个顶点
                            cube_vertices = [
                                # 底面（修正坐标:x,高度,y）
                                [pos[0] - cube_half_size, pos[2] - cube_half_size * 0.3, pos[1] - cube_half_size],
                                [pos[0] + cube_half_size, pos[2] - cube_half_size * 0.3, pos[1] - cube_half_size],
                                [pos[0] + cube_half_size, pos[2] - cube_half_size * 0.3, pos[1] + cube_half_size],
                                [pos[0] - cube_half_size, pos[2] - cube_half_size * 0.3, pos[1] + cube_half_size],
                                # 顶面
                                [pos[0] - cube_half_size, y_offset, pos[1] - cube_half_size],
                                [pos[0] + cube_half_size, y_offset, pos[1] - cube_half_size],
                                [pos[0] + cube_half_size, y_offset, pos[1] + cube_half_size],
                                [pos[0] - cube_half_size, y_offset, pos[1] + cube_half_size]
                            ]
                            
                            # 调整形状 - 模型类型特异性处理
                            if model_type in ["tree", "pine"]:
                                # 为树创建圆锥形状
                                cube_vertices[4:] = [
                                    [pos[0], y_offset * 1.5, pos[1]]  # 树尖
                                ] * 4
                            elif model_type in ["Predator", "Prey"]:
                                # 为生物创建扁平的几何体
                                y_offset = pos[2] + cube_half_size * 0.5  # 降低高度
                                cube_vertices[4:] = [
                                    [pos[0] - cube_half_size, y_offset, pos[1] - cube_half_size],
                                    [pos[0] + cube_half_size, y_offset, pos[1] - cube_half_size],
                                    [pos[0] + cube_half_size, y_offset, pos[1] + cube_half_size],
                                    [pos[0] - cube_half_size, y_offset, pos[1] + cube_half_size]
                                ]
                            
                            # 写入模型顶点
                            for vert in cube_vertices:
                                obj_file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
                            
                            # 为模型创建面
                            if model_type in ["tree", "pine"]:
                                # 树的三角锥面
                                obj_file.write(f"f {current_vertex_idx} {current_vertex_idx+1} {current_vertex_idx+4}\n")
                                obj_file.write(f"f {current_vertex_idx+1} {current_vertex_idx+2} {current_vertex_idx+4}\n")
                                obj_file.write(f"f {current_vertex_idx+2} {current_vertex_idx+3} {current_vertex_idx+4}\n")
                                obj_file.write(f"f {current_vertex_idx+3} {current_vertex_idx} {current_vertex_idx+4}\n")
                                
                                # 底面
                                obj_file.write(f"f {current_vertex_idx} {current_vertex_idx+3} {current_vertex_idx+2} {current_vertex_idx+1}\n")
                                
                                # 更新顶点索引
                                current_vertex_idx += 5
                            else:
                                # 立方体的6个面
                                # 底面
                                obj_file.write(f"f {current_vertex_idx} {current_vertex_idx+1} {current_vertex_idx+2} {current_vertex_idx+3}\n")
                                # 顶面
                                obj_file.write(f"f {current_vertex_idx+4} {current_vertex_idx+7} {current_vertex_idx+6} {current_vertex_idx+5}\n")
                                # 四个侧面
                                obj_file.write(f"f {current_vertex_idx} {current_vertex_idx+4} {current_vertex_idx+5} {current_vertex_idx+1}\n")
                                obj_file.write(f"f {current_vertex_idx+1} {current_vertex_idx+5} {current_vertex_idx+6} {current_vertex_idx+2}\n")
                                obj_file.write(f"f {current_vertex_idx+2} {current_vertex_idx+6} {current_vertex_idx+7} {current_vertex_idx+3}\n")
                                obj_file.write(f"f {current_vertex_idx+3} {current_vertex_idx+7} {current_vertex_idx+4} {current_vertex_idx}\n")
                                
                                # 更新顶点索引
                                current_vertex_idx += 8
                    else:
                        self.logger.warning("没有模型实例，使用简化集合体代替")
                        cube_half_size = scale * 0.5
                            
                        # 为了简化，在 y 轴上稍微偏移一点，确保模型不会完全嵌入地形
                        y_offset = pos[2] + cube_half_size

                        # 立方体的8个顶点
                        cube_vertices = [
                            # 底面（修正坐标:x,高度,y）
                            [pos[0] - cube_half_size, pos[2] - cube_half_size * 0.3, pos[1] - cube_half_size],
                            [pos[0] + cube_half_size, pos[2] - cube_half_size * 0.3, pos[1] - cube_half_size],
                            [pos[0] + cube_half_size, pos[2] - cube_half_size * 0.3, pos[1] + cube_half_size],
                            [pos[0] - cube_half_size, pos[2] - cube_half_size * 0.3, pos[1] + cube_half_size],
                            # 顶面
                            [pos[0] - cube_half_size, y_offset, pos[1] - cube_half_size],
                            [pos[0] + cube_half_size, y_offset, pos[1] - cube_half_size],
                            [pos[0] + cube_half_size, y_offset, pos[1] + cube_half_size],
                            [pos[0] - cube_half_size, y_offset, pos[1] + cube_half_size]
                        ]
                        
                        # 调整形状 - 模型类型特异性处理
                        if model_type in ["tree", "pine"]:
                            # 为树创建圆锥形状
                            cube_vertices[4:] = [
                                [pos[0], y_offset * 1.5, pos[1]]  # 树尖
                            ] * 4
                        elif model_type in ["Predator", "Prey"]:
                            # 为生物创建扁平的几何体
                            y_offset = pos[2] + cube_half_size * 0.5  # 降低高度
                            cube_vertices[4:] = [
                                [pos[0] - cube_half_size, y_offset, pos[1] - cube_half_size],
                                [pos[0] + cube_half_size, y_offset, pos[1] - cube_half_size],
                                [pos[0] + cube_half_size, y_offset, pos[1] + cube_half_size],
                                [pos[0] - cube_half_size, y_offset, pos[1] + cube_half_size]
                            ]
                        
                        # 写入模型顶点
                        for vert in cube_vertices:
                            obj_file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
                        
                        # 为模型创建面
                        if model_type in ["tree", "pine"]:
                            # 树的三角锥面
                            obj_file.write(f"f {current_vertex_idx} {current_vertex_idx+1} {current_vertex_idx+4}\n")
                            obj_file.write(f"f {current_vertex_idx+1} {current_vertex_idx+2} {current_vertex_idx+4}\n")
                            obj_file.write(f"f {current_vertex_idx+2} {current_vertex_idx+3} {current_vertex_idx+4}\n")
                            obj_file.write(f"f {current_vertex_idx+3} {current_vertex_idx} {current_vertex_idx+4}\n")
                            
                            # 底面
                            obj_file.write(f"f {current_vertex_idx} {current_vertex_idx+3} {current_vertex_idx+2} {current_vertex_idx+1}\n")
                            
                            # 更新顶点索引
                            current_vertex_idx += 5
                        else:
                            # 立方体的6个面
                            # 底面
                            obj_file.write(f"f {current_vertex_idx} {current_vertex_idx+1} {current_vertex_idx+2} {current_vertex_idx+3}\n")
                            # 顶面
                            obj_file.write(f"f {current_vertex_idx+4} {current_vertex_idx+7} {current_vertex_idx+6} {current_vertex_idx+5}\n")
                            # 四个侧面
                            obj_file.write(f"f {current_vertex_idx} {current_vertex_idx+4} {current_vertex_idx+5} {current_vertex_idx+1}\n")
                            obj_file.write(f"f {current_vertex_idx+1} {current_vertex_idx+5} {current_vertex_idx+6} {current_vertex_idx+2}\n")
                            obj_file.write(f"f {current_vertex_idx+2} {current_vertex_idx+6} {current_vertex_idx+7} {current_vertex_idx+3}\n")
                            obj_file.write(f"f {current_vertex_idx+3} {current_vertex_idx+7} {current_vertex_idx+4} {current_vertex_idx}\n")
                            
                            # 更新顶点索引
                            current_vertex_idx += 8

        # 压缩处理
        if self.config.compress_output:
            with open(filename, 'rb') as f_in:
                with gzip.open(f"{filename}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(filename)
            filename += ".gz"
        
        self.logger.info(f"成功导出OBJ文件到 {filename}，包含地形和所有模型")
        
        # 在最后添加:
        if auto_deploy:
            try:
                deployment_manager = DeploymentManager(self.logger)
                deployment_manager.deploy_to_blender(filename)
            except Exception as e:
                self.logger.error(f"自动部署到Blender失败: {e}")
                
        return filename

    def _generate_fallback_texture(self, map_data: Dict) -> Image.Image:
        """生成基于高度图的备用纹理"""
        try:
            w = map_data.get("width", 0)
            h = map_data.get("height", 0)
            height_map = map_data.get("height_map", [])
            
            if w <= 0 or h <= 0 or not height_map:
                return Image.new('RGB', (512, 512), (200, 200, 200))
            
            # 找出高度范围
            min_height = float('inf')
            max_height = float('-inf')
            for row in height_map:
                # 使用NumPy的min和max函数，避免直接比较数组
                if hasattr(row, '__iter__'):  # 确保是可迭代对象
                    if isinstance(row, np.ndarray):
                        row_min = np.min(row)
                        row_max = np.max(row)
                    else:
                        row_min = min(row)
                        row_max = max(row)
                    min_height = min(min_height, row_min)
                    max_height = max(max_height, row_max)
            
            height_range = max_height - min_height
            if height_range == 0:
                height_range = 1  # 避免除以零
            
            # 创建基于高度的简单纹理
            texture_w, texture_h = 512, 512
            img = Image.new('RGB', (texture_w, texture_h), (200, 200, 200))
            draw = ImageDraw.Draw(img)
            
            scale_x = texture_w / w
            scale_y = texture_h / h
            
            for j in range(h):
                for i in range(w):
                    # 根据高度生成颜色
                    normalized_height = (height_map[j][i] - min_height) / height_range
                    # 简单地图配色：低=蓝/绿，中=棕/绿，高=灰/白
                    if normalized_height < 0.3:  # 水域和低地
                        r, g, b = 100, 150 + int(100 * normalized_height), 200
                    elif normalized_height < 0.7:  # 平原和丘陵
                        r, g, b = 100 + int(155 * normalized_height), 150, 50
                    else:  # 山地
                        intensity = int(200 * normalized_height)
                        r, g, b = 150 + intensity//4, 150 + intensity//4, 150 + intensity//4
                    
                    x1, y1 = int(i * scale_x), int(j * scale_y)
                    x2, y2 = int((i+1) * scale_x), int((j+1) * scale_y)
                    draw.rectangle([x1, y1, x2, y2], fill=(r, g, b))
            
            return img
        except Exception as e:
            print(f"生成备用纹理失败: {e}")
            return Image.new('RGB', (512, 512), (200, 200, 200))
################################################
#unity导出器
################################################    
class UnityExporter(MapExporter, JSONExporterBase):
    """Unity专用导出器"""
    
    def __init__(self, config: MapExportConfig = None,logger=None):
        self.logger=logger
        # 确保两个父类都被正确初始化
        MapExporter.__init__(self, config,self.logger)
        JSONExporterBase.__init__(self, config,self.logger)
    
    def export(self, map_data: Dict, filename: str = None, auto_deploy: bool = False, project_path: str = None) -> str:
        """
        导出Unity格式的地图数据
        
        参数:
            map_data: 地图数据
            filename: 输出文件名
            auto_deploy: 是否自动部署到Unity项目
            project_path: Unity项目路径，如果为None则会提示选择
        """
        try:
            # 将 MapData 对象转换为字典
            map_data = self._prepare_map_data_dict(map_data)
            normalized_data = self._prepare_export_data(map_data)
            
            # 生成JSON文件
            if not filename:
                filename = self._get_output_path("unity_map", "json")
            
            # 构建导出数据
            export_data = {
                "terrain": self._build_terrain_data(normalized_data),
                "entities": self._build_entities(normalized_data),
                "metadata": self._generate_metadata(normalized_data)
            }
            
            if not self._validate_data(export_data):
                return ""
            
            self._write_json_file(export_data, filename)
            self.logger.info(f"已成功生成Unity引擎地图数据: {filename}")
            
            # 创建完整的Unity引擎包
            try:
                # 确保导入必要的模块
                import shutil
                
                # 创建一个完整的Unity包
                package_dir = self.create_unity_package(normalized_data)
                self.logger.info(f"已成功生成Unity引擎项目包: {package_dir}")
                
                # 在包目录中保存一份JSON数据以备参考
                json_in_package = os.path.join(package_dir, "Assets", "TerrainGenerator", "map_data.json")
                self._write_json_file(export_data, json_in_package)
                
                # 返回包目录路径
                return package_dir
            except ImportError as e:
                self.logger.warning(f"创建Unreal包时导入模块失败: {e}")
                self.logger.warning("仅生成JSON数据文件, 未创建完整UE包")
                return filename
            except Exception as e:
                self.logger.error(f"创建Unreal包失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 返回JSON文件路径而不是抛出异常，确保流程继续
                return filename
        except Exception as e:
            self.logger.error(f"Unreal导出失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        if auto_deploy and package_dir:
            try:
                deployment_manager = DeploymentManager(self.logger)
                deployment_manager.deploy_to_unity(package_dir, project_path)
            except Exception as e:
                self.logger.error(f"自动部署到Unity项目失败: {e}")
                
        return package_dir or filename
    
    def _build_terrain_data(self, data: Dict) -> Dict:
        """构建地形数据"""
        return {
            "width": data["width"],
            "height": data["height"],
            "height_map": data["height_map"],
            "texture_map": self._generate_texture_path(),
            "biomes": self._process_biomes(data["biome_map"]),
            "lightmap_uvs": self.config.lightmap_uvs,
            "collision": self.config.export_collision
        }
    
    def _build_entities(self, data: Dict) -> Dict:
        """构建实体数据"""
        return {
            "vegetation": self._process_entities(data["vegetation"]),
            "buildings": self._process_entities(data["buildings"]),
            "rivers": self._process_rivers(data["rivers"]),
            "caves": self._process_caves(data["caves"]),
            "roads": self._process_roads(data["road_coords"])
        }
        
    def _get_output_path(self, filename_base: str, extension: str) -> str:
        """构造输出文件路径"""
        if not filename_base:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_base = f"{self.config.base_filename}_{timestamp}"
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        return os.path.join(self.config.output_dir, f"{filename_base}.{extension}")
    
    def _validate_data(self, data: Dict) -> bool:
        """验证Unity导出数据的有效性"""
        if not data:
            self.logger.error("导出数据为空")
            return False
            
        # 验证必要字段是否存在
        if "terrain" not in data or "entities" not in data:
            self.logger.error("导出数据缺少必要字段: terrain 或 entities")
            return False
            
        # 验证地形数据
        terrain = data["terrain"]
        if not terrain or "width" not in terrain or "height" not in terrain or "height_map" not in terrain:
            self.logger.error("地形数据不完整")
            return False
            
        # 验证尺寸
        if terrain["width"] <= 0 or terrain["height"] <= 0:
            self.logger.error(f"无效的地图尺寸: {terrain['width']}x{terrain['height']}")
            return False
            
        return True
    
    def create_unity_package(self, map_data: Dict, filename: str = None) -> str:
        """创建Unity引擎专用包"""
        try:
            # 解包地图数据
            height_map = map_data.get("height_map", [])
            biome_map = map_data.get("biome_map", [])
            vegetation = map_data.get("vegetation", [])
            buildings = map_data.get("buildings", [])
            rivers = map_data.get("rivers", [])
            caves = map_data.get("caves", [])
            roads = map_data.get("road_coords", [])
            content_layout = map_data.get("content_layout", {})
            
            # 确保资源目录存在
            unity_dir = os.path.join(self.config.output_dir, "unity")
            os.makedirs(f"{unity_dir}/Assets/TerrainGenerator", exist_ok=True)
            os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Models", exist_ok=True)
            os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Textures", exist_ok=True)
            os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Materials", exist_ok=True)
            os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Prefabs", exist_ok=True)
            os.makedirs(f"{unity_dir}/Assets/TerrainGenerator/Scripts", exist_ok=True)
            
            # 创建临时资源目录
            temp_dir = os.path.join(self.config.output_dir, "temp")
            os.makedirs(f"{temp_dir}/textures", exist_ok=True)
            os.makedirs(f"{temp_dir}/models", exist_ok=True)
            os.makedirs(f"{temp_dir}/materials", exist_ok=True)
            
            # 创建高度图纹理
            try:
                heightmap_path = create_heightmap_texture(height_map,
                                                        os.path.join(temp_dir, "textures/heightmap.png"))
                shutil.copy(heightmap_path, f"{unity_dir}/Assets/TerrainGenerator/Textures/")
                self.logger.info(f"已创建并复制高度图：{heightmap_path}")
            except Exception as e:
                self.logger.error(f"创建高度图失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 创建默认高度图
                heightmap_path = os.path.join(temp_dir, "textures/default_heightmap.png")
                Image.new('L', (256, 256), 128).save(heightmap_path)
                shutil.copy(heightmap_path, f"{unity_dir}/Assets/TerrainGenerator/Textures/")
            
            # 创建法线贴图
            if self.config.export_normals:
                try:
                    normal_map_path = create_normal_map(height_map, 
                                                    os.path.join(temp_dir, "textures/normal_map.png"))
                    shutil.copy(normal_map_path, f"{unity_dir}/Assets/TerrainGenerator/Textures/")
                    self.logger.info(f"已创建并复制法线贴图：{normal_map_path}")
                except Exception as e:
                    self.logger.error(f"创建法线贴图失败: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    # 创建默认法线贴图
                    normal_map_path = os.path.join(temp_dir, "textures/default_normal_map.png")
                    # 创建蓝色的默认法线贴图(Z向上)
                    Image.new('RGB', (256, 256), (128, 128, 255)).save(normal_map_path)
                    shutil.copy(normal_map_path, f"{unity_dir}/Assets/TerrainGenerator/Textures/")

            # 创建材质混合贴图
            try:
                splat_map_path, biome_mapping = create_splat_map(biome_map, 
                                                        os.path.join(temp_dir, "textures/splat_map.png"))
                shutil.copy(splat_map_path, f"{unity_dir}/Assets/TerrainGenerator/Textures/")
                self.logger.info(f"已创建并复制混合贴图：{splat_map_path}")
            except Exception as e:
                self.logger.error(f"创建材质混合贴图失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 创建默认贴图和映射
                splat_map_path = os.path.join(temp_dir, "textures/default_splat_map.png")
                Image.new('RGB', (256, 256), (255, 0, 0)).save(splat_map_path)
                shutil.copy(splat_map_path, f"{unity_dir}/Assets/TerrainGenerator/Textures/")
                biome_mapping = {"default": 0}

            # 创建地形网格
            try:
                terrain_mesh_path = create_enhanced_terrain_mesh(height_map,
                                                        os.path.join(temp_dir, "models/terrain.obj"))
                shutil.copy(terrain_mesh_path, f"{unity_dir}/Assets/TerrainGenerator/Models/")
                self.logger.info(f"已创建并复制地形网格：{terrain_mesh_path}")
            except Exception as e:
                self.logger.error(f"创建地形网格失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 创建默认网格是复杂的，这里可以省略

            # 创建材质文件
            sea_level = 15.0  # 默认海平面高度
            try:
                materials_path = create_material_files(biome_mapping, sea_level,
                                                os.path.join(temp_dir, "materials/terrain_materials.json"))
                shutil.copy(materials_path, f"{unity_dir}/Assets/TerrainGenerator/Materials/")
                self.logger.info(f"已创建并复制材质文件：{materials_path}")
            except Exception as e:
                self.logger.error(f"创建材质文件失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 创建一个默认的材质文件
                materials_path = os.path.join(temp_dir, "materials/default_materials.json")
                with open(materials_path, "w") as f:
                    json.dump({"materials": {"default": MATERIAL_LIBRARY["grass"]}, 
                            "biome_mapping": biome_mapping, 
                            "sea_level": sea_level}, f, indent=2)
                shutil.copy(materials_path, f"{unity_dir}/Assets/TerrainGenerator/Materials/")

            # 放置3D模型
            try:
                # 使用临时文件，确保目录存在
                model_output_path = os.path.join(temp_dir, "models/placed_models.json")
                os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
                
                # 处理模型放置
                #models = place_3d_models(height_map, vegetation, buildings, caves, rivers, content_layout)
                model_output_path = os.path.join(temp_dir, "models/placed_models.json")
                os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
                models = place_3d_models(height_map, vegetation, buildings, caves, rivers, content_layout, model_output_path)
                
                # 保存模型数据到临时文件
                with open(model_output_path, "w") as f:
                    json.dump(models, f, indent=2)
                    
                # 复制到最终目录
                shutil.copy(model_output_path, f"{unity_dir}/Assets/TerrainGenerator/model_placements.json")
                self.logger.info(f"已生成模型放置数据：{model_output_path}")
            except Exception as e:
                self.logger.error(f"放置3D模型失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 创建一个空的模型列表作为默认值
                with open(f"{unity_dir}/Assets/TerrainGenerator/model_placements.json", "w") as f:
                    json.dump([], f, indent=2)

            # 创建Unity地形设置
            with open(f"{unity_dir}/Assets/TerrainGenerator/terrain_settings.json", "w") as f:
                json.dump({
                    "width": len(height_map[0]) if height_map and height_map[0] else 256,
                    "height": len(height_map) if height_map else 256,
                    "heightScale": 1.0,
                    "heightmapResolution": len(height_map) if height_map else 256,
                    "detailResolution": 1024,
                    "controlTextureResolution": 1024,
                    "baseTextureResolution": 1024,
                    "seaLevel": sea_level,
                    "unityVersion": self.config.unity_export_version,
                    "lightmapUVs": self.config.lightmap_uvs
                }, f, indent=2)

            # 创建Unity导入器脚本
            with open(f"{unity_dir}/Assets/TerrainGenerator/Scripts/TerrainImporter.cs", "w") as f:
                f.write("""
            using UnityEngine;
            using System.Collections.Generic;
            using System.IO;
            using UnityEditor;

            #if UNITY_EDITOR
            public class TerrainImporter : EditorWindow {
                [MenuItem("Tools/Import Generated Terrain")]
                static void Init() {
                    TerrainImporter window = GetWindow<TerrainImporter>();
                    window.Show();
                }

                void OnGUI() {
                    if(GUILayout.Button("Import Terrain and Assets")) {
                        ImportAll();
                    }
                }

                void ImportAll() {
                    // 读取配置文件
                    string settingsPath = "Assets/TerrainGenerator/terrain_settings.json";
                    string jsonText = File.ReadAllText(settingsPath);
                    var settings = JsonUtility.FromJson<TerrainSettings>(jsonText);
                    
                    // 创建地形
                    TerrainData terrainData = new TerrainData();
                    terrainData.heightmapResolution = settings.heightmapResolution;
                    terrainData.size = new Vector3(settings.width, settings.heightScale, settings.height);
                    
                    // 加载高度图
                    Texture2D heightmap = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/TerrainGenerator/Textures/heightmap.png");
                    float[,] heights = new float[settings.heightmapResolution, settings.heightmapResolution];
                    
                    // 设置高度图
                    for (int y = 0; y < settings.heightmapResolution; y++) {
                        for (int x = 0; x < settings.heightmapResolution; x++) {
                            float normX = (float)x / settings.heightmapResolution;
                            float normY = (float)y / settings.heightmapResolution;
                            Color pixelColor = heightmap.GetPixel(
                                Mathf.FloorToInt(normX * heightmap.width),
                                Mathf.FloorToInt(normY * heightmap.height)
                            );
                            heights[y, x] = pixelColor.grayscale;
                        }
                    }
                    terrainData.SetHeights(0, 0, heights);
                    
                    // 保存地形数据
                    AssetDatabase.CreateAsset(terrainData, "Assets/TerrainGenerator/TerrainData.asset");
                    
                    // 创建地形GameObject
                    GameObject terrainGO = Terrain.CreateTerrainGameObject(terrainData);
                    terrainGO.name = "GeneratedTerrain";
                    
                    // 放置模型
                    ImportModels();
                    
                    Debug.Log("Terrain import complete!");
                }
                
                void ImportModels() {
                    string modelDataPath = "Assets/TerrainGenerator/model_placements.json";
                    string jsonText = File.ReadAllText(modelDataPath);
                    ModelPlacementData placementData = JsonUtility.FromJson<ModelPlacementData>(jsonText);
                    
                    foreach (var model in placementData.models) {
                        // 加载模型预制体
                        string modelPath = "Assets/TerrainGenerator/Models/" + Path.GetFileName(model.model_path);
                        GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
                        
                        if (prefab != null) {
                            GameObject instance = Instantiate(prefab);
                            instance.transform.position = new Vector3(model.position[0], model.position[2], model.position[1]);
                            instance.transform.eulerAngles = new Vector3(model.rotation[0], model.rotation[1], model.rotation[2]);
                            instance.transform.localScale = Vector3.one * model.scale;
                            instance.name = model.type;
                        }
                    }
                }
            }

            [System.Serializable]
            public class TerrainSettings {
                public int width;
                public int height;
                public float heightScale;
                public int heightmapResolution;
                public int detailResolution;
                public int controlTextureResolution;
                public int baseTextureResolution;
                public float seaLevel;
            }

            [System.Serializable]
            public class ModelPlacement {
                public string type;
                public float[] position;
                public float[] rotation;
                public float scale;
                public string model_path;
            }

            [System.Serializable]
            public class ModelPlacementData {
                public List<ModelPlacement> models;
            }
            #endif
                    """)

            # 创建README.md
            with open(f"{unity_dir}/README.md", "w") as f:
                f.write(f"""
            # 生成的Unity地形包

            本包包含程序化生成的地形，及其模型、材质和游戏内容。

            ## 导入说明

            1. 创建新的Unity项目或打开现有项目（推荐版本 {self.config.unity_export_version}）
            2. 将本目录内容复制到Assets文件夹中
            3. 打开Unity编辑器
            4. 前往 工具 > 导入生成的地形
            5. 点击"导入地形和资源"
            6. 地形将按指定生成，所有模型都会放置到位

            ## 内容

            - 完整的3D地形高度数据
            - 材质和纹理
            - 植被、建筑和其他特征的3D模型
            - 游戏内容数据，包括生物属性和故事事件
            - 河流、道路和其他路径系统
                    """)

            self._copy_model_library_assets(temp_dir, unity_dir, "unity") 
            self._copy_texture_library_assets(temp_dir, unity_dir, "unity")
            
            return unity_dir
        except Exception as e:
            self.logger.error(f"创建Unity包失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _copy_model_library_assets(self, temp_dir, engine_dir, engine_type="unity"):
        """复制模型库中的所有模型文件到引擎项目目录"""
        self.logger.info("开始复制模型库资源...")
        
        # 确定目标目录
        if engine_type == "unreal":
            models_dir = os.path.join(engine_dir, "Content", "TerrainGenerator", "Models")
        else:  # unity
            models_dir = os.path.join(engine_dir, "Assets", "TerrainGenerator", "Models")
        
        # 确保目录存在
        os.makedirs(models_dir, exist_ok=True)
        
        # 查找基础模型目录
        base_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
        if not os.path.exists(base_model_dir):
            self.logger.warning(f"模型基础目录不存在: {base_model_dir}，将创建空白模型")
            base_model_dir = os.path.join(temp_dir, "default_models")
            os.makedirs(base_model_dir, exist_ok=True)
        
        # 对每个模型库中的模型
        for model_type, model_info in MODEL_LIBRARY.items():
            model_path = model_info.get("model", "")
            if not model_path:
                continue
                
            # 处理相对路径
            model_filename = os.path.basename(model_path)
            src_model_path = os.path.join(base_model_dir, model_filename)
            
            # 如果原始模型不存在，创建一个简单的默认模型
            if not os.path.exists(src_model_path):
                self.logger.warning(f"找不到模型: {model_filename}，将创建默认模型")
                # 创建一个简单的立方体OBJ
                with open(src_model_path, "w") as f:
                    f.write(f"""
    # Default {model_type} model
    v -0.5 0 -0.5
    v 0.5 0 -0.5
    v 0.5 0 0.5
    v -0.5 0 0.5
    v -0.5 1 -0.5
    v 0.5 1 -0.5
    v 0.5 1 0.5
    v -0.5 1 0.5
    f 1 2 3 4
    f 5 6 7 8
    f 1 5 8 4
    f 2 6 7 3
    f 1 2 6 5
    f 4 3 7 8
                    """)
            
            # 复制到目标目录
            dst_model_path = os.path.join(models_dir, model_filename)
            try:
                shutil.copy2(src_model_path, dst_model_path)
                self.logger.info(f"已复制模型: {model_filename}")
            except Exception as e:
                self.logger.error(f"复制模型失败 {model_filename}: {e}")

    def _copy_texture_library_assets(self, temp_dir, engine_dir, engine_type="unreal"):
        """复制材质库中的所有纹理文件到引擎项目目录"""
        self.logger.info("开始复制纹理库资源...")
        
        # 确定目标目录
        if engine_type == "unreal":
            textures_dir = os.path.join(engine_dir, "Content", "TerrainGenerator", "Textures")
        else:  # unity
            textures_dir = os.path.join(engine_dir, "Assets", "TerrainGenerator", "Textures")
        
        # 确保目录存在
        os.makedirs(textures_dir, exist_ok=True)
        
        # 查找基础纹理目录
        base_texture_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "textures")
        if not os.path.exists(base_texture_dir):
            self.logger.warning(f"纹理基础目录不存在: {base_texture_dir}，将创建默认纹理")
            base_texture_dir = os.path.join(temp_dir, "default_textures")
            os.makedirs(base_texture_dir, exist_ok=True)
        
        # 处理过的纹理，避免重复
        processed_textures = set()
        
        # 对每个材质库中的材质
        for material_type, material_info in MATERIAL_LIBRARY.items():
            for texture_type in ["diffuse", "normal", "roughness"]:
                texture_path = material_info.get(texture_type, "")
                # 添加类型检查，确保texture_path是字符串类型
                if not texture_path or texture_path in processed_textures or not isinstance(texture_path, str):
                    continue
                    
                processed_textures.add(texture_path)
                texture_filename = os.path.basename(texture_path)
                src_texture_path = os.path.join(base_texture_dir, texture_filename)
                
                # 如果原始纹理不存在，创建一个简单的默认纹理
                if not os.path.exists(src_texture_path):
                    self.logger.warning(f"找不到纹理: {texture_filename}，将创建默认纹理")
                    # 创建默认纹理
                    img = Image.new('RGB', (512, 512))
                    
                    # 根据纹理类型设置不同的默认颜色
                    if texture_type == "diffuse":
                        color = (100, 100, 100) if "rock" in material_type else (
                                (50, 150, 50) if "grass" in material_type else (
                                (200, 200, 250) if "water" in material_type else (
                                (200, 180, 150) if "sand" in material_type else (150, 150, 150))))
                    elif texture_type == "normal":
                        color = (128, 128, 255)  # 默认法线颜色 (Z up)
                    else:
                        color = (128, 128, 128)  # 灰色
                    
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([(0, 0), (512, 512)], fill=color)
                    
                    # 添加棋盘格以便于调试
                    sq_size = 64
                    for y in range(0, 512, sq_size*2):
                        for x in range(0, 512, sq_size*2):
                            draw.rectangle([(x, y), (x+sq_size, y+sq_size)], fill=(
                                color[0]//2, color[1]//2, color[2]//2))
                            draw.rectangle([(x+sq_size, y+sq_size), (x+sq_size*2, y+sq_size*2)], fill=(
                                color[0]//2, color[1]//2, color[2]//2))
                    
                    img.save(src_texture_path)
                
                # 复制到目标目录
                dst_texture_path = os.path.join(textures_dir, texture_filename)
                try:
                    shutil.copy2(src_texture_path, dst_texture_path)
                    self.logger.info(f"已复制纹理: {texture_filename}")
                except Exception as e:
                    self.logger.error(f"复制纹理失败 {texture_filename}: {e}")
###########################################
#UE导出器
###########################################
class UnrealExporter(MapExporter, JSONExporterBase):
    """Unreal Engine专用导出器"""
    def __init__(self, config: MapExportConfig = None,logger=None):
        self.logger=logger
        # 确保两个父类都被正确初始化
        MapExporter.__init__(self, config,self.logger)
        JSONExporterBase.__init__(self, config,self.logger)
    
    def export(self, map_data: Dict, filename: str = None, auto_deploy: bool = False, project_path: str = None) -> str:
        """
        导出Unreal Engine格式的地图数据
        
        参数:
            map_data: 地图数据
            filename: 输出文件名
            auto_deploy: 是否自动部署到Unreal项目
            project_path: Unreal项目文件路径(.uproject)，如果为None则会提示选择
        """
        try:
            # 将 MapData 对象转换为字典
            map_data = self._prepare_map_data_dict(map_data)
            normalized_data = self._prepare_export_data(map_data)
            
            # 生成JSON文件
            if not filename:
                filename = self._get_output_path("unreal_map", "json")
            
            export_data = {
                "map_data": self._build_map_data(normalized_data),
                "navigation": {
                    "navmesh": True,
                    "collision": self.config.export_collision
                },
                "metadata": self._generate_metadata(normalized_data)
            }
            
            if not self._validate_data(export_data):
                return ""
            
            self._write_json_file(export_data, filename)
            self.logger.info(f"已成功生成Unreal Engine地图数据: {filename}")
            
            # 创建完整的Unreal引擎包
            try:
                # 确保导入必要的模块
                import shutil
                
                # 创建一个完整的UE包
                package_dir = self.create_unreal_package(normalized_data)
                self.logger.info(f"已成功生成Unreal Engine项目包: {package_dir}")
                
                # 在包目录中保存一份JSON数据以备参考
                json_in_package = os.path.join(package_dir, "Content", "TerrainGenerator", "map_data.json")
                self._write_json_file(export_data, json_in_package)
                
                # 返回包目录路径
                return package_dir
            except ImportError as e:
                self.logger.warning(f"创建Unreal包时导入模块失败: {e}")
                self.logger.warning("仅生成JSON数据文件, 未创建完整UE包")
                return filename
            except Exception as e:
                self.logger.error(f"创建Unreal包失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 返回JSON文件路径而不是抛出异常，确保流程继续
                return filename
        except Exception as e:
            self.logger.error(f"Unreal导出失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        if auto_deploy and package_dir:
            try:
                deployment_manager = DeploymentManager(self.logger)
                deployment_manager.deploy_to_unreal(package_dir, project_path)
            except Exception as e:
                self.logger.error(f"自动部署到Unreal项目失败: {e}")
                
        return package_dir or filename
    
    def _build_map_data(self, data: Dict) -> Dict:
        """构建Unreal专用数据"""
        return {
            "terrain": {
                "dimensions": [data["width"], data["height"]],
                "height_data": data["height_map"],
                "texture_data": self._generate_texture_path()
            },
            "foliage": self._process_foliage(data["vegetation"]),
            "structures": self._process_structures(data["buildings"]),
            "water": self._process_water(data["rivers"]),
            "subterranean": {
                "caves": self._process_caves(data["caves"]),
                "tunnels": self._process_roads(data["road_coords"])
            }
        }
    
    def _process_foliage(self, vegetation):
        """处理植被数据（Unreal专用格式）"""
        result = []
        for item in vegetation:
            # 支持不同的数据格式
            if isinstance(item, dict) and "x" in item and "y" in item:
                x, y = item["x"], item["y"]
                item_type = item.get("type", "tree")
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                x, y, item_type = item[0], item[1], item[2]
            else:
                continue
                
            result.append({
                "position": [x, 0, y],  # Z-up坐标系
                "asset": f"SM_{item_type}",
                "scale": [1.0, 1.0, 1.0]
            })
        return result
    
    def _process_structures(self, buildings):
        """处理建筑数据（Unreal专用格式）"""
        return [{
            "position": [item["x"], 0, item["y"]],  # Z-up坐标系
            "asset": f"SM_Building_{item['type']}",
            "scale": [1.0, 1.0, 1.0],
            "rotation": [0, 0, 0]
        } for item in buildings]
    
    def _process_water(self, rivers):
        """处理水体数据（Unreal专用格式）"""
        # 创建水面网格
        h = len(rivers)
        w = len(rivers[0]) if h > 0 else 0
        water_regions = []
        
        # 扫描连续水域区块
        visited = set()
        for y in range(h):
            for x in range(w):
                if rivers[y][x] and (x, y) not in visited:
                    # 寻找连续水域
                    region = self._flood_fill_water(rivers, x, y, visited, w, h)
                    if region:
                        # 简化为矩形区域
                        min_x = min(p[0] for p in region)
                        max_x = max(p[0] for p in region)
                        min_y = min(p[1] for p in region)
                        max_y = max(p[1] for p in region)
                        
                        water_regions.append({
                            "type": "river",
                            "position": [min_x, 0, min_y],  # Unreal坐标系
                            "dimensions": [max_x - min_x + 1, 0.5, max_y - min_y + 1],
                            "material": "M_Water_River"
                        })
        
        return {
            "regions": water_regions,
            "material": "M_Water_Base"
        }
    def _get_output_path(self, filename_base: str, extension: str) -> str:
        """构造输出文件路径"""
        if not filename_base:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_base = f"{self.config.base_filename}_{timestamp}"
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        return os.path.join(self.config.output_dir, f"{filename_base}.{extension}")
    
    def _flood_fill_water(self, rivers, x, y, visited, w, h):
        """使用泛洪填充算法寻找连续水域"""
        if not (0 <= x < w and 0 <= y < h) or (x, y) in visited or not rivers[y][x]:
            return []
            
        region = []
        queue = [(x, y)]
        
        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) in visited:
                continue
                
            visited.add((cx, cy))
            region.append((cx, cy))
            
            # 检查四个方向
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited and rivers[ny][nx]:
                    queue.append((nx, ny))
        
        return region
    
    def _validate_data(self, data: Dict) -> bool:
        """验证Unreal导出数据的有效性"""
        if not data:
            self.logger.error("导出数据为空")
            return False
            
        # 验证必要字段是否存在
        if "map_data" not in data:
            self.logger.error("导出数据缺少必要字段: map_data")
            return False
            
        # 验证地形数据
        terrain = data["map_data"].get("terrain", {})
        if not terrain or "dimensions" not in terrain or "height_data" not in terrain:
            self.logger.error("地形数据不完整")
            return False
            
        # 验证尺寸
        dimensions = terrain["dimensions"]
        if len(dimensions) < 2 or dimensions[0] <= 0 or dimensions[1] <= 0:
            self.logger.error(f"无效的地图尺寸: {dimensions}")
            return False
            
        return True
    
    def create_unreal_package(self, map_data: Dict, filename: str = None) -> str:
        """创建Unreal引擎专用包"""
        try:
            # 解包地图数据
            height_map = map_data.get("height_map", [])
            biome_map = map_data.get("biome_map", [])
            vegetation = map_data.get("vegetation", [])
            buildings = map_data.get("buildings", [])
            rivers = map_data.get("rivers", [])
            caves = map_data.get("caves", [])
            roads = map_data.get("road_coords", [])
            content_layout = map_data.get("content_layout", {})
            
            # 确保资源目录存在
            unreal_dir = os.path.join(self.config.output_dir, "unreal")
            os.makedirs(f"{unreal_dir}/Content/TerrainGenerator", exist_ok=True)
            os.makedirs(f"{unreal_dir}/Content/TerrainGenerator/Models", exist_ok=True)
            os.makedirs(f"{unreal_dir}/Content/TerrainGenerator/Textures", exist_ok=True)
            os.makedirs(f"{unreal_dir}/Content/TerrainGenerator/Materials", exist_ok=True)
            os.makedirs(f"{unreal_dir}/Content/TerrainGenerator/Maps", exist_ok=True)
            os.makedirs(f"{unreal_dir}/Content/TerrainGenerator/Blueprints", exist_ok=True)
            
            # 创建临时资源目录
            temp_dir = os.path.join(self.config.output_dir, "temp")
            os.makedirs(f"{temp_dir}/textures", exist_ok=True)
            os.makedirs(f"{temp_dir}/models", exist_ok=True)
            os.makedirs(f"{temp_dir}/materials", exist_ok=True)
            
            # 创建高度图纹理
            try:
                heightmap_path = create_heightmap_texture(height_map, 
                                                        os.path.join(temp_dir, "textures/heightmap.png"))
                shutil.copy(heightmap_path, f"{unreal_dir}/Content/TerrainGenerator/Textures/")
                self.logger.info(f"已创建并复制高度图：{heightmap_path}")
            except Exception as e:
                self.logger.error(f"创建高度图失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 创建一个简单的默认高度图
                heightmap_path = os.path.join(temp_dir, "textures/default_heightmap.png")
                Image.new('L', (256, 256), 128).save(heightmap_path)
                shutil.copy(heightmap_path, f"{unreal_dir}/Content/TerrainGenerator/Textures/")
            
            # 创建法线贴图
            if self.config.export_normals:
                try:
                    normal_map_path = create_normal_map(height_map, 
                                                    os.path.join(temp_dir, "textures/normal_map.png"))
                    shutil.copy(normal_map_path, f"{unreal_dir}/Content/TerrainGenerator/Textures/")
                    self.logger.info(f"已创建并复制法线贴图：{normal_map_path}")
                except Exception as e:
                    self.logger.error(f"创建法线贴图失败: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            # 创建材质混合贴图
            try:
                splat_map_path, biome_mapping = create_splat_map(biome_map, 
                                                            os.path.join(temp_dir, "textures/splat_map.png"))
                shutil.copy(splat_map_path, f"{unreal_dir}/Content/TerrainGenerator/Textures/")
                self.logger.info(f"已创建并复制混合贴图：{splat_map_path}")
            except Exception as e:
                self.logger.error(f"创建材质混合贴图失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 创建默认贴图和映射
                splat_map_path = os.path.join(temp_dir, "textures/default_splat_map.png")
                Image.new('RGB', (256, 256), (255, 0, 0)).save(splat_map_path)
                biome_mapping = {"default": 0}
                shutil.copy(splat_map_path, f"{unreal_dir}/Content/TerrainGenerator/Textures/")
                
            # 创建地形网格
            try:
                terrain_mesh_path = create_enhanced_terrain_mesh(height_map,
                                                            os.path.join(temp_dir, "models/terrain.obj"))
                shutil.copy(terrain_mesh_path, f"{unreal_dir}/Content/TerrainGenerator/Models/")
                self.logger.info(f"已创建并复制地形网格：{terrain_mesh_path}")
            except Exception as e:
                self.logger.error(f"创建地形网格失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 创建一个简单的默认地形网格
                # 可以省略，因为后续步骤可能需要依赖于这个结果

            # 创建材质文件
            sea_level = 15.0  # 默认海平面高度
            try:
                materials_path = create_material_files(biome_mapping, sea_level,
                                                    os.path.join(temp_dir, "materials/terrain_materials.json"))
                shutil.copy(materials_path, f"{unreal_dir}/Content/TerrainGenerator/Materials/")
                self.logger.info(f"已创建并复制材质文件：{materials_path}")
            except Exception as e:
                self.logger.error(f"创建材质文件失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 创建一个默认的材质文件
                materials_path = os.path.join(temp_dir, "materials/default_materials.json")
                with open(materials_path, "w") as f:
                    json.dump({"materials": {"default": MATERIAL_LIBRARY["grass"]}, 
                            "biome_mapping": biome_mapping, 
                            "sea_level": sea_level}, f, indent=2)
                shutil.copy(materials_path, f"{unreal_dir}/Content/TerrainGenerator/Materials/")

            # 放置3D模型
            try:
                # 使用临时文件，确保目录存在
                model_output_path = os.path.join(temp_dir, "models/placed_models.json")
                os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
                
                # 手动处理模型放置过程
                #models = place_3d_models(height_map, vegetation, buildings, caves, rivers, content_layout)
                model_output_path = os.path.join(temp_dir, "models/placed_models.json")
                os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
                models = place_3d_models(height_map, vegetation, buildings, caves, rivers, content_layout, model_output_path)
                
                # 保存模型数据到临时文件
                with open(model_output_path, "w") as f:
                    json.dump(models, f, indent=2)
                    
                # 复制到最终目录
                shutil.copy(model_output_path, f"{unreal_dir}/Content/TerrainGenerator/model_placements.json")
                self.logger.info(f"已生成模型放置数据：{model_output_path}")
            except Exception as e:
                self.logger.error(f"放置3D模型失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 创建一个空的模型列表作为默认值
                with open(f"{unreal_dir}/Content/TerrainGenerator/model_placements.json", "w") as f:
                    json.dump([], f, indent=2)
            
            # 创建Unreal地形配置
            with open(f"{unreal_dir}/Content/TerrainGenerator/terrain_config.json", "w") as f:
                json.dump({
                    "TerrainWidth": len(height_map[0]) * 100,  # 转换为Unreal单位
                    "TerrainHeight": len(height_map) * 100,
                    "HeightScale": 100.0,
                    "NumSections": 1,
                    "SectionsPerComponent": 1,
                    "QuadsPerSection": len(height_map),
                    "MaxLOD": 8,
                    "UnrealVersion": self.config.unreal_export_version,
                    "ExportCollision": self.config.export_collision,
                    "GenerateLightmapUVs": self.config.lightmap_uvs
                }, f, indent=2)
            
            # 创建生物和事件数据
            with open(f"{unreal_dir}/Content/TerrainGenerator/game_content.json", "w") as f:
                json.dump({
                    "creatures": content_layout.get("creatures", []),
                    "story_events": content_layout.get("story_events", []),
                    "story_overview": content_layout.get("story_overview", ""),
                    "map_emotion": content_layout.get("map_emotion", {}),
                    "region_emotions": content_layout.get("region_emotions", {})
                }, f, indent=2)
            
            # 创建Unreal导入脚本
            with open(f"{unreal_dir}/Content/TerrainGenerator/ImportScript.py", "w") as f:
                f.write("""
        import unreal
        import json
        import os

        def import_terrain():
            # 读取配置
            with open(os.path.join(unreal.Paths.project_content_dir(), "TerrainGenerator/terrain_config.json"), 'r') as config_file:
                config = json.load(config_file)
            
            # 创建Heightfield
            heightfield_asset = unreal.LandscapeEditorUtils.create_landscape_from_heightmap(
                unreal.Paths.project_content_dir() + "/TerrainGenerator/Textures/heightmap.png",
                config["TerrainWidth"],
                config["TerrainHeight"],
                config["NumSections"],
                config["SectionsPerComponent"],
                config["QuadsPerSection"]
            )
            
            # 导入模型
            with open(os.path.join(unreal.Paths.project_content_dir(), "TerrainGenerator/model_placements.json"), 'r') as models_file:
                model_data = json.load(models_file)
            
            for model in model_data:
                task = unreal.AssetImportTask()
                task.filename = model["model_path"]
                task.destination_path = "/Game/TerrainGenerator/Models"
                task.replace_existing = True
                task.automated = True
                
                unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
                
                asset_path = task.get_imported_object_paths()[0]
                actor = unreal.EditorLevelLibrary.spawn_actor_from_object(
                    unreal.load_asset(asset_path),
                    unreal.Vector(model["position"][0] * 100, model["position"][1] * 100, model["position"][2] * 100)
                )
                actor.set_actor_scale3d(unreal.Vector(model["scale"], model["scale"], model["scale"]))
                actor.set_actor_rotation(
                    unreal.Rotator(model["rotation"][0], model["rotation"][1], model["rotation"][2]),
                    False
                )
            
            print("Terrain and models imported successfully!")

        # Run the import script
        import_terrain()
                """)
            
            # 创建README.md
            with open(f"{unreal_dir}/README.md", "w") as f:
                f.write(f"""
        # 生成的Unreal Engine地形包

        本资源包包含程序化生成的地形场景、配套模型、材质资源及游戏内容数据。

        ## 导入指引

        1. 创建新的Unreal Engine项目或打开现有项目（推荐版本 {self.config.unreal_export_version}）
        2. 将本目录所有内容复制到项目的Content目录
        3. 打开Unreal编辑器
        4. 前往 Window > Developer Tools > Python Console
        5. 运行以下命令：
        ```python
        exec(open("/Game/TerrainGenerator/ImportScript.py").read())
        ```
        6. 地形将按预设配置生成，所有模型自动完成布局

        ## 包含内容

        - 完整的3D地形数据（含高度图信息）
        - 材质与纹理资源
        - 预置的植被、建筑及其他场景元素的3D模型
        - 包含生物属性和剧情事件的游戏内容数据
        - 河流、道路等路径系统
                """)
            
            self._copy_model_library_assets(temp_dir, unreal_dir, "unreal")
            self._copy_texture_library_assets(temp_dir, unreal_dir, "unreal")
            
            # 返回Unreal目录路径
            return unreal_dir

        except Exception as e:
            self.logger.error(f"创建Unreal包失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        except Exception as e:
            self.logger.error(f"创建Unreal包失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        
    def _copy_model_library_assets(self, temp_dir, engine_dir, engine_type="unreal"):
        """复制模型库中的所有模型文件到引擎项目目录"""
        self.logger.info("开始复制模型库资源...")
        
        # 确定目标目录
        if engine_type == "unreal":
            models_dir = os.path.join(engine_dir, "Content", "TerrainGenerator", "Models")
        else:  # unity
            models_dir = os.path.join(engine_dir, "Assets", "TerrainGenerator", "Models")
        
        # 确保目录存在
        os.makedirs(models_dir, exist_ok=True)
        
        # 查找基础模型目录
        base_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
        if not os.path.exists(base_model_dir):
            self.logger.warning(f"模型基础目录不存在: {base_model_dir}，将创建空白模型")
            base_model_dir = os.path.join(temp_dir, "default_models")
            os.makedirs(base_model_dir, exist_ok=True)
        
        # 对每个模型库中的模型
        for model_type, model_info in MODEL_LIBRARY.items():
            model_path = model_info.get("model", "")
            if not model_path:
                continue
                
            # 处理相对路径
            model_filename = os.path.basename(model_path)
            src_model_path = os.path.join(base_model_dir, model_filename)
            
            # 如果原始模型不存在，创建一个简单的默认模型
            if not os.path.exists(src_model_path):
                self.logger.warning(f"找不到模型: {model_filename}，将创建默认模型")
                # 创建一个简单的立方体OBJ
                with open(src_model_path, "w") as f:
                    f.write(f"""
    # Default {model_type} model
    v -0.5 0 -0.5
    v 0.5 0 -0.5
    v 0.5 0 0.5
    v -0.5 0 0.5
    v -0.5 1 -0.5
    v 0.5 1 -0.5
    v 0.5 1 0.5
    v -0.5 1 0.5
    f 1 2 3 4
    f 5 6 7 8
    f 1 5 8 4
    f 2 6 7 3
    f 1 2 6 5
    f 4 3 7 8
                    """)
            
            # 复制到目标目录
            dst_model_path = os.path.join(models_dir, model_filename)
            try:
                shutil.copy2(src_model_path, dst_model_path)
                self.logger.info(f"已复制模型: {model_filename}")
            except Exception as e:
                self.logger.error(f"复制模型失败 {model_filename}: {e}")

    def _copy_texture_library_assets(self, temp_dir, engine_dir, engine_type="unreal"):
            """复制材质库中的所有纹理文件到引擎项目目录"""
            self.logger.info("开始复制纹理库资源...")
            
            # 确定目标目录
            if engine_type == "unreal":
                textures_dir = os.path.join(engine_dir, "Content", "TerrainGenerator", "Textures")
            else:  # unity
                textures_dir = os.path.join(engine_dir, "Assets", "TerrainGenerator", "Textures")
            
            # 确保目录存在
            os.makedirs(textures_dir, exist_ok=True)
            
            # 查找基础纹理目录
            base_texture_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "textures")
            if not os.path.exists(base_texture_dir):
                self.logger.warning(f"纹理基础目录不存在: {base_texture_dir}，将创建默认纹理")
                base_texture_dir = os.path.join(temp_dir, "default_textures")
                os.makedirs(base_texture_dir, exist_ok=True)
            
            # 处理过的纹理，避免重复
            processed_textures = set()
            
            # 对每个材质库中的材质
            for material_type, material_info in MATERIAL_LIBRARY.items():
                for texture_type in ["diffuse", "normal", "roughness"]:
                    texture_path = material_info.get(texture_type, "")
                    # 添加类型检查，确保texture_path是字符串类型
                    if not texture_path or texture_path in processed_textures or not isinstance(texture_path, str):
                        continue
                        
                    processed_textures.add(texture_path)
                    texture_filename = os.path.basename(texture_path)
                    src_texture_path = os.path.join(base_texture_dir, texture_filename)
                    
                    # 如果原始纹理不存在，创建一个简单的默认纹理
                    if not os.path.exists(src_texture_path):
                        self.logger.warning(f"找不到纹理: {texture_filename}，将创建默认纹理")
                        # 创建默认纹理
                        img = Image.new('RGB', (512, 512))
                        
                        # 根据纹理类型设置不同的默认颜色
                        if texture_type == "diffuse":
                            color = (100, 100, 100) if "rock" in material_type else (
                                    (50, 150, 50) if "grass" in material_type else (
                                    (200, 200, 250) if "water" in material_type else (
                                    (200, 180, 150) if "sand" in material_type else (150, 150, 150))))
                        elif texture_type == "normal":
                            color = (128, 128, 255)  # 默认法线颜色 (Z up)
                        else:
                            color = (128, 128, 128)  # 灰色
                        
                        draw = ImageDraw.Draw(img)
                        draw.rectangle([(0, 0), (512, 512)], fill=color)
                        
                        # 添加棋盘格以便于调试
                        sq_size = 64
                        for y in range(0, 512, sq_size*2):
                            for x in range(0, 512, sq_size*2):
                                draw.rectangle([(x, y), (x+sq_size, y+sq_size)], fill=(
                                    color[0]//2, color[1]//2, color[2]//2))
                                draw.rectangle([(x+sq_size, y+sq_size), (x+sq_size*2, y+sq_size*2)], fill=(
                                    color[0]//2, color[1]//2, color[2]//2))
                        
                        img.save(src_texture_path)
                    
                    # 复制到目标目录
                    dst_texture_path = os.path.join(textures_dir, texture_filename)
                    try:
                        shutil.copy2(src_texture_path, dst_texture_path)
                        self.logger.info(f"已复制纹理: {texture_filename}")
                    except Exception as e:
                        self.logger.error(f"复制纹理失败 {texture_filename}: {e}")
                        
##################################                        
#一键部署
##################################
class DeploymentManager:
    """管理向不同引擎和3D软件的一键部署功能"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging
        self.engine_paths = self._detect_installed_engines()
        
    def _detect_installed_engines(self):
        """检测已安装的游戏引擎和3D软件路径"""
        engines = {}
        
        # 检测Unity安装路径
        unity_paths = []
        if sys.platform == "win32":
            # 在Windows上检查常见Unity安装位置
            unity_program_files = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Unity", "Hub", "Editor")
            if os.path.exists(unity_program_files):
                for version in os.listdir(unity_program_files):
                    version_path = os.path.join(unity_program_files, version)
                    if os.path.isdir(version_path):
                        unity_paths.append((version, version_path))
        elif sys.platform == "darwin":
            # macOS上的Unity路径
            unity_app = "/Applications/Unity/Hub/Editor"
            if os.path.exists(unity_app):
                for version in os.listdir(unity_app):
                    version_path = os.path.join(unity_app, version)
                    if os.path.isdir(version_path):
                        unity_paths.append((version, version_path))
        
        if unity_paths:
            # 按版本号排序，选择最新版本
            unity_paths.sort(key=lambda x: x[0], reverse=True)
            engines["unity"] = unity_paths[0][1]
        
        # 检测Unreal Engine安装路径
        unreal_paths = []
        if sys.platform == "win32":
            # Windows上的Unreal路径
            epic_launcher_path = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Epic Games")
            if os.path.exists(epic_launcher_path):
                for folder in os.listdir(epic_launcher_path):
                    if folder.startswith("UE_"):
                        unreal_paths.append((folder.replace("UE_", ""), os.path.join(epic_launcher_path, folder)))
        elif sys.platform == "darwin":
            # macOS上的Unreal路径
            unreal_app = "/Applications/Epic Games"
            if os.path.exists(unreal_app):
                for folder in os.listdir(unreal_app):
                    if folder.startswith("UE_"):
                        unreal_paths.append((folder.replace("UE_", ""), os.path.join(unreal_app, folder)))
        
        if unreal_paths:
            # 按版本号排序，选择最新版本
            unreal_paths.sort(key=lambda x: x[0], reverse=True)
            engines["unreal"] = unreal_paths[0][1]
        
        # 检测Blender安装路径
        blender_path = None
        if sys.platform == "win32":
            # Windows上的Blender路径
            blender_program_files = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Blender Foundation")
            if os.path.exists(blender_program_files):
                for folder in os.listdir(blender_program_files):
                    if folder.startswith("Blender "):
                        blender_path = os.path.join(blender_program_files, folder, "blender.exe")
                        break
        elif sys.platform == "darwin":
            # macOS上的Blender路径
            blender_app = "/Applications/Blender.app"
            if os.path.exists(blender_app):
                blender_path = os.path.join(blender_app, "Contents/MacOS/Blender")
        
        if blender_path and os.path.exists(blender_path):
            engines["blender"] = blender_path
            
        return engines
    
    def deploy_to_unity(self, package_dir, project_path=None):
        """将导出的Unity包部署到指定项目"""
        if "unity" not in self.engine_paths and not project_path:
            self.logger.error("未检测到Unity安装，无法自动部署")
            return False
            
        if not project_path:
            # 提示用户创建或选择项目
            print("请创建或选择一个Unity项目...")
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            project_path = filedialog.askdirectory(title="选择Unity项目文件夹")
            root.destroy()
            
            if not project_path:
                self.logger.error("未选择Unity项目，取消部署")
                return False
        
        # 确保项目路径存在
        if not os.path.exists(project_path):
            self.logger.error(f"Unity项目路径不存在: {project_path}")
            return False
            
        # 将导出的资源复制到项目的Assets文件夹
        assets_path = os.path.join(project_path, "Assets")
        if not os.path.exists(assets_path):
            os.makedirs(assets_path)
            
        # 复制TerrainGenerator文件夹到Assets
        src_path = os.path.join(package_dir, "Assets", "TerrainGenerator")
        dst_path = os.path.join(assets_path, "TerrainGenerator")
        
        if os.path.exists(dst_path):
            # 备份现有文件夹
            backup_path = f"{dst_path}_backup_{int(time.time())}"
            shutil.move(dst_path, backup_path)
            self.logger.info(f"已备份现有TerrainGenerator文件夹到: {backup_path}")
        
        shutil.copytree(src_path, dst_path)
        self.logger.info(f"已将导出的地形资源部署到Unity项目: {project_path}")
        
        # 创建自动导入指南
        with open(os.path.join(dst_path, "README_AutoImport.txt"), "w") as f:
            f.write(f"""
Unity自动导入指南
================

资源已成功部署到项目中。请按照以下步骤导入地形:

1. 打开Unity编辑器并加载此项目
2. 在菜单中选择: Tools > Import Generated Terrain
3. 在弹出的窗口中点击 "Import Terrain and Assets"
4. 等待导入完成，地形和所有模型将被自动放置

注意: 导入后如果地形显示异常，可能需要调整光照或摄像机设置。
""")
            
        return True
    
    def deploy_to_unreal(self, package_dir, project_path=None):
        """将导出的Unreal包部署到指定项目"""
        if "unreal" not in self.engine_paths and not project_path:
            self.logger.error("未检测到Unreal Engine安装，无法自动部署")
            return False
            
        if not project_path:
            # 提示用户创建或选择项目
            print("请选择一个Unreal Engine项目...")
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            project_path = filedialog.askopenfilename(
                title="选择Unreal项目文件(.uproject)",
                filetypes=[("Unreal项目", "*.uproject")]
            )
            root.destroy()
            
            if not project_path:
                self.logger.error("未选择Unreal项目，取消部署")
                return False
        
        # 确保项目文件存在
        if not os.path.exists(project_path):
            self.logger.error(f"Unreal项目文件不存在: {project_path}")
            return False
            
        # 将导出的资源复制到项目的Content文件夹
        project_dir = os.path.dirname(project_path)
        content_path = os.path.join(project_dir, "Content")
        if not os.path.exists(content_path):
            os.makedirs(content_path)
            
        # 复制TerrainGenerator文件夹到Content
        src_path = os.path.join(package_dir, "Content", "TerrainGenerator")
        dst_path = os.path.join(content_path, "TerrainGenerator")
        
        if os.path.exists(dst_path):
            # 备份现有文件夹
            backup_path = f"{dst_path}_backup_{int(time.time())}"
            shutil.move(dst_path, backup_path)
            self.logger.info(f"已备份现有TerrainGenerator文件夹到: {backup_path}")
        
        shutil.copytree(src_path, dst_path)
        self.logger.info(f"已将导出的地形资源部署到Unreal项目: {project_path}")
        
        # 创建自动导入指南
        with open(os.path.join(dst_path, "README_AutoImport.txt"), "w") as f:
            f.write(f"""
Unreal Engine自动导入指南
=======================

资源已成功部署到项目中。请按照以下步骤导入地形:

1. 打开Unreal编辑器并加载此项目
2. 打开Python控制台: Window > Developer Tools > Python Console
3. 执行以下命令导入地形:
   exec(open("/Game/TerrainGenerator/ImportScript.py").read())
4. 等待导入完成，地形和所有模型将被自动放置

注意: 导入后可能需要重新构建光照才能获得最佳效果。
""")
            
        return True
        
    def deploy_to_blender(self, obj_file_path):
        """在Blender中打开导出的OBJ文件"""
        if "blender" not in self.engine_paths:
            self.logger.error("未检测到Blender安装，无法自动部署")
            return False
            
        blender_path = self.engine_paths["blender"]
        
        # 创建一个Blender Python脚本来导入OBJ文件
        script_path = os.path.join(os.path.dirname(obj_file_path), "import_to_blender.py")
        with open(script_path, "w") as f:
            f.write(f"""
import bpy
import os

# 清除默认场景
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 导入OBJ文件
obj_path = r"{obj_file_path.replace(os.sep, '/')}"
bpy.ops.import_scene.obj(filepath=obj_path)

# 选择导入的对象并居中
bpy.ops.object.select_all(action='SELECT')
bpy.ops.view3d.view_all(center=True)

# 设置渲染引擎为Cycles以获得更好的效果
bpy.context.scene.render.engine = 'CYCLES'

# 添加简单的环境光照
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)
bg.inputs[1].default_value = 1.0

print("地形已成功导入到Blender中!")
""")
        
        # 使用Blender运行此脚本
        import subprocess
        try:
            subprocess.Popen([blender_path, "--python", script_path])
            self.logger.info(f"正在使用Blender打开OBJ文件: {obj_file_path}")
            return True
        except Exception as e:
            self.logger.error(f"使用Blender打开文件失败: {e}")
            return False
        
def export_and_deploy(map_data, target_engine="obj", config=None, logger=None, project_path=None):
    """
    一键导出并部署地图到指定引擎
    
    参数:
        map_data: 地图数据
        target_engine: 目标引擎/格式 ("obj", "unity", "unreal", "blender")
        config: 导出配置
        logger: 日志记录器
        project_path: 引擎项目路径，如果为None则自动检测或提示选择
        
    返回:
        导出的文件路径或目录
    """
    logger = logger or logging.getLogger("MapExporter")
    config = config or MapExportConfig()
    
    # 如果target_engine是blender，实际上使用obj格式
    actual_format = "obj" if target_engine == "blender" else target_engine
    
    if actual_format == "obj":
        exporter = ObjExporter(config, logger)
        result = exporter.export(map_data, auto_deploy=(target_engine == "blender"))
        
    elif actual_format == "unity":
        exporter = UnityExporter(config, logger)
        result = exporter.export(map_data, auto_deploy=True, project_path=project_path)
        
    elif actual_format == "unreal":
        exporter = UnrealExporter(config, logger)
        result = exporter.export(map_data, auto_deploy=True, project_path=project_path)
        
    else:
        logger.error(f"不支持的目标引擎: {target_engine}")
        raise ValueError(f"不支持的目标引擎: {target_engine}")
        
    logger.info(f"已完成导出并部署到 {target_engine}")
    return result


########################
#导出我的世界风格
########################
class MinecraftExporter(MapExporter):
    """导出为Minecraft格式的地图导出器"""
    
    def __init__(self, config, logger=None):
        """初始化Minecraft导出器
        
        Args:
            config: 导出配置
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        # Minecraft方块映射表 - 将生物群系映射到Minecraft方块ID
        self.block_mapping = {
            "forest": "minecraft:grass_block",
            "taiga": "minecraft:podzol",
            "jungle": "minecraft:jungle_leaves",
            "desert": "minecraft:sand",
            "savanna": "minecraft:grass_block",
            "plains": "minecraft:grass_block",
            "swamp": "minecraft:grass_block",
            "tundra": "minecraft:snow_block",
            "snow": "minecraft:snow_block",
            "mountain": "minecraft:stone",
            "hills": "minecraft:grass_block",
            "water": "minecraft:water",
            "deep_water": "minecraft:water",
            "river": "minecraft:water",
            "beach": "minecraft:sand",
            "ocean": "minecraft:water",
            "grassland": "minecraft:grass_block",
            "rocky": "minecraft:stone",
            "volcanic": "minecraft:basalt",
            # 默认方块
            "default": "minecraft:grass_block"
        }
        
        # 植被映射表 - 将植被类型映射到Minecraft方块
        self.vegetation_mapping = {
            "tree": "minecraft:oak_log",
            "pine": "minecraft:spruce_log",
            "palm": "minecraft:jungle_log",
            "cactus": "minecraft:cactus",
            "bush": "minecraft:oak_leaves",
            # 默认植被
            "default": "minecraft:oak_leaves"
        }
        
        # 地下方块映射
        self.underground_mapping = {
            "stone": "minecraft:stone",
            "dirt": "minecraft:dirt",
            "cave": "minecraft:cave_air",
            "ore_iron": "minecraft:iron_ore",
            "ore_coal": "minecraft:coal_ore",
            "ore_gold": "minecraft:gold_ore",
            "ore_diamond": "minecraft:diamond_ore",
            "ore_emerald": "minecraft:emerald_ore",
            "ore_lapis": "minecraft:lapis_ore",
            "ore_redstone": "minecraft:redstone_ore",
            "lava": "minecraft:lava",
            "gravel": "minecraft:gravel",
            # 默认地下方块
            "default": "minecraft:stone"
        }

    def export(self, map_data):
        """导出地图数据为Minecraft格式
        
        Args:
            map_data: 要导出的地图数据
            
        Returns:
            str: 导出文件的路径
        """
        self.logger.info("开始导出为Minecraft格式...")
        
        # 确保输出目录存在
        output_dir = os.path.join(self.config.output_dir, "minecraft")
        os.makedirs(output_dir, exist_ok=True)
        
        # 确定导出文件名
        base_filename = self.config.base_filename or "map"
        schematic_path = os.path.join(output_dir, f"{base_filename}.schem")
        structure_path = os.path.join(output_dir, f"{base_filename}.nbt")
        
        # 根据Minecraft样式设置确定参数
        style_options = getattr(self.config, "style_options", {})
        max_height = style_options.get("max_height", 255)
        sea_level = style_options.get("sea_level", 63)
        block_resolution = style_options.get("block_resolution", 1)
        
        # 获取地图数据
        height_map = map_data.get_layer("height")
        biome_map = map_data.get_layer("biome")
        
        if height_map is None or biome_map is None:
            self.logger.error("缺少必要的地图数据层")
            return None
            
        # 计算Minecraft世界大小
        map_width = len(height_map[0])
        map_height = len(height_map)
        mc_width = map_width // block_resolution
        mc_height = map_height // block_resolution
        
        self.logger.info(f"Minecraft地图大小: {mc_width}x{mc_height}")
        
        try:
            # 调用两种格式的导出方法
            self._export_schematic(
                height_map, biome_map, map_data,
                schematic_path, mc_width, mc_height,
                max_height, sea_level, block_resolution
            )
            
            self._export_structure(
                height_map, biome_map, map_data,
                structure_path, mc_width, mc_height,
                max_height, sea_level, block_resolution
            )
            
            # 创建Minecraft方块映射表JSON配置文件
            mapping_path = os.path.join(output_dir, "block_mapping.json")
            with open(mapping_path, 'w') as f:
                json.dump({
                    "biome_to_block": self.block_mapping,
                    "vegetation_to_block": self.vegetation_mapping,
                    "underground_to_block": self.underground_mapping
                }, f, indent=2)
            
            # 如果启用了元数据导出，保存地图元数据
            if self.config.include_metadata:
                metadata_path = os.path.join(output_dir, f"{base_filename}_metadata.json")
                self._export_metadata(map_data, metadata_path, max_height, sea_level)
            
            # 创建预览图
            self._create_preview(height_map, biome_map, os.path.join(output_dir, f"{base_filename}_preview.png"))
            
            return output_dir
            
        except Exception as e:
            self.logger.error(f"Minecraft导出失败: {str(e)}")
            return None
    
    def _export_schematic(self, height_map, biome_map, map_data, filepath, width, height, 
                        max_height, sea_level, block_resolution):
        """导出为Schematic格式
        
        Args:
            height_map: 高度图数据
            biome_map: 生物群系数据
            map_data: 完整地图数据
            filepath: 输出文件路径
            width: Minecraft世界宽度
            height: Minecraft世界高度
            max_height: Minecraft最大高度
            sea_level: 海平面高度
            block_resolution: 方块分辨率 (每个地图单元对应多少方块)
        """
        try:
            import nbtlib
            from nbtlib.tag import ByteArray, String, List, Compound, IntArray, Int
            
            self.logger.info("正在创建Schematic文件...")
            
            # 计算方块数据
            blocks = []
            block_data = []
            block_entities = []
            
            # 跟踪已创建的方块位置
            block_positions = set()
            
            # 计算高度缩放因子
            height_min = np.min(height_map)
            height_max = np.max(height_map)
            height_scale = (max_height - sea_level) / (height_max - height_min + 1e-6)
            
            # 准备方块数据
            for z in range(height):
                for x in range(width):
                    map_x = min(int(x * block_resolution), len(height_map[0]) - 1)
                    map_z = min(int(z * block_resolution), len(height_map) - 1)
                    
                    # 获取生物群系和高度
                    biome_type = biome_map[map_z][map_x]
                    h_val = height_map[map_z][map_x]
                    
                    # 计算Minecraft中的高度 (将高度值映射到Minecraft的高度范围)
                    mc_y = int(sea_level + (h_val - height_min) * height_scale)
                    mc_y = max(1, min(mc_y, max_height - 1))
                    
                    # 记录这个位置有方块
                    block_positions.add((x, mc_y, z))
                    
                    # 根据生物群系选择方块类型
                    block_id = self.block_mapping.get(biome_type, self.block_mapping["default"])
                    
                    # 水下方块和陆地方块的处理
                    if mc_y < sea_level and biome_type not in ["water", "river", "deep_water", "ocean"]:
                        # 水下用沙子或砂砾
                        if random.random() < 0.7:
                            block_id = "minecraft:sand"
                        else:
                            block_id = "minecraft:gravel"
                    
                    # 添加方块数据
                    blocks.append(block_id)
                    block_data.append(0)  # 方块数据值
                    
                    # 为地面以下添加几层方块
                    for y_offset in range(1, 4):
                        new_y = mc_y - y_offset
                        if new_y > 0 and (x, new_y, z) not in block_positions:
                            block_positions.add((x, new_y, z))
                            if y_offset == 1:
                                # 紧挨着地表的一层
                                if biome_type in ["desert", "beach"]:
                                    blocks.append("minecraft:sand")
                                else:
                                    blocks.append("minecraft:dirt")
                            else:
                                # 更深的层
                                blocks.append("minecraft:stone")
                            block_data.append(0)
            
            # 处理植被
            if hasattr(map_data, "vegetation") and map_data.vegetation:
                for veg in map_data.vegetation:
                    # 确保X和Y在有效范围内
                    if 0 <= veg.get("x", 0) < len(height_map[0]) and 0 <= veg.get("y", 0) < len(height_map):
                        map_x = veg.get("x", 0)
                        map_z = veg.get("y", 0)
                        
                        # 转换为Minecraft坐标
                        mc_x = min(int(map_x / block_resolution), width - 1)
                        mc_z = min(int(map_z / block_resolution), height - 1)
                        
                        # 获取高度
                        h_val = height_map[map_z][map_x]
                        mc_y = int(sea_level + (h_val - height_min) * height_scale) + 1
                        
                        # 检查是否超出高度上限
                        if mc_y >= max_height:
                            continue
                            
                        # 获取植被类型
                        veg_type = veg.get("type", "tree")
                        block_id = self.vegetation_mapping.get(veg_type, self.vegetation_mapping["default"])
                        
                        # 添加植被方块
                        if (mc_x, mc_y, mc_z) not in block_positions:
                            block_positions.add((mc_x, mc_y, mc_z))
                            blocks.append(block_id)
                            block_data.append(0)
            
            # 创建Schematic NBT数据
            # 修改这里：使用nbtlib.List([])来替代List()
            entities_list = nbtlib.List([])
            tile_entities_list = nbtlib.List([])
            
            schematic = {
                "Width": nbtlib.Short(width),
                "Height": nbtlib.Short(max_height),
                "Length": nbtlib.Short(height),
                "Materials": nbtlib.String("Alpha"),
                "Blocks": ByteArray([0] * len(blocks)),  # 占位符
                "Data": ByteArray([0] * len(block_data)),
                "Entities": entities_list,
                "TileEntities": tile_entities_list
            }
            
            # 转换方块列表为ID索引
            block_id_map = {}
            next_id = 0
            
            for block in blocks:
                if block not in block_id_map:
                    block_id_map[block] = next_id
                    next_id += 1
            
            # 填充方块数据
            block_bytes = ByteArray([block_id_map[block] for block in blocks])
            schematic["Blocks"] = block_bytes
            
            # 保存Schematic文件
            nbt_file = nbtlib.File({"Schematic": nbtlib.Compound(schematic)})
            nbt_file.save(filepath)
            
            self.logger.info(f"Schematic文件已保存到: {filepath}")
            
        except ImportError:
            self.logger.warning("缺少nbtlib库，无法创建Schematic格式文件")
            # 创建一个占位文件表明需要nbtlib
            with open(filepath, 'w') as f:
                f.write("需要安装nbtlib库以创建Schematic文件\n")
                f.write("请运行: pip install nbtlib")
    
    def _export_structure(self, height_map, biome_map, map_data, filepath, width, height,
                        max_height, sea_level, block_resolution):
        """导出为Minecraft Structure NBT格式"""
        try:
            import nbtlib
            from nbtlib import File
            from nbtlib.tag import ByteArray, IntArray, String, Compound, Int, List
            
            self.logger.info("正在创建Structure文件...")
            
            # 计算高度缩放因子
            height_min = np.min(height_map)
            height_max = np.max(height_map)
            height_scale = (max_height - sea_level) / (height_max - height_min + 1e-6)
            
            # 准备Structure数据 - 使用nbtlib.List
            blocks_data = []
            
            # 跟踪已创建的方块位置
            block_positions = set()
            
            # 遍历地图创建方块
            for z in range(height):
                for x in range(width):
                    map_x = min(int(x * block_resolution), len(height_map[0]) - 1)
                    map_z = min(int(z * block_resolution), len(height_map) - 1)
                    
                    # 获取生物群系和高度
                    biome_type = biome_map[map_z][map_x]
                    h_val = height_map[map_z][map_x]
                    
                    # 计算Minecraft中的高度
                    mc_y = int(sea_level + (h_val - height_min) * height_scale)
                    mc_y = max(1, min(mc_y, max_height - 1))
                    
                    # 记录这个位置有方块
                    pos = (x, mc_y, z)
                    if pos in block_positions:
                        continue
                        
                    block_positions.add(pos)
                    
                    # 根据生物群系选择方块类型
                    block_id = self.block_mapping.get(biome_type, self.block_mapping["default"])
                    
                    # 水下方块和陆地方块的处理
                    if mc_y < sea_level and biome_type not in ["water", "river", "deep_water", "ocean"]:
                        if random.random() < 0.7:
                            block_id = "minecraft:sand"
                        else:
                            block_id = "minecraft:gravel"
                    
                    # 创建方块数据 - 使用nbtlib的Compound和Int
                    block = Compound({
                        "pos": IntArray([x, mc_y, z]),
                        "state": Int(len(blocks_data)),
                        "nbt": Compound({})
                    })
                    blocks_data.append(block)
                    
                    # 为地面以下添加几层方块
                    for y_offset in range(1, 4):
                        new_y = mc_y - y_offset
                        if new_y > 0 and (x, new_y, z) not in block_positions:
                            block_positions.add((x, new_y, z))
                            
                            underground_block = "minecraft:dirt" if y_offset == 1 else "minecraft:stone"
                            if biome_type in ["desert", "beach"] and y_offset == 1:
                                underground_block = "minecraft:sand"
                                
                            block = Compound({
                                "pos": IntArray([x, new_y, z]),
                                "state": Int(len(blocks_data)),
                                "nbt": Compound({})
                            })
                            blocks_data.append(block)
            
            # 处理植被和建筑
            self._add_vegetation_to_structure(height_map, map_data, blocks_data, block_positions, 
                                            width, height, max_height, sea_level, 
                                            height_min, height_scale, block_resolution)
            
            # 创建方块状态列表 (palette)
            palette = []
            
            # 收集所有使用到的方块ID
            block_ids = set(self.block_mapping.values()) | set(self.vegetation_mapping.values())
            for block_id in block_ids:
                palette.append(Compound({
                    "Name": String(block_id),
                    "Properties": Compound({})
                }))
            
            # 创建NBT结构的完整数据 - 使用nbtlib.List包装列表
            blocks_list = nbtlib.List(blocks_data)
            palette_list = nbtlib.List(palette)
            entities_list = nbtlib.List([])
            
            structure_data = {
                "DataVersion": Int(2584),  # Minecraft 1.16.5的DataVersion
                "size": IntArray([width, max_height, height]),
                "palette": palette_list,
                "blocks": blocks_list,
                "entities": entities_list
            }
            
            # 使用nbtlib创建和保存NBT文件
            nbt_file = nbtlib.File(structure_data)
            nbt_file.save(filepath)
            
            self.logger.info(f"Structure文件已保存到: {filepath}")
            
        except ImportError:
            self.logger.warning("缺少nbtlib库，无法创建Structure格式文件")
            with open(filepath, 'w') as f:
                f.write("需要安装nbtlib库以创建Structure文件\n")
                f.write("请运行: pip install nbtlib")
        except Exception as e:
            self.logger.error(f"创建Structure文件失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _add_vegetation_to_structure(self, height_map, map_data, blocks, block_positions, 
                                    width, height, max_height, sea_level, 
                                    height_min, height_scale, block_resolution):
        """向Structure中添加植被和建筑物
        
        Args:
            height_map: 高度图数据
            map_data: 地图数据
            blocks: 方块列表
            block_positions: 已存在的方块位置集合
            width: Minecraft世界宽度
            height: Minecraft世界高度
            max_height: Minecraft最大高度
            sea_level: 海平面高度
            height_min: 高度最小值
            height_scale: 高度缩放因子
            block_resolution: 方块分辨率
        """
        try:
            import nbtlib
            from nbtlib.tag import IntArray, Compound, Int
            
            # 处理植被
            if hasattr(map_data, "vegetation") and map_data.vegetation:
                for veg in map_data.vegetation:
                    # 确保坐标在有效范围内
                    if 0 <= veg.get("x", 0) < len(height_map[0]) and 0 <= veg.get("y", 0) < len(height_map):
                        map_x = veg.get("x", 0)
                        map_z = veg.get("y", 0)
                        
                        # 转换为Minecraft坐标
                        mc_x = min(int(map_x / block_resolution), width - 1)
                        mc_z = min(int(map_z / block_resolution), height - 1)
                        
                        # 获取高度
                        h_val = height_map[map_z][map_x]
                        mc_y = int(sea_level + (h_val - height_min) * height_scale) + 1
                        
                        # 检查是否超出高度上限
                        if mc_y >= max_height:
                            continue
                            
                        # 获取植被类型
                        veg_type = veg.get("type", "tree")
                        block_id = self.vegetation_mapping.get(veg_type, self.vegetation_mapping["default"])
                        
                        # 添加植被方块 - 修复：使用nbtlib的Compound
                        pos = (mc_x, mc_y, mc_z)
                        if pos not in block_positions:
                            block_positions.add(pos)
                            block = Compound({
                                "pos": IntArray([mc_x, mc_y, mc_z]),
                                "state": Int(len(blocks)),
                                "nbt": Compound({})
                            })
                            blocks.append(block)
                            
                            # 如果是树，添加树干和树叶
                            if veg_type in ["tree", "pine"]:
                                # 树干高度
                                trunk_height = random.randint(3, 6)
                                leaves_radius = random.randint(1, 3)
                                
                                # 添加树干 - 修复：使用nbtlib的Compound
                                for y_offset in range(1, trunk_height):
                                    if mc_y + y_offset < max_height and (mc_x, mc_y + y_offset, mc_z) not in block_positions:
                                        block_positions.add((mc_x, mc_y + y_offset, mc_z))
                                        trunk_block = Compound({
                                            "pos": IntArray([mc_x, mc_y + y_offset, mc_z]),
                                            "state": Int(len(blocks)),
                                            "nbt": Compound({})
                                        })
                                        blocks.append(trunk_block)
                                
                                # 添加树叶 - 修复：使用nbtlib的Compound
                                leaves_block = "minecraft:oak_leaves" if veg_type == "tree" else "minecraft:spruce_leaves"
                                for y_offset in range(trunk_height - 2, trunk_height + 2):
                                    for x_offset in range(-leaves_radius, leaves_radius + 1):
                                        for z_offset in range(-leaves_radius, leaves_radius + 1):
                                            # 计算到树干的距离
                                            dist = math.sqrt(x_offset**2 + z_offset**2)
                                            if dist <= leaves_radius and mc_y + y_offset < max_height:
                                                new_x = mc_x + x_offset
                                                new_y = mc_y + y_offset
                                                new_z = mc_z + z_offset
                                                
                                                if 0 <= new_x < width and 0 <= new_z < height and (new_x, new_y, new_z) not in block_positions:
                                                    block_positions.add((new_x, new_y, new_z))
                                                    leaf_block = Compound({
                                                        "pos": IntArray([new_x, new_y, new_z]),
                                                        "state": Int(len(blocks)),
                                                        "nbt": Compound({})
                                                    })
                                                    blocks.append(leaf_block)
            
            # 处理建筑物 (如果有)
            if hasattr(map_data, "buildings") and map_data.buildings:
                self._add_buildings_to_structure(height_map, map_data.buildings, blocks, block_positions,
                                            width, height, max_height, sea_level,
                                            height_min, height_scale, block_resolution)
            
        except ImportError:
            self.logger.warning("无法添加植被，缺少nbtlib库")
        except Exception as e:
            self.logger.error(f"添加植被时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _add_house_structure(self, blocks, block_positions, x, y, z, width, length, height):
        """添加简单的房屋结构
        
        Args:
            blocks: 方块列表
            block_positions: 已存在的方块位置集合
            x, y, z: 房屋左下角坐标
            width, length, height: 房屋尺寸
        """
        try:
            import nbtlib
            from nbtlib.tag import IntArray, Compound, Int
            
            # 房屋材料
            wall_block = "minecraft:oak_planks"
            floor_block = "minecraft:oak_planks"
            roof_block = "minecraft:oak_stairs"
            
            # 添加地板 - 修复：使用nbtlib的Compound
            for dx in range(width):
                for dz in range(length):
                    pos = (x + dx, y, z + dz)
                    if pos not in block_positions:
                        block_positions.add(pos)
                        block = Compound({
                            "pos": IntArray([pos[0], pos[1], pos[2]]),
                            "state": Int(len(blocks)),
                            "nbt": Compound({})
                        })
                        blocks.append(block)
            
            # 添加墙壁 - 修复：使用nbtlib的Compound
            for dy in range(1, height):
                for dx in range(width):
                    for dz in range(length):
                        # 只添加边缘的墙壁
                        if dx == 0 or dx == width - 1 or dz == 0 or dz == length - 1:
                            # 为门留出空间
                            if not (dy < 3 and dx == width // 2 and dz == 0):
                                pos = (x + dx, y + dy, z + dz)
                                if pos not in block_positions:
                                    block_positions.add(pos)
                                    block = Compound({
                                        "pos": IntArray([pos[0], pos[1], pos[2]]),
                                        "state": Int(len(blocks)),
                                        "nbt": Compound({})
                                    })
                                    blocks.append(block)
            
            # 添加屋顶 - 修复：使用nbtlib的Compound
            for dx in range(width):
                for dz in range(length):
                    pos = (x + dx, y + height, z + dz)
                    if pos not in block_positions:
                        block_positions.add(pos)
                        block = Compound({
                            "pos": IntArray([pos[0], pos[1], pos[2]]),
                            "state": Int(len(blocks)),
                            "nbt": Compound({})
                        })
                        blocks.append(block)
            
        except ImportError:
            self.logger.warning("无法添加房屋结构，缺少nbtlib库")

    def _add_buildings_to_structure(self, height_map, buildings, blocks, block_positions,
                                width, height, max_height, sea_level,
                                height_min, height_scale, block_resolution):
        """向Structure中添加建筑物"""
        try:
            import nbtlib
            from nbtlib.tag import IntArray, Compound, Int
            
            for building in buildings:
                # 确保坐标在有效范围内
                if 0 <= building.get("x", 0) < len(height_map[0]) and 0 <= building.get("y", 0) < len(height_map):
                    # 这里添加简单的房屋结构
                    map_x = building.get("x", 0)
                    map_z = building.get("y", 0)
                    
                    # 转换为Minecraft坐标
                    mc_x = min(int(map_x / block_resolution), width - 1)
                    mc_z = min(int(map_z / block_resolution), height - 1)
                    
                    # 获取高度
                    h_val = height_map[map_z][map_x]
                    mc_y = int(sea_level + (h_val - height_min) * height_scale)
                    
                    # 检查是否超出高度上限
                    if mc_y + 5 >= max_height:
                        continue
                        
                    # 创建简单的房屋结构
                    house_width = random.randint(3, 6)
                    house_length = random.randint(3, 6)
                    house_height = random.randint(3, 5)
                    
                    # 添加房屋方块
                    self._add_house_structure(blocks, block_positions, mc_x, mc_y, mc_z,
                                            house_width, house_length, house_height)
        
        except Exception as e:
            self.logger.error(f"添加建筑时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _export_metadata(self, map_data, filepath, max_height, sea_level):
        """导出地图元数据
        
        Args:
            map_data: 地图数据
            filepath: 输出文件路径
            max_height: Minecraft最大高度
            sea_level: 海平面高度
        """
        # 收集元数据
        metadata = {
            "minecraft_version": "1.16.5",
            "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "map_width": map_data.width,
            "map_height": map_data.height,
            "max_height": max_height,
            "sea_level": sea_level,
            "biomes": {},
            "block_counts": {}
        }
        
        # 如果有生物群系数据，计算生物群系统计信息
        biome_map = map_data.get_layer("biome")
        if biome_map is not None:
            unique_biomes, counts = np.unique(biome_map, return_counts=True)
            for biome, count in zip(unique_biomes, counts):
                if isinstance(biome, (int, float, str)):
                    biome_name = str(biome)
                    metadata["biomes"][biome_name] = int(count)
                    
                    # 记录方块数量估计
                    block_id = self.block_mapping.get(biome_name, self.block_mapping["default"])
                    if block_id not in metadata["block_counts"]:
                        metadata["block_counts"][block_id] = 0
                    metadata["block_counts"][block_id] += int(count)
        
        # 保存元数据文件
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"元数据已保存到: {filepath}")
    
    def _create_preview(self, height_map, biome_map, filepath):
        """创建预览图
        
        Args:
            height_map: 高度图数据
            biome_map: 生物群系数据
            filepath: 输出文件路径
        """
        try:
            from matplotlib import pyplot as plt
            import numpy as np
            
            # 创建RGB图像
            height = len(height_map)
            width = len(height_map[0])
            
            # 确定颜色映射
            biome_colors = {
                "forest": [34, 139, 34],     # 森林绿
                "taiga": [0, 100, 0],        # 针叶林深绿
                "jungle": [0, 128, 0],       # 丛林绿
                "desert": [210, 180, 140],   # 沙漠黄褐
                "savanna": [189, 183, 107],  # 草原黄绿
                "plains": [124, 252, 0],     # 平原浅绿
                "swamp": [107, 142, 35],     # 沼泽深绿棕
                "tundra": [192, 192, 192],   # 苔原灰
                "snow": [255, 250, 250],     # 雪白
                "mountain": [128, 128, 128], # 山脉灰
                "hills": [154, 205, 50],     # 丘陵亮绿
                "water": [65, 105, 225],     # 水蓝
                "deep_water": [0, 0, 139],   # 深海蓝
                "river": [30, 144, 255],     # 河流淡蓝
                "beach": [238, 214, 175],    # 沙滩浅黄
                "ocean": [25, 25, 112],      # 海洋深蓝
                "grassland": [173, 255, 47], # 草地亮绿
                "rocky": [169, 169, 169]     # 岩石灰
            }
            
            # 默认颜色
            default_color = [100, 100, 100]
            
            # 创建图像数组
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 填充图像
            for y in range(height):
                for x in range(width):
                    biome = biome_map[y][x]
                    color = biome_colors.get(biome, default_color)
                    
                    # 根据高度调整亮度
                    h = height_map[y][x]
                    h_min = np.min(height_map)
                    h_max = np.max(height_map)
                    
                    # 高度系数 (0.5-1.5)
                    height_factor = 0.5 + ((h - h_min) / (h_max - h_min + 1e-6)) * 1.0
                    
                    # 应用高度因子到颜色
                    adjusted_color = [min(255, int(c * height_factor)) for c in color]
                    img[y, x] = adjusted_color
            
            # 保存图像
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"预览图已保存到: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"创建预览图失败: {str(e)}")