"""
河流生成模块 - 基于水文学的逼真河流网络生成

此模块提供了生成逼真河流的功能，使用先进的水文学模拟算法。
主要特点:
- 基于D8流向模型计算流向场
- 精确的降水和流量累积模拟
- 自然的河流蜿蜒度和分叉
- 高性能并行计算
"""

# 标准库
import random
import hashlib
import time
from typing import Tuple, List, Optional, Dict, Any, Union, Callable

# 数据处理与科学计算
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, minimum_filter, maximum_filter, label, binary_dilation
from numba import jit, prange, int32, float32, float64
import joblib
from scipy.signal import convolve2d
from tqdm import tqdm

# 图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 日志
import logging
logger = logging.getLogger(__name__)

# 项目文件
from utils.tools import *
from core.services.map_tools import *
from .river_system import (
    RiverGenerator, 
    RiverGenerationConfig, 
    FlowModel,
    RiverExtractor
)


def generate_realistic_rivers(height_map, min_watershed_size=50, precipitation_factor=1.0, 
                           meander_factor=0.3, seed=None):
    """增强版河流生成算法，增加河流蜿蜒度和分叉自然性
    
    Args:
        height_map: 高度图数组
        min_watershed_size: 最小集水区面积
        precipitation_factor: 降水系数
        meander_factor: 河流弯曲度系数
        seed: 随机种子
        
    Returns:
        元组: (修改后的高度图, 河流路径列表, 流量累积图)
    """
    # 使用新的RiverGenerator类
    config = RiverGenerationConfig(
        min_watershed_size=min_watershed_size,
        precipitation_factor=precipitation_factor,
        meander_factor=meander_factor,
        seed=seed,
        verbose=True
    )
    
    generator = RiverGenerator(config)
    updated_height_map, river_features, flow_accum = generator.generate(height_map)
    
    return updated_height_map, river_features, flow_accum


def generate_rivers_package(height_map, min_watershed_size=50, num_attempts=100, 
                         precipitation_factor=1.0, meander_factor=0.3, seed=None):
    """生成逼真的河流网络，基于水文学方法
    
    Args:
        height_map: 高度图数组
        min_watershed_size: 最小集水区面积
        num_attempts: 尝试次数 (已不使用，保留参数兼容性)
        precipitation_factor: 降水系数
        meander_factor: 河流弯曲度系数
        seed: 随机种子
        
    Returns:
        元组: (修改后的高度图, 河流路径列表)
    """
    # 这是对generate_realistic_rivers函数的包装
    result_map, river_features, _ = generate_realistic_rivers(
        height_map, 
        min_watershed_size=min_watershed_size, 
        precipitation_factor=precipitation_factor,
        meander_factor=meander_factor,
        seed=seed
    )
    return result_map, river_features


def generate_rivers_map(height_map, width, height, seed=None, min_watershed_size=50, 
                      precipitation_factor=1.0, meander_factor=0.3):
    """生成河流图层和特征
    
    Args:
        height_map: 高度图数组
        width: 地图宽度
        height: 地图高度
        seed: 随机种子
        min_watershed_size: 最小集水区面积
        precipitation_factor: 降水系数
        meander_factor: 河流弯曲度系数
        
    Returns:
        元组: (河流布尔地图, 河流路径列表, 更新后的高度图)
    """
    # 使用generate_rivers_package函数生成河流
    start_time = time.time()
    logger.info("开始生成河流...")
    
    updated_height_map, river_features = generate_rivers_package(
        height_map,
        min_watershed_size=min_watershed_size,
        precipitation_factor=precipitation_factor,
        meander_factor=meander_factor,
        seed=seed
    )
    
    # 创建河流布尔地图
    rivers_map = np.zeros((height, width), dtype=bool)
    
    # 将河流特征转换为布尔地图
    for river_path in river_features:
        for y, x in river_path:
            if 0 <= y < height and 0 <= x < width:
                rivers_map[y, x] = True
    
    elapsed = time.time() - start_time
    logger.info(f"河流生成完成, 用时 {elapsed:.2f} 秒, 生成了 {len(river_features)} 条河流")
    
    return rivers_map, river_features, updated_height_map