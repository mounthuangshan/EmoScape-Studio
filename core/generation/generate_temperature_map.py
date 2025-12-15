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
from core.services.map_tools import *

def generate_temperature_map(height_map, width, height, seed=None, latitude_effect=0.5, use_frequency_optimization=True):
    """生成考虑高度和纬度的温度图
    
    Args:
        height_map: 高度图数组
        width: 地图宽度
        height: 地图高度
        seed: 随机种子
        latitude_effect: 纬度影响强度
        use_frequency_optimization: 是否使用频域优化
        
    Returns:
        温度图数组
    """
    # 使用参数种子或默认值
    temp_seed = 1234 if seed is None else seed+10
    
    # 基础温度噪声
    if use_frequency_optimization:
        temp_base = frequency_domain_noise(width, height, temp_seed, octaves=4, persistence=0.5, lacunarity=2.0)
    else:
        temp_base = advanced_fractal_noise_scalar(width, height, temp_seed, octaves=4, persistence=0.5, lacunarity=2.0)
    
    # 添加北半球纬度影响(基于y坐标，中心高温向极地逐渐降温)
    latitude = np.zeros_like(temp_base)
    for y in range(height):
        # 使用非线性映射来模拟更真实的纬度温度分布
        normalized_y = 2 * abs(y/height - 0.5)  # 0在中心，1在边缘
        
        # 哈德利环流影响 - 赤道和副热带高压(约30°)温度高，中纬度(约60°)温度较低
        if normalized_y < 0.33:  # 赤道到副热带
            latitude_temp = 1.0 - normalized_y * 0.2  # 较小降温
        elif normalized_y < 0.67:  # 副热带到温带
            latitude_temp = 0.93 - (normalized_y - 0.33) * 0.7  # 较大降温
        else:  # 温带到极地
            latitude_temp = 0.7 - (normalized_y - 0.67) * 0.7  # 极地区域温度低
        
        for x in range(width):
            latitude[y, x] = latitude_temp
    
    # 合并温度、高度和纬度影响
    temp_map = temp_base * (1-latitude_effect) + latitude * latitude_effect
    
    # 高度影响 - 自适应比例根据地图尺寸和高度范围调整
    height_min = np.min(height_map)
    height_max = np.max(height_map)
    height_range = max(1.0, height_max - height_min)
    
    # 标准大气温度梯度约为6.5°C/km，映射到0-1范围
    height_coefficient = 0.15 * (100.0 / height_range)  # 自适应系数
    
    # 计算高度影响
    height_effect = np.zeros_like(height_map)
    for y in range(height):
        for x in range(width):
            # 非线性高度效应 - 低海拔变化较小，高海拔变化更显著
            normalized_height = (height_map[y, x] - height_min) / height_range
            height_effect[y, x] = height_coefficient * np.power(normalized_height, 1.2)
    
    # 应用高度效应
    temp_map = temp_map - height_effect
    
    return normalize_array(temp_map)