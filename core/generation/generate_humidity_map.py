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

def generate_humidity_map(height_map, width, height, temp_map=None, seed=None, 
                       prevailing_wind=(1, 0), use_frequency_optimization=True):
    """生成考虑高度、温度和风向的湿度图
    
    Args:
        height_map: 高度图数组
        width: 地图宽度
        height: 地图高度
        temp_map: 温度图数组
        seed: 随机种子
        prevailing_wind: 主导风向(x,y)
        use_frequency_optimization: 是否使用频域优化
        
    Returns:
        湿度图数组
    """
    # 使用参数种子或默认值
    moisture_seed = 5678 if seed is None else seed+20
    
    # 生成基础湿度噪声
    if use_frequency_optimization:
        base_moisture = frequency_domain_noise(width, height, moisture_seed, octaves=4, persistence=0.6, lacunarity=2.0)
    else:
        base_moisture = advanced_fractal_noise_scalar(width, height, moisture_seed, octaves=4, persistence=0.6, lacunarity=2.0)
    
    # 计算风向影响的湿度分布
    wind_x, wind_y = prevailing_wind
    
    # 模拟雨影效应 - 风吹过高山时，大气湿度减少
    result = np.copy(base_moisture)
    
    # 简化的雨影效应模拟
    if abs(wind_x) > 0 or abs(wind_y) > 0:
        x_step = 1 if wind_x >= 0 else -1
        y_step = 1 if wind_y >= 0 else -1
        
        x_range = range(0, width) if x_step > 0 else range(width-1, -1, -1)
        y_range = range(0, height) if y_step > 0 else range(height-1, -1, -1)
        
        moisture_loss = np.zeros_like(height_map)
        
        for y in y_range:
            prev_height = 0
            for x in x_range:
                # 如果高于之前的高度，减少湿度
                if height_map[y, x] > prev_height:
                    moisture_loss[y, x] = (height_map[y, x] - prev_height) * 0.05
                prev_height = max(prev_height, height_map[y, x])
        
        # 应用湿度损失
        result = result - moisture_loss
    
    # 高温地区湿度较低 (蒸发效应)
    evaporation = temp_map * 0.3
    result = result - evaporation
    
    # 水体周围湿度高 (高度<10被认为是水)
    water_mask = height_map < 10
    water_influence = np.zeros_like(result)
    
    # 简单的水影响扩散
    kernel_size = 5
    for y in range(height):
        for x in range(width):
            if water_mask[y, x]:
                for ky in range(max(0, y-kernel_size), min(height, y+kernel_size+1)):
                    for kx in range(max(0, x-kernel_size), min(width, x+kernel_size+1)):
                        dist = np.sqrt((y-ky)**2 + (x-kx)**2)
                        if dist <= kernel_size:
                            water_influence[ky, kx] += 0.05 * (kernel_size - dist) / kernel_size
    
    result = result + water_influence
    
    return normalize_array(result)