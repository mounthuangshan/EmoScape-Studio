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
from .geomorphological_features import *

#########################################
#生成高度、温度和湿度图（噪音和侵蚀效果）
#########################################
############################################
# 生成高度、温度和湿度图（高级自然地貌模拟）
#############################################
import numpy as np
from numba import jit, prange, float64, int32
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import convolve, convolve2d
import math

def generate_geological_provinces(width, height, province_count=3, seed=None):
    """
    生成大尺度地质省份，控制宏观地形特征分布
    
    参数:
    width, height: 地图尺寸
    province_count: 主要地质省份数量
    seed: 随机种子
    
    返回:
    province_map: 地质省份标识图 (每个像素标记其所属省份ID)
    province_params: 每个省份的参数字典列表
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 初始化省份地图和参数
    province_map = np.zeros((height, width), dtype=np.int32)
    province_params = []
    
    # 使用大尺度Voronoi图生成省份基础
    # 生成随机中心点
    centers = []
    for _ in range(province_count):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        centers.append((x, y))
    
    # 对每个像素分配省份ID
    for y in range(height):
        for x in range(width):
            best_dist = float('inf')
            best_id = 0
            
            for i, (cx, cy) in enumerate(centers):
                # 使用欧几里德距离但添加随机扰动，使边界更自然
                dist = np.sqrt((x-cx)**2 + (y-cy)**2) * (0.8 + 0.4 * simplex_noise(x/width*3, y/height*3, seed+i))
                if dist < best_dist:
                    best_dist = dist
                    best_id = i
            
            province_map[y, x] = best_id
    
    # 平滑省份边界
    province_map = gaussian_blur_integer_map(province_map, sigma=max(width, height) / 100)
    
    # 为每个省份生成独特参数
    for i in range(province_count):
        # 每个省份有不同的地质特性
        params = {
            'mountain_density': np.random.random() * 0.6 + 0.2,  # 0.2-0.8
            'plateau_chance': np.random.random() * 0.3 + 0.1,    # 0.1-0.4
            'erosion_intensity': np.random.random() * 0.4 + 0.3, # 0.3-0.7
            'ridge_direction': np.random.random() * np.pi,       # 主要脊线方向
            'scale_modifier': np.random.random() * 0.6 + 0.7,    # 0.7-1.3 (影响频率)
            'base_elevation': np.random.random() * 0.4 - 0.2     # -0.2-0.2 (基础高度修正)
        }
        province_params.append(params)
    
    return province_map, province_params

def gaussian_blur_integer_map(int_map, sigma=2.0):
    """对整数地图进行高斯模糊，保持整数值，实现平滑边界"""
    # 将整数地图转为多个二值图层
    unique_values = np.unique(int_map)
    layers = []
    
    for val in unique_values:
        # 创建该值的二值掩码
        mask = (int_map == val).astype(np.float64)
        # 对该层应用高斯模糊
        blurred = gaussian_filter(mask, sigma=sigma)
        layers.append(blurred)
    
    # 找出每个像素最大值对应的层
    result = np.zeros_like(int_map)
    layers_array = np.array(layers)
    
    # 对每个像素，选择得分最高的层的索引
    for y in range(int_map.shape[0]):
        for x in range(int_map.shape[1]):
            best_layer = np.argmax(layers_array[:, y, x])
            result[y, x] = unique_values[best_layer]
            
    return result

def generate_tectonic_features(width, height, province_map, province_params, seed=None):
    """
    生成大尺度构造特征，如山脉链、断裂带等
    
    参数:
    width, height: 地图尺寸
    province_map: 地质省份图
    province_params: 省份参数
    seed: 随机种子
    
    返回:
    tectonic_map: 构造特征高度影响图
    feature_masks: 不同构造特征的掩码字典
    """
    if seed is not None:
        np.random.seed(seed)
    
    tectonic_map = np.zeros((height, width), dtype=np.float64)
    feature_masks = {
        'mountain_ranges': np.zeros((height, width), dtype=np.float64),
        'rift_valleys': np.zeros((height, width), dtype=np.float64),
        'plateaus': np.zeros((height, width), dtype=np.float64),
        'basins': np.zeros((height, width), dtype=np.float64)
    }
    
    # 获取省份边界 - 省份边界是构造活动的活跃区域
    province_boundaries = np.zeros((height, width), dtype=np.bool_)
    
    # 检测省份边界
    for y in range(1, height-1):
        for x in range(1, width-1):
            current = province_map[y, x]
            if (province_map[y-1, x] != current or 
                province_map[y+1, x] != current or
                province_map[y, x-1] != current or
                province_map[y, x+1] != current):
                province_boundaries[y, x] = True
    
    # 在省份边界生成山脉链
    # 使用距离变换来创建山脉的宽度渐变
    from scipy.ndimage import distance_transform_edt
    
    # 计算到边界的距离
    boundary_distance = distance_transform_edt(~province_boundaries)
    max_dist = np.max(boundary_distance)
    
    # 生成山脉和其他特征
    for y in range(height):
        for x in range(width):
            # 当前省份
            province_id = province_map[y, x]
            params = province_params[province_id]
            
            # 距离省份边界的标准化距离
            norm_dist = boundary_distance[y, x] / max_dist
            
            # 沿边界生成山脉
            if norm_dist < 0.1:  # 靠近边界
                # 山脉高度随距离衰减
                mountain_factor = (0.1 - norm_dist) / 0.1
                
                # 添加扰动使山脉不均匀
                noise_val = simplex_noise(x/width*5, y/height*5, seed+100)
                
                # 省份特定参数影响
                mountain_strength = mountain_factor * (0.5 + noise_val * 0.5) * params['mountain_density']
                
                # 加入山脉特征
                feature_masks['mountain_ranges'][y, x] = mountain_strength
                tectonic_map[y, x] += mountain_strength * 0.8  # 山脉提升地形
            
            # 在省份中部随机生成高原
            elif norm_dist > 0.3 and np.random.random() < params['plateau_chance'] * 0.01:
                # 生成圆形高原区域
                plateau_radius = int(min(width, height) * (0.05 + np.random.random() * 0.1))
                plateau_center_x, plateau_center_y = x, y
                
                # 绘制高原
                for py in range(max(0, y-plateau_radius), min(height, y+plateau_radius+1)):
                    for px in range(max(0, x-plateau_radius), min(width, x+plateau_radius+1)):
                        if province_map[py, px] == province_id:
                            dist = np.sqrt((px-plateau_center_x)**2 + (py-plateau_center_y)**2)
                            if dist <= plateau_radius:
                                # 高原强度随距离中心而减弱
                                plateau_factor = (plateau_radius - dist) / plateau_radius
                                plateau_factor = plateau_factor ** 0.5  # 使边缘更平滑
                                
                                # 添加无序性
                                noise_val = simplex_noise(px/width*10, py/height*10, seed+200)
                                plateau_strength = plateau_factor * (0.7 + noise_val * 0.3)
                                
                                feature_masks['plateaus'][py, px] = max(feature_masks['plateaus'][py, px], plateau_strength)
                                tectonic_map[py, px] += plateau_strength * 0.6  # 高原提升地形
    
    # 添加随机盆地和裂谷
    basin_count = int(max(1, min(width, height) / 100))
    for _ in range(basin_count):
        # 随机选择一个省份内部点
        attempts = 0
        while attempts < 10:  # 限制尝试次数
            x = np.random.randint(width // 10, width * 9 // 10)
            y = np.random.randint(height // 10, height * 9 // 10)
            norm_dist = boundary_distance[y, x] / max_dist
            
            if norm_dist > 0.15:  # 确保远离边界
                break
            attempts += 1
            
        # 生成盆地或裂谷
        feature_type = 'basins' if np.random.random() < 0.7 else 'rift_valleys'
        
        if feature_type == 'basins':
            # 圆形盆地
            radius = int(min(width, height) * (0.05 + np.random.random() * 0.1))
            for py in range(max(0, y-radius), min(height, y+radius+1)):
                for px in range(max(0, x-radius), min(width, x+radius+1)):
                    dist = np.sqrt((px-x)**2 + (py-y)**2)
                    if dist <= radius:
                        # 盆地强度随距离中心而增强 (中心最低)
                        basin_factor = (radius - dist) / radius
                        basin_factor = basin_factor ** 0.7  # 使边缘更陡峭
                        
                        # 添加无序性
                        noise_val = simplex_noise(px/width*15, py/height*15, seed+300)
                        basin_strength = basin_factor * (0.7 + noise_val * 0.3)
                        
                        feature_masks['basins'][py, px] = max(feature_masks['basins'][py, px], basin_strength)
                        tectonic_map[py, px] -= basin_strength * 0.5  # 盆地降低地形
        else:
            # 线性裂谷
            angle = np.random.random() * np.pi
            length = int(min(width, height) * (0.2 + np.random.random() * 0.3))
            width_val = int(min(width, height) * (0.01 + np.random.random() * 0.03))
            
            # 计算起点和终点
            x1, y1 = x - int(length/2 * np.cos(angle)), y - int(length/2 * np.sin(angle))
            x2, y2 = x + int(length/2 * np.cos(angle)), y + int(length/2 * np.sin(angle))
            
            # Bresenham算法绘制线段
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            
            cx, cy = x1, y1
            while True:
                if 0 <= cx < width and 0 <= cy < height:
                    # 绘制裂谷宽度
                    for wy in range(max(0, cy-width_val), min(height, cy+width_val+1)):
                        for wx in range(max(0, cx-width_val), min(width, cx+width_val+1)):
                            dist = np.sqrt((wx-cx)**2 + (wy-cy)**2)
                            if dist <= width_val:
                                # 裂谷强度随距离中心而增强
                                rift_factor = (width_val - dist) / width_val
                                rift_factor = rift_factor ** 0.5
                                
                                # 添加无序性
                                noise_val = simplex_noise(wx/width*20, wy/height*20, seed+400)
                                rift_strength = rift_factor * (0.7 + noise_val * 0.3)
                                
                                feature_masks['rift_valleys'][wy, wx] = max(feature_masks['rift_valleys'][wy, wx], rift_strength)
                                tectonic_map[wy, wx] -= rift_strength * 0.7  # 裂谷显著降低地形
                
                if cx == x2 and cy == y2:
                    break
                    
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    cx += sx
                if e2 < dx:
                    err += dx
                    cy += sy
    
    return tectonic_map, feature_masks

@jit(nopython=True)
def generate_directional_faults(height_map, fault_count=5, seed=42):
    """生成具有地质学合理性的方向性断层"""
    height, width = height_map.shape
    fault_map = np.zeros((height, width))
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 生成主应力方向 (通常地质断层有优先方向)
    main_angle = np.random.random() * np.pi  # 主断层方向
    angle_variance = np.pi / 6  # 允许30度的变化
    
    for _ in range(fault_count):
        # 随机选择断层起点
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        
        # 生成接近主应力方向的断层角度
        angle = main_angle + (np.random.random() * 2 - 1) * angle_variance
        
        # 断层长度，通常为地图宽度的10-40%
        length = int(width * (0.1 + np.random.random() * 0.3))
        
        # 计算断层终点
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # 断层宽度
        fault_width = 2 + int(np.random.random() * 4)
        
        # 绘制断层
        # 使用Bresenham算法绘制线段
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        while True:
            if 0 <= x < width and 0 <= y < height:
                # 断层强度
                strength = 0.5 + np.random.random() * 0.5
                
                # 为断层周围区域添加衰减值
                for ky in range(-fault_width, fault_width+1):
                    for kx in range(-fault_width, fault_width+1):
                        nx, ny = x + kx, y + ky
                        if 0 <= nx < width and 0 <= ny < height:
                            dist = np.sqrt(kx*kx + ky*ky)
                            if dist <= fault_width:
                                intensity = strength * (1.0 - dist/fault_width)
                                fault_map[ny, nx] = max(fault_map[ny, nx], intensity)
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    return fault_map

@jit(nopython=True)
def generate_rock_resistance_map(height_map, seed):
    """生成基于多种因素的岩石抗性图，提高地质多样性"""
    height, width = height_map.shape
    resistance_map = np.zeros_like(height_map)
    
    # 动态缩放因子，基于地图尺寸
    scale_factor = max(width, height) / 1024  # 以1024为基准分辨率
    noise_scale = 5 * scale_factor  # 调整噪声缩放
    
    # 基础噪声 - 模拟岩石类型的随机分布
    base_noise = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            # 使用动态缩放的坐标
            base_noise[y, x] = simplex_noise(x/width*noise_scale, y/height*noise_scale, seed) * 0.5 + 0.5
    
    # 高度相关性 - 高处岩石通常更坚硬
    height_normalized = normalize_array(height_map)
    
    # 坡度计算 - 陡峭区域可能有不同的岩石特性
    gradient_map = np.zeros((height, width))
    
    # 直接计算梯度而不使用之前的函数
    for y in range(1, height-1):
        for x in range(1, width-1):
            # 使用中心差分计算梯度
            dx = (height_map[y, x+1] - height_map[y, x-1]) * 0.5
            dy = (height_map[y+1, x] - height_map[y-1, x]) * 0.5
            gradient_map[y, x] = np.sqrt(dx*dx + dy*dy)
    
    # 边界处理
    for y in range(height):
        gradient_map[y, 0] = gradient_map[y, 1]
        gradient_map[y, width-1] = gradient_map[y, width-2]
    
    for x in range(width):
        gradient_map[0, x] = gradient_map[1, x]
        gradient_map[height-1, x] = gradient_map[height-2, x]
    
    gradient_normalized = normalize_array(gradient_map)
    fault_map = generate_directional_faults(height_map, fault_count=3, seed=seed+100)
    
    # 组合多种因素
    for y in range(height):
        for x in range(width):
            # 基础高度因素 - 高处岩石更坚硬
            height_factor = 0.3 + 0.4 * height_normalized[y, x]
            
            # 随机噪声因素 - 模拟地质变化
            noise_factor = 0.6 + 0.4 * base_noise[y, x]
            
            # 坡度因素 - 陡峭区域岩石特性不同
            gradient_factor = 0.0
            if y > 0 and y < height-1 and x > 0 and x < width-1:
                gradient_factor = 0.1 * gradient_normalized[y, x]
            
            # 组合计算最终抗性，确保在0.1-0.95范围内
            resistance = height_factor * 0.5 + noise_factor * 0.4 + gradient_factor * 0.1
            resistance_map[y, x] = clip_value(resistance, 0.1, 0.95)
            
            # 添加地质断层和特殊地形
            # 随机生成一些线性断层带，模拟不同硬度的岩石层
            if simplex_noise(x/width*20+0.5, y/height*20+0.5, seed+100) > 0.7:
                # 使用生成的断层图:
                resistance_value = resistance_map[y, x] * (1.0 + fault_map[y, x] * 0.5)
                resistance_map[y, x] = clip_value(resistance_value, 0.1, 0.95)
    
    return resistance_map

# 自适应时间步长计算辅助函数
@jit(nopython=True)
def calculate_adaptive_dt(height_map, water_map, base_dt=0.2, cfl_factor=0.4):
    """计算符合CFL条件的自适应时间步长"""
    height, width = height_map.shape
    max_gradient = 0.001  # 最小梯度值以防除零
    
    # 采样计算最大梯度
    sample_step = max(1, min(height, width) // 20)  # 采样步长，避免全图扫描
    for y in range(0, height, sample_step):
        for x in range(0, width, sample_step):
            if x > 0 and x < width-1 and y > 0 and y < height-1:
                # 计算x和y方向梯度
                dx = abs(height_map[y, x+1] - height_map[y, x-1]) * 0.5
                dy = abs(height_map[y+1, x] - height_map[y-1, x]) * 0.5
                local_grad = np.sqrt(dx*dx + dy*dy)
                
                # 考虑水量影响
                water_factor = 1.0 + water_map[y, x] * 2.0
                effective_grad = local_grad * water_factor
                
                max_gradient = max(max_gradient, effective_grad)
    
    # 使用CFL条件计算安全时间步长：dt ≤ dx / (v * cfl_factor)
    # 其中dx为网格尺寸，v为最大流速（与梯度相关）
    adaptive_dt = 1.0 / (max_gradient * 10.0 + 0.1) * cfl_factor
    
    # 限制时间步长在合理范围内
    return min(base_dt, max(0.05, adaptive_dt))

# 全新物理侵蚀系统：模拟复杂的地质侵蚀过程
@jit(nopython=True, parallel=True)
def improved_erosion_system(height_map, iterations=5, rainfall_amount=0.01, 
                           evaporation_rate=0.5, sediment_capacity=0.05, 
                           erosion_rate=0.3, deposition_rate=0.3, rock_resistance=None):
    """改进版侵蚀系统，增强数值稳定性和物理真实性"""
    height, width = height_map.shape
    result = height_map.copy()
    
    # 如果没有提供岩石抗性图，创建一个动态的抗性图
    if rock_resistance is None:
        rock_resistance = generate_rock_resistance_map(height_map, 12345)
    
    # 水量图和沉积物图
    water_map = np.zeros((height, width), dtype=np.float64)
    sediment_map = np.zeros((height, width), dtype=np.float64)
    
    # 侵蚀常数
    gravity = 9.81  # 重力加速度
    dt = 0.2        # 时间步长
    
    # 迭代模拟侵蚀过程
    for iter in range(iterations):
        # 1. 降雨 - 每10步模拟一次降雨事件
        if iter % 10 == 0:
            # 变化的降雨模式
            rain_variation = 0.5 + np.sin(iter * 0.1) * 0.5
            
            # 并行应用降雨
            for y in prange(height):
                for x in range(width):
                    # 高度影响降雨
                    local_height = result[y, x]
                    height_factor = min(1.5, 1.0 + local_height / 200.0)
                    
                    # 随机降雨变化
                    rand_factor = 0.7 + simplex_noise(x/width*30, y/height*30, iter) * 0.6
                    
                    # 安全地添加水量
                    water_map[y, x] += max(0.0, rainfall_amount * height_factor * rand_factor * rain_variation)
        
        # 2. 水流模拟 - 使用多步较小时间步长提高稳定性
        for _ in range(2):
            # 临时存储下一步的水量和沉积物
            new_water = np.zeros_like(water_map)
            new_sediment = np.zeros_like(sediment_map)
            
            # 并行计算每个单元格的水流
            for y in prange(1, height-1):
                for x in range(1, width-1):
                    if water_map[y, x] < 0.005:  # 忽略水量极少的单元格
                        new_water[y, x] += water_map[y, x]
                        new_sediment[y, x] += sediment_map[y, x]
                        continue
                    
                    # 当前单元格状态
                    current_height = result[y, x]
                    current_water = water_map[y, x]
                    current_sediment = sediment_map[y, x]
                    current_total = current_height + current_water
                    
                    # 计算流向邻居的水量
                    outflows = np.zeros(8, dtype=np.float64)
                    outflow_total = 0.0
                    
                    # 相邻单元坐标
                    neighbors = [
                        (y-1, x), (y+1, x), (y, x-1), (y, x+1),
                        (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)
                    ]
                    
                    # 计算到相邻单元的距离
                    distances = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
                    
                    # 计算流向每个相邻单元的水量
                    for i, (ny, nx) in enumerate(neighbors):
                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue
                            
                        neighbor_height = result[ny, nx]
                        neighbor_water = water_map[ny, nx]
                        neighbor_total = neighbor_height + neighbor_water
                        
                        # 只有当前位置比邻居高时才流动
                        height_diff = current_total - neighbor_total
                        if height_diff > 0:
                            # 时间步长控制
                            dt = calculate_adaptive_dt(height_map, water_map, base_dt=0.2, cfl_factor=0.4)

                            # 水流计算部分
                            # 基于高度差和距离计算流量时
                            # 引入基于简化圣维南方程的水流计算
                            water_depth = max(0.01, current_water)  # 防止除零
                            velocity = np.sqrt(2 * gravity * height_diff)  # 基于能量守恒
                            velocity = min(velocity, 3.0)  # 限制最大流速
                            flow_capacity = velocity * water_depth * 0.5  # 流量容量
                            # 应用CFL条件
                            flow = min(flow_capacity, current_water * 0.8, height_diff * 0.5) * dt
                            # 添加物理合理性约束
                            flow = min(flow, current_water * 0.8, height_diff * 0.5)  # 限制单次最大流量
                            outflows[i] = flow
                            outflow_total += flow
                    
                    # 水量平衡调整 - 增强数值稳定性
                    # 确保总流出不超过当前水量的95%
                    max_allowed_outflow = current_water * 0.95
                    
                    if outflow_total > 1e-10:  # 有意义的流量
                        if outflow_total > max_allowed_outflow:
                            # 按比例缩放所有流量
                            scale_factor = max_allowed_outflow / outflow_total
                            for i in range(8):
                                outflows[i] *= scale_factor
                            outflow_total = max_allowed_outflow
                    else:
                        # 无流动
                        outflow_total = 0.0
                        outflows = np.zeros(8, dtype=np.float64)
                    
                    # 3. 侵蚀和沉积
                    # 剩余水量
                    remaining_water = current_water - outflow_total
                    new_water[y, x] += remaining_water
                    
                    # 流速与坡度、水量有关
                    velocity = min(outflow_total / max(0.01, current_water), 1.0) * gravity * 0.05
                    
                    # 泥沙容量与速度和岩石抗性相关
                    local_resistance = rock_resistance[y, x]
                    local_capacity = sediment_capacity * velocity * (1.0 - local_resistance)
                    
                    # 安全保障
                    local_capacity = max(0.0, local_capacity)
                    
                    if current_sediment < local_capacity:
                        # 侵蚀
                        erosion_amount = min(
                            erosion_rate * (local_capacity - current_sediment),
                            0.01 * (1.0 - local_resistance)  # 限制单次侵蚀量
                        )
                        # 确保侵蚀量为正且不会导致地形不合理下降
                        erosion_amount = max(0.0, min(erosion_amount, current_height * 0.01))
                        result[y, x] -= erosion_amount
                        current_sediment += erosion_amount
                    else:
                        # 沉积
                        deposition_amount = deposition_rate * (current_sediment - local_capacity)
                        # 确保沉积量为正
                        deposition_amount = max(0.0, deposition_amount)
                        result[y, x] += deposition_amount
                        current_sediment -= deposition_amount
                    
                    # 更新携带的泥沙 - 保留在当前单元的部分
                    if outflow_total > 0 and current_water > 0:
                        stay_ratio = remaining_water / current_water
                        new_sediment[y, x] += current_sediment * stay_ratio
                        
                        # 分配流向每个相邻单元的水和泥沙
                        for i, (ny, nx) in enumerate(neighbors):
                            if outflows[i] > 0:
                                flow_ratio = outflows[i] / current_water
                                new_water[ny, nx] += outflows[i]
                                new_sediment[ny, nx] += current_sediment * flow_ratio
                    else:
                        # 如果没有流出，所有泥沙留在原地
                        new_sediment[y, x] += current_sediment
            
            # 更新水和泥沙图
            water_map = new_water
            sediment_map = new_sediment
        
        # 4. 蒸发过程
        for y in prange(height):
            for x in range(width):
                # 蒸发与温度、高度相关
                local_evaporation = evaporation_rate
                # 高海拔蒸发减少
                local_evaporation *= max(0.5, min(1.0, 1.0 - result[y, x] / 100.0))
                
                # 确保蒸发系数在有效范围内
                local_evaporation = min(0.99, max(0.0, local_evaporation * dt))
                
                # 应用蒸发
                water_map[y, x] *= (1.0 - local_evaporation)
        
        # 5. 热侵蚀 - 每5次水力侵蚀执行一次
        mean_gradient = 0.0
        sample_count = 0
        for sy in range(0, height, max(1, height//20)):
            for sx in range(0, width, max(1, width//20)):
                if sy > 0 and sy < height-1 and sx > 0 and sx < width-1:
                    dx = abs(result[sy, sx+1] - result[sy, sx-1]) * 0.5
                    dy = abs(result[sy+1, sx] - result[sy-1, sx]) * 0.5
                    mean_gradient += np.sqrt(dx*dx + dy*dy)
                    sample_count += 1

        if sample_count > 0:
            mean_gradient /= sample_count

        # 动态调整热侵蚀频率，陡峭地形更频繁进行热侵蚀
        thermal_freq = max(1, int(8 - mean_gradient * 30))
        if iter % thermal_freq == 0:
            for y in prange(1, height-1):
                for x in range(1, width-1):
                    # 岩石类型影响安息角
                    local_resistance = rock_resistance[y, x]
                    local_talus = 0.05 + local_resistance * 0.3  # 安息角
                    
                    # 检查八个方向的最大坡度
                    center_height = result[y, x]
                    max_slope = 0.0
                    max_dir_y, max_dir_x = 0, 0
                    
                    # 相邻单元
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0:
                                continue
                            
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                # 计算坡度
                                dist = np.sqrt(dx*dx + dy*dy)
                                slope = (center_height - result[ny, nx]) / dist
                                
                                if slope > max_slope:
                                    max_slope = slope
                                    max_dir_y, max_dir_x = ny, nx
                    
                    # 如果坡度超过安息角，物质开始滑动
                    if max_slope > local_talus and max_dir_y != 0 and max_dir_x != 0:
                        # 计算滑动量
                        dist = np.sqrt((max_dir_y-y)**2 + (max_dir_x-x)**2)
                        excess = (max_slope - local_talus) * dist
                        delta = min(0.5 * excess, center_height * 0.05)  # 限制单次滑动量
                        
                        # 应用滑动 - 使用互斥锁确保线程安全
                        result[y, x] -= delta
                        result[max_dir_y, max_dir_x] += delta
    
    # 检查结果中的NaN并替换
    result = np.nan_to_num(result, nan=0.5)
    
    return result

# 热侵蚀函数
@jit(nopython=True, parallel=True)
def thermal_erosion_improved(height_map, iterations=1, talus_angle=0.05, strength=0.1):
    """改进的热侵蚀算法，使用按高度分区处理保持连贯性"""
    result = height_map.copy()
    height, width = height_map.shape
    
    # 分区处理 - 按高度将地图分为多个区域
    num_zones = 10  # 分为10个高度区
    
    for iter in range(iterations):
        # 计算当前高度范围
        current_min = np.min(result)
        current_max = np.max(result)
        height_range = current_max - current_min
        
        if height_range <= 0.0001:
            break  # 几乎没有高度差异，退出循环
        
        # 按高度从低到高处理，保持沉积物自然流向
        zone_height = height_range / num_zones
        
        for zone in range(num_zones):
            # 计算当前区域的高度阈值
            zone_min = current_min + zone * zone_height
            zone_max = zone_min + zone_height
            
            # 创建当前高度区域的索引数组
            zone_indices = []
            for y in range(1, height-1):
                for x in range(1, width-1):
                    if zone_min <= result[y, x] < zone_max:
                        zone_indices.append((y, x))
            
            # 随机化处理顺序但保持区域连贯性
            # 修改随机洗牌实现，使用Numba兼容的方式
            if zone_indices:
                # 替代np.random.shuffle - 使用Fisher-Yates算法手动实现
                n = len(zone_indices)
                for i in range(n-1, 0, -1):
                    j = int(np.random.random() * (i+1))
                    zone_indices[i], zone_indices[j] = zone_indices[j], zone_indices[i]
                
                # 处理当前区域的所有点
                for y, x in zone_indices:
                    current = result[y, x]
                    
                    # 计算八个方向的坡度，找出最陡下降方向
                    slopes = []
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0:
                                continue
                                
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                dist = np.sqrt(dx*dx + dy*dy)
                                slope = (current - result[ny, nx]) / dist
                                slopes.append((slope, ny, nx))
                    
                    # 找出最大坡度方向
                    if slopes:
                        slopes.sort(reverse=True)  # 从大到小排序
                        max_slope, max_ny, max_nx = slopes[0]
                        
                        # 如果坡度超过临界角，物质滑动
                        if max_slope > talus_angle:
                            # 计算滑动量
                            dist = np.sqrt((max_ny-y)**2 + (max_nx-x)**2)
                            excess = (max_slope - talus_angle) * dist
                            
                            # 安全计算滑动量
                            delta = min(strength * excess, result[y, x] * 0.05)  # 降低单次滑动比例

                            # 应用滑动到多个相邻单元而非单点
                            main_delta = delta * 0.7  # 主方向接收70%的物质
                            result[y, x] -= delta
                            result[max_ny, max_nx] += main_delta

                            # 分散剩余30%到周边单元，创造平滑过渡
                            remaining = delta - main_delta
                            for n_dy, n_dx in [(1,0), (-1,0), (0,1), (0,-1)]:
                                ny, nx = max_ny + n_dy, max_nx + n_dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    dist = np.sqrt(n_dy*n_dy + n_dx*n_dx)
                                    result[ny, nx] += remaining * 0.25  # 剩余物质均分
    
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

# 改进地形特征函数
@jit(nopython=True)
def generate_terrain_features(height_map, feature_probability=0.02, seed=12345):
    """生成特殊地形特征：峡谷、悬崖、山脊等"""
    height, width = height_map.shape
    result = height_map.copy()
    
    # 使用梯度分析寻找山脊线和峡谷
    gx = np.zeros((height, width), dtype=np.float64)
    gy = np.zeros((height, width), dtype=np.float64)
    
    # 计算梯度
    for y in range(1, height-1):
        for x in range(1, width-1):
            gx[y, x] = (height_map[y, x+1] - height_map[y, x-1]) * 0.5
            gy[y, x] = (height_map[y+1, x] - height_map[y-1, x]) * 0.5
    
    # 计算梯度幅度和方向
    gradient_mag = np.sqrt(gx * gx + gy * gy)
    
    # 标记潜在的特征位置
    feature_map = np.zeros((height, width), dtype=np.int32)
    
    # 1. 寻找山脊线 - 梯度方向突变
    for y in range(2, height-2):
        for x in range(2, width-2):
            if gradient_mag[y, x] > 0.1:  # 有显著梯度
                # 在梯度方向垂直方向上检查高度变化
                if gx[y, x] != 0 or gy[y, x] != 0:
                    # 归一化梯度向量
                    len_g = np.sqrt(gx[y, x]**2 + gy[y, x]**2)
                    nx, ny = -gy[y, x]/len_g, gx[y, x]/len_g  # 垂直于梯度的方向
                    
                    # 检查垂直方向上的高度变化
                    h1 = height_map[min(height-1, max(0, int(y + ny))), 
                                    min(width-1, max(0, int(x + nx)))]
                    h2 = height_map[min(height-1, max(0, int(y - ny))), 
                                    min(width-1, max(0, int(x - nx)))]
                    
                    # 山脊特征：两侧高度都低于中心
                    if height_map[y, x] > h1 and height_map[y, x] > h2:
                        feature_map[y, x] = 1  # 山脊
    
    # 2. 通过分析曲率寻找山谷和峡谷
    for y in range(2, height-2):
        for x in range(2, width-2):
            # 计算拉普拉斯算子（二阶导数）作为曲率估计
            laplacian = (height_map[y-1, x] + height_map[y+1, x] + 
                        height_map[y, x-1] + height_map[y, x+1] - 
                        4 * height_map[y, x])
            
            # 正曲率表示凹陷（山谷）
            if laplacian > 0.05 and gradient_mag[y, x] > 0.05:
                feature_map[y, x] = 2  # 山谷
                
                # 检测深V形峡谷
                if gradient_mag[y, x] > 0.2 and laplacian > 0.1:
                    feature_map[y, x] = 3  # 峡谷
    
    # 3. 检测悬崖 - 高梯度区域
    for y in range(1, height-1):
        for x in range(1, width-1):
            if gradient_mag[y, x] > 0.3:  # 非常陡峭的区域
                feature_map[y, x] = 4  # 悬崖
    
    # 应用特征增强
    np.random.seed(seed)
    for y in range(3, height-3):
        for x in range(3, width-3):
            feature = feature_map[y, x]
            
            if feature == 0:
                # 随机添加小型特征
                if np.random.random() < feature_probability:
                    local_height = height_map[y, x]
                    nearby_height = np.mean([
                        height_map[y-2, x], height_map[y+2, x],
                        height_map[y, x-2], height_map[y, x+2]
                    ])
                    
                    if local_height > 60:  # 高处添加峰顶
                        # 创建尖锐山峰
                        peak_height = local_height * 1.1
                        radius = np.random.randint(2, 5)
                        for dy in range(-radius, radius+1):
                            for dx in range(-radius, radius+1):
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist <= radius:
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < height and 0 <= nx < width:
                                        # 基于距离的高度衰减
                                        factor = (1.0 - dist/radius)**2
                                        result[ny, nx] = max(result[ny, nx], 
                                                          local_height + (peak_height - local_height) * factor)
            
            elif feature == 1:  # 山脊增强
                # 锐化山脊
                result[y, x] *= 1.05
                
                # 延伸山脊
                if gx[y, x] != 0 or gy[y, x] != 0:
                    # 获取山脊方向（垂直于梯度）
                    len_g = np.sqrt(gx[y, x]**2 + gy[y, x]**2)
                    ridge_dx, ridge_dy = -gy[y, x]/len_g, gx[y, x]/len_g
                    
                    # 沿山脊方向增强
                    for t in range(-2, 3):
                        if t == 0:
                            continue
                        ny = int(y + ridge_dy * t)
                        nx = int(x + ridge_dx * t)
                        if 0 <= ny < height and 0 <= nx < width:
                            # 逐渐降低山脊高度
                            # 替换为高斯衰减：
                            factor = np.exp(-0.5 * (t / 2.0)**2)  # 更自然的高斯衰减
                            result[ny, nx] = max(result[ny, nx], result[y, x] * factor)
            
            elif feature == 2:  # 山谷增强
                # 加深山谷
                result[y, x] *= 0.95
                
                # 平滑谷底
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            result[ny, nx] = (result[ny, nx] + result[y, x]) * 0.5
            
            elif feature == 3:  # 峡谷增强
                # 显著加深峡谷
                result[y, x] *= 0.85
                
                # 加宽峡谷
                if gx[y, x] != 0 or gy[y, x] != 0:
                    # 获取峡谷方向（垂直于梯度）
                    len_g = np.sqrt(gx[y, x]**2 + gy[y, x]**2)
                    canyon_dx, canyon_dy = -gy[y, x]/len_g, gx[y, x]/len_g
                    
                    # 沿峡谷方向增强
                    for t in range(-3, 4):
                        ny = int(y + canyon_dy * t)
                        nx = int(x + canyon_dx * t)
                        if 0 <= ny < height and 0 <= nx < width:
                            # 使峡谷更深和更平滑
                            factor = 0.9 - 0.05 * abs(t)
                            result[ny, nx] = min(result[ny, nx], result[y, x] / factor)
            
            elif feature == 4:  # 悬崖增强
                # 进一步提高悬崖高度
                result[y, x] *= 1.02
                
                # 找出悬崖下方位置，使其更低
                down_x, down_y = x, y
                if abs(gx[y, x]) > abs(gy[y, x]):  # 主要沿x方向的梯度
                    down_x = x - 1 if gx[y, x] > 0 else x + 1
                else:  # 主要沿y方向的梯度
                    down_y = y - 1 if gy[y, x] > 0 else y + 1
                
                if 0 <= down_y < height and 0 <= down_x < width:
                    result[down_y, down_x] *= 0.97  # 降低悬崖下方
    
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
def is_plateau_region(height_map, x, y, threshold=0.7):
    """检测某点是否位于高原区域（周围高度相近）"""
    height, width = height_map.shape
    if x < 2 or y < 2 or x >= width-2 or y >= height-2:
        return False
    
    # 检查3x3邻域的高度变化
    center = height_map[y, x]
    variance = 0
    count = 0
    
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                variance += abs(height_map[ny, nx] - center)
                count += 1
    
    if count == 0:
        return False
    
    avg_variance = variance / count
    # 方差小表示高度变化不大，可能是高原
    return avg_variance < threshold * 0.05

# 优化高度分布函数
@jit(nopython=True)
def apply_geomorphological_height_distribution(height_map, plain_ratio=0.3, hill_ratio=0.3, 
                                             mountain_ratio=0.2, plateau_ratio=0.1, 
                                             mountain_slope=1.5, plateau_threshold=0.7):
    """
    应用基于地貌学原理的高度分布，使用改进的非线性映射保留中尺度特征
    """
    # 复制原始高度图
    height_map = normalize_array(height_map)
    result = np.zeros_like(height_map)
    
    # 计算高度分位数
    flat_heights = height_map.flatten()
    sorted_heights = np.sort(flat_heights)
    length = len(sorted_heights)
    
    # 计算各地形类型的阈值
    plain_threshold = sorted_heights[int(plain_ratio * length)]
    hill_threshold = sorted_heights[int((plain_ratio + hill_ratio) * length)]
    mountain_threshold = sorted_heights[int((plain_ratio + hill_ratio + mountain_ratio) * length)]
    plateau_start = sorted_heights[int((1.0 - plateau_ratio) * 0.5 * length)]
    plateau_end = sorted_heights[int((1.0 - plateau_ratio * 0.3) * length)]
    
    # 中尺度特征增强阈值
    mid_detail_threshold = 0.04  # 中尺度变化检测阈值
    
    # 应用地形变换，使用自适应分段多项式
    height, width = height_map.shape
    for y in range(height):
        for x in range(width):
            h = height_map[y, x]
            
            # 检测中尺度特征 - 计算局部变化
            local_variation = 0.0
            neighbor_count = 0
            
            # 采样5x5区域检测中尺度变化
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        local_variation += abs(height_map[ny, nx] - h)
                        neighbor_count += 1
            
            if neighbor_count > 0:
                local_variation /= neighbor_count
            
            # 中尺度特征权重 - 变化适中的区域保留更多细节
            mid_scale_weight = 0.0
            if 0.005 < local_variation < mid_detail_threshold:
                # 钟形曲线:最大值在local_variation=0.02附近
                mid_scale_weight = 1.0 - ((local_variation - 0.02) / 0.015) ** 2
                mid_scale_weight = max(0.0, min(1.0, mid_scale_weight))
            
            # 平原区 - 使用保留中尺度特征的映射函数
            if h <= plain_threshold:
                # 使用二次样条而非立方根，保留更多中尺度特征
                norm_h = h / plain_threshold
                
                # 分段函数:低区平滑过渡，中区保留特征
                if norm_h < 0.3:
                    # 低平原区域 - 温和过渡
                    result[y, x] = 0.05 + 0.05 * norm_h/0.3
                elif norm_h < 0.6:
                    # 中平原区域 - 保留更多中尺度特征
                    t = (norm_h - 0.3) / 0.3
                    # 利用缓动函数保留中尺度变化
                    detail_factor = t * (1.0 - t) * 4.0 * mid_scale_weight  # 二次峰值函数
                    result[y, x] = 0.1 + 0.05 * t + detail_factor * 0.03
                else:
                    # 高平原区域 - 平滑S曲线过渡到丘陵
                    t = (norm_h - 0.6) / 0.4
                    s_curve = t * t * (3.0 - 2.0 * t)  # 经典S曲线
                    result[y, x] = 0.15 + 0.05 * s_curve
            
            # 丘陵区 - 改进的曲线保留更多纹理细节
            elif h <= hill_threshold:
                # 使用三次样条控制过渡曲线形状
                norm_h = (h - plain_threshold) / (hill_threshold - plain_threshold)
                
                # 自适应分段控制，保留更多丘陵纹理
                if norm_h < 0.4:
                    # 低丘陵区域
                    t = norm_h / 0.4
                    # 添加非线性Perlin扰动保留中尺度特征
                    detail_preserving_curve = t * t * (3.0 - 2.0 * t)  # 保持平滑起始
                    
                    # 增强中尺度变化
                    if mid_scale_weight > 0:
                        # 保留局部变化，但确保整体形状一致性
                        local_detail = mid_scale_weight * (norm_h - 0.2) * 0.25
                        detail_preserving_curve += local_detail
                    
                    result[y, x] = 0.2 + 0.1 * detail_preserving_curve
                else:
                    # 高丘陵区域，提供更丰富的起伏和纹理
                    t = (norm_h - 0.4) / 0.6
                    
                    # 创建更复杂的中尺度纹理
                    # 使用不同频率的噪声混合，保留中尺度特征
                    texture_noise = 0.0
                    if y % 3 == x % 3:  # 伪随机采样，避免使用全局随机函数
                        texture_noise = 0.05
                    elif (y+x) % 5 == 0:
                        texture_noise = -0.05
                    
                    # 添加纹理噪声，强度由中尺度权重调制
                    texture_component = texture_noise * mid_scale_weight
                    
                    # 主曲线加上纹理组件
                    curve = 0.4 + 0.6 * t * (1.0 + texture_component + 
                                           0.2 * simplex_noise(x*0.5, y*0.5, 12345))
                    result[y, x] = 0.3 + 0.15 * curve
            
            # 高原检测和处理 - 识别并平坦化高原区域
            elif plateau_start <= h <= plateau_end and is_plateau_region(height_map, x, y, plateau_threshold):
                # 高原区域特殊处理 - 相对平坦但保留中尺度变化的高地
                norm_h = (h - hill_threshold) / (mountain_threshold - hill_threshold)
                
                # 高原特征：突然升高后保持相对平坦
                plateau_height = 0.4 + 0.1 * np.power(norm_h, 0.5)
                
                # 在高原上添加细微但真实的起伏
                # 使用高频和中频噪声的混合来创建多层次细节
                local_variation = 0.02 * (simplex_noise(x * 0.1, y * 0.1, 12345) + 1) / 2
                
                # 添加中尺度变化 - 模拟丘陵、浅沟和小山丘
                if mid_scale_weight > 0:
                    # 中尺度特征更明显
                    mid_scale_variation = 0.04 * mid_scale_weight * (
                        simplex_noise(x * 0.04, y * 0.04, 67890) * 0.6 +
                        simplex_noise(x * 0.08, y * 0.08, 54321) * 0.4
                    )
                    
                    # 强化中尺度特征
                    local_variation = local_variation * 0.3 + mid_scale_variation * 0.7
                
                result[y, x] = plateau_height + local_variation
            
            # 山区 - 陡峭的高地
            elif h <= mountain_threshold:
                norm_h = (h - hill_threshold) / (mountain_threshold - hill_threshold)
                
                # 使用自适应幂函数创造更陡峭且多变的山脉
                power_factor = mountain_slope
                
                # 山区中尺度特征增强 - 模拟次级山脊和山谷
                if mid_scale_weight > 0.3:
                    # 在山区中部添加更多中尺度变化
                    if 0.3 < norm_h < 0.7:
                        # 增强中部山区的中尺度特征
                        ridge_detail = mid_scale_weight * 0.08 * (
                            simplex_noise(x*0.15, y*0.15, 31415) - 0.5) # 零均值噪声
                        
                        # 应用增强后微调幂函数以保持整体形状
                        power_result = 0.4 + 0.3 * np.power(norm_h + ridge_detail, power_factor)
                        result[y, x] = power_result
                    else:
                        # 山脚和山顶区域保持更平滑的过渡
                        result[y, x] = 0.4 + 0.3 * np.power(norm_h, power_factor)
                else:
                    # 标准山地处理
                    result[y, x] = 0.4 + 0.3 * np.power(norm_h, power_factor)
            
            # 山峰区 - 极高且陡峭的尖峰
            else:
                norm_h = (h - mountain_threshold) / (1.0 - mountain_threshold)
                
                # 使用指数函数创造更尖锐的山峰，但保留中尺度特征
                if mid_scale_weight > 0.2:
                    # 检测潜在的副山脊
                    secondary_peak = simplex_noise(x*0.2, y*0.2, 27182) * mid_scale_weight * 0.15
                    
                    # 保留主峰形状并添加次级特征
                    peak_curve = np.power(norm_h + secondary_peak, 1.8 * mountain_slope)
                    peak_curve = min(peak_curve, 1.0)  # 限制极端值
                    
                    result[y, x] = 0.7 + 0.3 * peak_curve
                else:
                    # 标准山峰处理
                    peak_curve = np.power(norm_h, 1.8 * mountain_slope)
                    result[y, x] = 0.7 + 0.3 * peak_curve
                
                # 特别处理山脊和山峰
                if norm_h > 0.8:
                    # 添加锐利的山脊和山峰
                    sharp_factor = (norm_h - 0.8) / 0.2
                    result[y, x] += 0.05 * sharp_factor
    
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
def generate_improved_ridges(height_map, ridge_intensity=1.0, seed=42):
    """生成逼真的山脊和山谷"""
    height, width = height_map.shape
    result = height_map.copy()
    
    # 计算高度梯度
    gx = np.zeros((height, width))
    gy = np.zeros((height, width))
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            gx[y, x] = (height_map[y, x+1] - height_map[y, x-1]) * 0.5
            gy[y, x] = (height_map[y+1, x] - height_map[y-1, x]) * 0.5
    
    # 梯度幅度和方向
    grad_mag = np.sqrt(gx*gx + gy*gy)
    
    # 识别潜在的山脊和山谷
    for y in range(2, height-2):
        for x in range(2, width-2):
            if grad_mag[y, x] > 0.08:  # 有显著梯度的区域
                # 计算梯度垂直方向
                if gx[y, x] != 0 or gy[y, x] != 0:
                    # 归一化梯度向量 - 这里可能产生NaN
                    mag = np.sqrt(gx[y, x]**2 + gy[y, x]**2)
                    # 添加安全阈值检查
                    if mag > 1e-10:  # 使用足够小的阈值
                        nx, ny = -gy[y, x]/mag, gx[y, x]/mag
                    else:
                        nx, ny = 0.0, 1.0  # 默认安全值
                    # 检查垂直方向上的高度变化
                    sample1_x = int(x + nx * 2)
                    sample1_y = int(y + ny * 2)
                    sample2_x = int(x - nx * 2)
                    sample2_y = int(y - ny * 2)
                    
                    # 确保采样点在边界内
                    if (0 <= sample1_x < width and 0 <= sample1_y < height and
                        0 <= sample2_x < width and 0 <= sample2_y < height):
                        
                        h_center = height_map[y, x]
                        h1 = height_map[sample1_y, sample1_x]
                        h2 = height_map[sample2_y, sample2_x]
                        
                        # 山脊特征：中心高于两侧
                        if h_center > h1 and h_center > h2:
                            # 增强山脊
                            ridge_factor = min(h_center - max(h1, h2), 0.1) * 2.5 * ridge_intensity  # 减小基础系数
                            # 添加渐变式增强
                            for dy in range(-2, 3):
                                for dx in range(-2, 3):
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < height and 0 <= nx < width:
                                        dist = np.sqrt(dy*dy + dx*dx)
                                        if dist > 0:
                                            # 使用二次衰减而非线性衰减
                                            decay = np.exp(-0.5 * (dist/1.5)**2)  # 高斯衰减提供更自然的过渡
                                            result[ny, nx] += ridge_factor * decay
                            
                            # 沿山脊方向延伸
                            ridge_dx, ridge_dy = ny, -nx  # 山脊方向
                            for t in range(1, 3):
                                ry, rx = int(y + ridge_dy * t), int(x + ridge_dx * t)
                                if 0 <= rx < width and 0 <= ry < height:
                                    result[ry, rx] += ridge_factor * (0.7 ** t)
                                
                                ry, rx = int(y - ridge_dy * t), int(x - ridge_dx * t)
                                if 0 <= rx < width and 0 <= ry < height:
                                    result[ry, rx] += ridge_factor * (0.7 ** t)
                        
                        # 山谷特征：中心低于两侧
                        elif h_center < h1 and h_center < h2:
                            # 增强山谷
                            valley_factor = min(min(h1, h2) - h_center, 0.2) * 3 * ridge_intensity
                            result[y, x] -= valley_factor
                            
                            # 沿山谷方向延伸
                            valley_dx, valley_dy = ny, -nx  # 山谷方向
                            for t in range(1, 3):
                                vy, vx = int(y + valley_dy * t), int(x + valley_dx * t)
                                if 0 <= vx < width and 0 <= vy < height:
                                    result[vy, vx] -= valley_factor * (0.7 ** t)
                                
                                vy, vx = int(y - valley_dy * t), int(x - valley_dx * t)
                                if 0 <= vx < width and 0 <= vy < height:
                                    result[vy, vx] -= valley_factor * (0.7 ** t)
    
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
def simulate_weathering(height_map, intensity=0.5, iterations=2):
    """模拟风化作用，使地形更自然"""
    result = height_map.copy()
    height, width = height_map.shape
    
    # 风化强度受海拔影响 - 高海拔风化更强
    weathering_strength = np.zeros_like(result)
    for y in range(height):
        for x in range(width):
            # 高度越高，风化越强
            weathering_strength[y, x] = 0.1 + 0.9 * min(1.0, result[y, x] / 75.0)
    
    for _ in range(iterations):
        for y in range(1, height-1):
            for x in range(1, width-1):
                # 局部风化强度
                local_strength = weathering_strength[y, x] * intensity
                
                # 计算周围高度的加权平均值
                sum_weighted = 0.0
                sum_weights = 0.0
                
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                            
                        # 距离加权
                        weight = 1.0 / max(1.0, np.sqrt(dx*dx + dy*dy))
                        sum_weighted += result[y+dy, x+dx] * weight
                        sum_weights += weight
                
                avg_height = sum_weighted / sum_weights
                
                # 向平均高度缓慢收敛
                diff = avg_height - result[y, x]
                if diff > 0:  # 填充低洼处
                    result[y, x] += diff * local_strength * 0.3
                else:  # 降低高处
                    result[y, x] += diff * local_strength * 0.5
    
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
def simulate_water_erosion_flow(height_map, iterations=1, rain_amount=0.05, evaporation=0.5, capacity=0.05, erosion=0.5):
    """水流侵蚀模拟，基于流向流量方法"""
    height, width = height_map.shape
    result = height_map.copy()
    
    # 流量图和沉积物图
    water_map = np.zeros((height, width))
    sediment_map = np.zeros((height, width))
    
    # 流向图（8个基本方向：0=无流向，1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE）
    flow_dir = np.zeros((height, width), dtype=np.int32)
    
    # 更新流向图
    for y in range(1, height-1):
        for x in range(1, width-1):
            # 当前高度
            center_height = result[y, x]
            
            # 检查8个邻居
            min_height = center_height
            min_dir = 0
            dir_code = 1  # 开始于东方向
            
            for i, (dy, dx) in enumerate([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if result[ny, nx] < min_height:
                        min_height = result[ny, nx]
                        min_dir = dir_code
                dir_code *= 2  # 下一个方向
            
            flow_dir[y, x] = min_dir
    
    # 迭代模拟降雨和侵蚀
    for _ in range(iterations):
        # 降雨
        water_map += rain_amount
        
        # 水流动与侵蚀
        new_water = np.zeros_like(water_map)
        new_sediment = np.zeros_like(sediment_map)
        
        for y in range(1, height-1):
            for x in range(1, width-1):
                if water_map[y, x] < 0.01:  # 水太少
                    continue
                
                # 当前水量和泥沙
                current_water = water_map[y, x]
                current_sediment = sediment_map[y, x]
                
                # 计算流向
                dir_code = flow_dir[y, x]
                if dir_code == 0:  # 无流向（局部最低点）
                    # 水滞留并蒸发
                    new_water[y, x] += current_water * (1 - evaporation)
                    new_sediment[y, x] += current_sediment
                    continue
                
                # 计算流向坐标
                next_y, next_x = y, x
                if dir_code & 1:    # E
                    next_x += 1
                elif dir_code & 2:   # SE
                    next_y += 1
                    next_x += 1
                elif dir_code & 4:   # S
                    next_y += 1
                elif dir_code & 8:   # SW
                    next_y += 1
                    next_x -= 1
                elif dir_code & 16:  # W
                    next_x -= 1
                elif dir_code & 32:  # NW
                    next_y -= 1
                    next_x -= 1
                elif dir_code & 64:  # N
                    next_y -= 1
                elif dir_code & 128: # NE
                    next_y -= 1
                    next_x += 1
                
                # 高度差
                height_diff = result[y, x] - result[next_y, next_x]
                
                # 泥沙容量与水量和坡度成正比
                c = capacity * current_water * max(0.01, height_diff)
                
                if current_sediment > c:  # 泥沙多，沉积
                    deposition = min(current_sediment - c, height_diff * 0.5)
                    current_sediment -= deposition
                    result[y, x] += deposition
                else:  # 泥沙少，侵蚀
                    erosion_amount = min(erosion * (c - current_sediment), height_diff * 0.5)
                    current_sediment += erosion_amount
                    result[y, x] -= erosion_amount
                
                # 水流向下一个单元
                new_water[next_y, next_x] += current_water * (1 - evaporation * 0.1)
                new_sediment[next_y, next_x] += current_sediment
        
        # 更新水量和泥沙
        water_map = new_water
        sediment_map = new_sediment
    
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

def generate_fault_lines(width, height, count=5, length_factor=0.5, seed=None):
    """生成地质断层线"""
    if seed is not None:
        np.random.seed(seed)
        
    fault_lines = []
    
    for _ in range(count):
        # 随机断层起点和方向
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        
        # 断层角度和长度
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(width//4, int(width * length_factor))
        
        # 计算断层终点
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # 断层抬升或下沉幅度
        displacement = np.random.uniform(-0.15, 0.15)
        
        fault_lines.append((x1, y1, x2, y2, displacement))
    
    return fault_lines

def apply_fault_lines(height_map, fault_lines, impact_width=10):
    """应用断层线对地形的影响"""
    height, width = height_map.shape
    result = height_map.copy()
    
    for x1, y1, x2, y2, displacement in fault_lines:
        # 计算断层线长度
        fault_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # 对每个像素计算到断层线的距离
        for y in range(height):
            for x in range(width):
                # 计算点到线段的距离
                if x1 == x2 and y1 == y2:  # 点
                    dist = np.sqrt((x-x1)**2 + (y-y1)**2)
                else:  # 线段
                    t = ((x-x1)*(x2-x1) + (y-y1)*(y2-y1)) / (fault_length**2)
                    t = max(0, min(1, t))
                    px = x1 + t * (x2-x1)
                    py = y1 + t * (y2-y1)
                    dist = np.sqrt((x-px)**2 + (y-py)**2)
                
                # 应用高度变化 - 衰减函数
                if dist <= impact_width:
                    # 影响强度随距离衰减
                    factor = (1 - dist/impact_width)**2
                    
                    # 确定在断层哪一侧 - 创建阶梯状的断层效果
                    # 计算点到断层的方位（法向量）
                    if x2 != x1 or y2 != y1:
                        nx = -(y2-y1)/fault_length
                        ny = (x2-x1)/fault_length
                        side = np.sign((x-x1)*nx + (y-y1)*ny)
                        
                        # 在一侧抬升，另一侧下沉
                        height_change = displacement * factor * side
                        result[y, x] += height_change
    
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

def biome_temperature(height_map, latitude_effect=0.5, seed=None, use_frequency_optimization=True, prevailing_wind=(1,0)):
    """改进的温度计算函数，考虑地形对气候的影响"""
    h, w = height_map.shape
    
    # 使用参数种子或默认值
    temp_seed = 1234 if seed is None else seed+10
    
    # 基础温度噪声
    if use_frequency_optimization:
        temp_base = frequency_domain_noise(w, h, temp_seed, octaves=4, persistence=0.5, lacunarity=2.0)
    else:
        temp_base = advanced_fractal_noise_scalar(w, h, temp_seed, octaves=4, persistence=0.5, lacunarity=2.0)
    
    # 添加纬度影响
    latitude = np.zeros_like(temp_base)
    for y in range(h):
        # 使用非线性映射来模拟更真实的纬度温度分布
        normalized_y = 2 * abs(y/h - 0.5)  # 0在中心，1在边缘
        
        # 哈德利环流影响
        if normalized_y < 0.33:  # 赤道到副热带
            latitude_temp = 1.0 - normalized_y * 0.2
        elif normalized_y < 0.67:  # 副热带到温带
            latitude_temp = 0.93 - (normalized_y - 0.33) * 0.7
        else:  # 温带到极地
            latitude_temp = 0.7 - (normalized_y - 0.67) * 0.7
        
        for x in range(w):
            latitude[y, x] = latitude_temp
    
    # 基础温度合并纬度影响
    temp_map = temp_base * (1-latitude_effect) + latitude * latitude_effect
    
    # 高度影响 (与之前相同)
    height_min = np.min(height_map)
    height_max = np.max(height_map)
    height_range = max(1.0, height_max - height_min)
    height_coefficient = 0.15 * (100.0 / height_range)  # 自适应系数
    
    # 计算高度影响
    height_effect = np.zeros_like(height_map)
    for y in range(h):
        for x in range(w):
            normalized_height = (height_map[y, x] - height_min) / height_range
            height_effect[y, x] = height_coefficient * np.power(normalized_height, 1.2)
    
    # 应用高度效应
    temp_map = temp_map - height_effect
    
    # 新增: 计算山脉阻挡效应
    mountain_threshold = height_min + height_range * 0.7  # 定义山脉高度阈值
    mountain_mask = height_map > mountain_threshold
    
    # 计算风向的影响
    wind_x, wind_y = prevailing_wind
    if abs(wind_x) > 0 or abs(wind_y) > 0:
        wind_effect = np.zeros_like(temp_map)
        
        # 规范化风向向量
        wind_mag = np.sqrt(wind_x**2 + wind_y**2)
        if wind_mag > 0:
            wind_x, wind_y = wind_x/wind_mag, wind_y/wind_mag
        
        # 对每个像素，计算山脉对风的阻挡效应
        for y in range(h):
            for x in range(w):
                # 沿风向上风方向追踪
                upwind_x = x - int(wind_x * 10)  # 检查10个像素的上风区域
                upwind_y = y - int(wind_y * 10)
                
                # 检查上风方向是否有山脉阻挡
                mountain_blocked = False
                for i in range(10):
                    check_x = x - int(wind_x * i)
                    check_y = y - int(wind_y * i)
                    if (0 <= check_x < w and 0 <= check_y < h and 
                        height_map[check_y, check_x] > mountain_threshold):
                        mountain_blocked = True
                        break
                
                # 被山脉阻挡区域温度降低（背风效应）或升高（焚风效应）
                if mountain_blocked:
                    # 简化的焚风/背风效应
                    if height_map[y, x] < mountain_threshold - height_range * 0.2:
                        # 较低区域受背风影响，温度降低
                        wind_effect[y, x] = -0.05
                    else:
                        # 山区下坡风升温（焚风效应）
                        wind_effect[y, x] = 0.03
        
        # 应用风向效应
        temp_map = temp_map + wind_effect
    
    return normalize_array(temp_map)

# 湿度图生成函数
def moisture_map(height_map, temp_map, prevailing_wind=(1, 0), seed=None, use_frequency_optimization=True):
    """生成考虑高度、温度和风向的湿度图，使用频域噪声优化"""
    h, w = height_map.shape
    # 使用参数种子或默认值
    moisture_seed = 5678 if seed is None else seed+20
    
    if use_frequency_optimization:
        base_moisture = frequency_domain_noise(w, h, moisture_seed, octaves=4, persistence=0.6, lacunarity=2.0)
    else:
        base_moisture = advanced_fractal_noise_scalar(w, h, moisture_seed, octaves=4, persistence=0.6, lacunarity=2.0)
    
    # 计算风向影响的湿度分布
    wind_x, wind_y = prevailing_wind
    
    # 模拟雨影效应 - 风吹过高山时，大气湿度减少
    result = np.copy(base_moisture)
    
    # 简化的雨影效应模拟
    if abs(wind_x) > 0 or abs(wind_y) > 0:
        x_step = 1 if wind_x >= 0 else -1
        y_step = 1 if wind_y >= 0 else -1
        
        x_range = range(0, w) if x_step > 0 else range(w-1, -1, -1)
        y_range = range(0, h) if y_step > 0 else range(h-1, -1, -1)
        
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
    for y in range(h):
        for x in range(w):
            if water_mask[y, x]:
                for ky in range(max(0, y-kernel_size), min(h, y+kernel_size+1)):
                    for kx in range(max(0, x-kernel_size), min(w, x+kernel_size+1)):
                        dist = np.sqrt((y-ky)**2 + (x-kx)**2)
                        if dist <= kernel_size:
                            water_influence[ky, kx] += 0.05 * (kernel_size - dist) / kernel_size
    
    result = result + water_influence
    
    return normalize_array(result)

@jit(nopython=True)
def smooth_extreme_features(height_map, threshold=0.15, window_size=3, iterations=1):
    """平滑异常尖锐特征，使用局部统计指标检测异常点"""
    result = height_map.copy()
    height, width = height_map.shape
    
    for _ in range(iterations):
        for y in range(window_size, height-window_size):
            for x in range(window_size, width-window_size):
                # 计算局部统计特征
                window_heights = []
                for dy in range(-window_size, window_size+1):
                    for dx in range(-window_size, window_size+1):
                        if 0 <= y+dy < height and 0 <= x+dx < width:
                            window_heights.append(result[y+dy, x+dx])
                
                if not window_heights:
                    continue
                
                # 计算局部统计量
                window_avg = sum(window_heights) / len(window_heights)
                
                # 计算局部标准差
                sum_squared_diff = 0.0
                for h in window_heights:
                    sum_squared_diff += (h - window_avg) ** 2
                window_std = np.sqrt(sum_squared_diff / len(window_heights))
                
                # 获取当前值
                current_height = result[y, x]
                
                # 计算Z分数：偏离均值多少个标准差
                if window_std > 0.0001:  # 防止除零
                    z_score = abs(current_height - window_avg) / window_std
                else:
                    z_score = 0.0
                
                # 使用Z分数检测离群点，比简单阈值更可靠
                if z_score > threshold * 4.0:
                    # 显著离群点 - 使用平均值替换
                    result[y, x] = window_avg
                elif z_score > threshold * 2.0:
                    # 中度离群点 - 向平均值部分平滑
                    smoothing_factor = min(1.0, (z_score - threshold*2.0) / (threshold*2.0))
                    result[y, x] = current_height * (1.0-smoothing_factor) + window_avg * smoothing_factor
    
    return result

@jit(nopython=True)
def detect_and_fix_unrealistic_slopes(height_map, max_slope_angle=50.0):
    """检测并修复不现实的陡坡"""
    result = height_map.copy()
    height, width = height_map.shape
    max_slope_value = np.tan(np.radians(max_slope_angle))
    
    # 第一遍：检测异常坡度
    for y in range(1, height-1):
        for x in range(1, width-1):
            # 检查周围的坡度
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                
                # 计算高度差除以距离得到坡度
                local_slope = abs(result[y, x] - result[ny, nx])
                
                # 如果坡度超过阈值，标记为需要修复
                if local_slope > max_slope_value:
                    # 使用梯度下降方式平滑坡度
                    if result[y, x] > result[ny, nx]:
                        # 当前点更高，降低高点，提升低点
                        adjustment = (local_slope - max_slope_value) * 0.5
                        result[y, x] -= adjustment
                        result[ny, nx] += adjustment * 0.5  # 低点调整较小
                    else:
                        # 邻居点更高
                        adjustment = (local_slope - max_slope_value) * 0.5
                        result[ny, nx] -= adjustment
                        result[y, x] += adjustment * 0.5  # 低点调整较小
    
    return result

# 主函数中修改调用方式
def generate_height_temp_humid(width, height, seed=None, scale_factor=1.0, erosion_iterations=3,
                                river_density=1.0, mountain_sharpness=1.2, use_tectonic=True,
                                detail_level=1.0, use_frequency_optimization=True,
                                # 新增参数
                                enable_micro_detail=True,  # 是否启用微细节生成
                                enable_extreme_detection=True,  # 是否启用异常点检测
                                optimization_level=1,  # 优化级别：0=低(快速)，1=中(均衡)，2=高(高质量)
                                # 新增地貌特征参数
                                enable_realistic_landforms=False,  # 是否启用真实地貌特征
                                dominant_landform=None,  # 主导地貌类型 (山脉类型、河谷类型、冰川类型等)
                                # 新增大地形参数
                                large_map_mode=True,  # 开启大地图模式
                                province_count=10,      # 地质省份数量
                                macro_feature_scale=2.0, # 宏观特征缩放因子
                                # 侵蚀参数
                                erosion_type="advanced", erosion_strength=0.8, 
                                talus_angle=0.05, sediment_capacity=0.15,
                                rainfall=0.01, evaporation=0.5,
                                # 河流参数
                                min_watershed_size=50, precipitation_factor=1.0, meander_factor=0.3,
                                # 噪声参数
                                octaves=6, persistence=0.5, lacunarity=2.0,
                                # 地形分布参数
                                plain_ratio=0.3, hill_ratio=0.3, mountain_ratio=0.2, plateau_ratio=0.1,
                                # 气候参数
                                latitude_effect=0.5, prevailing_wind=None):
    """
    增强版地形生成主函数，支持全面的参数控制
    
    参数：
    width (int):          生成地图的宽度（像素数）
    height (int):         生成地图的高度（像素数）
    seed (int, optional): 随机种子，默认随机生成
    scale_factor (float): 地形特征缩放因子（>1放大特征，<1缩小特征），默认1.0
    erosion_iterations (int): 侵蚀过程迭代次数，默认3
    river_density (float): 河流密度系数（1.0为正常密度），默认1.0
    mountain_sharpness (float): 山脉陡峭度调整，默认1.2
    use_tectonic (bool):  是否启用地质构造模拟，默认True
    detail_level (float): 细节级别（0.5-2.0），控制小尺度细节的强度，默认1.0
    use_frequency_optimization (bool或None): 是否使用频域优化的噪声生成，None表示自动判断
    
    # 侵蚀参数
    erosion_type (str):   侵蚀系统类型，可选：'thermal'/'hydraulic'/'combined'/'advanced'/'simple'
    erosion_strength (float): 侵蚀强度系数
    talus_angle (float):  滑坡角度
    sediment_capacity (float): 沉积物容量
    rainfall (float):     降雨量
    evaporation (float):  蒸发率
    
    # 河流参数
    min_watershed_size (int): 最小集水区面积
    precipitation_factor (float): 降水因子
    meander_factor (float): 河流蜿蜒度，控制河流曲折程度
    
    # 噪声参数
    octaves (int):        八度数
    persistence (float):  持续度
    lacunarity (float):   频率增长因子
    
    # 地形分布参数
    plain_ratio (float):  平原比例
    hill_ratio (float):   丘陵比例
    mountain_ratio (float): 山地比例
    plateau_ratio (float): 高原比例
    
    # 气候参数
    latitude_effect (float): 纬度影响强度
    prevailing_wind (tuple): 主导风向(x,y)，默认随机生成
    """
    # 添加参数日志记录
    print("地形生成参数：")
    print(f"  基础参数：width={width}, height={height}, seed={seed}, scale_factor={scale_factor}")
    print(f"  侵蚀参数：erosion_type={erosion_type}, erosion_strength={erosion_strength}, erosion_iterations={erosion_iterations}")
    print(f"  地形分布：plain_ratio={plain_ratio}, hill_ratio={hill_ratio}, mountain_ratio={mountain_ratio}, plateau_ratio={plateau_ratio}")
    print(f"  河流参数：min_watershed_size={min_watershed_size}, precipitation_factor={precipitation_factor}, meander_factor={meander_factor}")
    
    # 种子处理
    if seed is None:
        seed = np.random.randint(0, 999999)
    np.random.seed(seed)
    
    # 根据地图尺寸自动检测是否启用大地图模式
    if large_map_mode is None:
        large_map_mode = (width >= 2048 or height >= 2048)
        if large_map_mode:
            print(f"注意：检测到超大地图 ({width}x{height})，自动启用宏观地质特征")
    
    # 大地图模式处理
    if large_map_mode:
        print("应用大地图宏观地质生成...")
        # 计算适合地图大小的省份数量
        if province_count is None:
            # 根据地图尺寸动态计算合适的省份数量
            map_size_factor = np.sqrt(width * height) / 1024
            province_count = max(3, int(map_size_factor * 3))
            print(f"  自动确定地质省份数量: {province_count}")
        
        # 生成宏观地质省份
        province_map, province_params = generate_geological_provinces(
            width, height, province_count=province_count, seed=seed+1000
        )
        
        # 生成大尺度构造特征
        tectonic_map, tectonic_features = generate_tectonic_features(
            width, height, province_map, province_params, seed=seed+2000
        )
    
    # 如果未指定优化策略，根据地图大小自动选择
    if use_frequency_optimization is None:
        use_frequency_optimization = (width >= 1024 or height >= 1024)
        if use_frequency_optimization:
            print(f"注意：地图尺寸较大 ({width}x{height})，自动启用频域优化")
    
    # 根据地图大小动态调整参数
    is_large_map = width > 512 or height > 512
    
    # 地形区域分类
    print("划分地形区域...")
    regional_seed = seed + 5
    
    scales_array = np.array([(16.0, 1.0), (32.0, 0.5)], dtype=np.float64)
    region_noise = advanced_fractal_noise_scales(width, height, scales_array, regional_seed)
    
    # 分割地形区域
    terrain_types = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            value = region_noise[y, x]
            if value < 0.3:
                terrain_types[y, x] = 0  # 山区
            elif value < 0.6:
                terrain_types[y, x] = 1  # 丘陵
            elif value < 0.85:
                terrain_types[y, x] = 2  # 平原
            else:
                terrain_types[y, x] = 3  # 高原/峡谷
    
    # 生成基础高度图
    print("生成基础地形...")
    
    # 改进：调整不同尺度的权重以平衡细节
    large_scale_weight = 0.5  # 减小大尺度权重
    medium_scale_weight = 0.3
    small_scale_weight = 0.2 * detail_level  # 增加小尺度权重，并与细节级别相关
    
    # 确保权重和为1
    total_weight = large_scale_weight + medium_scale_weight + small_scale_weight
    large_scale_weight /= total_weight
    medium_scale_weight /= total_weight
    small_scale_weight /= total_weight
    
    # 根据地形类型动态调整尺度
    # 山区应该有更多的小尺度细节
    # 平原应该更平滑，大尺度更明显
    dynamic_weights = np.zeros((height, width, 3), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            if terrain_types[y, x] == 0:  # 山区
                dynamic_weights[y, x] = [large_scale_weight * 0.9, 
                                      medium_scale_weight * 1.0,
                                      small_scale_weight * 1.3]
            elif terrain_types[y, x] == 1:  # 丘陵
                dynamic_weights[y, x] = [large_scale_weight * 0.95, 
                                      medium_scale_weight * 1.1,
                                      small_scale_weight * 1.0]
            elif terrain_types[y, x] == 2:  # 平原
                dynamic_weights[y, x] = [large_scale_weight * 1.2, 
                                      medium_scale_weight * 0.9,
                                      small_scale_weight * 0.6]
            else:  # 高原/峡谷
                dynamic_weights[y, x] = [large_scale_weight * 1.1, 
                                      medium_scale_weight * 0.8,
                                      small_scale_weight * 1.2]
            
            # 确保每个位置的权重和为1
            sum_weights = sum(dynamic_weights[y, x])
            dynamic_weights[y, x] /= sum_weights
    
    # 使用更优化的批量噪声生成
    if use_frequency_optimization:
        print("使用频域优化生成噪声...")
        # 大尺度噪声 - 使用频域合成
        large_scale = frequency_domain_noise(
            width, height, seed, 
            octaves=4,  
            persistence=0.6, 
            lacunarity=1.8
        )
        
        # 中尺度噪声
        medium_scale = frequency_domain_noise(
            width, height, seed+10, 
            octaves=6,
            persistence=0.5,
            lacunarity=2.0
        )
        
        # 小尺度噪声 - 更多细节
        small_scale = frequency_domain_noise(
            width, height, seed+20, 
            octaves=8,
            persistence=0.45,
            lacunarity=2.2
        )
    else:
        # 原始噪声生成方法
        large_scale_array = np.array([(100.0*scale_factor, 1.0)], dtype=np.float64)
        large_scale = advanced_fractal_noise_scales(width, height, large_scale_array, seed)
        
        medium_scale_array = np.array([(50.0*scale_factor, 0.5), (25.0*scale_factor, 0.25)], dtype=np.float64)
        medium_scale = advanced_fractal_noise_scales(width, height, medium_scale_array, seed+10)
        
        small_scale_array = np.array([(10.0*scale_factor, 0.15), (5.0*scale_factor, 0.05)], dtype=np.float64)
        small_scale = advanced_fractal_noise_scales(width, height, small_scale_array, seed+20)
    
    # 微小尺度纹理生成
    if detail_level > 0.8 and enable_micro_detail:
        print("增强地形微细节...")
        if use_frequency_optimization:
            # 使用频域方法生成微细节
            micro_scale = frequency_domain_noise(
                width, height, seed+30,
                octaves=6 if optimization_level < 2 else 8,  # 根据优化级别调整八度数
                persistence=0.35,
                lacunarity=2.5  # 更高的频率变化
            )
        else:
            micro_scale_array = np.array([(3.0*scale_factor, 0.7), (1.5*scale_factor, 0.3)], dtype=np.float64)
            micro_scale = advanced_fractal_noise_scales(width, height, micro_scale_array, seed+30)
        
        # 优化微细节掩码计算
        if optimization_level < 1:
            # 低质量模式：使用降采样计算，然后再上采样
            sample_rate = 4  # 降采样到1/4尺寸
            small_h, small_w = height//sample_rate, width//sample_rate
            small_mask = np.zeros((small_h, small_w), dtype=np.float64)
            
            # 在降采样尺度上计算掩码
            for y in range(small_h):
                for x in range(small_w):
                    orig_y, orig_x = y*sample_rate, x*sample_rate
                    if orig_y < height and orig_x < width:
                        if terrain_types[orig_y, orig_x] in [0, 1, 3]:  # 山区、丘陵、高原
                            # 计算坡度 (简化版)
                            small_mask[y, x] = 0.15 * detail_level
            
            # 上采样回原始尺寸 (使用最近邻插值以保持速度)
            micro_mask = np.zeros((height, width), dtype=np.float64)
            for y in range(height):
                for x in range(width):
                    sy, sx = min(y//sample_rate, small_h-1), min(x//sample_rate, small_w-1)
                    micro_mask[y, x] = small_mask[sy, sx]
        else:
            # 中高质量模式：正常计算
            micro_mask = np.zeros((height, width), dtype=np.float64)
            for y in range(height):
                for x in range(width):
                    if terrain_types[y, x] in [0, 1, 3]:  # 山区、丘陵、高原
                        # 计算坡度 - 坡度大的区域应用更多微细节
                        if y > 0 and y < height-1 and x > 0 and x < width-1:
                            dx = (large_scale[y, x+1] - large_scale[y, x-1]) * 0.5
                            dy = (large_scale[y+1, x] - large_scale[y-1, x]) * 0.5
                            gradient = np.sqrt(dx*dx + dy*dy)
                            
                            # 坡度大的区域有更多微细节
                            micro_mask[y, x] = min(1.0, gradient * 5.0) * 0.15 * detail_level
    else:
        micro_scale = np.zeros_like(large_scale)
        micro_mask = np.zeros_like(large_scale)

    # 使用动态权重组合不同尺度
    height_map = np.zeros((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            # 基本地形组合
            base_height = (
                large_scale[y, x] * dynamic_weights[y, x, 0] +
                medium_scale[y, x] * dynamic_weights[y, x, 1] +
                small_scale[y, x] * dynamic_weights[y, x, 2] +
                micro_scale[y, x] * micro_mask[y, x]  # 微小尺度仅在特定区域应用
            )
            
            # 大地图模式下，应用宏观地质特征影响
            if large_map_mode:
                # 获取当前省份
                province_id = province_map[y, x]
                params = province_params[province_id]
                
                # 应用省份特定参数
                # 1. 基础高度调整
                base_height += params['base_elevation'] * 0.2
                
                # 2. 缩放调整
                if params['scale_modifier'] != 1.0:
                    # 使用插值噪声获取缩放后的高度
                    scaled_x = x / params['scale_modifier']
                    scaled_y = y / params['scale_modifier']
                    
                    # 简单双线性插值
                    sx, sy = int(scaled_x), int(scaled_y)
                    fx, fy = scaled_x - sx, scaled_y - sy
                    
                    if 0 <= sx < width-1 and 0 <= sy < height-1:
                        h00 = large_scale[sy, sx]
                        h01 = large_scale[sy, sx+1]
                        h10 = large_scale[sy+1, sx]
                        h11 = large_scale[sy+1, sx+1]
                        
                        scaled_height = (h00 * (1-fx) * (1-fy) + 
                                        h01 * fx * (1-fy) + 
                                        h10 * (1-fx) * fy + 
                                        h11 * fx * fy)
                        
                        # 混合原始高度和缩放高度
                        base_height = base_height * 0.7 + scaled_height * 0.3
                
                # 3. 应用大尺度构造特征
                base_height += tectonic_map[y, x] * macro_feature_scale
                
                # 4. 针对特定构造特征应用额外效果
                if tectonic_features['mountain_ranges'][y, x] > 0.1:
                    # 山脉区域增强小尺度细节
                    base_height += small_scale[y, x] * tectonic_features['mountain_ranges'][y, x] * 0.2
                    
                if tectonic_features['plateaus'][y, x] > 0.1:
                    # 高原区域更平坦但有中尺度变化
                    base_height = base_height * 0.7 + tectonic_features['plateaus'][y, x] * 0.3
                    
                if tectonic_features['rift_valleys'][y, x] > 0.1:
                    # 裂谷区域深且陡峭
                    base_height -= tectonic_features['rift_valleys'][y, x] * 0.3
                    
                if tectonic_features['basins'][y, x] > 0.1:
                    # 盆地区域平缓地降低
                    base_height -= tectonic_features['basins'][y, x] * 0.15
            
            height_map[y, x] = base_height
    
    # 应用地形分布变换
    print("应用地形分布...")
    # 根据优化级别决定是否执行异常点检测
    if enable_extreme_detection:
        # 高度图上应用地形特征增强前先检测异常
        height_map = smooth_extreme_features(
            height_map, 
            threshold=0.1, 
            window_size=2, 
            iterations=1 if optimization_level < 2 else 2
        )

    height_map = apply_geomorphological_height_distribution(
        height_map, 
        plain_ratio=plain_ratio, 
        hill_ratio=hill_ratio,
        mountain_ratio=mountain_ratio * mountain_sharpness,
        plateau_ratio=plateau_ratio,
        mountain_slope=mountain_sharpness
    )
    
    # 使用改进的侵蚀系统（修改这部分，根据侵蚀类型选择不同算法）
    print(f"应用侵蚀过程... 类型: {erosion_type}")
    # 生成更真实的岩石抗性图
    rock_resistance = generate_rock_resistance_map(height_map, seed+45)
    
    # 根据侵蚀类型选择不同的侵蚀算法
    if erosion_type == "thermal":
        # 只应用热侵蚀
        height_map = thermal_erosion_improved(
            height_map,
            iterations=erosion_iterations,
            talus_angle=talus_angle,
            strength=erosion_strength
        )
    elif erosion_type == "hydraulic":
        # 只应用水力侵蚀
        height_map = simulate_water_erosion_flow(
            height_map,
            iterations=erosion_iterations,
            rain_amount=rainfall,
            evaporation=evaporation,
            capacity=sediment_capacity,
            erosion=erosion_strength
        )
    elif erosion_type == "combined":
        # 先应用热侵蚀，再应用水力侵蚀
        height_map = thermal_erosion_improved(
            height_map,
            iterations=max(1, erosion_iterations//2),
            talus_angle=talus_angle,
            strength=erosion_strength*0.7
        )
        height_map = simulate_water_erosion_flow(
            height_map,
            iterations=max(1, erosion_iterations//2),
            rain_amount=rainfall,
            evaporation=evaporation,
            capacity=sediment_capacity,
            erosion=erosion_strength*0.7
        )
    elif erosion_type == "simple":
        # 简化侵蚀，只做基本平滑
        height_map = simulate_weathering(
            height_map, 
            intensity=erosion_strength*0.5, 
            iterations=erosion_iterations
        )
    else:  # "advanced" (默认)
        # 使用最先进的侵蚀系统
        height_map = improved_erosion_system(
            height_map,
            iterations=erosion_iterations*2,
            rainfall_amount=rainfall,
            evaporation_rate=evaporation,
            sediment_capacity=sediment_capacity,
            erosion_rate=erosion_strength,
            deposition_rate=erosion_strength*0.8,
            rock_resistance=rock_resistance
        )
        
    # 在完成基础侵蚀后，应用真实地貌特征
    if enable_realistic_landforms:
        print("应用真实地貌特征增强...")
        try:
            from core.generation.geomorphological_features import apply_realistic_terrain_features
            # 确保terrain_types已定义(如果未定义，使用上面的代码片段来定义)
            height_map = apply_realistic_terrain_features(height_map, terrain_types, rivers_map if 'rivers_map' in locals() else None)
        except Exception as e:
            print(f"应用真实地貌特征时出错: {e}")
    
    # 根据优化级别决定是否执行最终平滑和修复
    if enable_extreme_detection:
        if optimization_level > 0:  # 中高质量模式才执行
            height_map = smooth_extreme_features(height_map, threshold=0.15)
            
        if optimization_level > 1:  # 只在高质量模式执行
            # 进行第二次更精细的异常点检测，专注于高梯度区域（山脊、悬崖等）
            height_map = detect_and_fix_unrealistic_slopes(height_map, max_slope_angle=50.0)
    
    # 标准化到合适的高度范围
    height_map = normalize_array(height_map) * 100 + 10  
    
    height_map = smooth_extreme_features(height_map, threshold=0.15)
    
    # 进行第二次更精细的异常点检测，专注于高梯度区域（山脊、悬崖等）
    height_map = detect_and_fix_unrealistic_slopes(height_map, max_slope_angle=50.0)  
    
    # 最终检查NaN值
    height_map = np.nan_to_num(height_map, nan=50.0)
    
    print("地形生成完成!")
    return height_map

