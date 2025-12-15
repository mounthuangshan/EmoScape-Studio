#from __future__ import annotations
#标准库
import random
from numba import jit
import hashlib
from collections import namedtuple

#数据处理与科学计算
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, minimum_filter, maximum_filter
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

# 定义坐标点结构，统一坐标表示
Point = namedtuple('Point', ['y', 'x'])

#######################
#生成洞穴和裂谷
#######################
def carve_caves_and_ravines(height_map, seed=None, cave_density=0.5, ravine_density=0.5):
    """
    生成地质学上合理的洞穴和峡谷系统
    
    参数:
    - height_map: 高度图 (2D numpy数组)
    - seed: 随机种子
    - cave_density: 洞穴密度 (0-1)
    - ravine_density: 峡谷密度 (0-1)
    
    返回:
    - 修改后的高度图
    - 洞穴系统的拓扑信息 (入口点、连接通道)
    """
    # 确保高度图是numpy数组
    height_map = np.array(height_map, dtype=np.float32)
    h, w = height_map.shape
    
    # 常规种子处理
    seed = validate_seed(seed)
    random.seed(seed)  # 全局设置随机种子
    
    # 基于地图尺寸动态调整参数
    map_size = np.sqrt(h * w)
    
    # 动态调整洞穴数量和参数
    num_caves = max(3, int(map_size * 0.03 * cave_density))
    ravine_threshold = 0.75 - (ravine_density * 0.25)  # 密度越高，阈值越低
    ravine_intensity = 0.3 + (ravine_density * 0.4)  # 更平滑的密度映射
    min_cave_height = 15
    max_cave_height = 45
    
    # 1. 峡谷生成 - 使用多层噪声和地质侵蚀模型
    carved_map = _create_ravines(height_map, ravine_threshold, ravine_intensity, seed)
    
    # 2. 基于修改后的地图重新计算适宜性 - 避免在峡谷中生成洞穴
    suitability_map = _generate_cave_suitability_map(carved_map, seed + 1)
    
    # 3. 洞穴系统生成 - 基于连接的组件创建真实的洞穴网络
    caves, cave_entrances = _create_cave_system(carved_map, suitability_map, num_caves, 
                                              min_cave_height, max_cave_height, seed + 2)
    
    # 4. 细节增强 - 添加小型分支和地质细节
    final_map = _enhance_cave_details(carved_map, caves, seed + 3)
    
    # 转换洞穴数据格式为应用程序所需的格式
    formatted_caves = []
    for cave in caves:
        for point in cave:
            # 统一使用 x,y 坐标顺序输出
            formatted_caves.append({"x": point.x, "y": point.y})
    
    # 同样转换洞穴入口格式
    formatted_entrances = [{"x": pos.x, "y": pos.y} for pos in cave_entrances]
    
    # 合并洞穴信息
    cave_data = {
        "caves": formatted_caves,
        "entrances": formatted_entrances
    }
    
    return final_map, cave_data

@jit(nopython=True)
def _smooth_edges(height_map, affected_points, radius=2, smooth_factor=0.6):
    """平滑洞穴和峡谷边缘，使其更自然"""
    h, w = height_map.shape
    result = height_map.copy()
    
    for y, x in affected_points:
        # 局部平滑操作
        for j in range(max(0, y-radius), min(h, y+radius+1)):
            for i in range(max(0, x-radius), min(w, x+radius+1)):
                dist = np.sqrt((y-j)**2 + (x-i)**2)
                if dist <= radius:
                    weight = (radius - dist) / radius * smooth_factor
                    # 边缘平滑: 当前高度向周围高度平均值靠拢
                    neighbors = []
                    for nj in range(max(0, j-1), min(h, j+2)):
                        for ni in range(max(0, i-1), min(w, i+2)):
                            if nj != j or ni != i:
                                neighbors.append(result[nj, ni])
                    
                    if neighbors:
                        avg_height = sum(neighbors) / len(neighbors)
                        result[j, i] = result[j, i] * (1-weight) + avg_height * weight
    
    return result

def _generate_cave_suitability_map(height_map, seed):
    """基于地质特性生成适合形成洞穴的区域图"""
    h, w = height_map.shape
    noise_gen = OpenSimplex(seed=seed)
    
    # 使用网格生成坐标向量化
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_scaled = x_coords/30.0
    y_scaled = y_coords/30.0
    
    # 批量计算各层噪声 (向量化)
    def apply_noise(x, y, scale=1.0, offset=0.0):
        result = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                result[i, j] = noise_gen.noise2((x[i, j] + offset) * scale, 
                                               (y[i, j] + offset) * scale)
        return result
    
    # 计算三层噪声
    base_rock = apply_noise(x_scaled, y_scaled, 1/3.0, 0)
    fractures = apply_noise(x_scaled, y_scaled, 1.0, 100) * 0.5
    dissolution = apply_noise(x_scaled, y_scaled, 2.0, 300) * 0.25
    
    # 组合噪声层
    combined = base_rock + fractures + dissolution
    
    # 考虑高度因素 (向量化)
    height_factor = np.ones((h, w), dtype=np.float32)
    # 水下或近水区域不适合
    height_factor[height_map < 10] = 0.1
    # 高山区域不适合
    height_factor[height_map > 80] = 0.3
    # 中等高度最适合
    mask = (height_map >= 10) & (height_map <= 80)
    height_diff = np.abs(height_map[mask] - 35)
    height_factor[mask] = 1.0 - np.minimum(1.0, height_diff / 35.0) * 0.7
    
    # 计算最终适宜性
    suitability = (combined * 0.5 + 0.5) * height_factor
    
    # 应用高斯平滑使过渡更自然
    suitability = gaussian_filter(suitability, sigma=1.0)
    
    # 确保没有NaN值
    return np.nan_to_num(suitability, nan=0.5)

def _calculate_local_slope(height_map, x, y, window=3):
    """计算局部坡度，用于洞穴入口选择"""
    h, w = height_map.shape
    if x <= 0 or x >= w-1 or y <= 0 or y >= h-1:
        return 0  # 边界处返回0
    
    # 计算小窗口内的坡度
    window_size = window // 2
    local_area = height_map[max(0, y-window_size):min(h, y+window_size+1),
                           max(0, x-window_size):min(w, x+window_size+1)]
    
    # 计算最大高度差作为坡度指标
    if local_area.size > 0:
        height_center = height_map[y, x]
        max_diff = np.max(np.abs(local_area - height_center))
        return max_diff
    return 0

def _create_ravines(height_map, threshold=0.7, intensity=0.5, seed=None):
    """创建逼真的峡谷系统，考虑水文和地质特性"""
    h, w = height_map.shape
    result = height_map.copy()
    
    # 创建噪声发生器
    noise_gen = OpenSimplex(seed=seed)
    
    # 准备多层噪声
    base_scale = 50.0 / intensity  # 基础尺度，受强度参数影响
    detail_scale = base_scale / 3.0  # 细节尺度
    
    # 计算侵蚀强度映射 - 使用噪声确定每个位置的侵蚀强度
    erosion_map = np.zeros((h, w), dtype=np.float32)
    direction_map = np.zeros((h, w, 2), dtype=np.float32)  # 存储水流方向
    
    # 向量化计算峡谷路径和侵蚀强度
    for y in range(h):
        for x in range(w):
            # 基础峡谷形态
            base_val = noise_gen.noise2(x/base_scale, y/base_scale)
            
            # 添加细节变化
            detail_val = noise_gen.noise2(x/detail_scale + 500, y/detail_scale + 500) * 0.3
            
            # 计算水流方向 - 用于形成连续峡谷
            flow_x = noise_gen.noise2(x/base_scale + 1000, y/base_scale + 1000)
            flow_y = noise_gen.noise2(x/base_scale + 2000, y/base_scale + 2000)
            direction_map[y, x] = [flow_x, flow_y]
            
            # 组合噪声值
            combined = base_val * 0.7 + detail_val
            
            # 计算侵蚀强度 - 阈值越高，峡谷越稀疏
            if combined > threshold:
                # 平滑的侵蚀强度曲线
                erosion_strength = ((combined - threshold) / (1 - threshold)) ** 1.5
                erosion_map[y, x] = erosion_strength * 15 * intensity
    
    # 应用流体模拟，创建连续的峡谷路径
    flow_accumulated = np.zeros_like(erosion_map)
    _simulate_water_flow(direction_map, erosion_map, flow_accumulated, iterations=5, seed=seed)
    
    # 应用侵蚀效果 - 高度减少
    for y in range(h):
        for x in range(w):
            # 只在符合高度条件的区域侵蚀
            if result[y, x] > 15:
                erosion = flow_accumulated[y, x] + erosion_map[y, x]
                # 应用侵蚀，并确保高度不会低于合理值
                result[y, x] -= erosion
                result[y, x] = max(result[y, x], 3)
                
                # 如果侵蚀强度大，考虑附近也会被水冲刷
                if erosion > 5:
                    radius = int(min(3, erosion / 3))
                    for j in range(max(0, y-radius), min(h, y+radius+1)):
                        for i in range(max(0, x-radius), min(w, x+radius+1)):
                            if (i != x or j != y) and result[j, i] > 10:
                                dist = np.sqrt((y-j)**2 + (x-i)**2)
                                if dist <= radius:
                                    side_erosion = erosion * (1 - dist/radius) * 0.5
                                    result[j, i] -= side_erosion
                                    result[j, i] = max(result[j, i], 3)
    
    # 平滑处理，使峡谷边缘更自然
    result = gaussian_filter(result, sigma=0.7)
    
    return result

def _simulate_water_flow(direction_map, erosion_map, accumulation, iterations=3, seed=None):
    """模拟水流侵蚀过程，形成连续的峡谷系统"""
    h, w = direction_map.shape[0:2]
    rng = random.Random(seed)  # 创建独立的随机数生成器
    
    # 初始水源点 - 选择侵蚀强度高的点作为起点
    sources = []
    for y in range(h):
        for x in range(w):
            if erosion_map[y, x] > 8:
                sources.append((y, x))
    
    # 如果没有足够的源点，添加一些随机点
    if len(sources) < 5:
        for _ in range(5 - len(sources)):
            y = rng.randint(0, h-1)
            x = rng.randint(0, w-1)
            sources.append((y, x))
    
    # 水流模拟 - 从源点开始，沿流向前进
    for _ in range(iterations):
        for src_y, src_x in sources:
            # 初始化当前位置和水量
            y, x = src_y, src_x
            water = 1.0
            
            # 跟踪水流路径
            for step in range(100):  # 最多流动100步
                if y < 0 or y >= h or x < 0 or x >= w or water < 0.01:
                    break
                
                # 增加该点的累积侵蚀量
                accumulation[int(y), int(x)] += water * 0.2
                
                # 确定流向
                dx, dy = direction_map[int(y), int(x)]
                
                # 更新位置，沿流向前进
                new_x = x + dx * 2
                new_y = y + dy * 2
                
                # 确保新位置在地图范围内
                if new_y < 0 or new_y >= h or new_x < 0 or new_x >= w:
                    break
                
                # 减少水量 (模拟蒸发和渗透)
                water *= 0.95
                
                # 更新位置
                x, y = new_x, new_y

def _create_cave_system(height_map, suitability_map, num_caves, min_height, max_height, seed):
    """创建相互连接的真实洞穴系统"""
    h, w = height_map.shape
    result = height_map.copy()
    
    # 创建独立的随机数生成器
    rng = random.Random(seed)
    
    # 确保没有NaN值
    suitability_map = np.nan_to_num(suitability_map, nan=0.5)
    
    # 1. 选择洞穴入口位置 - 考虑适宜性、高度条件和坡度
    potential_entrances = []
    for y in range(h):
        for x in range(w):
            height = result[y, x]
            if min_height <= height <= max_height and suitability_map[y, x] > 0.5:
                # 计算局部坡度
                slope = _calculate_local_slope(result, x, y)
                if slope < 5:  # 避免陡峭区域
                    # 评分: 适宜性 + 高度适中度 + 坡度适中度 + 少许随机性
                    height_score = 1.0 - abs((height - (min_height + max_height)/2)) / ((max_height - min_height)/2)
                    slope_score = 1.0 - min(1.0, slope / 5.0)
                    score = (suitability_map[y, x] * 0.5 + 
                             height_score * 0.25 + 
                             slope_score * 0.15 + 
                             rng.random() * 0.1)
                    potential_entrances.append((score, Point(y, x)))
    
    # 确保有足够的候选点
    if len(potential_entrances) < num_caves:
        # 放宽条件但保持地质合理性
        for y in range(h):
            for x in range(w):
                height = result[y, x]
                if 10 < height < 60 and not any(p.y == y and p.x == x for _, p in potential_entrances):
                    # 计算局部坡度，偏好较平缓的区域
                    slope = _calculate_local_slope(result, x, y)
                    slope_score = max(0.1, 1.0 - min(1.0, slope / 5.0))
                    potential_entrances.append((slope_score * 0.4 + rng.random() * 0.2, Point(y, x)))
                    if len(potential_entrances) >= num_caves * 2:
                        break
            if len(potential_entrances) >= num_caves * 2:
                break
    
    # 选择最佳入口点
    potential_entrances.sort(reverse=True)  # 按分数降序
    entrances = [p for _, p in potential_entrances[:num_caves]]
    
    # 2. 创建洞穴及连接通道
    caves = []
    valid_entrances = []  # 存储成功生成洞穴的入口
    
    # 第一次尝试创建洞穴
    for i, point in enumerate(entrances):
        # 安全地获取适宜性值并确保其有效
        suit_value = max(0.0, min(1.0, float(suitability_map[point.y, point.x])))
        
        # 每个洞穴占据的区域大小
        cave_size = rng.randint(3, 8) + int(suit_value * 5)
        cave_depth = rng.uniform(5, 15) * suit_value
        
        # 创建洞穴
        cave_points = _create_single_cave(result, point.x, point.y, cave_size, cave_depth, rng)
        
        # 只有当洞穴包含点时才添加
        if cave_points:
            caves.append(cave_points)
            valid_entrances.append(point)
    
    # 如果生成的洞穴太少，重试生成（降低要求）
    retry_count = 0
    while len(caves) < max(2, num_caves // 2) and retry_count < 3:
        retry_count += 1
        # 对未成功的入口点再次尝试
        for point in entrances:
            if point not in valid_entrances:
                # 降低条件尝试再生成
                cave_size = rng.randint(2, 5)
                cave_depth = rng.uniform(3, 8)
                cave_points = _create_single_cave(result, point.x, point.y, cave_size, cave_depth, rng)
                if cave_points:
                    caves.append(cave_points)
                    valid_entrances.append(point)
                    if len(caves) >= num_caves:
                        break
    
    # 使用有效入口进行连接
    for i in range(len(valid_entrances) - 1):
        # 连接洞穴，考虑高度和适宜性
        _connect_caves(result, suitability_map, valid_entrances[i], valid_entrances[i+1], rng)
    
    # 3. 添加额外的随机连接，形成网络
    if len(valid_entrances) > 3:
        for _ in range(min(len(valid_entrances) // 2, 3)):  # 增加连接数量
            idx1 = rng.randint(0, len(valid_entrances)-1)
            idx2 = rng.randint(0, len(valid_entrances)-1)
            if idx1 != idx2:
                _connect_caves(result, suitability_map, valid_entrances[idx1], valid_entrances[idx2], rng)
    
    return caves, valid_entrances  # 返回有效的洞穴和入口

def _create_single_cave(height_map, x, y, size, depth, rng=None):
    """在指定位置创建单个洞穴"""
    h, w = height_map.shape
    points = []
    
    # 使用随机椭圆而非圆形，更自然
    if rng is None:
        rng = random
    
    # 添加一些随机性到中心
    offset_x = rng.uniform(-1, 1)
    offset_y = rng.uniform(-1, 1)
    center_x, center_y = x + offset_x, y + offset_y
    
    # 随机椭圆参数
    a = size * rng.uniform(0.8, 1.2)  # x轴半径
    b = size * rng.uniform(0.8, 1.2)  # y轴半径
    angle = rng.uniform(0, np.pi)  # 旋转角度
    
    # 检测当前位置是否已经被侵蚀（避免在已有峡谷中重复挖掘）
    if height_map[int(y), int(x)] < 5:
        # 寻找附近更合适的位置
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and height_map[ny, nx] > 10:
                    center_x, center_y = nx, ny
                    break
    
    # 生成椭圆洞穴
    for j in range(max(0, int(y-size*1.2)), min(h, int(y+size*1.2+1))):
        for i in range(max(0, int(x-size*1.2)), min(w, int(x+size*1.2+1))):
            # 旋转后的椭圆距离计算
            dx = i - center_x
            dy = j - center_y
            rotated_x = dx * np.cos(angle) - dy * np.sin(angle)
            rotated_y = dx * np.sin(angle) + dy * np.cos(angle)
            
            # 椭圆内距离
            normalized_dist = np.sqrt((rotated_x/a)**2 + (rotated_y/b)**2)
            
            if normalized_dist <= 1:
                # 加入一些噪声使洞穴边缘不规则
                noise = rng.uniform(0.85, 1.15)
                
                # 距离中心越远，深度效应越小
                depth_effect = depth * (1 - normalized_dist * 0.7) * noise
                
                # 只将高于一定高度的点降低，避免水下洞穴
                if height_map[j, i] > 5:
                    # 避免过度侵蚀（检查是否已经被侵蚀过）
                    current_height = height_map[j, i]
                    new_height = max(current_height - depth_effect, 0)
                    
                    # 如果变化明显，记录这个点
                    if current_height - new_height > 1:
                        height_map[j, i] = new_height
                        points.append(Point(j, i))
    
    # 平滑洞穴边缘，但只在点列表非空时执行
    if points:
        affected_points = [(p.y, p.x) for p in points]
        height_map = _smooth_edges(height_map, affected_points, radius=2)
    else:
        # 如果没有点，创建一个小的默认洞穴
        if height_map[y, x] > 5:
            height_map[y, x] -= 3
            points.append(Point(y, x))  # 至少添加一个点
    
    return points

def _connect_caves(height_map, suitability_map, pos1, pos2, rng=None):
    """创建两个洞穴间的连接通道，考虑地质适宜性和高度"""
    if rng is None:
        rng = random
        
    h, w = height_map.shape
    
    # 计算连接路径的控制点
    dist = np.sqrt((pos2.x-pos1.x)**2 + (pos2.y-pos1.y)**2)
    num_points = max(5, int(dist / 3))
    
    # 生成基本路径点
    points = []
    for t in np.linspace(0, 1, num_points):
        # 线性插值基本路径
        x = int(pos1.x + (pos2.x - pos1.x) * t)
        y = int(pos1.y + (pos2.y - pos1.y) * t)
        
        # 添加随机扰动，使通道不那么直
        deviation = int(min(5, dist / 10))
        if 0.1 < t < 0.9 and deviation > 0:  # 避免扰动起点和终点
            x += rng.randint(-deviation, deviation)
            y += rng.randint(-deviation, deviation)
            
        # 确保点在地图范围内
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))
        
        points.append(Point(y, x))
    
    # 根据地质适宜性和高度调整路径
    for i in range(1, len(points) - 1):
        point = points[i]
        
        # 在3x3区域内寻找更适宜的点
        best_x, best_y = point.x, point.y
        
        # 综合评分：适宜性 * 0.6 + 高度倒数 * 0.4（高度越低越好）
        height_factor = 1.0 / (height_map[point.y, point.x] + 1)
        best_score = suitability_map[point.y, point.x] * 0.6 + height_factor * 0.4
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = point.y + dy, point.x + dx
                if 0 <= nx < w and 0 <= ny < h:
                    # 高度越低，越适合作为通道（更自然）
                    height_factor = 1.0 / (height_map[ny, nx] + 1)
                    score = suitability_map[ny, nx] * 0.6 + height_factor * 0.4
                    if score > best_score:
                        best_score = score
                        best_x, best_y = nx, ny
        
        # 更新为更适宜的点
        points[i] = Point(best_y, best_x)
    
    # 创建隧道 - 使用贝塞尔曲线平滑
    tunnel_points = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        
        # 添加中间点
        steps = max(3, int(np.sqrt((end.x-start.x)**2 + (end.y-start.y)**2)))
        for t in np.linspace(0, 1, steps):
            x = int(start.x + (end.x - start.x) * t)
            y = int(start.y + (end.y - start.y) * t)
            tunnel_points.append(Point(y, x))
    
    # 创建实际的隧道
    for point in tunnel_points:
        radius = rng.uniform(1.0, 2.5)  # 隧道宽度随机变化
        
        for j in range(max(0, int(point.y-radius)), min(h, int(point.y+radius+1))):
            for i in range(max(0, int(point.x-radius)), min(w, int(point.x+radius+1))):
                dist = np.sqrt((j-point.y)**2 + (i-point.x)**2)
                if dist <= radius:
                    # 计算深度效应
                    depth_factor = (1 - dist/radius) * rng.uniform(0.8, 1.2)
                    
                    # 检查避免过度侵蚀
                    if height_map[j, i] > 5:  # 避免水下隧道
                        # 更加平滑的深度过渡
                        reduction = 8 * depth_factor
                        height_map[j, i] -= reduction
                        height_map[j, i] = max(height_map[j, i], 1)  # 保留一点高度
    
    return tunnel_points

def _enhance_cave_details(height_map, caves, seed):
    """增强洞穴和峡谷的细节，添加钟乳石、水池等特征"""
    h, w = height_map.shape
    result = height_map.copy()
    
    # 创建独立的随机数生成器
    rng = random.Random(seed)
    
    # 合并所有洞穴点
    all_cave_points = []
    for cave in caves:
        all_cave_points.extend(cave)
    
    # 遍历所有洞穴点，随机添加细节
    for point in all_cave_points:
        if rng.random() < 0.05:  # 5%概率添加特殊特征
            feature_type = rng.choice(['stalagmite', 'pool', 'collapse'])
            x, y = point.x, point.y
            
            if feature_type == 'stalagmite' and result[y, x] > 5:
                # 添加小型钟乳石（高度微小增加）
                size = rng.randint(1, 2)
                height = rng.uniform(0.5, 1.5)  # 减小高度，避免过度突起
                for j in range(max(0, y-size), min(h, y+size+1)):
                    for i in range(max(0, x-size), min(w, x+size+1)):
                        dist = np.sqrt((j-y)**2 + (i-x)**2)
                        if dist <= size:
                            # 平滑过渡的钟乳石
                            effect = height * (1 - dist/size)**2  # 使用二次曲线平滑过渡
                            result[j, i] += effect
            
            elif feature_type == 'pool' and result[y, x] > 3:
                # 创建小水池（局部降低）
                size = rng.randint(1, 3)
                depth = rng.uniform(0.5, 1.5)  # 减小深度，避免过度凹陷
                for j in range(max(0, y-size), min(h, y+size+1)):
                    for i in range(max(0, x-size), min(w, x+size+1)):
                        dist = np.sqrt((j-y)**2 + (i-x)**2)
                        if dist <= size:
                            effect = depth * (1 - dist/size)
                            result[j, i] = max(1, result[j, i] - effect)
            
            elif feature_type == 'collapse':
                # 塌方区域（周围高度不规则变化）
                size = rng.randint(2, 4)
                for j in range(max(0, y-size), min(h, y+size+1)):
                    for i in range(max(0, x-size), min(w, x+size+1)):
                        dist = np.sqrt((j-y)**2 + (i-x)**2)
                        if dist <= size:
                            # 降低噪声幅度，使变化更自然
                            noise = (rng.random() - 0.5) * 1.5  # 减小随机变化范围
                            # 距离中心越远，变化越小
                            noise_scaled = noise * (1 - dist/size * 0.7)
                            result[j, i] += noise_scaled
                            result[j, i] = max(0, result[j, i])
    
    # 最终平滑处理，确保一切特征融合自然
    result = gaussian_filter(result, sigma=0.5)
    
    return result