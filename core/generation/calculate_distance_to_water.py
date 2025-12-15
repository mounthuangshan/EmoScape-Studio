#from __future__ import annotations
#标准库
import random
import math
import hashlib

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

#项目文件
from utils.tools import *

##################
#放置植被和建筑
##################
def calculate_distance_to_water(x, y, water_map, max_search_radius):
    """使用广度优先搜索计算到最近水域的距离"""
    if water_map is None or not isinstance(water_map, (list, np.ndarray)) or (isinstance(water_map, list) and len(water_map) == 0) or (isinstance(water_map, np.ndarray) and water_map.size == 0):
        return max_search_radius  # 无水域时返回最大值

    h = len(water_map)
    w = len(water_map[0]) if isinstance(water_map[0], (list, np.ndarray)) else 1
    
    # 起点超出边界检查
    if x < 0 or y < 0 or x >= w or y >= h:
        return max_search_radius
    
    # 如果起点就是水域，直接返回0
    if _is_water_cell(water_map[y][x]):
        return 0.0
    
    # 使用BFS搜索
    from collections import deque
    queue = deque([(x, y, 0)])  # (x, y, distance)
    visited = set([(x, y)])
    
    # 四/八方向移动
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while queue:
        cx, cy, dist = queue.popleft()
        
        # 超过最大搜索半径，返回最大值
        if dist > max_search_radius:
            return max_search_radius
        
        # 检查邻居
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            # 边界检查
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            
            # 避免重复检查
            if (nx, ny) in visited:
                continue
                
            visited.add((nx, ny))
            
            # 对角线移动距离为√2
            move_dist = 1.0 if abs(dx) + abs(dy) == 1 else 1.414
            
            # 找到水域
            if _is_water_cell(water_map[ny][nx]):
                return dist + move_dist
                
            # 继续搜索
            queue.append((nx, ny, dist + move_dist))
    
    return max_search_radius  # 找不到水域

def _is_water_cell(cell):
    if isinstance(cell, (list, np.ndarray)):
        return np.any(np.array(cell) > 0)  # 使用NumPy的any判断整个数组
    return bool(cell)

def apply_gaussian_blur(data, radius):
    """使用SciPy实现的高斯模糊，处理边界更平滑"""
    
    # 确保输入是NumPy数组
    arr = np.array(data, dtype=float)
    
    # 计算合适的sigma值，通常为半径/2.5左右
    sigma = radius / 2.5
    
    # 使用SciPy的高斯滤波，mode='reflect'可以更好地处理边界
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(arr, sigma=sigma, mode='reflect')
    
    # 根据需要返回列表或保持NumPy数组
    return blurred if isinstance(data, np.ndarray) else blurred.tolist()

def generate_fractal_noise_2d(noise_generator, x, y, scale, octaves, persistence, lacunarity):
    """多倍频噪声叠加，用于地形/植被分布"""
    amplitude = 1.0
    frequency = 1.0
    value = 0.0
    for _ in range(octaves):
        value += amplitude * noise_generator.noise2d(x * frequency * scale, y * frequency * scale)
        frequency *= lacunarity
        amplitude *= persistence
    return (value + 1.0) / 2.0  # 将值范围映射到[0, 1]

class SimplexNoise:
    """简单的占位Simplex噪声实现，可自行替换为专业库"""
    def __init__(self, seed=0):
        random.seed(seed)
        self.seed_val = seed
    
    def noise2d(self, x, y):
        # 占位实现，可根据需要替换
        random.seed(int(x*10000 + y*10000 + self.seed_val))
        return random.uniform(-1, 1)

def generate_city_size_distribution(city_count):
    """为城市生成规模因子分布"""

    sizes = []
    for _ in range(city_count):
        sizes.append(random.uniform(0.3, 1.0))
    sizes.sort(reverse=True)
    return sizes

def calculate_building_count(settlement_type, size):
    """计算建筑数量的简易方法"""
    base_count = {
        "city": 100,
        "town": 50,
        "village": 20,
        "hamlet": 10,
        "oasis": 15,
        "mining_camp": 15,
        "logging_camp": 15
    }.get(settlement_type, 10)
    return int(base_count * size)

def generate_city_center(cx, cy, size, width, height):
    buildings = []
    count = random.randint(1, 3)
    for _ in range(count):
        x_offset = random.randint(-2, 2)
        y_offset = random.randint(-2, 2)
        buildings.append((cx + x_offset, cy + y_offset, "city_hall", 0, 1.0, 5, 5))  # 添加尺寸
    return buildings

def generate_residential_area(cx, cy, size, building_count, width, height):
    """改进版居住区生成，增强碰撞检测"""
    buildings = []
    max_attempts = building_count * 3  # 增加尝试次数
    
    # 根据网格尺寸动态调整参数
    grid_size_factor = min(width, height) / 100  # 基准网格100x100的缩放因子
    max_building_size = max(3, int(6 * grid_size_factor))  # 最大建筑尺寸
    min_offset = 1  # 建筑物之间的最小间隔
    max_offset = int(15 * grid_size_factor)
    
    # 使用集合记录已占用的格子，提高碰撞检测效率
    occupied_cells = set()
    
    for _ in range(max_attempts):
        # 动态生成建筑尺寸，小型民居应是主要类型
        size_factor = random.random()
        if size_factor < 0.7:  # 70%几率是小房子
            building_w = random.randint(2, 4)
            building_h = random.randint(2, 4)
        elif size_factor < 0.9:  # 20%几率是中等房子
            building_w = random.randint(3, 5)
            building_h = random.randint(3, 5)
        else:  # 10%几率是大房子
            building_w = random.randint(4, max_building_size)
            building_h = random.randint(4, max_building_size)
        
        # 计算放置区域（向城市中心密集分布）
        dist_factor = random.random()
        if dist_factor < 0.5:  # 50%的建筑靠近中心
            max_dist = int(max_offset * 0.5)
        else:
            max_dist = max_offset
        
        # 随机选择方向和距离
        angle = random.uniform(0, 2 * np.pi)
        distance = random.randint(0, max_dist)
        
        # 计算目标位置
        x = int(cx + distance * np.cos(angle))
        y = int(cy + distance * np.sin(angle))
        
        # 边界检查
        if x < 0 or y < 0 or x + building_w >= width or y + building_h >= height:
            continue
            
        # 全面的碰撞检测
        collision = False
        for dy in range(-min_offset, building_h + min_offset):
            for dx in range(-min_offset, building_w + min_offset):
                if (x+dx, y+dy) in occupied_cells:
                    collision = True
                    break
            if collision:
                break
                
        if not collision:
            # 建筑物的随机朝向(0-3)和重要性(0.3-1.0)
            orientation = random.randint(0, 3)
            importance = random.uniform(0.3, 1.0)
            
            # 记录新建筑
            buildings.append((x, y, "house", orientation, importance, building_w, building_h))
            
            # 更新已占用格子集合
            for dy in range(building_h):
                for dx in range(building_w):
                    occupied_cells.add((x+dx, y+dy))
                    
            # 达到建筑数量后退出
            if len(buildings) >= building_count:
                break
    
    return buildings

def generate_commercial_area(cx, cy, size, building_count, width, height):
    buildings = []
    for _ in range(int(building_count)):
        x_offset = random.randint(-10, 10)
        y_offset = random.randint(-10, 10)
        buildings.append((
            cx + x_offset,
            cy + y_offset,
            "shop",
            random.randint(0,3),
            random.uniform(0.5,1.0),
            3,  # 默认宽度
            3    # 默认高度
        ))  # 新增尺寸参数
    return buildings

def generate_industrial_area(cx, cy, size, building_count, width, height):
    buildings = []
    for _ in range(int(building_count)):
        x_offset = random.randint(-8, 8)
        y_offset = random.randint(-8, 8)
        buildings.append((
            cx + x_offset,
            cy + y_offset,
            "factory",
            random.randint(0,3),
            random.uniform(0.4,1.0),
            4,  # 默认宽度
            4    # 默认高度
        ))  # 新增尺寸参数
    return buildings

def generate_town_center(cx, cy, size, width, height):
    return [(cx, cy, "town_center", 0, 1.0, 4, 4)]  # 添加尺寸

def generate_village_buildings(cx, cy, size, building_count, width, height):
    buildings = []
    for _ in range(int(building_count)):
        x_offset = random.randint(-5, 5)
        y_offset = random.randint(-5, 5)
        buildings.append((cx + x_offset, cy + y_offset, "village_house", 
                        random.randint(0,3), random.uniform(0.2,1.0), 3, 3))  # 添加尺寸
    return buildings

def generate_oasis_buildings(cx, cy, size, width, height):
    buildings = []
    # 生成水井、商队营地等
    buildings.append((cx, cy, "water_well", 0, 1.0, 2, 2))  # 添加尺寸
    for _ in range(random.randint(2,6)):
        x_offset = random.randint(-3,3)
        y_offset = random.randint(-3,3)
        buildings.append((cx + x_offset, cy + y_offset, "tent", 
                        random.randint(0,3), random.uniform(0.2,1.0), 2, 2))  # 添加尺寸
    return buildings

def generate_mining_buildings(cx, cy, size, width, height):
    """修复版矿业营地生成"""
    buildings = []
    
    # 添加主矿井入口
    buildings.append((cx, cy, "mine_entrance", 0, 1.0, 4, 4))
    
    # 添加周围的辅助建筑
    for i in range(random.randint(2, 6)):
        # 随机选择方位
        angle = random.uniform(0, 2 * np.pi)
        distance = random.randint(3, 6)
        
        x = cx + int(distance * np.cos(angle))
        y = cy + int(distance * np.sin(angle))
        
        # 随机决定建筑类型
        if i == 0:
            btype = "miners_lodge"
        elif i == 1:
            btype = "ore_storage"
        else:
            btype = "miners_hut"
            
        buildings.append((
            x, y, btype, 
            random.randint(0, 3),  # 朝向
            random.uniform(0.3, 1.0),  # 重要性
            3, 3  # 尺寸
        ))
    
    # 添加矿道和轨道
    for i in range(2):
        x = cx + random.randint(3, 5) * (1 if i==0 else -1) 
        y = cy + random.randint(-2, 2)
        
        buildings.append((
            x, y, 
            "mine_tracks" if i==0 else "ore_pile",
            0,  # 朝向
            0.6,  # 重要性
            2, 2  # 尺寸  
        ))
    
    return buildings

def generate_logging_buildings(cx, cy, size, width, height):
    buildings = []
    buildings.append((cx, cy, "lumber_camp", 0, 1.0, 4, 4))  # 添加尺寸
    for _ in range(random.randint(2,5)):
        x_offset = random.randint(-4,4)
        y_offset = random.randint(-4,4)
        buildings.append((cx + x_offset, cy + y_offset, "woodcutter_hut", 
                        random.randint(0,3), random.uniform(0.3,1.0), 3, 3))  # 添加尺寸
    return buildings

def generate_road_network(settlements, height_map, water_map, w, h):
    """使用A*寻路算法生成考虑地形的道路网络"""
    roads = []
    
    # 创建代价地图 - 水域不可通行，高度差增加代价
    cost_map = np.ones((h, w), dtype=float)
    
    # 设置水域为高代价（不可通行）
    for y in range(h):
        for x in range(w):
            if water_map is not None and x < water_map.shape[1] and y < water_map.shape[0]:
                if _is_water_cell(water_map[y][x]):
                    cost_map[y, x] = float('inf')  # 水域设为不可通过
    
    # 添加高度代价
    for y in range(h-1):
        for x in range(w-1):
            # 计算相邻格子高度差，增加爬坡代价
            if x+1 < w and y < h:
                height_diff = abs(height_map[y, x] - height_map[y, x+1])
                cost_map[y, x] += height_diff * 2
            if x < w and y+1 < h:
                height_diff = abs(height_map[y, x] - height_map[y+1, x])
                cost_map[y, x] += height_diff * 2
    
    # 定义简化版A*算法的启发式函数
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # 曼哈顿距离
    
    # A*寻路实现
    def a_star_path(start, goal):
        from heapq import heappush, heappop
        
        # 检查起点和终点是否有效
        if (not (0 <= start[0] < w and 0 <= start[1] < h) or 
            not (0 <= goal[0] < w and 0 <= goal[1] < h)):
            return []
            
        # 避开水域的起终点
        if (cost_map[start[1], start[0]] == float('inf') or 
            cost_map[goal[1], goal[0]] == float('inf')):
            return bresenham_line(start[0], start[1], goal[0], goal[1])
        
        # 定义移动方向（只使用四方向简化寻路）
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # 初始化A*数据结构
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        open_set_hash = {start}
        
        while open_set:
            current = heappop(open_set)[1]
            open_set_hash.remove(current)
            
            if current == goal:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查边界
                if not (0 <= neighbor[0] < w and 0 <= neighbor[1] < h):
                    continue
                
                # 计算新的g_score
                tentative_g_score = g_score[current] + cost_map[neighbor[1], neighbor[0]]
                
                if (neighbor not in g_score or 
                    tentative_g_score < g_score[neighbor]):
                    
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        # 如果找不到路径，返回直线路径
        return bresenham_line(start[0], start[1], goal[0], goal[1])
    
    # 生成所有聚居点之间的路径
    for i in range(len(settlements)):
        for j in range(i+1, len(settlements)):
            # 获取起点和终点
            sx, sy, *_ = settlements[i]
            tx, ty, *_ = settlements[j]
            start = (sx, sy)
            goal = (tx, ty)
            
            # 使用A*寻找路径
            path = a_star_path(start, goal)
            
            # 添加到道路列表
            for x, y in path:
                if 0 <= x < w and 0 <= y < h:
                    if not any(r[0] == x and r[1] == y for r in roads):
                        roads.append((x, y, "road", 0))
    
    return roads

def bresenham_line(x0, y0, x1, y1):
    """Bresenham算法，返回从(x0, y0)到(x1, y1)的直线网格点列表"""
    points = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def place_vegetation_and_buildings(biome_map, height_map, water_map, preferences, map_params):
    """放置植被和建筑物，基于生物群系特性、地形条件和城市规划原理
    
    Args:
        biome_map: 生物群系地图
        height_map: 高度图
        water_map: 水资源地图(河流/湖泊)
        preferences: 玩家偏好设置
        map_params: 地图参数
        
    Returns:
        tuple: (vegetation, buildings, roads, settlements)
    """
    #h = len(biome_map)
    #w = len(biome_map[0])
    h, w = biome_map.shape

    # === 新增代码：确保 water_map 有效 ===
    # 如果 water_map 为空或形状不匹配，重新创建
    if water_map is None or not isinstance(water_map, np.ndarray) or water_map.size == 0:
        water_map = np.zeros((h, w), dtype=np.bool_)
    elif water_map.shape != biome_map.shape:
        # 创建适合形状的新水域地图
        new_water_map = np.zeros((h, w), dtype=water_map.dtype)
        # 如果原来的水域地图有数据，尝试复制
        if isinstance(water_map, np.ndarray) and water_map.size > 0:
            min_h = min(h, water_map.shape[0])
            min_w = min(w, water_map.shape[1] if water_map.ndim > 1 else 1)
            if water_map.ndim > 1:
                new_water_map[:min_h, :min_w] = water_map[:min_h, :min_w]
            else:
                # 处理一维水域地图
                new_water_map[:, :min_w] = water_map[:min_w].reshape(1, -1)
        water_map = new_water_map
    
    # 初始化数据结构
    vegetation = []     # 格式: (x, y, type, size, age)
    buildings = []      # 格式: (x, y, type, orientation, importance)
    roads = []          # 格式: (x, y, type, direction)
    settlements = []    # 格式: (center_x, center_y, type, size, buildings_count)

    # 分析地形和环境
    flatness_map = calculate_terrain_flatness(height_map)

    resource_map = analyze_resources(biome_map, water_map, height_map)

    habitability_map = calculate_habitability(biome_map, height_map, water_map)

    # 智能确定主要聚居点位置
    city_count = map_params["city_count"]
    settlement_locations = determine_settlement_locations(
        habitability_map, 
        resource_map,
        water_map,
        city_count,
        min_distance=max(w, h) // (city_count * 2)  # 确保城市之间有足够距离
    )

    # 生成聚居点和城市
    city_size_distribution = generate_city_size_distribution(city_count)
    
    for i, (cx, cy) in enumerate(settlement_locations):
        settlement_type, size = determine_settlement_type(
            city_size_distribution[i], 
            biome_map[cy][cx], 
            habitability_map[cy][cx]
        )
        
        # 生成城市布局
        settlement_buildings = generate_settlement(
            cx, cy, 
            settlement_type, 
            size,
            biome_map,
            flatness_map, 
            w, h
        )
        
        buildings.extend(settlement_buildings)
        settlements.append((cx, cy, settlement_type, size, len(settlement_buildings)))

    # 生成连接主要聚居点的道路网络
    road_network = generate_road_network(settlements, height_map, water_map, w, h)
    roads.extend(road_network)

    # 基于生物群系和噪声函数放置植被
    vegetation = place_biome_specific_vegetation(
        biome_map, 
        buildings, 
        roads,
        map_params["vegetation_coverage"],
        preferences.get("vegetation_diversity", 0.5),
        w, h
    )

    # 添加特殊地标
    landmarks = generate_landmarks(
        biome_map, 
        height_map,
        settlements,
        preferences.get("landmark_frequency", 0.1),
        w, h
    )
    buildings.extend(landmarks)

    return vegetation, buildings, roads, settlements

def calculate_terrain_flatness(height_map):
    """计算地形平坦度，对建筑选址至关重要"""
    h, w = len(height_map), len(height_map[0])
    flatness = [[0.0 for _ in range(w)] for _ in range(h)]
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            # 计算当前点与周围8个点的最大高度差
            center_height = height_map[y][x]
            max_diff = 0
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    diff = abs(center_height - height_map[y+dy][x+dx])
                    max_diff = max(max_diff, diff)
            
            # 平坦度是高度差的反函数，越平坦值越高
            flatness[y][x] = 1.0 / (1.0 + max_diff * 5.0)
    
    return flatness

def analyze_resources(biome_map, water_map, height_map):
    """分析地图上的资源分布，影响聚居点选址"""
    #h, w = len(biome_map), len(biome_map[0])
    h, w = biome_map.shape
    resources = np.zeros((h, w), dtype=np.float32)  # 改用NumPy数组
    
    # === 新增代码：确保 water_map 有效 ===
    if water_map is None or not isinstance(water_map, np.ndarray) or water_map.size == 0:
        water_map = np.zeros((h, w), dtype=np.bool_)
    
    # 资源评估因子
    BIOME_RESOURCE_VALUE = {
        "Forest": 0.8,
        "Plains": 0.7,
        "Grassland": 0.6,
        "Savanna": 0.5,
        "Desert": 0.2,
        "Tundra": 0.3,
        "Taiga": 0.6,
        "Jungle": 0.9,
        "Mountain": 0.4,
        "Ocean": 0.0
    }
    
    # 计算基础资源值
    for y in range(h):
        for x in range(w):
            # 修复1: 使用正确索引语法 biome_map[y, x]
            biome_name = biome_map[y, x]  # 假设 biome_map 存储字符串
            resources[y, x] = BIOME_RESOURCE_VALUE.get(biome_name, 0.5)
            
            # 靠近水源的地方资源更丰富
            water_dist = calculate_distance_to_water(x, y, water_map, 10)
            resources[y, x] += max(0, (1.0 - water_dist / 10.0) * 0.5)
    
    # 使用卷积平滑资源图
    # 使用NumPy实现高斯模糊（替代原列表操作）
    from scipy.ndimage import gaussian_filter
    resources = gaussian_filter(resources, sigma=1)
    
    return resources

def calculate_habitability(biome_map, height_map, water_map):
    """计算适宜人类居住的程度"""
    #h, w = len(biome_map), len(biome_map[0])
    h, w = biome_map.shape
    habitability = np.zeros((h, w), dtype=np.float32)  # 改用NumPy数组   
    #habitability = [[0.0 for _ in range(w)] for _ in range(h)]
    
    # === 新增代码：确保 water_map 有效 ===
    if water_map is None or not isinstance(water_map, np.ndarray) or water_map.size == 0:
        water_map = np.zeros((h, w), dtype=np.bool_)
    
    # 生物群系宜居度基础值
    BIOME_HABITABILITY = {
        "Forest": 0.7,
        "Plains": 0.9,
        "Grassland": 0.8,
        "Savanna": 0.6,
        "Desert": 0.3,
        "Tundra": 0.4,
        "Taiga": 0.5,
        "Jungle": 0.6,
        "Mountain": 0.2,
        "Ocean": 0.0
    }
    
    # 计算宜居度
    for y in range(h):
        for x in range(w):
            # 修复1: 使用正确索引语法 biome_map[y, x]
            biome_name = biome_map[y, x]  # 假设 biome_map 存储字符串
            
            # 基础宜居度
            # 修复2: 直接使用字符串作为键
            habitability[y, x] = BIOME_HABITABILITY.get(biome_name, 0.5)
            
            # 极端高度修正（太高或太低都不适合居住）
            height = height_map[y, x]  # 正确索引
            if height > 80:  # 高山
                habitability[y, x] *= max(0.1, 1.0 - (height - 80) / 20.0)
            elif height < 20:  # 低洼地，可能洪涝
                habitability[y, x] *= max(0.3, height / 20.0)
            
            # 水源可达性（既不要太远也不要太近）
            water_dist = calculate_distance_to_water(x, y, water_map, 15)
            if 2 <= water_dist <= 8:
                habitability[y, x] *= 1.2  # 适当距离的水源加成
            elif water_dist < 2:
                habitability[y, x] *= 0.7  # 太近可能有洪水风险
            else:
                habitability[y, x] *= max(0.5, 1.0 - (water_dist - 8) / 15.0)  # 远离水源逐渐降低
    
    return habitability
            
def determine_settlement_locations(habitability_map, resource_map, water_map, count, min_distance):
    """智能确定聚居点位置，使用加权随机算法和最小距离约束"""
    h, w = len(habitability_map), len(habitability_map[0])
    
    # 计算综合适宜性评分
    suitability = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            # 权重可以根据实际需要调整
            suitability[y][x] = habitability_map[y][x] * 0.6 + resource_map[y][x] * 0.4
            
            # ==== 关键修复：处理 water_map 的多维数据 ====
            #获取当前水域单元格状态
            is_water = water_map[y, x]  # 直接获取布尔值
            
            # 处理多维单元格（如果有）
            if isinstance(is_water, np.ndarray):
                is_water = np.any(is_water)  # 多维数组时判断任意True
            
            if is_water:
                suitability[y][x] = 0
    
    # 使用加权随机选择算法
    locations = []
    attempts = 0
    max_attempts = count * 50  # 防止无限循环
    
    # 扁平化并创建候选位置
    candidates = []
    for y in range(h):
        for x in range(w):
            if suitability[y][x] > 0.4:  # 只考虑适宜性足够高的位置
                candidates.append((x, y, suitability[y][x]))
    
    # 根据适宜性排序
    candidates.sort(key=lambda c: c[2], reverse=True)
    
    # 保留前20%的候选点
    candidates = candidates[:max(count*3, int(len(candidates)*0.2))]
    
    while len(locations) < count and attempts < max_attempts:
        attempts += 1
        
        if not candidates:
            break
            
        # 随机选择一个候选点，优先选择适宜性高的
        weights = [c[2] for c in candidates]
        total_weight = sum(weights)
        
        if total_weight <= 0:
            break
            
        # 加权随机选择
        r = random.random() * total_weight
        cumulative = 0
        selected_idx = 0
        
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                selected_idx = i
                break
        
        x, y, _ = candidates[selected_idx]
        
        # 检查与已有位置的距离
        if all(((x-lx)**2 + (y-ly)**2)**0.5 > min_distance for lx, ly in locations):
            locations.append((x, y))
        
        # 从候选列表中移除已选点及其附近点
        candidates = [c for c in candidates if ((c[0]-x)**2 + (c[1]-y)**2)**0.5 > min_distance/2]
    
    return locations

def determine_settlement_type(size_factor, biome, habitability):
    """根据规模因子和环境条件确定聚居点类型"""
    if size_factor > 0.8:
        return "city", 4 + int(size_factor * 3)
    elif size_factor > 0.6:
        return "town", 3 + int(size_factor * 2)
    elif size_factor > 0.3:
        return "village", 2 + int(size_factor * 2)
    else:
        # 特殊生物群落的特殊聚居点
        biome_name = biome["name"]
        if biome_name == "Desert" and habitability > 0.5:
            return "oasis", 2
        elif biome_name == "Mountain" and habitability > 0.4:
            return "mining_camp", 2
        elif biome_name in ["Forest", "Taiga"] and habitability > 0.5:
            return "logging_camp", 2
        else:
            return "hamlet", 1 + int(size_factor * 2)

def generate_settlement(cx, cy, settlement_type, size, biome_map, flatness_map, width, height):
    """完整版聚居点生成函数，包含所有类型处理"""
    buildings = []
    building_count = calculate_building_count(settlement_type, size)
    
    # 基础参数配置
    base_radius = {
        "city": 25,
        "town": 20,
        "village": 15,
        "hamlet": 10,
        "oasis": 12,
        "mining_camp": 8,
        "logging_camp": 8
    }[settlement_type]
    
    # 根据地形平坦度调整中心位置
    best_flat_spot = _find_flat_spot(cx, cy, flatness_map, width, height)
    if best_flat_spot:
        cx, cy = best_flat_spot
    
    # 生成核心建筑
    if settlement_type == "city":
        # 城市：多中心结构
        main_center = generate_city_center(cx, cy, size, width, height)
        sub_centers = [(
            cx + random.randint(-base_radius, base_radius),
            cy + random.randint(-base_radius, base_radius),
            "district_center",
            0,
            0.8,
            5,  # 新增宽度
            5   # 新增高度
        ) for _ in range(2)]
        
        buildings.extend(main_center)
        buildings.extend(sub_centers)
        
        # 分区域生成
        # 修改循环解包逻辑
        for building in [main_center[0]] + sub_centers:
            center_x, center_y, *_ = building  # 使用通配符解包
            residential = generate_residential_area(
                center_x, center_y, 
                size, 
                int(building_count * 0.6), 
                width, height
            )
            commercial = generate_commercial_area(
                center_x, center_y,
                size,
                int(building_count * 0.25),
                width, height
            )
            industrial = generate_industrial_area(
                center_x + 10,  # 工业区偏移
                center_y + 10,
                size,
                int(building_count * 0.15),
                width, height
            )
            buildings.extend(residential)
            buildings.extend(commercial)
            buildings.extend(industrial)
            
    elif settlement_type == "town":
        # 城镇：单中心结构
        town_center = generate_town_center(cx, cy, size, width, height)
        buildings.extend(town_center)
        
        # 居住区和商业区
        residential = generate_residential_area(
            cx, cy,
            size,
            int(building_count * 0.7),
            width, height
        )
        commercial = generate_commercial_area(
            cx, cy,
            size,
            int(building_count * 0.3),
            width, height
        )
        buildings.extend(residential)
        buildings.extend(commercial)
        
    elif settlement_type == "village":
        # 村庄：紧凑型布局
        village_center = [(cx, cy, "village_square", 0, 1.0)]
        buildings.extend(village_center)
        
        houses = generate_village_buildings(
            cx, cy,
            size,
            building_count,
            width, height
        )
        buildings.extend(houses)
        
        # 添加公共设施
        if size > 1:
            public_building = [
                (cx + 5, cy + 5, "communal_well", 0, 0.7, 2, 2),  # 新增尺寸
                (cx - 5, cy - 5, "grain_storage", 0, 0.6, 3, 2)   # 新增尺寸
            ]
            buildings.extend(public_building)
            
    elif settlement_type == "hamlet":
        # 小村落：随机散布
        houses = generate_village_buildings(
            cx, cy,
            size,
            building_count,
            width, height
        )
        buildings.extend(houses)
        
        # 至少一个公共建筑
        if buildings:
            buildings.append((
                cx + random.randint(-5,5),
                cy + random.randint(-5,5),
                "shared_shed",
                0,
                0.5
            ))
            
    elif settlement_type == "oasis":
        # 绿洲：中心水井+帐篷
        oasis_buildings = generate_oasis_buildings(cx, cy, size, width, height)
        buildings.extend(oasis_buildings)
        
        # 添加棕榈树
        for _ in range(int(size * 3)):
            x = cx + random.randint(-8,8)
            y = cy + random.randint(-8,8)
            if 0 <= x < width and 0 <= y < height:
                buildings.append((x, y, "palm_tree", 0, 0.3, 1, 1))  # 新增尺寸
                
    elif settlement_type == "mining_camp":
        # 采矿营地：矿井+工棚
        buildings.append((
            cx + i*3,
            cy + random.randint(-2,2),
            "mine_entrance" if i==0 else "ore_pile",
            0,
            0.6,
            2, 2  # 新增尺寸
        ))
        
        # 添加矿洞和轨道
        for i in range(2):
            buildings.append((
                cx + i*3,
                cy + random.randint(-2,2),
                "mine_entrance" if i==0 else "ore_pile",
                0,
                0.6
            ))
            
    elif settlement_type == "logging_camp":
        # 伐木营地：木材堆+工棚
        logging_buildings = generate_logging_buildings(cx, cy, size, width, height)
        buildings.extend(logging_buildings)
        
        # 添加木材堆放区
        for angle in np.linspace(0, 2*np.pi, 6):
            x = cx + int(5 * np.cos(angle))
            y = cy + int(5 * np.sin(angle))
            buildings.append((x, y, "log_pile", 0, 0.4, 2, 2))
            
    # 边界保护和碰撞检测
    # 修改建筑生成后的数据结构处理部分
    # 修改碰撞检测逻辑增强兼容性
    valid_buildings = []
    occupied = set()
    for b in buildings:
        try:
            # 使用灵活解包方式
            x, y, btype, orient, importance, w, h = b
        except ValueError:
            # 自动补充默认尺寸
            x, y, btype, orient, importance = b[:5]
            w = b[5] if len(b)>5 else 3  # 默认宽度
            h = b[6] if len(b)>6 else 3  # 默认高度
            
        # 边界检查（需考虑建筑尺寸）
        if x < 0 or y < 0 or x + w > width or y + h > height:
            continue
            
        # 碰撞检测（基于建筑占地面积）
        collision = False
        for dx in range(w):
            for dy in range(h):
                if (x+dx, y+dy) in occupied:
                    collision = True
                    break
            if collision:
                break
                
        if not collision:
            valid_buildings.append( (x, y, btype, orient, importance, w, h) )
            # 标记整个建筑区域
            for dx in range(w):
                for dy in range(h):
                    occupied.add( (x+dx, y+dy) )
    
    return valid_buildings

def _find_flat_spot(cx, cy, flatness_map, w, h, search_radius=5):
    """在指定范围内寻找最平坦的位置"""
    best_score = 0
    best_pos = (cx, cy)
    
    for dx in range(-search_radius, search_radius+1):
        for dy in range(-search_radius, search_radius+1):
            x = cx + dx
            y = cy + dy
            if 0 <= x < w and 0 <= y < h:
                score = flatness_map[y][x]
                if score > best_score:
                    best_score = score
                    best_pos = (x, y)
    return best_pos

def place_biome_specific_vegetation(biome_map, buildings, roads, coverage, diversity, w, h):
    """修复版生物群系植被放置算法"""
    vegetation = []
    building_positions = {(x, y) for x, y, *_ in buildings}
    road_positions = {(x, y) for x, y, *_ in roads}
    
    # 使用 NumPy 向量化操作创建植被概率图
    # 创建随机种子
    seed = random.randint(1, 1000)
    noise_generator = SimplexNoise(seed=seed)
    
    # 创建噪声图
    noise_map = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            noise_map[y, x] = generate_fractal_noise_2d(
                noise_generator, x, y, 0.1, 4, 0.5, 2.0
            )
    
    # 植被类型定义
    BIOME_VEGETATION = {
        "Forest": [
            ("oak_tree", 0.5, 0.8), 
            ("pine_tree", 0.3, 0.7),
            ("bush", 0.15, 0.4),
            ("flowers", 0.05, 0.2)
        ],
        "Plains": [
            ("grass", 0.6, 0.9),
            ("bush", 0.3, 0.3), 
            ("tree", 0.1, 0.2)
        ],
        "Desert": [
            ("cactus", 0.5, 0.2),
            ("desert_shrub", 0.4, 0.15),
            ("dead_tree", 0.1, 0.05)
        ],
        "Jungle": [
            ("jungle_tree", 0.4, 0.9),
            ("bamboo", 0.3, 0.7),
            ("fern", 0.2, 0.8),
            ("exotic_flowers", 0.1, 0.5)
        ],
        "Savanna": [
            ("acacia_tree", 0.3, 0.4),
            ("tall_grass", 0.5, 0.8),
            ("shrub", 0.2, 0.5)
        ],
        "Tundra": [
            ("sparse_pine", 0.3, 0.3),
            ("tundra_grass", 0.5, 0.4),
            ("berry_bush", 0.2, 0.2)
        ],
        "Taiga": [
            ("spruce_tree", 0.6, 0.7),
            ("fir_tree", 0.3, 0.6),
            ("small_bush", 0.1, 0.3)
        ],
        "Mountain": [
            ("mountain_pine", 0.4, 0.3),
            ("rock_vegetation", 0.4, 0.2),
            ("alpine_flowers", 0.2, 0.2)
        ],
        "Grassland": [
            ("tall_grass", 0.6, 0.7),
            ("wild_flowers", 0.3, 0.5),
            ("small_tree", 0.1, 0.3)
        ],
        "Ocean": []  # 海洋没有植被
    }
    
    # 生物群系基础植被密度配置 (取值范围: 0.0-1.5，值越高植被越密集)
    BIOME_DENSITY = {
        # 森林类
        "Forest": 1.0,    # 温带森林，中等密度树木
        "Jungle": 1.3,    # 热带雨林，植被极其茂密
        "Taiga": 0.9,     # 针叶林，树木密集但种类单一
        
        # 平原类
        "Plains": 0.7,    # 开阔平原，以草地为主
        "Grassland": 0.8, # 高草草原，植被覆盖度高
        "Savanna": 0.6,   # 热带草原，树木稀疏但有灌木丛
        
        # 特殊地形
        "Mountain": 0.4,  # 高山地区，仅岩石缝隙有植被
        "Tundra": 0.3,    # 冻土苔原，只有低矮耐寒植物
        
        # 恶劣环境
        "Desert": 0.2,    # 沙漠，零星仙人掌类植物
        "Wasteland": 0.1, # 废土，几乎没有植被
        
        # 水域
        "Ocean": 0.0,     # 海洋，无陆生植被
        "Swamp": 1.1,     # 沼泽，潮湿环境下密集植被
        
        # 特殊生物群系（如有）
        "Mystic_Forest": 1.4,  # 魔法森林，超自然密集植被
        "Volcanic": 0.05       # 火山区域，仅有耐热菌类
    }

    # 使用说明：
    # 1. 密度值会参与计算公式：adjusted_threshold = (1 - coverage) / biome_density
    # 2. 密度>1.0的生物群系（如Jungle）会降低阈值，使植被更容易生成
    # 3. 特殊生物群系可根据游戏设定调整密度（如魔法区域设置超高密度）
    
    
    # 使用柏林噪声实现自然分布
    noise_scale = 0.1
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0
    
    noise_generator = SimplexNoise(seed=random.randint(1, 1000))
    
    # 使用双线性模拟植被分布，确保主要植被区块与单个植被的自然过渡
    # 逐个栅格决定是否放置植被
    for y in range(h):
        for x in range(w):
            # 跳过已有建筑或道路的位置
            if (x, y) in building_positions or (x, y) in road_positions:
                continue
            
            # 获取当前生物群系
            biome_name = biome_map[y, x]
            
            # 检查是否有适用的植被类型
            if biome_name not in BIOME_VEGETATION or not BIOME_VEGETATION[biome_name]:
                continue
            
            # 获取噪声值
            noise_value = noise_map[y, x]
            
            # 根据生物群系调整阈值
            biome_density = BIOME_DENSITY.get(biome_name, 1.0)
            density_threshold = (1.0 - coverage) / biome_density
            
            # 决定是否放置植被
            if noise_value > density_threshold:
                # 选择植被类型
                veg_options = BIOME_VEGETATION[biome_name]
                
                # 修复权重计算，正确处理多样性因子
                base_weights = [freq for _, freq, _ in veg_options]
                diversity_factors = [var for _, _, var in veg_options]
                veg_types = [veg_type for veg_type, _, _ in veg_options]
                
                # 根据多样性参数调整权重
                adjusted_weights = []
                for i, (base_w, div_factor) in enumerate(zip(base_weights, diversity_factors)):
                    # 多样性越高，权重差异越小；多样性越低，高频率植被权重越大
                    if diversity >= 0.5:  # 高多样性，增加稀有植物机会
                        adj = base_w * (1.0 + (diversity - 0.5) * 2 * (1.0 - base_w))
                    else:  # 低多样性，增强主要植被优势
                        adj = base_w * (1.0 + (0.5 - diversity) * 2 * base_w)
                    adjusted_weights.append(adj)
                
                # 确保权重归一化
                total = sum(adjusted_weights)
                if total > 0:
                    normalized_weights = [w/total for w in adjusted_weights]
                    
                    # 加权随机选择植被类型
                    veg_type = random.choices(veg_types, weights=normalized_weights, k=1)[0]
                    
                    # 添加大小、年龄变化
                    size = random.uniform(0.8, 1.2)
                    age = random.randint(1, 100)
                    
                    vegetation.append((x, y, veg_type, size, age))
    
    return vegetation

def generate_landmarks(biome_map, height_map, settlements, frequency, w, h):
    """生成特殊地标，如遗迹、神殿、自然奇观等"""
    landmarks = []
    
    # 输入校验
    if not isinstance(biome_map, np.ndarray) or biome_map.ndim != 2:
        raise ValueError("生物群系地图必须是二维NumPy数组")
    if not isinstance(height_map, np.ndarray) or height_map.ndim != 2:
        raise ValueError("高度图必须是二维NumPy数组")
    if biome_map.shape != height_map.shape:
        raise ValueError("生物群系地图与高度图尺寸不一致")
    
    # 使用通配符适配新的settlements数据结构
    settlement_positions = {(x, y) for x, y, *_ in settlements}  # 正确获取聚居点坐标  # 添加地标尺寸
    
    # 地标类型定义，按生物群系
    BIOME_LANDMARKS = {
        "Forest": ["ancient_tree", "druid_circle", "ranger_outpost"],
        "Mountain": ["mountain_temple", "dragon_cave", "dwarven_mine", "mountaintop_altar"],
        "Desert": ["pyramid", "sphinx", "desert_obelisk", "ancient_ruins"],
        "Plains": ["standing_stones", "burial_mound", "abandoned_farm"],
        "Jungle": ["overgrown_temple", "sacrificial_altar", "vine_covered_statue"],
        "Ocean": [],  # 海洋没有地标
        "Tundra": ["ice_sculpture", "frozen_tower", "mammoth_graveyard"],
        "Savanna": ["tribal_monument", "hunting_grounds", "pride_rock"],
        "Taiga": ["winter_cabin", "frozen_lake", "wolf_den"],
        "Grassland": ["ancient_battlefield", "nomad_camp", "wildflower_field"]
    }
    
    # 生成地标数量
    landmark_count = int(w * h * frequency / 1000)
    attempts = 0
    max_attempts = landmark_count * 10
    min_landmark_distance = max(w, h) // 10
    
    # 使用NumPy优化随机坐标生成
    candidate_x = np.random.randint(0, w, size=max_attempts)
    candidate_y = np.random.randint(0, h, size=max_attempts)
    
    for i in range(max_attempts):
        if len(landmarks) >= landmark_count:
            break
            
        x = candidate_x[i]
        y = candidate_y[i]
        
        # 检查距离聚居点
        if any(np.sqrt((x-sx)**2 + (y-sy)**2) < min_landmark_distance 
               for (sx, sy) in settlement_positions):
            continue
            
        # 修复1: 使用正确的二维数组索引语法
        biome_name = biome_map[y, x]  # 假设 biome_map 存储字符串类型
        if biome_name not in BIOME_LANDMARKS or not BIOME_LANDMARKS[biome_name]:
            continue
            
        # 修复2: 使用正确的二维数组索引语法获取高度
        height = height_map[y, x]
        
        # 地标类型验证（保持原逻辑）
        landmark_type = random.choice(BIOME_LANDMARKS[biome_name])
        valid_placement = True
        if "temple" in landmark_type and height < 50:
            valid_placement = False
        elif "cave" in landmark_type and height < 70:
            valid_placement = False
        elif "ruins" in landmark_type and height < 30:
            valid_placement = False
            
        if valid_placement:
            # 使用统一数据结构 (x, y, type, orientation, importance, width, height)
            landmarks.append((
                x, 
                y, 
                landmark_type, 
                random.randint(0, 3),   # orientation
                random.uniform(0.5, 1.0),  # importance
                5,  # width
                5   # height
            ))
    
    return landmarks