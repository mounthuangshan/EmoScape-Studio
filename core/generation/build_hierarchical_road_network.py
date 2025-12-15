#from __future__ import annotations
#标准库
import random
import math
import heapq
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
from collections import defaultdict

#项目文件
from utils.tools import *

################
#构建道路
################
################
# 构建现实地理感知路网系统
################

class GeographicPathfinder:
    """地理感知路径规划系统"""
    
    def __init__(self, height_map, water_map=None, biome_map=None, urban_density=None):
        """
        初始化路径规划器
        
        Args:
            height_map: 高度图数据
            water_map: 水域分布图(可选)
            biome_map: 生物群系图(可选)
            urban_density: 城市密度图(可选)
        """
        self.height_map = np.array(height_map)
        self.h, self.w = self.height_map.shape
        
        # 将额外地理数据转换为numpy数组
        self.water_map = np.array(water_map) if water_map is not None else np.zeros((self.h, self.w), dtype=bool)
        self.biome_map = biome_map
        self.urban_density = urban_density if urban_density is not None else np.zeros((self.h, self.w))
        
        # 计算地形复杂度(坡度)
        self.slope_map = self._calculate_slope_map()
        
        # 定义不同地形的基本通行成本
        self.terrain_cost = {
            'water': 100.0,      # 水域成本极高
            'mountain': 5.0,     # 山地成本高
            'forest': 1.5,       # 森林略有成本
            'urban': 0.8,        # 城市地区优先
            'plain': 1.0,        # 平原正常
            'base': 1.0,         # 基础成本
        }
        
        # 创建路网连通图
        # 初始化道路网络为布尔数组，道路类型为整数数组
        self.road_network = np.zeros_like(height_map, dtype=np.bool_)
        self.road_types = np.zeros_like(height_map, dtype=np.uint8)  # 0=无路, 1=主干道, 2=次干道, 3=小路
        
        # 路径搜索优化
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]  # 8方向移动
        self.dir_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]  # 对角线距离为√2
        
        # 空间哈希索引优化
        self.cell_size = 10
        self.spatial_index = self._build_spatial_index()

    def _calculate_slope_map(self):
        """计算每个单元格的坡度图，用于评估路径成本"""
        gradient_y, gradient_x = np.gradient(self.height_map)
        return np.sqrt(gradient_x**2 + gradient_y**2)

    def _build_spatial_index(self):
        """构建空间哈希索引，加速最近点查询"""
        index = defaultdict(list)
        for y in range(self.h):
            for x in range(self.w):
                cell_x, cell_y = x // self.cell_size, y // self.cell_size
                index[(cell_x, cell_y)].append((x, y))
        return index

    def _is_potential_bridge_location(self, p1, p2):
        """检测潜在的桥梁位置(窄水域)"""
        x1, y1 = p1
        x2, y2 = p2
        
        # 检查水域宽度
        if not self.water_map[y1, x1] and self.water_map[y2, x2]:
            # 从陆地到水域，检查前方水域宽度
            direction = (x2 - x1, y2 - y1)
            norm = max(abs(direction[0]), abs(direction[1]))
            dx, dy = direction[0]/norm, direction[1]/norm
            
            # 检查前方20个单位是否能到达对岸
            water_width = 0
            max_width = 15  # 可建桥的最大宽度
            
            for dist in range(1, 20):
                nx, ny = int(x2 + dx * dist), int(y2 + dy * dist)
                if not (0 <= nx < self.w and 0 <= ny < self.h):
                    break
                    
                if self.water_map[ny, nx]:
                    water_width += 1
                else:
                    # 找到对岸
                    return water_width <= max_width
                    
            return False
        return False

    def _get_movement_cost(self, x1, y1, x2, y2):
        """计算从(x1,y1)到(x2,y2)的移动成本，考虑多种地理因素"""
        # 计算直线距离
        dx, dy = abs(x2-x1), abs(y2-y1)
        base_cost = math.sqrt(dx*dx + dy*dy) if (dx and dy) else (dx + dy)
        
        # 高度差异惩罚（降低惩罚系数）
        height_diff = abs(self.height_map[y2, x2] - self.height_map[y1, x1])
        slope_cost = (height_diff / base_cost)**2 * 1.0 if base_cost > 0 else 0  # 从2.0降至1.0
        
        # 改进水域处理，降低水域惩罚以便必要时能够穿过
        water_cost = 0
        if self.water_map[y2, x2]:
            # 检查是否是窄水域(可建桥点)
            is_narrow_crossing = False
            if self._is_potential_bridge_location((x1, y1), (x2, y2)):
                water_cost = 2.0  # 穿越窄水域的成本降低 (从3.0降至2.0)
                is_narrow_crossing = True
            else:
                water_cost = 10.0  # 降低宽水域的成本(从20.0降至10.0)
        
        # 综合成本计算 - 降低各种因素权重
        total_cost = base_cost * (1.0 + slope_cost * 0.7) + water_cost
        
        # 避免过于陡峭的坡度（提高容忍度）
        if height_diff / max(1.0, base_cost) > 0.7:  # 从0.5提高到0.7
            total_cost *= 5.0  # 从10.0降至5.0
            
        return total_cost

    def find_path_with_fallback(self, start, goal):
        """带回退机制的路径寻找，如果A*失败则使用直线路径"""
        try:
            # 尝试使用A*寻路
            path = self.find_path(start, goal, max_iterations=5000)  # 限制最大迭代次数避免卡死
            if path:
                return path
            
            # A*寻路失败，使用直线路径
            print(f"[INFO] A*寻路失败，尝试使用直线路径 {start} → {goal}")
            sx, sy = int(round(start[0])), int(round(start[1]))
            gx, gy = int(round(goal[0])), int(round(goal[1]))
            
            # 创建直线路径
            line_path = []
            # 使用Bresenham算法生成直线路径
            dx, dy = abs(gx - sx), abs(gy - sy)
            sx, sy = int(sx), int(sy)
            gx, gy = int(gx), int(gy)
            
            sx_step = 1 if sx < gx else -1
            sy_step = 1 if sy < gy else -1
            err = dx - dy
            
            while sx != gx or sy != gy:
                if 0 <= sx < self.w and 0 <= sy < self.h and not self.water_map[sy, sx]:
                    line_path.append((sx, sy))
                
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    sx += sx_step
                if e2 < dx:
                    err += dx
                    sy += sy_step
            
            # 添加目标点
            if 0 <= gx < self.w and 0 <= gy < self.h and not self.water_map[gy, gx]:
                line_path.append((gx, gy))
            
            if len(line_path) > 1:
                print(f"[INFO] 使用直线路径作为替代，长度: {len(line_path)}")
                return line_path
            return None
        except Exception as e:
            # 捕获任何异常，确保道路生成不会完全失败
            print(f"[ERROR] 路径寻找出错: {e}")
            return None

    def find_path(self, start, goal, max_iterations=100000):
        """基于地理感知的A*寻路算法，考虑地形和道路自然性"""
        # 强制坐标转换为整数
        sx, sy = int(round(start[0])), int(round(start[1]))
        gx, gy = int(round(goal[0])), int(round(goal[1]))

        print(f"[DEBUG] find_path start → goal: { (sx,sy) } → { (gx,gy) }")
        
        # 边界检查（确保转换后坐标有效）
        if not (0 <= sx < self.w and 0 <= sy < self.h and 0 <= gx < self.w and 0 <= gy < self.h):
            print("[DEBUG] find_path 越界，直接返回 None")
            return None
        
        # 检查起点和终点是否在水域
        if self.water_map[sy, sx]:
            print(f"[DEBUG] 起点 ({sx},{sy}) 在水域中，尝试找到附近陆地点")
            # 尝试在附近找一个非水域点作为起点
            # 扩大搜索范围至10
            for radius in range(1, 10):
                found = False
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        if dx*dx + dy*dy > radius*radius:
                            continue  # 圆形搜索
                        nx, ny = sx + dx, sy + dy
                        if 0 <= nx < self.w and 0 <= ny < self.h and not self.water_map[ny, nx]:
                            sx, sy = nx, ny
                            print(f"[DEBUG] 使用新起点 ({sx},{sy})")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        if self.water_map[gy, gx]:
            print(f"[DEBUG] 终点 ({gx},{gy}) 在水域中，尝试找到附近陆地点")
            # 扩大搜索范围至10
            for radius in range(1, 10):
                found = False
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        if dx*dx + dy*dy > radius*radius:
                            continue  # 圆形搜索
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.w and 0 <= ny < self.h and not self.water_map[ny, nx]:
                            gx, gy = nx, ny
                            print(f"[DEBUG] 使用新终点 ({gx},{gy})")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        # 检查是否仍在水域，如果是则放弃寻路
        if self.water_map[sy, sx] or self.water_map[gy, gx]:
            print(f"[DEBUG] 无法找到合适的非水域起终点，寻路失败")
            return None
        
        # 启发式函数（添加坐标校验）
        def heuristic(x, y):
            # 强制转换为整数
            x, y = int(x), int(y)
            if not (0 <= x < self.w and 0 <= y < self.h):
                return float('inf')  # 无效坐标返回极大值
            
            # 计算高度差
            try:
                current_height = self.height_map[y, x]
                goal_height = self.height_map[gy, gx]
            except IndexError:
                return float('inf')
            
            dx, dy = abs(x - gx), abs(y - gy)
            direct_dist = math.sqrt(dx**2 + dy**2)
            height_diff = abs(current_height - goal_height)
            # 降低高度差的影响
            return direct_dist * (1.0 + min(0.5, height_diff / max(20.0, direct_dist * 0.1)))
        
        # A*算法核心（修改邻居生成逻辑）
        open_set = []
        heapq.heappush(open_set, (0, (sx, sy)))
        closed_set = set()
        came_from = {}
        g_score = {(sx, sy): 0}
        f_score = {(sx, sy): heuristic(sx, sy)}
        
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            cx, cy = current
            
            if iterations % 1000 == 0:
                print(f"[DEBUG] 寻路迭代中: {iterations}次，open_set大小={len(open_set)}")
            
            if current == (gx, gy):
                path = self._reconstruct_path(came_from, current)
                smoothed_path = self._smooth_path(path)
                print(f"[DEBUG] 成功找到路径! 迭代次数={iterations}, 路径长度={len(path)}")
                return smoothed_path
            
            closed_set.add(current)
            
            # 使用全方向搜索，优先级更灵活
            for i, (dx, dy) in enumerate(self.directions):
                nx = int(cx + dx)
                ny = int(cy + dy)
                neighbor = (nx, ny)
                
                # 严格边界检查
                if not (0 <= nx < self.w and 0 <= ny < self.h):
                    continue
                if neighbor in closed_set:
                    continue
                
                # 计算移动成本（添加异常处理）
                try:
                    move_cost = self._get_movement_cost(cx, cy, nx, ny)
                    if move_cost > 1000:  # 设置一个合理的成本上限
                        continue  # 跳过成本过高的点
                except Exception as e:
                    print(f"[ERROR] 计算移动成本出错: {e}")
                    continue
                    
                tentative_g = g_score[current] + move_cost * self.dir_costs[i]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(nx, ny)
                    f_score[neighbor] = f
                    
                    if neighbor not in (item[1] for item in open_set):
                        heapq.heappush(open_set, (f, neighbor))

        # 到这里说明没找到路径
        print(f"[DEBUG] find_path 失败: {(sx,sy)} → {(gx,gy)}, 遍历次数={iterations}, open_set大小={len(open_set)}")
        
        # 如果迭代次数为1，表示首次迭代就失败了
        if iterations <= 1:
            print("[CRITICAL] A*算法在第一次迭代就失败，检查邻居生成和成本计算")
        
        return None
    
    def _reconstruct_path(self, came_from, current):
        """重建从起点到终点的路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _smooth_path(self, path, smoothing_factor=0.3):
        """改进的路径平滑算法，确保平滑后的路径可行性"""
        if len(path) <= 2:
            return path
                    
        smoothed = path.copy()
        
        # 使用Chaikin平滑，但增加地形验证
        for _ in range(2):
            new_path = [smoothed[0]]  # 保持起点不变
            
            for i in range(len(smoothed) - 1):
                p0x, p0y = smoothed[i]
                p1x, p1y = smoothed[i+1]
                
                # 计算平滑点
                q_x = p0x + (p1x - p0x) * 0.25
                q_y = p0y + (p1y - p0y) * 0.25
                r_x = p0x + (p1x - p0x) * 0.75
                r_y = p0y + (p1y - p0y) * 0.75
                
                # 边界检查和取整
                q_x = max(0, min(self.w - 1, int(round(q_x))))
                q_y = max(0, min(self.h - 1, int(round(q_y))))
                r_x = max(0, min(self.w - 1, int(round(r_x))))
                r_y = max(0, min(self.h - 1, int(round(r_y))))
                
                # 关键改进：验证平滑点地形可行性
                # 检查q点可行性
                q_valid = not self.water_map[q_y, q_x] and self._get_movement_cost(p0x, p0y, q_x, q_y) < 10.0
                
                # 检查r点可行性
                r_valid = not self.water_map[r_y, r_x] and self._get_movement_cost(q_x, q_y, r_x, r_y) < 10.0
                
                # 只添加有效的平滑点
                if q_valid and new_path[-1] != (q_x, q_y):
                    new_path.append((q_x, q_y))
                
                if r_valid:
                    new_path.append((r_x, r_y))
                elif new_path[-1] != (p1x, p1y):
                    # 如果r点无效，保留原始终点
                    new_path.append((p1x, p1y))
            
            # 确保终点不变
            if new_path[-1] != smoothed[-1]:
                new_path.append(smoothed[-1])
            
            smoothed = new_path
        
        # 优化冗余点移除，保留关键地形转折点
        if len(smoothed) > 3:
            i = 1
            while i < len(smoothed) - 1:
                prev_x, prev_y = smoothed[i-1]
                curr_x, curr_y = smoothed[i]
                next_x, next_y = smoothed[i+1]
                
                # 计算方向变化
                dir1 = math.atan2(curr_y - prev_y, curr_x - prev_x)
                dir2 = math.atan2(next_y - curr_y, next_x - curr_x)
                angle_change = abs(dir1 - dir2)
                if angle_change > math.pi:
                    angle_change = 2 * math.pi - angle_change
                
                # 计算当前点与直线替代路径的地形差异
                terrain_diff = abs(self._get_movement_cost(prev_x, prev_y, next_x, next_y) - 
                                (self._get_movement_cost(prev_x, prev_y, curr_x, curr_y) + 
                                self._get_movement_cost(curr_x, curr_y, next_x, next_y)))
                
                # 如果点非常接近且方向变化小，且地形差异小，则可移除
                is_close = abs(curr_x - prev_x) <= 1 and abs(curr_y - prev_y) <= 1
                if is_close and angle_change < 0.3 and terrain_diff < 2.0:
                    smoothed.pop(i)
                else:
                    i += 1
        
        # 添加生态群系/地形边界检测，增强道路转弯的自然性
        def is_terrain_boundary(x, y):
            """检测是否位于地形边界或特殊地形特征点"""
            if not (0 <= x < self.w-1 and 0 <= y < self.h-1):
                return False
                
            # 检查高度变化率
            height_diff_x = abs(self.height_map[y, x+1] - self.height_map[y, x])
            height_diff_y = abs(self.height_map[y+1, x] - self.height_map[y, x])
            
            # 高度变化率大说明是地形过渡点
            if height_diff_x > 0.1 or height_diff_y > 0.1:
                return True
                
            # 检查是否是生态群系边界
            if self.biome_map is not None:
                current_biome = self.biome_map[y][x]["name"]
                if (x < self.w-1 and self.biome_map[y][x+1]["name"] != current_biome or
                    y < self.h-1 and self.biome_map[y+1][x]["name"] != current_biome):
                    return True
                    
            return False
        
        # 在平滑后，增加额外的自然扰动点
        if len(smoothed) > 5:
            # 找到潜在的扰动点(地形变化处)
            potential_disruption = []
            for i, (x, y) in enumerate(smoothed[1:-1], 1):
                if is_terrain_boundary(x, y):
                    potential_disruption.append(i)
            
            # 在地形边界处增加小扰动，使道路更自然地适应地形
            for idx in potential_disruption:
                x, y = smoothed[idx]
                # 在边界点周围小范围(±1)随机扰动
                offset_x = random.randint(-1, 1)
                offset_y = random.randint(-1, 1)
                nx, ny = x + offset_x, y + offset_y
                
                # 确保坐标有效且不在水域
                if (0 <= nx < self.w and 0 <= ny < self.h and not self.water_map[ny, nx]):
                    smoothed[idx] = (nx, ny)
        
        return smoothed
    
    def update_road_network(self, path, road_type=1):
        """将新路径添加到道路网络"""
        if not path:
            print("尝试更新空 path，road_type=", road_type)
            return
        print("add path len=", len(path), " type=", road_type)        
        for x, y in path:
            self.road_network[y, x] = True
            # 如果现有道路类型值较大(次要道路)，保留新的较小值(主要道路)
            if road_type < self.road_types[y, x] or self.road_types[y, x] == 0:
                self.road_types[y, x] = road_type
    
    def find_nearest_road(self, point, max_range=30):
        """找到距离给定点最近的现有道路点"""
        px, py = point
        best_dist = float('inf')
        best_point = None
        
        # 使用空间索引加速搜索
        cell_x, cell_y = px // self.cell_size, py // self.cell_size
        search_cells = []
        search_radius = max_range // self.cell_size + 1
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                search_cells.append((cell_x + dx, cell_y + dy))
        
        for cell in search_cells:
            if cell not in self.spatial_index:
                continue
                
            for x, y in self.spatial_index[cell]:
                if self.road_network[y, x]:
                    dist = math.sqrt((x - px)**2 + (y - py)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_point = (x, y)
        
        return best_point if best_dist <= max_range else None

def build_hierarchical_road_network(height_map, settlements, buildings, water_map=None, biome_map=None):
    """构建层次化道路网络，包含主干道、次干道和小路
    
    Args:
        height_map: 高度图
        settlements: 聚居点列表[(x, y, type, size, building_count), ...]
        buildings: 建筑物列表[(x, y, type, orientation, importance), ...]
        water_map: 水域地图(可选)
        biome_map: 生物群系地图(可选)
        
    Returns:
        道路网络(二维布尔数组)和道路类型(二维整数数组)
    """
    print("Settlements:", settlements)
    
    # 增加数据验证
    print(f"[DEBUG] 高度图形状: {height_map.shape}, 范围: {height_map.min():.2f}-{height_map.max():.2f}")
    
    if water_map is not None:
        water_pct = np.mean(water_map) * 100
        print(f"[DEBUG] 水域图形状: {water_map.shape}, 水域覆盖率: {water_pct:.2f}%")
        if water_pct > 80:
            print("[警告] 水域覆盖率过高，可能影响道路生成")
    
    # 1. 输入校验
    if not isinstance(height_map, np.ndarray) or height_map.ndim != 2:
        raise ValueError("高度图必须是二维NumPy数组")
    #if water_map is not None and not isinstance(water_map, np.ndarray):
        #raise ValueError("水域地图必须是NumPy数组")
    if biome_map is not None and not isinstance(biome_map, np.ndarray):
        raise ValueError("生物群系地图必须是NumPy数组")

    # 1. 初始化路径规划器
    pathfinder = GeographicPathfinder(height_map, water_map, biome_map)
    
    # 优化城市密度计算为向量化操作
    h, w = height_map.shape
    urban_density = np.zeros((h, w), dtype=np.float32)
    
    # 预处理建筑信息，减少循环中的重复计算
    building_points = []
    importance_values = []
    radii = []
    
    for bx, by, _, _, importance in buildings:
        bx, by = int(bx), int(by)
        if 0 <= bx < w and 0 <= by < h:
            radius = int(max(3, importance * 10))
            building_points.append((bx, by))
            importance_values.append(importance)
            radii.append(radius)
    
    # 创建辅助数组避免二重循环
    if building_points:
        # 创建一个临时密度图
        for i, (bx, by) in enumerate(building_points):
            radius = radii[i]
            importance = importance_values[i]
            
            # 计算影响区域
            y_start = max(0, by - radius)
            y_end = min(h, by + radius + 1)
            x_start = max(0, bx - radius)
            x_end = min(w, bx + radius + 1)
            
            # 构建距离矩阵
            y_coords, x_coords = np.ogrid[y_start:y_end, x_start:x_end]
            dist_squared = (x_coords - bx)**2 + (y_coords - by)**2
            
            # 创建影响掩码
            mask = dist_squared <= radius**2
            influence = np.zeros((y_end-y_start, x_end-x_start), dtype=np.float32)
            influence[mask] = (1 - np.sqrt(dist_squared[mask])/radius) * importance
            
            # 累加到密度图
            urban_density[y_start:y_end, x_start:x_end] += influence
    
    # 平滑处理
    urban_density = gaussian_filter(urban_density, sigma=3.0)
    pathfinder.urban_density = urban_density
    
    # 3. 根据重要性排序聚居点
    # 5. 聚居点处理（保持原有逻辑，修复索引方式）
    important_settlements = []
    for i, (x, y, stype, size, count) in enumerate(settlements):
        importance = size * count * 0.01
        if stype in ["city", "town"]:
            importance *= 2.0
        important_settlements.append((x, y, importance, i))
   
    # 按重要性降序排序
    important_settlements.sort(key=lambda s: s[2], reverse=True)
   
    # 4. 构建主干道网络（连接主要城市）
    main_settlements = important_settlements[:min(len(important_settlements), 5)]
   
    # 使用最小生成树算法连接主要聚居点
    mst_edges = calculate_mst_edges(main_settlements, height_map)

    print("MST edges:", mst_edges)
   
    # 构建主干道
    for edge in mst_edges:
        from_idx, to_idx, _ = edge
        from_x, from_y = main_settlements[from_idx][0], main_settlements[from_idx][1]
        to_x, to_y = main_settlements[to_idx][0], main_settlements[to_idx][1]

        # A*寻路
        path = pathfinder.find_path_with_fallback((from_x, from_y), (to_x, to_y))

        if path:
            pathfinder.update_road_network(path, road_type=1)  # 主干道
   
    # 5. 连接次要聚居点到最近的主干道
    secondary_settlements = important_settlements[len(main_settlements):]
   
    for x, y, importance, idx in secondary_settlements:
        # 初始化nearest_road变量
        nearest_road = None
        
        # 使用NumPy加速最近道路搜索
        if pathfinder.road_network.any():
            y_coords, x_coords = np.where(pathfinder.road_network)
            if len(x_coords) > 0:
                distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
                nearest_idx = np.argmin(distances)
                nearest_road = (x_coords[nearest_idx], y_coords[nearest_idx])
        
        if nearest_road:
            # 连接到现有道路
            path = pathfinder.find_path_with_fallback((x, y), nearest_road)
            if path:
                # 根据重要性决定道路类型
                road_type = 2 if importance > 0.5 else 3
                pathfinder.update_road_network(path, road_type)
        else:
            # 如果找不到附近的道路，连接到最近的主要聚居点
            best_dist = float('inf')
            best_target = None
            
            for tx, ty, _, _ in main_settlements:
                dist = math.sqrt((x - tx)**2 + (y - ty)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_target = (tx, ty)
            
            if best_target:
                path = pathfinder.find_path_with_fallback((x, y), best_target)
                if path:
                    pathfinder.update_road_network(path, road_type=2)  # 次干道
    
    # 2. 改进重要建筑物连接 - 根据重要性分配道路类型
    important_buildings = []
    for i, (x, y, type, _, importance) in enumerate(buildings):
        if importance > 0.7 or type in ["city_hall", "town_center", "mine_entrance", "water_well"]:
            important_buildings.append((x, y, importance))
    
    # 随机选择一部分重要建筑物进行道路连接
    random.shuffle(important_buildings)
    selected_buildings = important_buildings[:min(len(important_buildings), 30)]
    
    for x, y, importance in selected_buildings:
        nearest_road = pathfinder.find_nearest_road((x, y), max_range=20)
        if nearest_road:
            path = pathfinder.find_path_with_fallback((x, y), nearest_road)
            if path:
                # 根据建筑重要性决定道路类型
                if importance > 0.9:
                    road_type = 2  # 非常重要的建筑用次干道
                else:
                    road_type = 3  # 普通重要建筑用小路
                pathfinder.update_road_network(path, road_type)
    
    # 7. 完善城区内部道路网格
    for x, y, type, size, _ in settlements:
        if type in ["city", "town"] and size > 3:
            generate_urban_road_grid(pathfinder, (x, y), size)
    
    # 8. 自然化道路，增加曲折性
    naturalize_road_network(pathfinder)
    
    return pathfinder.road_network, pathfinder.road_types

def calculate_mst_edges(points, height_map):
    """计算连接点的最小生成树边集
    
    Args:
        points: [(x, y, importance, idx), ...]
        height_map: 高度图，用于计算距离代价
        
    Returns:
        [(from_idx, to_idx, weight), ...]
    """
    if len(points) <= 1:
        return []
        
    # 计算所有点对之间的距离
    edges = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            x1, y1 = points[i][0], points[i][1]
            x2, y2 = points[j][0], points[j][1]
            
            # 直线距离
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # 高度差异修正
            height_diff = abs(height_map[y2][x2] - height_map[y1][x1])
            adjusted_dist = dist * (1 + height_diff * 0.01)
            
            edges.append((i, j, adjusted_dist))
    
    # 按距离排序
    edges.sort(key=lambda e: e[2])
    
    # Kruskal算法构建最小生成树
    parent = list(range(len(points)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    mst = []
    for u, v, w in edges:
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            mst.append((u, v, w))
            parent[root_v] = root_u
            # 如果已经有足够的边，提前退出
            if len(mst) >= len(points) - 1:
                break
    
    return mst

def generate_urban_road_grid(pathfinder, center, size):
    """改进的城市区域网格状道路生成，优化道路间距和地形适应性
    
    Args:
        pathfinder: GeographicPathfinder实例
        center: 城市中心坐标(x, y)
        size: 城市规模
    """
    cx, cy = center
    
    # 动态调整网格尺寸和间距
    if size > 10:
        grid_size = int(size * 2.5)
        spacing = max(3, min(8, int(size / 2)))
    else:
        grid_size = int(size * 3)
        spacing = max(2, min(5, int(size)))
    
    # 减小主次干道间隔
    main_road_interval = max(2, spacing * 2)
    
    # 新增：分析城市区域地形，确定主要道路方向
    # 计算城市区域的地形主导方向(沿等高线)
    elevation_samples = []
    samples_radius = min(grid_size, 15)  # 采样范围
    
    for dx in range(-samples_radius, samples_radius+1, 2):
        for dy in range(-samples_radius, samples_radius+1, 2):
            x, y = cx + dx, cy + dy
            if (0 <= x < pathfinder.w and 0 <= y < pathfinder.h):
                elevation_samples.append((x, y, pathfinder.height_map[y, x]))
    
    # 对高度进行排序分析
    elevation_samples.sort(key=lambda s: s[2])
    
    # 检查坡度主方向
    if len(elevation_samples) > 4:
        low_points = elevation_samples[:len(elevation_samples)//4]
        high_points = elevation_samples[-len(elevation_samples)//4:]
        
        # 计算坡度方向(高到低)
        if low_points and high_points:
            low_center_x = sum(p[0] for p in low_points) / len(low_points)
            low_center_y = sum(p[1] for p in low_points) / len(low_points)
            high_center_x = sum(p[0] for p in high_points) / len(high_points)
            high_center_y = sum(p[1] for p in high_points) / len(high_points)
            
            # 坡度方向向量
            slope_direction = (low_center_x - high_center_x, low_center_y - high_center_y)
            
            # 坡度显著时，调整网格方向与坡度垂直和平行
            slope_magnitude = math.sqrt(slope_direction[0]**2 + slope_direction[1]**2)
            if slope_magnitude > grid_size * 0.2:  # 坡度足够显著
                # 单位化方向
                slope_direction = (slope_direction[0]/slope_magnitude, slope_direction[1]/slope_magnitude)
                # 计算垂直方向（主要道路沿等高线）
                perp_direction = (-slope_direction[1], slope_direction[0])
                
                # 使用地形适应的道路方向替代直角网格
                road_plans = []
                
                # 沿等高线(垂直于坡度)的主要道路
                for i in range(-grid_size, grid_size+1, main_road_interval):
                    # 计算偏移基点
                    offset = i * perp_direction[1]  # 沿垂直方向偏移
                    start_x = int(cx - grid_size * perp_direction[0] + i * slope_direction[0])
                    start_y = int(cy - grid_size * perp_direction[1] + i * slope_direction[1])
                    end_x = int(cx + grid_size * perp_direction[0] + i * slope_direction[0])
                    end_y = int(cy + grid_size * perp_direction[1] + i * slope_direction[1])
                    
                    road_plans.append({
                        'start': (start_x, start_y),
                        'end': (end_x, end_y),
                        'type': 2  # 次干道
                    })
                
                # 沿坡度方向的次要道路，间距更大
                for i in range(-grid_size, grid_size+1, spacing*2):
                    # 计算偏移基点
                    start_x = int(cx - grid_size * slope_direction[0] + i * perp_direction[0])
                    start_y = int(cy - grid_size * slope_direction[1] + i * perp_direction[1])
                    end_x = int(cx + grid_size * slope_direction[0] + i * perp_direction[0])
                    end_y = int(cy + grid_size * slope_direction[1] + i * perp_direction[1])
                    
                    road_plans.append({
                        'start': (start_x, start_y),
                        'end': (end_x, end_y),
                        'type': 3  # 小路
                    })
                
                # 按道路类型排序
                road_plans.sort(key=lambda x: x['type'])
                
                # 执行道路规划
                for plan in road_plans:
                    start_x, start_y = plan['start']
                    end_x, end_y = plan['end']
                    road_type = plan['type']
                    
                    # 边界检查
                    if (0 <= start_y < pathfinder.h and 0 <= end_y < pathfinder.h and
                        0 <= start_x < pathfinder.w and 0 <= end_x < pathfinder.w):
                        
                        # 验证起终点可行性
                        start_ok = not pathfinder.water_map[start_y, start_x]
                        end_ok = not pathfinder.water_map[end_y, end_x]
                        
                        if start_ok and end_ok:
                            path = pathfinder.find_path_with_fallback((start_x, start_y), (end_x, end_y))
                            if path:
                                pathfinder.update_road_network(path, road_type)
                return
    
    # 优化生成顺序 - 先主干道再次干道
    road_plans = []
    
    # 1. 规划主要干道
    for i in range(-grid_size, grid_size+1, main_road_interval):
        # 水平主干道
        road_plans.append({
            'start': (cx - grid_size, cy + i),
            'end': (cx + grid_size, cy + i),
            'type': 2  # 次干道(在城市内部)
        })
        
        # 垂直主干道
        road_plans.append({
            'start': (cx + i, cy - grid_size),
            'end': (cx + i, cy + grid_size),
            'type': 2  # 次干道(在城市内部)
        })
    
    # 2. 规划次要道路
    for i in range(-grid_size, grid_size+1, spacing):
        # 跳过已经作为主干道的位置
        if i % main_road_interval == 0:
            continue
            
        # 水平次要道路
        road_plans.append({
            'start': (cx - grid_size, cy + i),
            'end': (cx + grid_size, cy + i),
            'type': 3  # 小路
        })
        
        # 垂直次要道路
        road_plans.append({
            'start': (cx + i, cy - grid_size),
            'end': (cx + i, cy + grid_size),
            'type': 3  # 小路
        })
    
    # 3. 根据地形优势执行规划
    # 先主干道后小路，确保骨架先形成
    road_plans.sort(key=lambda x: x['type'])
    
    for plan in road_plans:
        start_x, start_y = plan['start']
        end_x, end_y = plan['end']
        road_type = plan['type']
        
        # 边界检查
        if (0 <= start_y < pathfinder.h and 0 <= end_y < pathfinder.h and
            0 <= start_x < pathfinder.w and 0 <= end_x < pathfinder.w):
            
            # 计算起终点地形是否可行
            start_ok = not pathfinder.water_map[start_y, start_x]
            end_ok = not pathfinder.water_map[end_y, end_x]
            
            # 只在两端都可行时建路
            if start_ok and end_ok:
                path = pathfinder.find_path_with_fallback((start_x, start_y), (end_x, end_y))
                if path:
                    pathfinder.update_road_network(path, road_type)

def naturalize_road_network(pathfinder):
    """改进的道路网络自然化处理，保持连通性和地形合理性
    
    Args:
        pathfinder: GeographicPathfinder实例
    """
    # 校验输入类型
    if not isinstance(pathfinder.road_network, np.ndarray) or not isinstance(pathfinder.road_types, np.ndarray):
        raise TypeError("道路网络和类型必须为NumPy数组")
    
    # 创建副本
    road_map = pathfinder.road_network.astype(np.bool_).copy()
    road_types = pathfinder.road_types.astype(np.uint8).copy()
    
    # 获取尺寸
    h, w = road_map.shape
    
    # 定义方向数组 - 修复：之前使用了未定义的directions变量
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
    
    # 改进：记录原始连通性信息(用于验证)
    def get_connected_components(road_map):
        """获取道路网络的连通分量"""
        visited = np.zeros_like(road_map, dtype=bool)
        components = []
        
        for y in range(h):
            for x in range(w):
                if road_map[y, x] and not visited[y, x]:
                    # 新连通分量
                    component = []
                    stack = [(x, y)]
                    while stack:
                        cx, cy = stack.pop()
                        if visited[cy, cx]:
                            continue
                        
                        visited[cy, cx] = True
                        component.append((cx, cy))
                        
                        # 检查8个方向 - 使用本地directions变量
                        for dx, dy in directions:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < w and 0 <= ny < h and 
                                road_map[ny, nx] and not visited[ny, nx]):
                                stack.append((nx, ny))
                    
                    components.append(component)
        
        return components
   
    # 获取原始连通分量
    original_components = get_connected_components(road_map)
    
    # 识别关键点
    junction_points = np.zeros((h, w), dtype=bool)
    endpoint_points = np.zeros((h, w), dtype=bool)
    
    # 计算每个点的连接数
    # 这里继续使用前面定义的directions变量
    for y in range(1, h-1):
        for x in range(1, w-1):
            if road_map[y, x]:
                neighbors = sum(1 for dx, dy in directions 
                              if 0 <= y+dy < h and 0 <= x+dx < w and road_map[y+dy, x+dx])
                if neighbors >= 3:
                    junction_points[y, x] = True
                elif neighbors == 1:
                    endpoint_points[y, x] = True
    
    # 2. 自然化处理 - 只处理非关键点
    # 使用4方向而不是8方向进行道路自然化
    road_directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if road_map[y, x] and not junction_points[y, x] and not endpoint_points[y, x]:
                # 主干道保持不变
                if road_types[y, x] == 1:
                    continue
                
                # 计算道路段长度和曲率(影响偏移概率)
                road_segment_length = 0
                curved = False
                
                # 寻找当前道路段
                nx, ny = x, y
                last_dir = None
                for _ in range(10):  # 最多检查10个相邻点
                    road_neighbors = []
                    for dx, dy in road_directions:
                        tx, ty = nx + dx, ny + dy
                        if (0 <= tx < w and 0 <= ty < h and 
                            road_map[ty, tx] and not junction_points[ty, tx]):
                            road_neighbors.append((tx, ty))
                    
                    if len(road_neighbors) != 2:  # 不是直线段的一部分
                        break
                        
                    # 检查方向变化
                    new_dir = None
                    for tx, ty in road_neighbors:
                        if (tx, ty) != (nx, ny):
                            new_dir = (tx - nx, ty - ny)
                            break
                    
                    if last_dir and last_dir != new_dir:
                        curved = True
                        
                    last_dir = new_dir
                    road_segment_length += 1
                
                # 动态计算偏移概率，直线段偏移概率更高
                base_prob = 0.05 if road_types[y, x] == 2 else 0.15
                if road_segment_length > 5 and not curved:
                    shift_prob = min(0.9, base_prob * (1 + road_segment_length * 0.05))
                else:
                    shift_prob = base_prob
                
                if random.random() < shift_prob:
                    
                    # 尝试找到最佳偏移方向
                    best_cost = float('inf')
                    best_pos = None
                    
                    # 随机打乱方向以避免系统性偏移
                    random_dirs = road_directions.copy()
                    random.shuffle(random_dirs)
                    
                    for dx, dy in random_dirs:
                        nx, ny = x + dx, y + dy
                        
                        # 基本边界和冲突检查
                        if (0 <= nx < w and 0 <= ny < h and 
                            not road_map[ny, nx] and 
                            not pathfinder.water_map[ny, nx]):
                            
                            # 确保不会创建奇怪的转弯 - 检查两个方向的邻居
                            neighbors_before = 0
                            neighbors_after = 0
                            
                            for ndx, ndy in road_directions:
                                ox, oy = x + ndx, y + ndy
                                if 0 <= ox < w and 0 <= oy < h and road_map[oy, ox]:
                                    neighbors_before += 1
                                    
                                ox, oy = nx + ndx, ny + ndy
                                if 0 <= ox < w and 0 <= oy < h and road_map[oy, ox]:
                                    neighbors_after += 1
                            
                            # 只有当移动后连通性相似时才考虑该位置
                            if abs(neighbors_after - neighbors_before) <= 1:
                                # 检查地形成本
                                try:
                                    cost = pathfinder._get_movement_cost(x-dx, y-dy, nx, ny)
                                    if cost < best_cost and cost < 5.0:  # 避免高成本地形
                                        best_cost = cost
                                        best_pos = (nx, ny)
                                except:
                                    continue
                    
                    # 改进：应用偏移前保存原始状态
                    original_state = road_map.copy()
                    
                    # 应用最佳偏移
                    if best_pos:
                        nx, ny = best_pos
                        road_map[ny, nx] = True
                        road_types[ny, nx] = road_types[y, x]
                        road_map[y, x] = False
                        road_types[y, x] = 0
                        
                        # 验证拓扑完整性
                        new_components = get_connected_components(road_map)
                        
                        # 如果连通分量数量增加，说明产生了断路，回滚变更
                        if len(new_components) > len(original_components):
                            road_map = original_state
                            road_types[ny, nx] = 0
                            road_types[y, x] = road_types[y, x]
    
    # 更新回路径规划器
    pathfinder.road_network = road_map
    pathfinder.road_types = road_types

def build_roads(height_map, buildings, order, water_map=None, biome_map=None, settlements=None):
    """优化版道路构建函数，构建层次化和自然的道路网络
    
    Args:
        height_map: 高度图
        buildings: 建筑物列表，每个元素为(x, y, type, orientation, importance, width, height)
        order: 建筑物的连接顺序（索引列表）
        water_map: 水域地图(可选)
        biome_map: 生物群系地图(可选)
        settlements: 聚居点列表(可选)，每个元素为(x, y, type, size, building_count)
        
    Returns:
        (road_network, road_types): 
            road_network - 二维布尔数组，表示道路位置
            road_types - 二维整数数组，表示道路类型（1-主干道，2-次干道，3-小路）
    """
    
    print(f"开始构建道路: buildings={len(buildings)}, settlement={settlements is not None}")
    # 确保至少有两个聚居点，这是构建道路网的基础
    h, w = height_map.shape
    if settlements is None or len(settlements) < 2:
        print(f"添加默认聚居点")
        # 添加至少两个位于地图对角的聚居点作为路网基础
        settlements = [
            (w//4, h//4, "village", 5, 10),
            (3*w//4, 3*h//4, "village", 5, 10)
        ]
    
    # 生物群系格式转换 - 如果是原始格式，进行转换为预期格式
    if biome_map is not None and isinstance(biome_map, np.ndarray):
        if biome_map.size > 0:
            test_value = biome_map[0, 0]
            if isinstance(test_value, (str, int, float)) and not isinstance(test_value, dict):
                print(f"转换biome_map格式")
                # 转换为字典格式
                converted_biome_map = np.empty(biome_map.shape, dtype=object)
                for y in range(biome_map.shape[0]):
                    for x in range(biome_map.shape[1]):
                        converted_biome_map[y, x] = {"name": biome_map[y, x]}
                biome_map = converted_biome_map
    
    # 确保水域地图格式正确
    if water_map is not None:
        print(f"处理water_map，类型: {type(water_map)}")
        # 转换为布尔类型
        water_map = np.array(water_map).astype(bool)
    
    # 改进：自动生成聚居点添加地形适宜性评估
    if settlements is None:
        settlements = []
        # 临时记录潜在聚居点
        potential_settlements = []
        
        for building in buildings:
            x, y, b_type, orient, imp, width, height = building
            
            # 检查地形适宜性
            if 0 <= y < height_map.shape[0] and 0 <= x < height_map.shape[1]:
                # 避免在陡峭地形和水域上生成聚居点
                gradient_y, gradient_x = np.gradient(height_map)
                local_slope = np.sqrt(gradient_x[y, x]**2 + gradient_y[y, x]**2)
                
                is_suitable = (not water_map[y, x]) and local_slope < 0.3
                
                if is_suitable and (imp > 0.7 or b_type in ["city_hall", "town_center"]):
                    # 计算周边建筑密度
                    building_count = sum(
                        1 for (bx, by, *_) in buildings 
                        if (bx - x)**2 + (by - y)**2 <= 25
                    )
                    potential_settlements.append((x, y, "village", max(2, building_count//5), building_count, imp))
        
        # 按重要性排序并选择前N个避免过多聚居点
        potential_settlements.sort(key=lambda s: s[5], reverse=True)
        max_settlements = min(len(potential_settlements), 10)
        
        # 确保聚居点空间分布
        for i, settlement in enumerate(potential_settlements[:max_settlements]):
            x, y = settlement[0], settlement[1]
            # 检查是否与现有聚居点太近
            too_close = False
            for sx, sy, *_ in settlements:
                if (sx-x)**2 + (sy-y)**2 < 400:  # 距离<20
                    too_close = True
                    break
                    
            if not too_close:
                settlements.append(settlement[:5])  # 去掉临时添加的重要性

    # 改进建筑优先级计算
    enhanced_buildings = []
    for idx, building in enumerate(buildings):
        # 解包全部参数
        x, y, typ, orient, imp, width, height = building
        
        # 改进：保持原有重要性，仅适度提升order中的建筑
        if idx in order:
            # 提升比例而非固定值，避免低重要性建筑被过度提升
            enhanced_imp = min(1.0, imp * 1.5)  # 最多提升50%，上限为1.0
        else:
            enhanced_imp = imp
        enhanced_buildings.append((x, y, typ, orient, enhanced_imp))

    # 生成基础路网
    base_road_net, base_road_types = build_hierarchical_road_network(
        height_map, settlements, enhanced_buildings, water_map, biome_map)

    # 初始化路径规划器并加载生成的路网
    pathfinder = GeographicPathfinder(height_map, water_map, biome_map)
    pathfinder.road_network = base_road_net
    pathfinder.road_types   = base_road_types.copy()

    # 按照指定顺序连接建筑物
    for connect_order in order:
        if connect_order >= len(buildings):
            continue
        # 获取完整建筑信息
        building = buildings[connect_order]
        x, y = building[0], building[1]
        
        # 获取建筑重要性（如果可用）
        importance = building[4] if len(building) > 4 else 0.5
        
        nearest_road = pathfinder.find_nearest_road((x, y), max_range=30)
        if nearest_road:
            path = pathfinder.find_path_with_fallback((x, y), nearest_road)
            if path:
                # 根据建筑重要性决定道路类型
                if importance > 0.8:
                    road_type = 2  # 重要建筑连接次干道
                else:
                    road_type = 3  # 普通建筑连接小路
                pathfinder.update_road_network(path, road_type)

    # 最终自然化处理
    naturalize_road_network(pathfinder)
    print("最终道路点总数:", np.count_nonzero(pathfinder.road_network))
    return pathfinder.road_network.astype(np.uint8), pathfinder.road_types.astype(np.uint8)