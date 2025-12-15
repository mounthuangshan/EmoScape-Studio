#from __future__ import annotations
#标准库
import os
import random
import time
import logging

#数据处理与科学计算
import numpy as np

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#网络与并发
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

#其他工具
import pickle
from collections import defaultdict
from sklearn.cluster import KMeans
from .biome_patterns import BiomePatternLibrary, UserPreferenceModel



###################
#交互式进化调整地图
###################
class BiomeEvolutionEngine:
    """高级生物群系进化引擎，支持多样性保持、自适应变异和用户偏好学习"""
    
    # 生物群系相邻兼容性表（影响连贯性）
    BIOME_COMPATIBILITY = {
        "Desert": ["Savanna", "Plains", "Desert", "Mountain"],
        "Savanna": ["Desert", "Grassland", "Plains", "Forest", "Savanna"],
        "Plains": ["Desert", "Savanna", "Grassland", "Forest", "Taiga", "Tundra", "Mountain", "Plains"],
        "Grassland": ["Plains", "Forest", "Savanna", "Grassland", "Taiga", "Tundra", "Mountain"],
        "Forest": ["Plains", "Grassland", "Taiga", "Forest", "Jungle", "Mountain"],
        "Jungle": ["Forest", "Taiga", "Jungle"],
        "Taiga": ["Forest", "Plains", "Grassland", "Tundra", "Mountain", "Taiga"],
        "Tundra": ["Taiga", "Plains", "Grassland", "Mountain", "Ice", "Tundra"],
        "Mountain": ["Desert", "Plains", "Grassland", "Forest", "Taiga", "Tundra", "Mountain", "Ice"],
        "Ice": ["Tundra", "Mountain", "Ice"],
        "Ocean": ["Ocean", "Coast", "Beach"],
        "Coast": ["Ocean", "Beach", "Coast"],
        "Beach": ["Coast", "Plains", "Desert", "Grassland", "Beach"],
        "TemperateForest": ["Forest", "Plains", "Grassland", "Taiga", "TemperateForest"]
    }
    
    # 生物群系权重（决定变异概率）
    BIOME_WEIGHTS = {
        "Desert": 0.7,
        "Savanna": 0.7,
        "Plains": 0.9,
        "Grassland": 1.0,
        "Forest": 0.9,
        "Jungle": 0.6,
        "Taiga": 0.8,
        "Tundra": 0.7,
        "Mountain": 0.6,
        "Ice": 0.5,
        "Ocean": 0.4,
        "TemperateForest": 0.9
    }
    
    # 拓展的生物群系颜色映射
    BIOME_COLORS = {
        "Desert": (0.94, 0.87, 0.6),
        "Savanna": (0.85, 0.77, 0.36),
        "Plains": (0.8, 0.9, 0.6),
        "Grassland": (0.5, 0.8, 0.3),
        "Forest": (0.2, 0.6, 0.2),
        "Jungle": (0.0, 0.7, 0.2),
        "Taiga": (0.5, 0.7, 0.4),
        "Tundra": (0.8, 0.85, 0.8),
        "Mountain": (0.6, 0.6, 0.6),
        "Ice": (0.9, 0.95, 0.98),
        "Ocean": (0.0, 0.3, 0.8),
        "TemperateForest": (0.3, 0.65, 0.3)
    }
    
    def __init__(self, biome_map, height_map=None, temperature_map=None, 
                 moisture_map=None, memory_path=".evolution_memory"):
        """初始化进化引擎
        
        Args:
            biome_map: 当前生物群系地图
            height_map: 高度图（可选）
            temperature_map: 温度图（可选）
            moisture_map: 湿度图（可选）
            memory_path: 用户偏好记忆存储路径
        """
        self.biome_map = biome_map
        self.height_map = height_map
        self.temperature_map = temperature_map
        self.moisture_map = moisture_map
        self.h, self.w = biome_map.shape

        # 提取所有可用的生物群系类型
        self.available_biomes = set()
        for y in range(self.h):
            for x in range(self.w):
                biome_name = biome_map[y, x]
                self.available_biomes.add(biome_name)
            
        # 用户偏好学习系统
        self.memory_path = memory_path
        self.preference_memory = self._load_preference_memory()
        self.biome_preference_scores = defaultdict(lambda: 0.5)  # 默认中性偏好
        self.region_preference_scores = np.ones((self.h, self.w)) * 0.5  # 空间偏好
        
        # 初始化模式库和高级用户偏好模型
        self.pattern_library = BiomePatternLibrary(patterns_file=os.path.join(os.path.dirname(memory_path), "biome_patterns.pkl"))
        self.preference_model = UserPreferenceModel(feature_dim=11)  # 10个生物群系+1个多样性指标
        model_path = os.path.join(os.path.dirname(memory_path), "user_preference_model.pkl")
        self.preference_model.load_model(model_path)

        # 创建种群（多个地图方案）
        self.population_size = min(6, max(4, int(np.sqrt(self.h * self.w) // 20)))
        self.population = []
        self.fitness_scores = []
        self.current_generation = 0
        self.best_individual = None
        self.best_fitness = -1
        
        # 记录使用的模式
        self.used_patterns = {}  # 映射个体索引到使用的模式名称

        # 现在安全初始化种群
        self._initialize_population()

        # 高级特征
        self.landscape_metrics = {}  # 存储景观指标
        self.history = []  # 进化历史
        self.executor = ThreadPoolExecutor(max_workers=4)  # 并行处理
    
    def _initialize_population(self):
        """初始化多样化种群"""
        self.population = []
        
        # 第一个个体是原始地图的深拷贝
        original = np.copy(self.biome_map)  # 明确拷贝为数组
        self.population.append(original)
        
        # 生成变异个体
        for i in range(1, self.population_size):
            mutation_rate = 0.05 + 0.15 * (i / self.population_size)
            new_individual = np.copy(self.biome_map)

            new_individual = self._mutate(new_individual, mutation_rate)

            self.population.append(new_individual)
        
        self.fitness_scores = [0.0] * self.population_size
    
    def _deep_copy_biome_map(self, biome_map):
        """创建生物群系地图的深拷贝"""

        return np.copy(biome_map)  # 使用 np.copy 创建独立副本
    
    def _mutate(self, biome_map, mutation_rate=0.1):
        """智能变异操作，保持生物群系连贯性
        
        Args:
            biome_map: 要变异的生物群系地图（NumPy数组，元素为字符串）
            mutation_rate: 变异率（0-1之间）
                
        Returns:
            变异后的生物群系地图
        """
        # 创建副本以避免修改原始数据
        mutated_map = np.copy(biome_map)
        
        # 计算变异格子数
        cells_to_mutate = int(self.h * self.w * mutation_rate)
        
        # 选择连续区域而非随机点
        if cells_to_mutate > 10 and random.random() < 0.7:
            # 选择1-3个种子点
            seed_count = random.randint(1, min(3, cells_to_mutate // 10))
            seeds = []
            for _ in range(seed_count):
                x = random.randint(0, self.w - 1)
                y = random.randint(0, self.h - 1)
                seeds.append((x, y))
            
            # 从种子点扩散
            changed_cells = set()
            for sx, sy in seeds:
                # 获取当前生物群系（直接访问字符串）
                current_biome = mutated_map[sy, sx]
                
                # 获取兼容生物群系
                compatible_biomes = self.BIOME_COMPATIBILITY.get(current_biome, list(self.available_biomes))
                
                # 基于偏好选择变异目标
                target_biome = self._weighted_choice(compatible_biomes)
                
                # 扩散变异（模拟洪水填充）
                pending = [(sx, sy)]
                region_size = random.randint(cells_to_mutate // (2 * seed_count), 
                                            cells_to_mutate // seed_count)
                
                while pending and len(changed_cells) < region_size:
                    x, y = pending.pop(0)
                    if (x, y) in changed_cells:
                        continue
                    
                    changed_cells.add((x, y))
                    mutated_map[y, x] = target_biome  # 正确索引方式
                    
                    # 添加相邻格子
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.w and 0 <= ny < self.h and 
                            (nx, ny) not in changed_cells and
                            random.random() < 0.7):  # 扩散概率
                            pending.append((nx, ny))
        else:
            # 常规随机变异
            for _ in range(cells_to_mutate):
                # 智能选点：倾向于选择边界区域
                if random.random() < 0.7 and self.height_map is not None:
                    # 使用高度梯度寻找边界
                    y = random.randint(1, self.h - 2)
                    x = random.randint(1, self.w - 2)
                    neighbors = [
                        mutated_map[y+1, x],  # 正确索引方式
                        mutated_map[y-1, x],
                        mutated_map[y, x+1],
                        mutated_map[y, x-1]
                    ]
                    # 如果周围有不同生物群系，增加选择概率
                    if len(set(neighbors)) > 1:
                        pass  # 保持当前选择
                    else:
                        # 重新随机选点
                        x = random.randint(0, self.w - 1)
                        y = random.randint(0, self.h - 1)
                else:
                    x = random.randint(0, self.w - 1)
                    y = random.randint(0, self.h - 1)
                
                # 获取当前和兼容的生物群系
                current_biome = mutated_map[y, x]
                compatible_biomes = self.BIOME_COMPATIBILITY.get(current_biome, list(self.available_biomes))
                
                # 基于用户偏好选择新生物群系
                target_biome = self._weighted_choice(compatible_biomes)
                mutated_map[y, x] = target_biome  # 直接赋值字符串
        
        # 应用平滑滤波器，确保生物群系区域连贯性
        if random.random() < 0.3:
            mutated_map = self._smooth_biomes(mutated_map)
                
        return mutated_map
    
    def _weighted_choice(self, biomes):
        """根据权重和用户偏好选择生物群系"""
        weights = []
        for biome in biomes:
            # 结合默认权重和学习到的用户偏好
            weight = self.BIOME_WEIGHTS.get(biome, 0.5) * (0.5 + self.biome_preference_scores[biome])
            weights.append(weight)
            
        # 归一化权重
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
            
        # 加权选择
        r = random.random()
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return biomes[i]
        
        # 后备选择
        return random.choice(biomes)
    
    def _smooth_biomes(self, biome_map):
        """修正后的平滑方法"""
        smoothed = np.copy(biome_map)
        kernel = [(i,j) for i in (-1,0,1) for j in (-1,0,1) if not (i == j == 0)]
        
        for y in range(1, self.h-1):
            for x in range(1, self.w-1):
                neighbors = [biome_map[y+dy, x+dx] for dy, dx in kernel]
                counts = defaultdict(int)
                for n in neighbors:
                    counts[n] += 1
                
                current = biome_map[y, x]
                if counts[current] < 5:  # 如果周围超过一半不同
                    most_common = max(counts.items(), key=lambda x:x[1])[0]
                    smoothed[y, x] = most_common
        return smoothed
    
    def _crossover(self, parent1, parent2):
        """修正后的交叉方法，支持字符串类型的生物群系"""
        # 创建child数组时显式指定dtype=object，以支持任意类型值（包括字符串）
        child = np.empty_like(parent1, dtype=object)
        # 先复制parent1的所有内容
        for y in range(self.h):
            for x in range(self.w):
                child[y, x] = parent1[y, x]
        
        # 随机选择交叉策略
        strategy = random.choice(["horizontal", "vertical", "quadrant"])
        
        if strategy == "horizontal":
            split_y = random.randint(1, self.h-2)
            for y in range(split_y, self.h):
                for x in range(self.w):
                    child[y, x] = parent2[y, x]
        elif strategy == "vertical":
            split_x = random.randint(1, self.w-2)
            for y in range(self.h):
                for x in range(split_x, self.w):
                    child[y, x] = parent2[y, x]
        else:
            # 四象限交叉
            mid_x, mid_y = self.w//2, self.h//2
            # 左上象限
            for y in range(0, mid_y):
                for x in range(0, mid_x):
                    child[y, x] = parent2[y, x]
            # 左下象限
            for y in range(mid_y, self.h):
                for x in range(0, mid_x):
                    child[y, x] = parent1[y, x]
            # 右上象限
            for y in range(0, mid_y):
                for x in range(mid_x, self.w):
                    child[y, x] = parent1[y, x]
            # 右下象限
            for y in range(mid_y, self.h):
                for x in range(mid_x, self.w):
                    child[y, x] = parent2[y, x]
        
        return self._smooth_biomes(child)
    
    def _select_parents(self):
        """基于适应度选择父代"""
        if not any(self.fitness_scores):  # 避免全零分
            return random.sample(range(self.population_size), 2)
            
        # 转换为选择概率
        probs = np.array(self.fitness_scores)
        min_val = min(probs)
        probs = probs - min_val + 1e-6  # 避免负数和零
        probs = probs / probs.sum()
        
        # 轮盘赌选择
        selected = np.random.choice(self.population_size, 2, p=probs, replace=False)
        return selected
    
    def _update_preference_model(self, individual, score):
        """更新用户偏好模型"""
        # 记录每种生物群系的受欢迎程度
        biome_counts = defaultdict(int)
        total_cells = self.h * self.w
        
        for y in range(self.h):
            for x in range(self.w):
                # 直接使用生物群系名称字符串，而不是尝试访问字典的"name"键
                biome = individual[y, x]  # 使用正确的numpy数组索引方式
                biome_counts[biome] += 1
        
        # 归一化到0-1
        normalized_score = score / 10.0
        
        # 更新每种生物群系的偏好分数
        learn_rate = 0.2  # 学习率
        for biome, count in biome_counts.items():
            # 权重与生物群系数量成正比
            weight = count / total_cells
            old_score = self.biome_preference_scores[biome]
            # 平滑更新
            self.biome_preference_scores[biome] = old_score * (1 - learn_rate) + normalized_score * learn_rate * weight
        
        # 保存用户偏好
        self._save_preference_memory()
    
    def _load_preference_memory(self):
        """加载用户偏好记忆"""
        try:
            if os.path.exists(self.memory_path):
                with open(self.memory_path, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        return {"biome_preferences": {}, "region_preferences": None}
    
    def _save_preference_memory(self):
        """保存用户偏好记忆"""
        try:
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
            data = {
                "biome_preferences": dict(self.biome_preference_scores),
                "region_preferences": self.region_preference_scores,
                "last_update": time.time()
            }
            with open(self.memory_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"保存偏好记忆失败: {e}")
    
    def calculate_landscape_metrics(self, individual):
        """修正后的景观指标计算"""
        metrics = {}
        
        # 斑块分析
        visited = np.zeros((self.h, self.w), dtype=bool)
        patch_sizes = []
        
        for y in range(self.h):
            for x in range(self.w):
                if not visited[y, x]:
                    biome = individual[y, x]  # 直接使用生物群系名称
                    queue = [(x, y)]
                    size = 0
                    
                    while queue:
                        cx, cy = queue.pop(0)
                        if visited[cy, cx]:
                            continue
                        visited[cy, cx] = True
                        size += 1
                        
                        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nx, ny = cx+dx, cy+dy
                            if 0<=nx<self.w and 0<=ny<self.h:
                                if individual[ny, nx] == biome and not visited[ny, nx]:
                                    queue.append((nx, ny))
                    
                    patch_sizes.append(size)
        
        metrics["num_patches"] = len(patch_sizes)
        metrics["max_patch"] = max(patch_sizes) if patch_sizes else 0
        metrics["avg_patch"] = sum(patch_sizes)/len(patch_sizes) if patch_sizes else 0
        
        return metrics
    
    def _flood_fill(self, biome_map, x, y, target_biome, visited):
        """用于斑块大小计算的填充算法"""
        queue = [(x, y)]
        visited.add((x, y))
        count = 0

        while queue:
            cx, cy = queue.pop(0)
            count += 1

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.w and 0 <= ny < self.h and
                    (nx, ny) not in visited and
                    biome_map[ny, nx] == target_biome):  # 直接比较生物群系名称
                    queue.append((nx, ny))
                    visited.add((nx, ny))

        return count
    
    def _create_diversity_individual(self):
        """创建一个高度多样化的个体，使用模式库"""
        # 有一定几率使用预设模式
        if random.random() < 0.7 and self.height_map is not None:
            # 使用复合模式生成地图
            composite_pattern = self.pattern_library.generate_composite_pattern(
                self.w, self.h, complexity=random.uniform(0.3, 0.8)
            )
            new_map = self._apply_composite_pattern(composite_pattern)
            
            # 记录使用的模式，便于后续更新质量分数
            pattern_names = []
            for region in composite_pattern.get('regions', []):
                pattern_names.append(region.get('name'))
            
            # 存储使用的模式名称
            idx = len(self.population) if self.population else 0
            self.used_patterns[idx] = pattern_names
            
            return new_map
        else:
            # 使用原有的环境或聚类生成方法作为后备
            if self.height_map is not None and self.temperature_map is not None:
                return self._generate_environmental_biomes()
            else:
                return self._generate_clustered_biomes()
    
    def _apply_composite_pattern(self, composite):
        """将复合模式应用到地图上"""
        # 创建新地图
        new_map = np.zeros((self.h, self.w), dtype=object)
        
        # 首先使用现有地图填充基础
        for y in range(self.h):
            for x in range(self.w):
                new_map[y, x] = self.biome_map[y, x]
        
        # 应用每个区域的模式
        for region in composite.get('regions', []):
            shape = region.get('shape')
            pattern = region.get('pattern')
            
            if shape == 'radial':
                # 径向扩散模式
                center_x, center_y = region.get('center')
                radius = region.get('radius')
                core_biome = pattern.get('core')
                
                # 应用核心和过渡层
                self._apply_radial_pattern(new_map, center_x, center_y, radius, pattern)
                
            elif shape == 'linear':
                # 线性模式（如山脉）
                center_x, center_y = region.get('center')
                length = region.get('length')
                width = region.get('width')
                angle = region.get('angle')
                
                # 应用线性模式
                self._apply_linear_pattern(new_map, center_x, center_y, 
                                          length, width, angle, pattern)
                
            elif shape == 'cluster':
                # 聚类模式（如森林斑块）
                clusters = region.get('clusters', [])
                size = region.get('size')
                
                # 应用聚类模式
                self._apply_cluster_pattern(new_map, clusters, size, pattern)
        
        # 平滑处理，确保区域之间过渡自然
        new_map = self._smooth_biomes(new_map)
        
        return new_map
    
    def _apply_radial_pattern(self, map_data, cx, cy, radius, pattern):
        """应用径向模式"""
        core_biome = pattern.get('core')
        transitions = pattern.get('transition_layers', [])
        hotspots = pattern.get('hotspots', [])
        
        # 将过渡层排序，从远到近
        transitions.sort(key=lambda x: x.get('weight', 0))
        
        # 计算每个过渡层的半径
        layer_radii = []
        for i, layer in enumerate(transitions):
            # 每层占总半径的比例
            proportion = 1.0 - layer.get('weight', 0.5)
            layer_radius = radius * proportion
            layer_radii.append(layer_radius)
        
        # 应用核心和过渡层
        for y in range(max(0, cy-radius), min(self.h, cy+radius+1)):
            for x in range(max(0, cx-radius), min(self.w, cx+radius+1)):
                # 计算到中心的距离
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                
                # 小于核心区，赋值核心生物群系
                if dist <= radius * 0.4:  # 核心区占半径的40%
                    map_data[y, x] = core_biome
                    continue
                
                # 在过渡层中，按距离分配生物群系
                for i, (layer, layer_radius) in enumerate(zip(transitions, layer_radii)):
                    if dist <= layer_radius:
                        # 随机性，增加自然感
                        if random.random() < 0.8:  # 80%概率使用该层生物群系
                            map_data[y, x] = layer.get('biome')
                        break
        
        # 应用热点（如沙漠中的绿洲）
        for hotspot in hotspots:
            # 随机选择热点位置
            offset_x = random.randint(-radius//2, radius//2)
            offset_y = random.randint(-radius//2, radius//2)
            hx = min(self.w-1, max(0, cx + offset_x))
            hy = min(self.h-1, max(0, cy + offset_y))
            
            hotspot_radius = hotspot.get('radius', 2)
            hotspot_biome = hotspot.get('biome')
            
            # 应用热点生物群系
            for y in range(max(0, hy-hotspot_radius), min(self.h, hy+hotspot_radius+1)):
                for x in range(max(0, hx-hotspot_radius), min(self.w, hx+hotspot_radius+1)):
                    dist = np.sqrt((x-hx)**2 + (y-hy)**2)
                    if dist <= hotspot_radius:
                        map_data[y, x] = hotspot_biome
    
    def _apply_linear_pattern(self, map_data, cx, cy, length, width, angle, pattern):
        """应用线性模式（如山脉）"""
        core_biome = pattern.get('core')
        transitions = pattern.get('transition_layers', [])
        
        # 计算方向向量
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # 为每个点计算到线的距离
        for y in range(self.h):
            for x in range(self.w):
                # 计算点到线的投影点
                t = (x - cx) * dx + (y - cy) * dy
                
                # 计算投影点坐标
                px = cx + t * dx
                py = cy + t * dy
                
                # 计算点到线的距离和在线上的位置
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                pos = abs(t)
                
                # 如果在范围内，应用模式
                if dist <= width and pos <= length / 2:
                    # 在核心区
                    if dist <= width * 0.4:
                        map_data[y, x] = core_biome
                    else:
                        # 在过渡区
                        for transition in transitions:
                            twidth = width * transition.get('weight', 0.5)
                            if dist <= twidth:
                                if random.random() < 0.85:  # 一定随机性
                                    map_data[y, x] = transition.get('biome')
                                break
    
    def _apply_cluster_pattern(self, map_data, clusters, size, pattern):
        """应用聚类模式（如森林斑块）"""
        core_biome = pattern.get('core')
        
        for cx, cy in clusters:
            # 创建不规则形状的斑块
            points = set()
            pending = [(cx, cy)]
            
            while pending and len(points) < size * size:
                x, y = pending.pop(0)
                if (x, y) in points:
                    continue
                    
                points.add((x, y))
                
                # 四向扩散，不规则形状
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.w and 0 <= ny < self.h and
                        (nx, ny) not in points and
                        random.random() < 0.75):  # 随机分支概率
                        pending.append((nx, ny))
            
            # 应用斑块
            for x, y in points:
                if 0 <= x < self.w and 0 <= y < self.h:
                    map_data[y, x] = core_biome
    
    def evolve_generation(self, user_scores=None):
        """进化到下一代，支持高级用户偏好模型"""
        if user_scores:
            self.fitness_scores = user_scores
            
            # 使用高级偏好模型进行学习
            for i, score in enumerate(user_scores):
                if i < len(self.population) and score > 0:
                    # 更新用户偏好模型
                    self.preference_model.update_model(self.population[i], score)
                    
                    # 更新使用的模式的质量分数
                    if i in self.used_patterns:
                        for pattern_name in self.used_patterns[i]:
                            self.pattern_library.update_pattern_quality(pattern_name, score/10.0)
  
        # 保存最佳个体
        max_fitness_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[max_fitness_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[max_fitness_idx]
            self.best_individual = self._deep_copy_biome_map(self.population[max_fitness_idx])

        # 更新原有用户偏好模型（为了向后兼容）
        for i, score in enumerate(self.fitness_scores):
            if score > 0:
                self._update_preference_model(self.population[i], score)

        # 生成新种群的逻辑保持不变
        new_population = []
        sorted_indices = np.argsort(self.fitness_scores)[::-1]

        # 保留25%的精英
        elite_count = max(1, self.population_size // 4)
        for i in range(elite_count):
            idx = sorted_indices[i]
            new_population.append(self._deep_copy_biome_map(self.population[idx]))
        
        # 通过交叉和变异生成新个体
        while len(new_population) < self.population_size:
            # 选择父代
            parent_indices = self._select_parents()
            parent1 = self.population[parent_indices[0]]
            parent2 = self.population[parent_indices[1]]
            
            # 交叉
            if random.random() < 0.7:  # 交叉概率
                child = self._crossover(parent1, parent2)
            else:
                child = self._deep_copy_biome_map(random.choice([parent1, parent2]))
            
            # 变异
            if random.random() < 0.8:  # 变异概率
                mutation_rate = 0.05 + random.random() * 0.15  # 5%-20%变异率
                child = self._mutate(child, mutation_rate)
            
            new_population.append(child)

        # 确保多样性 - 替换一个个体为高多样性方案
        if self.current_generation > 0 and len(new_population) >= self.population_size:
            # 一定概率使用模式库生成多样性个体
            if random.random() < 0.4:  # 40%概率
                diversity_individual = self._create_diversity_individual()
                # 替换最不适应的非精英个体
                new_population[-1] = diversity_individual
        
        self.population = new_population
        self.current_generation += 1
        self.used_patterns = {}  # 重置使用的模式记录
        
        # 重置适应度分数
        self.fitness_scores = [0] * self.population_size
        
        # 对每个个体预测分数
        predicted_scores = []
        for ind in self.population:
            predicted = self.preference_model.predict_score(ind)
            predicted_scores.append(predicted)
        
        logging.info(f"第{self.current_generation}代预测分数: {[round(s, 2) for s in predicted_scores]}")
        
        # 计算并存储景观指标
        for i, ind in enumerate(self.population):
            self.landscape_metrics[i] = self.calculate_landscape_metrics(ind)

        # 保存更新后的模式库和用户偏好模型
        self.pattern_library.save_patterns()
        self.preference_model.save_model(
            os.path.join(os.path.dirname(self.memory_path), "user_preference_model.pkl")
        )
        
        return self.population
    
    def _generate_environmental_biomes(self):
        """修正后的环境生成方法"""
        new_map = np.zeros((self.h, self.w), dtype=object)
        
        # 标准化环境参数
        height_norm = (self.height_map - np.min(self.height_map)) / (np.ptp(self.height_map)+1e-6)
        temp_norm = (self.temperature_map - np.min(self.temperature_map)) / (np.ptp(self.temperature_map)+1e-6)
        moisture_norm = (self.moisture_map - np.min(self.moisture_map)) / (np.ptp(self.moisture_map)+1e-6)
        
        # 并行生成
        def _assign_biome(y, x):
            h = height_norm[y, x]
            t = temp_norm[y, x]
            m = moisture_norm[y, x]
            
            if h < 0.2: return "Ocean"
            if h > 0.8: return "Mountain" if t > 0.3 else "Ice"
            if t < 0.3: return "Tundra"
            if t > 0.7: return "Desert" if m < 0.3 else "Jungle"
            if m > 0.6: return "Forest"
            if m > 0.4: return "Grassland"
            return "Plains"
        
        # 使用线程池并行处理
        with ThreadPoolExecutor() as executor:
            futures = []
            for y in range(self.h):
                for x in range(self.w):
                    futures.append(executor.submit(_assign_biome, y, x))
            
            for y in range(self.h):
                for x in range(self.w):
                    new_map[y, x] = futures[y*self.w + x].result()
        
        return new_map

    def _generate_clustered_biomes(self, num_clusters=8):
        """基于聚类算法生成生物群系"""
        
        # 准备特征矩阵（包含空间坐标和环境参数）
        features = []
        valid_points = []
        
        for y in range(self.h):
            for x in range(self.w):
                # 标准化特征
                norm_x = x / self.w
                norm_y = y / self.h
                # 修复索引方式，使用numpy数组正确的索引方式
                h = self.height_map[y, x] if self.height_map is not None else 0
                t = self.temperature_map[y, x] if self.temperature_map is not None else 0
                m = self.moisture_map[y, x] if self.moisture_map is not None else 0
                
                features.append([norm_x, norm_y, h, t, m])
                valid_points.append((y, x))
        
        # 执行聚类
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        clusters = kmeans.fit_predict(features)
        
        # 将聚类结果映射回二维网格
        cluster_grid = np.zeros((self.h, self.w), dtype=int)
        for idx, (y, x) in enumerate(valid_points):
            cluster_grid[y, x] = clusters[idx]
        
        # 为每个聚类分配生物群系
        biome_choices = list(self.available_biomes)
        cluster_biomes = {}
        
        for cluster_id in range(num_clusters):
            # 获取聚类中心的环境参数
            center = kmeans.cluster_centers_[cluster_id]
            _, _, h, t, m = center
            
            # 根据中心特征选择合适生物群系
            if h < 0.2:
                candidates = ["Ocean", "Coast"]
            elif h > 0.7:
                candidates = ["Mountain", "Ice"]
            else:
                if t > 0.6:
                    candidates = ["Desert", "Savanna", "Jungle"]
                elif t < 0.4:
                    candidates = ["Taiga", "Tundra"]
                else:
                    if m > 0.6:
                        candidates = ["Forest", "Jungle"]
                    elif m > 0.3:
                        candidates = ["Grassland", "Plains"]
                    else:
                        candidates = ["Plains", "Desert"]
            
            # 过滤可用的生物群系
            valid_biomes = [b for b in candidates if b in biome_choices]
            if not valid_biomes:
                valid_biomes = biome_choices
            
            cluster_biomes[cluster_id] = random.choice(valid_biomes)
        
        # 生成新地图
        new_map = self._deep_copy_biome_map(self.biome_map)
        for y in range(self.h):
            for x in range(self.w):
                cluster_id = cluster_grid[y, x]
                # 修复：直接赋值字符串，不使用字典结构
                new_map[y, x] = cluster_biomes[cluster_id]
        
        # 后处理：确保聚类边界自然过渡
        return self._smooth_cluster_boundaries(new_map, cluster_grid, cluster_biomes)

    def _smooth_cluster_boundaries(self, biome_map, cluster_grid, cluster_biomes):
        """平滑聚类边界，增加自然过渡"""
        for y in range(1, self.h-1):
            for x in range(1, self.w-1):
                current_cluster = cluster_grid[y, x]
                
                # 检查周围聚类
                neighbors = set()
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        neighbors.add(cluster_grid[y+dy, x+dx])
                
                # 如果处于多个聚类交界处
                if len(neighbors) > 2:
                    # 收集相邻生物群系
                    adjacent_biomes = [cluster_biomes[c] for c in neighbors if c != current_cluster]
                    
                    # 选择兼容的生物群系
                    # 修复：直接使用生物群系名称字符串
                    current_biome = biome_map[y, x]
                    valid_biomes = [
                        b for b in adjacent_biomes
                        if b in self.BIOME_COMPATIBILITY.get(current_biome, [])
                    ]
                    
                    if valid_biomes:
                        # 修复：直接赋值字符串
                        biome_map[y, x] = random.choice(valid_biomes)
        
        return biome_map

    def _process_environment_chunk(self, new_map, height, temp, moisture, 
                                y_start, y_end, x_start, x_end,
                                sea_level, mountain_level, ice_temp, desert_humid):
        """处理环境参数分块映射"""
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                h = height[y, x]  # 修正索引方式
                t = temp[y, x]
                m = moisture[y, x]
                
                # 海洋判定
                if h < sea_level:
                    # 修复：直接赋值字符串
                    new_map[y, x] = "Ocean"
                    continue
                
                # 山地和冰原
                if h > mountain_level:
                    if t < ice_temp:
                        biome = "Ice"
                    else:
                        biome = "Mountain"
                    # 修复：直接赋值字符串    
                    new_map[y, x] = biome
                    continue
                
                # 温带区域判定
                if 0.4 <= t <= 0.6:
                    if m > 0.6:
                        biome = "Forest"
                    elif m > 0.3:
                        biome = "Grassland"
                    else:
                        biome = "Plains"
                # 热带区域
                elif t > 0.6:
                    if m < desert_humid:
                        biome = "Desert"
                    elif m > 0.7:
                        biome = "Jungle"
                    else:
                        biome = "Savanna"
                # 寒带区域
                else:
                    if m > 0.5:
                        biome = "Taiga"
                    else:
                        biome = "Tundra"
                
                # 边界平滑处理
                if x > 0 and y > 0:
                    neighbors = [
                        new_map[y-1, x],  # 修正索引方式并直接访问值
                        new_map[y, x-1],
                        new_map[y-1, x-1]
                    ]
                    if biome not in neighbors:
                        biome = random.choice(neighbors)
                
                # 修复：直接赋值字符串
                new_map[y, x] = biome