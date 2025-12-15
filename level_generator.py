import logging
import random
import math
import numpy as np
import networkx as nx
from datetime import datetime
from collections import deque
from scipy.ndimage import gaussian_filter, sobel

class LevelGenerator:
    """游戏关卡生成器
    
    负责创建结构化、平衡的游戏关卡。支持两种生成策略:
    - graph: 基于图论的复杂关卡结构生成
    - spatial: 基于空间布局的关卡生成
    """
    
    def __init__(self, map_data, params=None, preferences=None, logger=None, strategy="graph"):
        """初始化关卡生成器
        
        Args:
            map_data: 地图数据对象
            params: 空间策略的关卡生成参数(spatial)
            preferences: 图策略的关卡生成参数(graph)
            logger: 日志记录器
            strategy: 生成策略，"graph" 或 "spatial"
        """
        self.map_data = map_data
        self.params = params or {}
        self.preferences = preferences or {}
        self.logger = logger or logging.getLogger("LevelGenerator")
        self.strategy = strategy
        self.level_data = {}
        self.node_graph = None
        
        # 空间策略的附加属性
        if hasattr(map_data, 'width') and hasattr(map_data, 'height'):
            self.width = map_data.width
            self.height = map_data.height
        else:
            # 尝试从地图数据推断尺寸
            height_map = self._get_height_map()
            if height_map is not None and hasattr(height_map, 'shape'):
                self.height, self.width = height_map.shape
            else:
                self.width = 512
                self.height = 512
                self.logger.warning("无法确定地图尺寸，使用默认值 512x512")
        
        # 通用节点类型
        self.node_types = ["combat", "puzzle", "treasure", "rest", "elite", "discovery"]
        
        # 初始化空间策略的参数
        if strategy == "spatial":
            self._init_spatial_strategy()
            # 确保有高度图可用
            height_map = self._get_height_map()
            if height_map is not None:
                self.heights = height_map
            else:
                # 创建默认高度图
                self.logger.warning("无法获取高度图，创建默认高度图")
                self.heights = np.random.rand(self.height, self.width) * 100
        
        # 版本信息
        self.version = "2.0-unified"
    
    def _init_spatial_strategy(self):
        """初始化空间策略特定的属性"""
        # 节点权重 - 用于不同节点类型的放置概率
        self.node_weights = {
            "combat": self.params.get("node_ratios", {}).get("combat", 0.4),
            "puzzle": self.params.get("node_ratios", {}).get("puzzle", 0.2),
            "treasure": self.params.get("node_ratios", {}).get("treasure", 0.2),
            "rest": 0.15,
            "elite": 0.1,
            "discovery": 0.1
        }
        
        # 关卡类型特定配置
        self.level_type_configs = {
            "balanced": {
                "path_count": 2,
                "node_count_range": (10, 15),
                "branch_probability": 0.5
            },
            "exploration": {
                "path_count": 3,
                "node_count_range": (15, 25),
                "branch_probability": 0.7
            },
            "combat": {
                "path_count": 1,
                "node_count_range": (8, 12),
                "branch_probability": 0.3,
                "node_weights": {"combat": 0.6, "elite": 0.2}
            },
            "puzzle": {
                "path_count": 2,
                "node_count_range": (10, 18),
                "branch_probability": 0.4,
                "node_weights": {"puzzle": 0.5, "discovery": 0.2}
            }
        }
    
    def _get_height_map(self):
        """获取高度图，兼容两种方式的地图数据"""
        if hasattr(self.map_data, 'get_layer'):
            return self.map_data.get_layer("height")
        elif hasattr(self.map_data, 'layers') and "height" in self.map_data.layers:
            return self.map_data.layers["height"]
        return None
    
    def _get_biome_map(self):
        """获取生物群系图，兼容两种方式的地图数据"""
        if hasattr(self.map_data, 'get_layer'):
            return self.map_data.get_layer("biome")
        elif hasattr(self.map_data, 'layers') and "biome" in self.map_data.layers:
            return self.map_data.layers["biome"]
        return None
    
    def generate_level(self, level_type="balanced", difficulty=0.5):
        """生成游戏关卡
        
        Args:
            level_type: 关卡类型 ("exploration", "combat", "puzzle", "balanced")
            difficulty: 难度系数 (0.0-1.0)
            
        Returns:
            生成的关卡数据字典
        """
        self.logger.info(f"开始生成{level_type}类型关卡，难度:{difficulty}，使用{self.strategy}策略")
        
        if self.strategy == "graph":
            return self._generate_graph_level(level_type, difficulty)
        else:
            return self._generate_spatial_level(level_type, difficulty)
    
    #######################################################################
    # 图论策略的实现 (来自pygamemap.py)
    #######################################################################
    
    def _generate_graph_level(self, level_type="balanced", difficulty=0.5):
        """使用图论策略生成关卡"""
        self.level_data = {}
        
        # 1. 分析地图数据，找到适合的区域
        suitable_regions = self._graph_analyze_map_for_level()
        
        # 2. 创建关卡结构图
        self.node_graph = self._graph_create_level_structure(level_type, difficulty)
        
        # 3. 放置关键节点(起点、终点、检查点)
        self._graph_place_key_nodes(suitable_regions)
        
        # 4. 设计关卡路径和连接
        self._graph_design_paths()
        
        # 5. 根据难度曲线放置挑战和奖励
        self._graph_place_challenges_and_rewards(difficulty)
        
        # 6. 添加故事元素和任务
        self._graph_add_story_elements(level_type)
        
        # 7. 验证关卡可玩性和平衡性
        self._graph_validate_level()
        
        return self.level_data
    
    def _graph_analyze_map_for_level(self):
        """分析地图找出适合作为关卡区域的位置"""
        suitable_regions = {
            "start_areas": [],
            "end_areas": [],
            "challenge_areas": [],
            "reward_areas": [],
            "path_areas": []
        }
        
        # 获取必要的地图数据
        height_map = self._get_height_map()
        biome_map = self._get_biome_map()
        
        if height_map is None or biome_map is None:
            self.logger.warning("缺少高度图或生物群系图，使用默认区域划分")
            return suitable_regions
            
        height, width = height_map.shape
        
        # 分析高度图和生物群系找到合适的区域
        for region_type in suitable_regions:
            areas = []
            
            # 根据不同区域类型使用不同的选择标准
            if region_type == "start_areas":
                # 起点区域：平坦、安全、接近水源
                for y in range(0, height, height//10):
                    for x in range(0, width, width//10):
                        if self._graph_is_suitable_start_area(x, y, height_map, biome_map):
                            areas.append({"x": x, "y": y, "score": self._graph_score_start_area(x, y, height_map, biome_map)})
            
            elif region_type == "end_areas":
                # 终点区域：较高、独特、视野开阔
                for y in range(0, height, height//10):
                    for x in range(0, width, width//10):
                        if self._graph_is_suitable_end_area(x, y, height_map, biome_map):
                            areas.append({"x": x, "y": y, "score": self._graph_score_end_area(x, y, height_map, biome_map)})
            
            elif region_type == "challenge_areas":
                # 挑战区域：复杂地形、独特环境
                for y in range(0, height, height//10):
                    for x in range(0, width, width//10):
                        if 0 <= x < width-10 and 0 <= y < height-10:
                            # 检查区域适合挑战
                            region = height_map[y:y+10, x:x+10]
                            # 寻找地形变化较大的区域
                            if np.std(region) > 7.0:
                                areas.append({"x": x, "y": y, "score": 10.0 + np.std(region)})
            
            elif region_type == "reward_areas":
                # 奖励区域：隐蔽、美丽、特殊地形
                for y in range(0, height, height//10):
                    for x in range(0, width, width//10):
                        if 0 <= x < width-10 and 0 <= y < height-10:
                            # 检查区域是否适合奖励
                            region_biomes = set(biome_map[y:y+10, x:x+10].flat)
                            special_biomes = {2, 5, 9, 14}  # 假设这些ID是特殊生物群系
                            if any(biome in special_biomes for biome in region_biomes):
                                areas.append({"x": x, "y": y, "score": 10.0})
            
            elif region_type == "path_areas":
                # 路径区域：相对平坦，适合移动
                for y in range(0, height, height//10):
                    for x in range(0, width, width//10):
                        if 0 <= x < width-10 and 0 <= y < height-10:
                            # 检查区域是否适合路径
                            region = height_map[y:y+10, x:x+10]
                            if np.std(region) < 10.0:  # 相对平坦
                                areas.append({"x": x, "y": y, "score": 10.0 - np.std(region)})
            
            # 按分数排序
            areas.sort(key=lambda area: area["score"], reverse=True)
            suitable_regions[region_type] = areas[:max(5, len(areas))]
            
        return suitable_regions
    
    def _graph_is_suitable_start_area(self, x, y, height_map, biome_map):
        """判断位置是否适合作为起点区域"""
        # 检查区域是否在地图范围内
        h, w = height_map.shape
        if not (0 <= x < w-10 and 0 <= y < h-10):
            return False
            
        # 检查区域是否平坦(高度变化小)
        region = height_map[y:y+10, x:x+10]
        if np.std(region) > 5.0:
            return False
            
        # 检查生物群系是否适合(非危险生物群系)
        region_biomes = set(biome_map[y:y+10, x:x+10].flat)
        dangerous_biomes = {7, 12, 16}  # 假设这些ID是危险生物群系
        if any(biome in dangerous_biomes for biome in region_biomes):
            return False
            
        return True
    
    def _graph_score_start_area(self, x, y, height_map, biome_map):
        """评分起点区域的适合度"""
        # 基础分数
        score = 10.0
        
        # 区域平坦度加分(标准差越小越好)
        region = height_map[y:y+10, x:x+10]
        flatness = 1.0 / (np.std(region) + 0.1)
        score += flatness * 5.0
        
        # 靠近水源加分
        has_water_nearby = False
        for dy in range(-20, 20):
            for dx in range(-20, 20):
                ny, nx = y+dy, x+dx
                if 0 <= ny < height_map.shape[0] and 0 <= nx < height_map.shape[1]:
                    if height_map[ny, nx] < 5:  # 假设高度<5是水域
                        dist = np.sqrt(dx*dx + dy*dy)
                        if 5 <= dist <= 15:  # 不要太近也不要太远
                            has_water_nearby = True
                            score += 2.0
                            break
            if has_water_nearby:
                break
        
        # 生物群系适合度
        preferred_biomes = {1, 4, 6}  # 假设这些ID是适合起点的生物群系
        region_biomes = set(biome_map[y:y+10, x:x+10].flat)
        score += sum(2.0 for biome in region_biomes if biome in preferred_biomes)
        
        return score
        
    def _graph_is_suitable_end_area(self, x, y, height_map, biome_map):
        """判断位置是否适合作为终点区域"""
        # 检查区域是否在地图范围内
        h, w = height_map.shape
        if not (0 <= x < w-10 and 0 <= y < h-10):
            return False
            
        # 终点通常在高处或特殊地点
        region = height_map[y:y+10, x:x+10]
        avg_height = np.mean(region)
        if avg_height < np.mean(height_map) + 5:  # 比平均高度高一些
            return False
            
        # 检查视野 - 周围应该开阔
        surroundings = height_map[max(0, y-20):min(h, y+20), max(0, x-20):min(w, x+20)]
        if np.std(surroundings) > 15.0:  # 太多起伏不适合作为终点
            return False
            
        return True
    
    def _graph_score_end_area(self, x, y, height_map, biome_map):
        """评分终点区域的适合度"""
        # 基础分数
        score = 10.0
        
        # 区域高度加分(越高越好)
        region = height_map[y:y+10, x:x+10]
        avg_height = np.mean(region)
        height_factor = avg_height / np.mean(height_map)
        score += height_factor * 10.0
        
        # 独特性加分(如果附近有特殊生物群系)
        region_biomes = set(biome_map[y:y+10, x:x+10].flat)
        special_biomes = {8, 10, 15}  # 假设这些ID是特殊生物群系
        score += sum(3.0 for biome in region_biomes if biome in special_biomes)
        
        # 视野开阔度
        h, w = height_map.shape
        surroundings = height_map[max(0, y-20):min(h, y+20), max(0, x-20):min(w, x+20)]
        visibility = 15.0 - min(15.0, np.std(surroundings))
        score += visibility
        
        return score
        
    def _graph_create_level_structure(self, level_type, difficulty):
        """创建关卡结构图"""
        # 创建有向图表示关卡结构
        G = nx.DiGraph()
        
        # 根据关卡类型确定结构复杂度
        if level_type == "exploration":
            node_count = int(10 + difficulty * 15)  # 探索型关卡节点更多
            branching_factor = 0.6 + difficulty * 0.3  # 分支较多
        elif level_type == "combat":
            node_count = int(8 + difficulty * 12)
            branching_factor = 0.4 + difficulty * 0.2  # 相对线性
        elif level_type == "puzzle":
            node_count = int(6 + difficulty * 10)
            branching_factor = 0.5 + difficulty * 0.25
        else:  # balanced
            node_count = int(8 + difficulty * 12)
            branching_factor = 0.5 + difficulty * 0.25
            
        # 添加起点和终点
        G.add_node("start", type="start")
        G.add_node("end", type="end")
        
        # 添加中间节点
        for i in range(node_count):
            # 节点类型根据关卡类型和随机因素确定
            if level_type == "exploration":
                node_types = ["discovery", "vista", "treasure", "rest"]
                weights = [0.4, 0.3, 0.2, 0.1]
            elif level_type == "combat":
                node_types = ["combat", "elite", "rest", "treasure"]
                weights = [0.5, 0.2, 0.2, 0.1]
            elif level_type == "puzzle":
                node_types = ["puzzle", "clue", "treasure", "rest"]
                weights = [0.5, 0.3, 0.1, 0.1]
            else:  # balanced
                node_types = ["combat", "puzzle", "discovery", "rest", "treasure"]
                weights = [0.3, 0.2, 0.2, 0.2, 0.1]
                
            node_type = np.random.choice(node_types, p=weights)
            G.add_node(f"node_{i}", type=node_type)
            
        # 连接节点形成关卡进程图
        nodes = list(G.nodes())
        nodes.remove("start")
        nodes.remove("end")
        
        # 从起点连接到第一层节点
        first_layer = np.random.choice(nodes, size=max(1, int(len(nodes) * 0.2)), replace=False)
        for node in first_layer:
            G.add_edge("start", node)
            nodes.remove(node)
            
        # 处理中间层
        current_layer = list(first_layer)
        remaining = nodes.copy()
        
        while remaining:
            next_layer = []
            for node in current_layer:
                # 决定从这个节点出发的边数
                edge_count = np.random.binomial(len(remaining), branching_factor) if remaining else 0
                edge_count = min(edge_count, len(remaining))
                
                if edge_count > 0:
                    targets = np.random.choice(remaining, size=edge_count, replace=False)
                    for target in targets:
                        G.add_edge(node, target)
                        next_layer.append(target)
                        remaining.remove(target)
                        
            if not next_layer and remaining:
                # 防止图断开，确保所有节点都连接
                source = np.random.choice(current_layer)
                target = np.random.choice(remaining)
                G.add_edge(source, target)
                next_layer.append(target)
                remaining.remove(target)
                
            current_layer = next_layer
            
        # 连接到终点
        for node in current_layer:
            G.add_edge(node, "end")
            
        # 确保图是连通的
        if not nx.is_weakly_connected(G):
            self.logger.warning("生成的关卡图不连通，进行修复")
            # 添加必要的边使图连通
            components = list(nx.weakly_connected_components(G))
            main_component = max(components, key=len)
            
            for component in components:
                if component != main_component:
                    source = np.random.choice(list(main_component))
                    target = np.random.choice(list(component))
                    G.add_edge(source, target)
                    
        return G
    
    def _graph_place_key_nodes(self, suitable_regions):
        """在地图上放置关键节点"""
        if not self.node_graph:
            self.logger.error("节点图未初始化")
            return
            
        # 放置起点
        if suitable_regions["start_areas"]:
            start_area = suitable_regions["start_areas"][0]
            self.level_data["start_point"] = {
                "x": start_area["x"], 
                "y": start_area["y"],
                "type": "start",
                "description": "关卡起点",
                "connections": []
            }
        else:
            # 找不到合适区域，使用默认位置
            self.level_data["start_point"] = {
                "x": 50, 
                "y": 50,
                "type": "start",
                "description": "关卡起点",
                "connections": []
            }
            
        # 放置终点
        if suitable_regions["end_areas"]:
            end_area = suitable_regions["end_areas"][0]
            self.level_data["end_point"] = {
                "x": end_area["x"], 
                "y": end_area["y"],
                "type": "end",
                "description": "关卡终点",
                "connections": []
            }
        else:
            # 找不到合适区域，使用默认位置
            self.level_data["end_point"] = {
                "x": 450, 
                "y": 450,
                "type": "end",
                "description": "关卡终点",
                "connections": []
            }
            
        # 放置其他节点
        self.level_data["nodes"] = []
        
        # 为每个图中的节点在地图上找位置
        node_positions = {}
        for node in self.node_graph.nodes():
            if node == "start":
                node_positions[node] = (self.level_data["start_point"]["x"], self.level_data["start_point"]["y"])
            elif node == "end":
                node_positions[node] = (self.level_data["end_point"]["x"], self.level_data["end_point"]["y"])
            else:
                # 为其他节点找合适的位置
                node_data = self.node_graph.nodes[node]
                node_type = node_data.get("type", "generic")
                
                # 根据节点类型选择合适的区域
                if node_type in ["combat", "elite"]:
                    areas = suitable_regions.get("challenge_areas", [])
                elif node_type in ["treasure", "discovery"]:
                    areas = suitable_regions.get("reward_areas", [])
                else:
                    areas = suitable_regions.get("path_areas", [])
                    
                if areas:
                    # 从合适区域中随机选择并稍微偏移位置避免重叠
                    area = random.choice(areas)
                    x = area["x"] + random.randint(-10, 10)
                    y = area["y"] + random.randint(-10, 10)
                else:
                    # 没有合适区域，在地图上随机选择
                    x = random.randint(50, 450)
                    y = random.randint(50, 450)
                    
                node_positions[node] = (x, y)
                
                # 添加到关卡数据
                self.level_data["nodes"].append({
                    "id": node,
                    "x": x,
                    "y": y,
                    "type": node_type,
                    "description": f"{node_type.capitalize()} 节点",
                    "connections": []
                })
                
        # 添加边连接信息
        for u, v in self.node_graph.edges():
            # 找到对应的节点对象
            source = None
            target = None
            
            if u == "start":
                source = self.level_data["start_point"]
            elif u == "end":
                source = self.level_data["end_point"]
            else:
                for node in self.level_data["nodes"]:
                    if node["id"] == u:
                        source = node
                        break
                        
            if v == "start":
                target = self.level_data["start_point"]
            elif v == "end":
                target = self.level_data["end_point"]
            else:
                for node in self.level_data["nodes"]:
                    if node["id"] == v:
                        target = node
                        break
                        
            if source and target:
                # 添加连接信息
                source["connections"].append({
                    "target": v,
                    "x": target["x"],
                    "y": target["y"]
                })
    
    def _graph_design_paths(self):
        """设计关卡路径和连接"""
        # 创建所有节点之间的路径
        self.level_data["paths"] = []
        
        # 收集所有节点
        all_nodes = [self.level_data["start_point"], self.level_data["end_point"]] + self.level_data["nodes"]
        
        # 为每个连接创建路径
        for source in all_nodes:
            for connection in source["connections"]:
                # 找到目标节点
                target = None
                for node in all_nodes:
                    if node.get("id", "") == connection["target"] or (
                        node is self.level_data["start_point"] and connection["target"] == "start") or (
                        node is self.level_data["end_point"] and connection["target"] == "end"):
                        target = node
                        break
                
                if not target:
                    continue
                    
                # 创建简单的直线路径（实际应用中可以使用寻路算法）
                path_points = self._create_path_between(source, target)
                
                # 添加路径数据
                self.level_data["paths"].append({
                    "from": source.get("id", "start" if source is self.level_data["start_point"] else "end"),
                    "to": target.get("id", "start" if target is self.level_data["start_point"] else "end"),
                    "points": path_points,
                    "difficulty": random.uniform(0.3, 0.7)  # 简单随机难度
                })
                
    def _graph_place_challenges_and_rewards(self, difficulty):
        """放置挑战和奖励"""
        # 为每个节点添加相应的挑战和奖励
        for node in self.level_data["nodes"]:
            node_type = node.get("type", "generic")
            
            # 根据节点类型添加不同内容
            if node_type == "combat":
                # 添加战斗挑战
                node["challenge"] = self._create_combat_challenge(difficulty)
                node["reward"] = self._create_reward(difficulty * 0.8)
                
            elif node_type == "elite":
                # 添加精英挑战
                node["challenge"] = self._create_elite_challenge(difficulty * 1.3)
                node["reward"] = self._create_reward(difficulty * 1.2)
                
            elif node_type == "puzzle":
                # 添加解谜挑战
                node["challenge"] = self._create_puzzle_challenge(difficulty)
                node["reward"] = self._create_reward(difficulty)
                
            elif node_type == "discovery":
                # 添加探索区域
                node["discovery"] = {
                    "type": random.choice(["lore", "map", "secret"]),
                    "description": "一处值得探索的区域",
                    "value": random.uniform(0.5, 1.0)
                }
                node["reward"] = self._create_reward(difficulty * 0.5)
                
            elif node_type == "treasure":
                # 添加宝藏
                node["reward"] = self._create_treasure(difficulty * 1.5)
                
            elif node_type == "rest":
                # 添加休息点
                node["rest"] = {
                    "recovery": 0.3 + difficulty * 0.2,
                    "buffs": ["恢复", "休息"] if random.random() > 0.5 else ["恢复"]
                }
        
        # 为路径添加障碍和事件
        for path in self.level_data["paths"]:
            if random.random() < 0.3 + difficulty * 0.4:
                # 添加路径障碍
                path["obstacles"] = [
                    self._create_obstacle(difficulty)
                    for _ in range(random.randint(1, 3))
                ]
                
            if random.random() < 0.2 + difficulty * 0.3:
                # 添加路径事件
                path["event"] = self._create_event()
        
    def _graph_add_story_elements(self, level_type):
        """添加故事元素和任务"""
        # 基于关卡类型生成合适的故事元素
        if level_type == "exploration":
            story_theme = random.choice(["lost_civilization", "natural_wonder", "mysterious_ruins"])
        elif level_type == "combat":
            story_theme = random.choice(["invasion", "rebellion", "monster_hunt"])
        elif level_type == "puzzle":
            story_theme = random.choice(["ancient_trial", "magical_mystery", "forgotten_knowledge"])
        else:  # balanced
            story_theme = random.choice(["hero_journey", "rescue_mission", "treasure_hunt"])
            
        # 添加故事信息
        self.level_data["story"] = {
            "theme": story_theme,
            "title": f"The {story_theme.replace('_', ' ').title()}",
            "description": f"一个关于{story_theme.replace('_', ' ')}的冒险",
            "objective": {
                "primary": "到达终点",
                "secondary": random.choice(["收集所有宝藏", "击败所有精英敌人", "解开所有谜题"])
            }
        }
        
        # 添加故事点
        self.level_data["story_points"] = []
        
        # 在起点添加故事介绍
        self.level_data["story_points"].append({
            "location": "start",
            "type": "introduction",
            "text": f"你开始了{self.level_data['story']['title']}的旅程。{self.level_data['story']['description']}"
        })
        
        # 在关键节点添加故事进展
        key_nodes = []
        for node in self.level_data["nodes"]:
            if node.get("type") in ["elite", "discovery", "puzzle"]:
                key_nodes.append(node["id"])
                
        # 选择2-3个关键点
        if len(key_nodes) > 0:
            story_node_count = min(len(key_nodes), random.randint(2, 3))
            story_nodes = random.sample(key_nodes, story_node_count)
            
            for i, node_id in enumerate(story_nodes):
                self.level_data["story_points"].append({
                    "location": node_id,
                    "type": "development" if i < len(story_nodes) - 1 else "climax",
                    "text": f"故事发展点 {i+1}: 这里有关于{story_theme.replace('_', ' ')}的更多信息。"
                })
                
        # 在终点添加故事结论
        self.level_data["story_points"].append({
            "location": "end",
            "type": "conclusion",
            "text": f"你完成了{self.level_data['story']['title']}的旅程。"
        })
        
    def _graph_validate_level(self):
        """验证关卡的可玩性和平衡性"""
        # 检查起点到终点是否有路径
        if not self.node_graph:
            self.logger.warning("无法验证关卡结构，节点图未初始化")
            return
            
        import networkx as nx
        
        # 检查可达性
        if not nx.has_path(self.node_graph, "start", "end"):
            self.logger.error("验证失败: 从起点无法到达终点")
            # 紧急修复:添加直接连接
            self.node_graph.add_edge("start", "end")
            self.level_data["start_point"]["connections"].append({
                "target": "end",
                "x": self.level_data["end_point"]["x"],
                "y": self.level_data["end_point"]["y"]
            })
            
        # 检查奖励和挑战平衡
        total_challenge = 0
        total_reward = 0
        challenge_count = 0
        reward_count = 0
        
        for node in self.level_data["nodes"]:
            if "challenge" in node:
                total_challenge += node["challenge"].get("difficulty", 0.5)
                challenge_count += 1
                
            if "reward" in node:
                if isinstance(node["reward"], dict):
                    total_reward += node["reward"].get("value", 0.5)
                    reward_count += 1
                elif isinstance(node["reward"], list):
                    for reward in node["reward"]:
                        total_reward += reward.get("value", 0.5)
                        reward_count += 1
                        
        # 计算平均难度和奖励
        avg_challenge = total_challenge / max(1, challenge_count)
        avg_reward = total_reward / max(1, reward_count)
        
        # 记录平衡性分析
        self.level_data["balance_analysis"] = {
            "avg_challenge": avg_challenge,
            "avg_reward": avg_reward,
            "challenge_reward_ratio": avg_challenge / max(0.1, avg_reward),
            "total_nodes": len(self.level_data["nodes"]),
            "challenge_nodes": challenge_count,
            "reward_nodes": reward_count
        }
        
        # 如果比例严重失衡，进行修正
        ratio = self.level_data["balance_analysis"]["challenge_reward_ratio"]
        if ratio > 2.0:
            # 挑战太高或奖励太低
            self.logger.warning(f"关卡失衡: 挑战/奖励比例为 {ratio:.2f} (过高)")
            self._graph_rebalance_level("increase_rewards")
        elif ratio < 0.5:
            # 挑战太低或奖励太高
            self.logger.warning(f"关卡失衡: 挑战/奖励比例为 {ratio:.2f} (过低)")
            self._graph_rebalance_level("increase_challenges")
        else:
            self.logger.info(f"关卡平衡度良好: 挑战/奖励比例为 {ratio:.2f}")
        
    def _graph_rebalance_level(self, strategy):
        """重新平衡关卡"""
        if strategy == "increase_rewards":
            # 增加奖励
            for node in self.level_data["nodes"]:
                if "reward" in node:
                    if isinstance(node["reward"], dict):
                        node["reward"]["value"] = node["reward"].get("value", 0.5) * 1.5
                    elif isinstance(node["reward"], list):
                        for reward in node["reward"]:
                            reward["value"] = reward.get("value", 0.5) * 1.5
        else:  # increase_challenges
            # 增加挑战
            for node in self.level_data["nodes"]:
                if "challenge" in node:
                    node["challenge"]["difficulty"] = node["challenge"].get("difficulty", 0.5) * 1.5
    
    #######################################################################
    # 空间策略的实现 (来自level_generator.py)
    #######################################################################
    
    def _generate_spatial_level(self, level_type="balanced", difficulty=0.5):
        """使用空间布局策略生成关卡"""
        self.logger.info(f"使用空间策略生成{level_type}类型关卡，难度:{difficulty}")
        
        # 初始化关卡数据
        level_data = {
            "level_type": level_type,
            "difficulty": difficulty,
            "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "generator_version": self.version,
            "nodes": [],
            "paths": []
        }
        
        # 获取关卡类型配置
        config = self.level_type_configs.get(level_type, self.level_type_configs["balanced"])
        
        # 确定节点数量（根据难度调整）
        min_nodes, max_nodes = config["node_count_range"]
        node_count = int(min_nodes + (max_nodes - min_nodes) * difficulty)
        
        # 调整节点权重
        node_weights = self.node_weights.copy()
        if "node_weights" in config:
            for key, value in config["node_weights"].items():
                node_weights[key] = value
                
        # 确定路径数量（根据参数和难度）
        path_count = config["path_count"]
        if self.params.get("multi_path", True):
            path_count = max(1, int(path_count * (0.8 + difficulty * 0.4)))
        else:
            path_count = 1
            
        # 路径复杂度
        path_complexity = self.params.get("path_complexity", 0.5)
        
        # 确定起点和终点
        if self.params.get("auto_endpoints", True):
            start_point, end_point = self._spatial_find_suitable_endpoints(difficulty)
        else:
            # 使用指定或默认位置
            start_point = {"x": self.params.get("start_x", 50), "y": self.params.get("start_y", 50)}
            end_point = {"x": self.params.get("end_x", 450), "y": self.params.get("end_y", 450)}
        
        # 设置起点和终点类型
        start_point["type"] = "start"
        end_point["type"] = "end"
        level_data["start_point"] = start_point
        level_data["end_point"] = end_point
        
        # 生成主路径节点
        main_path_nodes = self._spatial_generate_path_nodes(
            start_point, end_point, 
            int(node_count * 0.7), path_complexity,
            difficulty
        )
        
        # 生成分支路径
        branch_nodes = []
        remaining_nodes = node_count - len(main_path_nodes)
        if remaining_nodes > 0 and self.params.get("multi_path", True):
            # 实现分支路径生成
            branch_sources = random.sample(main_path_nodes, min(len(main_path_nodes), remaining_nodes // 2))
            
            for i, source in enumerate(branch_sources):
                # 为每个分支源创建1-3个分支节点
                branch_count = min(remaining_nodes, random.randint(1, 3))
                remaining_nodes -= branch_count
                
                # 创建分支节点，从源节点向周围扩散
                for j in range(branch_count):
                    # 计算分支节点位置(从源节点向周围随机方向扩散)
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(30, 80)
                    
                    x = source["x"] + distance * math.cos(angle)
                    y = source["y"] + distance * math.sin(angle)
                    
                    # 确保坐标在地图范围内
                    x = max(0, min(self.width - 1, x))
                    y = max(0, min(self.height - 1, y))
                    
                    # 创建分支节点
                    branch_node = {
                        "x": int(x),
                        "y": int(y),
                        "type": "generic",  # 初始类型，后续更新
                        "branch_from": main_path_nodes.index(source)  # 记录分支源
                    }
                    
                    # 避免节点重叠
                    if not self._is_node_overlapping(branch_node, main_path_nodes + branch_nodes + [start_point, end_point]):
                        branch_nodes.append(branch_node)
                
                if remaining_nodes <= 0:
                    break
        
        # 合并所有节点
        all_nodes = main_path_nodes + branch_nodes
        
        # 根据敌人密度和奖励多样性分配节点类型
        self._spatial_assign_node_types(
            all_nodes, node_weights, 
            self.params.get("enemy_density", 0.5), 
            self.params.get("reward_variety", 0.5)
        )
        
        # 将节点添加到关卡数据
        level_data["nodes"] = all_nodes
        
        # 生成路径连接
        paths = self._spatial_generate_paths(start_point, end_point, all_nodes, difficulty)
        level_data["paths"] = paths
        
        if self.logger:
            self.logger.info(f"空间策略生成完成，共{len(all_nodes)}个节点，{len(paths)}条路径")
        
        return level_data
    
    # 辅助方法
    def _is_node_overlapping(self, node, other_nodes, min_distance=10):
        """检查节点是否与其他节点重叠"""
        for other in other_nodes:
            dist = math.sqrt((node["x"] - other["x"])**2 + (node["y"] - other["y"])**2)
            if dist < min_distance:
                return True
        return False
    
    def _create_path_points(self, start, end, difficulty):
        """创建两个节点之间的路径点"""
        points = []
        
        # 计算起点和终点之间的距离
        distance = math.sqrt((end["x"] - start["x"])**2 + (end["y"] - start["y"])**2)
        
        # 决定路径点数量（根据距离和难度）
        point_count = max(2, int(distance / 50))
        
        # 创建路径点
        for i in range(point_count - 1):
            progress = (i + 1) / point_count
            
            # 基本位置（直线插值）
            x = start["x"] + (end["x"] - start["x"]) * progress
            y = start["y"] + (end["y"] - start["y"]) * progress
            
            # 添加随机偏移，难度越高偏移越小（更直接的路径）
            max_offset = distance * 0.2 * (1 - difficulty * 0.5)
            x += random.uniform(-max_offset, max_offset)
            y += random.uniform(-max_offset, max_offset)
            
            # 确保坐标在地图范围内
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            
            points.append({"x": int(x), "y": int(y)})
        
        return points
    
    def _create_path_between(self, source, target):
        """在两个节点之间创建路径"""
        # 获取起点和终点坐标
        start_x, start_y = source["x"], source["y"]
        end_x, end_y = target["x"], target["y"]
        
        # 计算中间点数量（基于距离）
        dist = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        point_count = max(1, int(dist / 50))  # 每50个单位添加一个中间点
        
        path_points = []
        
        # 添加起点
        path_points.append({"x": start_x, "y": start_y})
        
        # 添加中间点
        for i in range(1, point_count):
            # 插值计算基础位置
            t = i / point_count
            base_x = start_x + (end_x - start_x) * t
            base_y = start_y + (end_y - start_y) * t
            
            # 添加随机偏移以使路径不是直线
            offset = dist * 0.1  # 最大偏移为距离的10%
            rand_x = random.uniform(-offset, offset)
            rand_y = random.uniform(-offset, offset)
            
            path_points.append({
                "x": int(base_x + rand_x),
                "y": int(base_y + rand_y)
            })
        
        # 添加终点
        path_points.append({"x": end_x, "y": end_y})
        
        return path_points
    
    # 添加原level_generator.py中的所有方法，添加_spatial_前缀
    def _spatial_find_suitable_endpoints(self, difficulty):
        """寻找适合的起点和终点位置
        
        根据地形高度和生物群系寻找合适的起点和终点
        
        Args:
            difficulty: 难度系数，影响起点和终点的选择

        Returns:
            (start_point, end_point) 元组
        """
        # 对于低难度，寻找平坦的区域
        # 对于高难度，可以考虑高山或特殊地形
        
        height_map = self.heights
        
        # 计算地形梯度，用于找到平坦区域
        from scipy.ndimage import sobel
        gradient_x = sobel(height_map, axis=1)
        gradient_y = sobel(height_map, axis=0)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 根据难度调整坡度阈值
        slope_threshold = 0.5 - difficulty * 0.3  # 难度越高，坡度阈值越低
        
        # 寻找平坦区域（梯度低的区域）
        flat_areas = gradient_magnitude < slope_threshold
        
        # 将地图分为四个象限，在左侧找起点，右侧找终点
        half_width = self.width // 2
        half_height = self.height // 2
        
        # 左侧区域
        left_flat = flat_areas[:, :half_width]
        left_y, left_x = np.where(left_flat)
        
        # 右侧区域
        right_flat = flat_areas[:, half_width:]
        right_y, right_x = np.where(right_flat)
        right_x += half_width  # 调整右侧x坐标
        
        # 如果找不到平坦区域，则使用随机位置
        if len(left_y) == 0 or len(right_y) == 0:
            start_x = random.randint(0, half_width - 1)
            start_y = random.randint(0, self.height - 1)
            end_x = random.randint(half_width, self.width - 1)
            end_y = random.randint(0, self.height - 1)
        else:
            # 随机选择起点和终点
            left_idx = random.randint(0, len(left_y) - 1)
            right_idx = random.randint(0, len(right_y) - 1)
            
            start_x = left_x[left_idx]
            start_y = left_y[left_idx]
            end_x = right_x[right_idx]
            end_y = right_y[right_idx]
        
        # 创建起点和终点
        start_point = {"x": int(start_x), "y": int(start_y)}
        end_point = {"x": int(end_x), "y": int(end_y)}
        
        return start_point, end_point
    
    def _spatial_generate_path_nodes(self, start, end, count, complexity, difficulty):
        """在起点和终点之间生成路径节点
        
        Args:
            start: 起点坐标
            end: 终点坐标
            count: 节点数量
            complexity: 路径复杂度
            difficulty: 难度
            
        Returns:
            节点列表
        """
        nodes = []
        
        # 计算起点和终点之间的距离
        distance = math.sqrt((end["x"] - start["x"])**2 + (end["y"] - start["y"])**2)
        
        # 沿着起点到终点的方向生成节点
        for i in range(count):
            # 计算节点位置
            progress = (i + 1) / (count + 1)  # 节点在路径上的进度 (0-1)
            
            # 基本位置（直线插值）
            base_x = start["x"] + (end["x"] - start["x"]) * progress
            base_y = start["y"] + (end["y"] - start["y"]) * progress
            
            # 根据复杂度添加随机偏移
            max_offset = distance * complexity * 0.2  # 最大偏移量
            offset_x = random.uniform(-max_offset, max_offset)
            offset_y = random.uniform(-max_offset, max_offset)
            
            # 应用偏移
            x = base_x + offset_x
            y = base_y + offset_y
            
            # 确保坐标在地图范围内
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            
            # 创建节点 (不分配类型，后续统一处理)
            node = {
                "x": int(x),
                "y": int(y),
                "type": "generic"  # 初始类型，后续会更新
            }
            
            # 避免节点重叠
            if not self._is_node_overlapping(node, nodes + [start, end], min_distance=5):
                nodes.append(node)
        
        return nodes
    
    def _spatial_assign_node_types(self, nodes, weights, enemy_density, reward_variety):
        """为节点分配类型
        
        Args:
            nodes: 节点列表
            weights: 节点类型权重
            enemy_density: 敌人密度
            reward_variety: 奖励多样性
        """
        # 调整权重
        adjusted_weights = weights.copy()
        
        # 根据敌人密度调整战斗和精英节点的比例
        adjusted_weights["combat"] *= enemy_density
        adjusted_weights["elite"] *= enemy_density
        
        # 根据奖励多样性调整宝藏和发现节点的比例
        adjusted_weights["treasure"] *= reward_variety
        adjusted_weights["discovery"] *= reward_variety
        
        # 确保休息点分布均匀
        rest_count = max(1, int(len(nodes) * 0.15))  # 至少一个休息点
        
        # 休息点位置 (均匀分布)
        rest_indices = []
        if len(nodes) > 0:
            step = len(nodes) / (rest_count + 1)
            for i in range(rest_count):
                index = min(int((i + 1) * step), len(nodes) - 1)
                rest_indices.append(index)
        
        # 为每个节点分配类型
        for i, node in enumerate(nodes):
            if i in rest_indices:
                node["type"] = "rest"
            else:
                # 根据权重随机选择节点类型
                types = list(adjusted_weights.keys())
                weights_list = [adjusted_weights[t] for t in types]
                
                # 归一化权重
                total = sum(weights_list)
                if total > 0:
                    weights_list = [w/total for w in weights_list]
                    node_type = random.choices(types, weights=weights_list, k=1)[0]
                    node["type"] = node_type
                else:
                    # 默认为战斗节点
                    node["type"] = "combat"
    
    def _spatial_generate_paths(self, start, end, nodes, difficulty):
        """生成连接节点的路径
        
        Args:
            start: 起点
            end: 终点
            nodes: 节点列表
            difficulty: 难度
            
        Returns:
            路径列表
        """
        paths = []
        
        # 将起点和终点添加到节点列表（仅用于路径生成）
        all_nodes = [start] + nodes + [end]
        
        # 构建路径连接
        for i, node in enumerate(all_nodes[:-1]):
            # 寻找下一个节点
            next_idx = i + 1
            
            # 如果是分支节点，连接到分支源
            if "branch_from" in node:
                branch_from = node["branch_from"] + 1  # +1是因为all_nodes包含start
                
                # 创建分支路径
                path = {
                    "from": i,
                    "to": branch_from,
                    "difficulty": min(0.9, difficulty + random.uniform(-0.2, 0.2)),
                    "points": self._create_path_points(all_nodes[i], all_nodes[branch_from], difficulty)
                }
                paths.append(path)
                continue
            
            # 创建到下一个节点的路径
            path = {
                "from": i,
                "to": next_idx,
                "difficulty": min(0.9, difficulty + random.uniform(-0.2, 0.2)),
                "points": self._create_path_points(node, all_nodes[next_idx], difficulty)
            }
            paths.append(path)
            
            # 为了增加复杂度，可能添加一些额外连接
            if self.params["multi_path"] and random.random() < self.params["branch_factor"] * 0.3:
                # 查找可能的额外连接目标 (前方的节点)
                for j in range(next_idx + 1, min(next_idx + 3, len(all_nodes))):
                    path = {
                        "from": i,
                        "to": j,
                        "difficulty": min(0.95, difficulty + 0.15 + random.uniform(-0.1, 0.1)),
                        "points": self._create_path_points(node, all_nodes[j], difficulty)
                    }
                    paths.append(path)
                    break
        
        return paths
    
    def _create_combat_challenge(self, difficulty):
        """创建战斗挑战
        
        Args:
            difficulty: 难度系数 (0.0-1.0)
            
        Returns:
            战斗挑战的数据字典
        """
        # 根据难度确定敌人数量和强度
        enemy_count = max(1, int(1 + difficulty * 3))
        enemy_strength = 0.3 + difficulty * 0.7
        
        challenge = {
            "type": "combat",
            "difficulty": difficulty,
            "enemies": []
        }
        
        # 决定敌人类型
        enemy_types = ["minion", "ranged", "elite", "boss"]
        enemy_weights = [0.7 - difficulty * 0.3, 0.2, difficulty * 0.5, difficulty * 0.1]
        
        # 确保权重和为1
        total_weight = sum(enemy_weights)
        enemy_weights = [w / total_weight for w in enemy_weights]
        
        # 生成敌人列表
        for i in range(enemy_count):
            enemy_type = random.choices(enemy_types, weights=enemy_weights, k=1)[0]
            
            # 根据难度和类型确定敌人属性
            enemy = {
                "type": enemy_type,
                "level": max(1, int(difficulty * 10)),
                "strength": enemy_strength * (1.5 if enemy_type == "elite" else 
                                            2.0 if enemy_type == "boss" else 1.0),
                "abilities": []
            }
            
            # 添加敌人能力
            ability_count = random.randint(0, 2 if difficulty > 0.5 else 1)
            abilities = ["quick", "strong", "armored", "regeneration", "ranged"]
            
            if ability_count > 0:
                enemy["abilities"] = random.sample(abilities, ability_count)
            
            challenge["enemies"].append(enemy)
        
        return challenge

    def _create_elite_challenge(self, difficulty):
        """创建精英挑战
        
        Args:
            difficulty: 难度系数 (0.0-1.0)
            
        Returns:
            精英挑战的数据字典
        """
        # 精英挑战通常是高难度战斗或特殊机制
        challenge = self._create_combat_challenge(min(1.0, difficulty * 1.3))
        challenge["type"] = "elite"
        
        # 添加精英特性
        elite_traits = ["高伤害", "高防御", "特殊能力", "环境优势"]
        trait_count = min(len(elite_traits), 1 + int(difficulty * 2))
        challenge["elite_traits"] = random.sample(elite_traits, trait_count)
        
        # 如果难度足够高，添加特殊机制
        if difficulty > 0.6:
            special_mechanics = ["阶段性", "环境伤害", "召唤增援", "状态效果"]
            mechanic = random.choice(special_mechanics)
            challenge["special_mechanic"] = {
                "type": mechanic,
                "description": f"精英特殊机制: {mechanic}"
            }
        
        return challenge

    def _create_puzzle_challenge(self, difficulty):
        """创建解谜挑战
        
        Args:
            difficulty: 难度系数 (0.0-1.0)
            
        Returns:
            解谜挑战的数据字典
        """
        # 难度决定谜题复杂度和解决时间
        puzzle_types = ["sequence", "pattern", "logic", "riddle", "mechanical"]
        complexity = 0.3 + difficulty * 0.7
        
        puzzle = {
            "type": "puzzle",
            "puzzle_type": random.choice(puzzle_types),
            "difficulty": difficulty,
            "complexity": complexity,
            "estimated_time": int(30 + difficulty * 90),  # 秒
            "hints_available": max(0, 3 - int(difficulty * 3))
        }
        
        # 添加解谜步骤
        step_count = max(1, int(1 + difficulty * 4))
        puzzle["steps"] = []
        
        for i in range(step_count):
            step = {
                "step": i + 1,
                "description": f"谜题步骤 {i+1}",
                "difficulty": 0.2 + difficulty * 0.8 * ((i+1) / step_count)  # 难度递增
            }
            puzzle["steps"].append(step)
        
        return puzzle

    def _create_reward(self, difficulty):
        """创建标准奖励
        
        Args:
            difficulty: 难度系数，影响奖励质量和数量
            
        Returns:
            奖励数据字典或奖励列表
        """
        # 根据难度决定奖励数量
        reward_count = max(1, int(1 + difficulty * 2))
        
        # 如果只有一个奖励，返回单个奖励对象
        if reward_count == 1:
            return self._create_single_reward(difficulty)
        
        # 否则返回奖励列表
        rewards = []
        for i in range(reward_count):
            # 分配难度以确保总价值符合预期
            item_difficulty = difficulty * 0.8 + random.uniform(-0.2, 0.2)
            item_difficulty = max(0.1, min(1.0, item_difficulty))
            rewards.append(self._create_single_reward(item_difficulty))
        
        return rewards

    def _create_single_reward(self, difficulty):
        """创建单个奖励
        
        Args:
            difficulty: 难度系数 (0.0-1.0)
            
        Returns:
            单个奖励的数据字典
        """
        # 根据难度决定奖励类型
        reward_types = ["currency", "item", "resource", "experience", "skill"]
        weights = [
            0.5 - difficulty * 0.3,  # 低难度更多货币
            0.2 + difficulty * 0.3,  # 高难度更多物品
            0.2,
            0.1 + difficulty * 0.1,
            difficulty * 0.2
        ]
        
        # 归一化权重
        total = sum(weights)
        weights = [w/total for w in weights]
        
        reward_type = random.choices(reward_types, weights=weights, k=1)[0]
        
        # 根据难度计算奖励价值(0.1-1.0)
        value = 0.3 + difficulty * 0.7 + random.uniform(-0.1, 0.1)
        value = max(0.1, min(1.0, value))
        
        # 创建基本奖励数据
        reward = {
            "type": reward_type,
            "value": value
        }
        
        # 根据奖励类型添加特定属性
        if reward_type == "currency":
            reward["amount"] = int(10 + value * 90)
            reward["currency_type"] = random.choice(["gold", "gems", "tokens"])
            
        elif reward_type == "item":
            rarity_types = ["common", "uncommon", "rare", "epic", "legendary"]
            rarity_idx = min(len(rarity_types) - 1, int(value * len(rarity_types)))
            reward["rarity"] = rarity_types[rarity_idx]
            reward["item_type"] = random.choice(["weapon", "armor", "accessory", "consumable"])
            
        elif reward_type == "resource":
            reward["amount"] = int(5 + value * 20)
            reward["resource_type"] = random.choice(["wood", "stone", "metal", "crystal", "herb"])
            
        elif reward_type == "experience":
            reward["amount"] = int(50 + value * 200)
            
        elif reward_type == "skill":
            reward["skill_type"] = random.choice(["attack", "defense", "utility", "movement"])
            reward["power_level"] = value
        
        return reward

    def _create_treasure(self, difficulty):
        """创建宝藏（高价值奖励）
        
        Args:
            difficulty: 难度系数 (0.0-1.0)
            
        Returns:
            宝藏数据的列表
        """
        # 宝藏通常包含多个高质量奖励
        treasure_count = max(2, int(2 + difficulty * 3))
        
        # 创建宝藏列表，提升平均价值
        treasures = []
        for i in range(treasure_count):
            # 宝藏价值应该显著高于标准奖励
            item_difficulty = min(1.0, difficulty * 1.2 + random.uniform(0, 0.3))
            reward = self._create_single_reward(item_difficulty)
            
            # 增加稀有度
            if "rarity" in reward:
                rarities = ["common", "uncommon", "rare", "epic", "legendary"]
                current_idx = rarities.index(reward["rarity"])
                new_idx = min(len(rarities) - 1, current_idx + 1)
                reward["rarity"] = rarities[new_idx]
                
            # 增加数量
            if "amount" in reward:
                reward["amount"] = int(reward["amount"] * 1.5)
                
            treasures.append(reward)
        
        return treasures

    def _create_obstacle(self, difficulty):
        """创建路径障碍
        
        Args:
            difficulty: 难度系数 (0.0-1.0)
            
        Returns:
            障碍数据字典
        """
        obstacle_types = ["physical", "trap", "puzzle", "environmental"]
        weights = [0.4, 0.3, 0.2, 0.1]
        
        obstacle_type = random.choices(obstacle_types, weights=weights, k=1)[0]
        
        # 基本障碍数据
        obstacle = {
            "type": obstacle_type,
            "difficulty": difficulty,
            "description": f"{obstacle_type.capitalize()} 类型障碍"
        }
        
        # 根据类型添加特定属性
        if obstacle_type == "physical":
            obstacle["strength"] = 0.3 + difficulty * 0.7
            obstacle["clearance_methods"] = random.sample(
                ["strength", "agility", "tool", "magic"], 
                k=max(1, int(1 + difficulty))
            )
            
        elif obstacle_type == "trap":
            obstacle["detection_difficulty"] = 0.2 + difficulty * 0.8
            obstacle["damage_potential"] = 0.3 + difficulty * 0.7
            obstacle["avoidance_methods"] = random.sample(
                ["perception", "agility", "tool", "magic"], 
                k=max(1, int(1 + difficulty))
            )
            
        elif obstacle_type == "puzzle":
            obstacle["complexity"] = 0.2 + difficulty * 0.8
            obstacle["solution_steps"] = max(1, int(1 + difficulty * 3))
            
        elif obstacle_type == "environmental":
            obstacle["effect"] = random.choice(["slow", "damage", "confusion", "fear"])
            obstacle["intensity"] = 0.3 + difficulty * 0.7
        
        return obstacle

    def _create_event(self):
        """创建随机事件
        
        Returns:
            事件数据字典
        """
        event_types = ["discovery", "encounter", "choice", "blessing", "curse"]
        event_type = random.choice(event_types)
        
        event = {
            "type": event_type,
            "title": f"{event_type.capitalize()} 事件",
            "description": f"一个 {event_type} 类型的随机事件"
        }
        
        # 根据事件类型添加特定属性
        if event_type == "discovery":
            event["discovery_type"] = random.choice(["item", "lore", "secret", "resource"])
            event["value"] = random.uniform(0.2, 0.8)
            
        elif event_type == "encounter":
            encounter_types = ["friendly", "neutral", "hostile"]
            weights = [0.3, 0.4, 0.3]
            event["encounter_type"] = random.choices(encounter_types, weights=weights, k=1)[0]
            event["npc_type"] = random.choice(["merchant", "traveler", "creature", "spirit", "local"])
            
        elif event_type == "choice":
            event["options"] = []
            option_count = random.randint(2, 3)
            
            for i in range(option_count):
                option = {
                    "text": f"选项 {i+1}",
                    "outcome_type": random.choice(["reward", "penalty", "mixed", "neutral"]),
                    "risk_level": random.uniform(0, 1.0)
                }
                event["options"].append(option)
                
        elif event_type == "blessing" or event_type == "curse":
            effect_types = ["stat", "ability", "restriction", "requirement"]
            event["effect_type"] = random.choice(effect_types)
            event["duration"] = random.choice(["temporary", "permanent"])
            event["strength"] = random.uniform(0.3, 0.8)
        
        return event