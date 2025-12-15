import os
import random
import math
import pickle
import logging
import numpy as np
from collections import defaultdict

class BiomePatternLibrary:
    """生物群系模式库，存储预设和自适应生成的生态模式"""

    def __init__(self, patterns_file="biome_patterns.pkl"):
        self.patterns = {}  # 模式集合
        self.usage_stats = defaultdict(int)  # 使用统计
        self.quality_scores = {}  # 质量评分
        self.pattern_file = patterns_file
        self.load_patterns()
        
    def load_patterns(self):
        """加载已存储的生物群系模式"""
        if os.path.exists(self.pattern_file):
            try:
                with open(self.pattern_file, 'rb') as f:
                    data = pickle.load(f)
                    self.patterns = data.get('patterns', {})
                    self.usage_stats = data.get('usage_stats', defaultdict(int))
                    self.quality_scores = data.get('quality_scores', {})
                    logging.info(f"加载了 {len(self.patterns)} 个生物群系模式")
            except Exception as e:
                logging.error(f"加载生物群系模式出错: {e}")
                self._initialize_default_patterns()
        else:
            self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """初始化默认生物群系模式"""
        # 预定义的高质量模式
        self.patterns = {
            "mountain_range": {
                "core": "Mountain",
                "transition_layers": [
                    {"biome": "Taiga", "weight": 0.7},
                    {"biome": "Forest", "weight": 0.5},
                    {"biome": "Plains", "weight": 0.3}
                ],
                "shape": "linear",
                "params": {"length": 10, "width": 5}
            },
            "desert_oasis": {
                "core": "Desert",
                "hotspots": [{"biome": "Grassland", "radius": 2}],
                "shape": "radial",
                "params": {"radius": 7}
            },
            "forest_patches": {
                "core": "Forest",
                "shape": "cluster",
                "params": {"count": 3, "size": 5}
            },
            "coastal_region": {
                "core": "Ocean",
                "transition_layers": [
                    {"biome": "Coast", "weight": 0.9},
                    {"biome": "Beach", "weight": 0.7},
                    {"biome": "Plains", "weight": 0.4}
                ],
                "shape": "coastline",
                "params": {"smoothness": 0.3}
            },
            "savanna_plain": {
                "core": "Savanna",
                "hotspots": [{"biome": "Jungle", "radius": 3}],
                "shape": "plain",
                "params": {"size": 8}
            }
        }
        
        # 初始评分
        for pattern in self.patterns:
            self.quality_scores[pattern] = 0.7  # 初始评分较高
    
    def save_patterns(self):
        """保存生物群系模式到文件"""
        try:
            with open(self.pattern_file, 'wb') as f:
                pickle.dump({
                    'patterns': self.patterns,
                    'usage_stats': dict(self.usage_stats),
                    'quality_scores': self.quality_scores
                }, f)
            logging.info(f"保存了 {len(self.patterns)} 个生物群系模式")
        except Exception as e:
            logging.error(f"保存生物群系模式出错: {e}")
    
    def select_weighted_pattern(self, exploration_factor=0.3):
        """根据质量和多样性需求选择模式"""
        if not self.patterns:
            return None
            
        patterns = list(self.patterns.keys())
        
        # 计算选择权重
        weights = []
        for name in patterns:
            # 基于质量和使用频率的权重
            quality = self.quality_scores.get(name, 0.5)
            usage = math.log(self.usage_stats.get(name, 0) + 1)
            
            # 探索项：减少过度使用的模式权重
            exploration = 1.0 / (usage + 1)
            
            # 综合权重
            weight = quality * (1 - exploration_factor) + exploration * exploration_factor
            weights.append(max(0.1, weight))  # 保证最小权重
        
        # 归一化权重
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
                
        # 随机选择
        selected = random.choices(patterns, weights=weights, k=1)[0]
        # 确保键存在，即使使用了defaultdict也添加这个检查以增加代码健壮性
        if selected not in self.usage_stats:
            self.usage_stats[selected] = 0
        self.usage_stats[selected] += 1
        return selected

    def generate_composite_pattern(self, w, h, complexity=0.5):
        """生成复合模式，组合多个基本模式"""
        composite = {}
        
        # 决定使用的模式数量
        pattern_count = max(1, int(3 * complexity))
        
        # 选择要使用的模式
        selected_patterns = []
        for _ in range(pattern_count):
            pattern_name = self.select_weighted_pattern(exploration_factor=0.2)
            if pattern_name and self.get_pattern(pattern_name):
                selected_patterns.append((pattern_name, self.get_pattern(pattern_name)))
        
        # 随机分配每个模式的影响区域
        regions = []
        for i, (name, pattern) in enumerate(selected_patterns):
            # 确定影响中心点
            cx = random.randint(0, w-1)
            cy = random.randint(0, h-1)
            
            # 根据形状确定影响区域
            shape = pattern.get('shape', 'radial')
            if shape == 'radial':
                radius = pattern.get('params', {}).get('radius', 5)
                radius = max(3, min(radius, min(w, h) // 3))
                regions.append({
                    'name': name,
                    'pattern': pattern,
                    'center': (cx, cy),
                    'shape': shape,
                    'radius': radius
                })
            elif shape == 'linear':
                length = pattern.get('params', {}).get('length', 7)
                width = pattern.get('params', {}).get('width', 3)
                angle = random.uniform(0, 2*math.pi)
                regions.append({
                    'name': name,
                    'pattern': pattern,
                    'center': (cx, cy),
                    'shape': shape,
                    'length': length,
                    'width': width,
                    'angle': angle
                })
            elif shape == 'cluster':
                count = pattern.get('params', {}).get('count', 3)
                size = pattern.get('params', {}).get('size', 4)
                clusters = []
                for _ in range(count):
                    cluster_x = min(w-1, max(0, cx + random.randint(-10, 10)))
                    cluster_y = min(h-1, max(0, cy + random.randint(-10, 10)))
                    clusters.append((cluster_x, cluster_y))
                regions.append({
                    'name': name,
                    'pattern': pattern,
                    'clusters': clusters,
                    'shape': shape,
                    'size': size
                })
        
        composite['regions'] = regions
        return composite

    def get_pattern(self, name):
        """获取特定模式"""
        return self.patterns.get(name)
        
    def add_pattern(self, name, pattern, initial_score=0.5):
        """添加新模式到库中"""
        self.patterns[name] = pattern
        self.quality_scores[name] = initial_score
        
    def update_pattern_quality(self, pattern_name, score):
        """更新模式质量评分"""
        if pattern_name in self.quality_scores:
            # 指数移动平均更新
            alpha = 0.3  # 新评分权重
            old_score = self.quality_scores[pattern_name]
            new_score = old_score * (1-alpha) + score * alpha
            self.quality_scores[pattern_name] = new_score
            return True
        return False

class UserPreferenceModel:
    """用户偏好建模系统，通过机器学习捕获用户评价行为"""
    
    def __init__(self, feature_dim=10, learning_rate=0.01):
        # 用户偏好模型参数
        self.weights = np.random.normal(0, 0.1, feature_dim)
        self.bias = 0.0
        self.lr = learning_rate
        self.history = []  # 评分历史
        self.feature_cache = {}  # 缓存提取的特征
        
    def extract_features(self, biome_map):
        """从生物群落图中提取特征向量"""
        # 为提高性能，检查缓存
        map_hash = hash(str(biome_map))
        if map_hash in self.feature_cache:
            return self.feature_cache[map_hash]
            
        h, w = biome_map.shape
        
        # 生物群系统计
        biome_counts = defaultdict(int)
        for y in range(h):
            for x in range(w):
                biome_name = biome_map[y, x]
                biome_counts[biome_name] += 1
        
        # 计算生物群系比例
        total_cells = w * h
        features = []
        
        # 主要生物群系比例
        key_biomes = ["Forest", "Grassland", "Desert", "Mountain", "Ocean", 
                       "Jungle", "Taiga", "Tundra", "Savanna", "Plains"]
        for biome in key_biomes:
            features.append(biome_counts.get(biome, 0) / total_cells)
        
        # 生物多样性指标
        if total_cells > 0:
            diversity = len(biome_counts) / len(key_biomes)
            features.append(diversity)
        else:
            features.append(0)
        
        # 确保维度一致
        while len(features) < len(self.weights):
            features.append(0.0)
        features = features[:len(self.weights)]
        
        # 保存到缓存
        feature_array = np.array(features)
        self.feature_cache[map_hash] = feature_array
        return feature_array
    
    def predict_score(self, biome_map):
        """预测用户对给定生物群落图的评分"""
        features = self.extract_features(biome_map)
        raw_score = np.dot(features, self.weights) + self.bias
        
        # 归一化到0-10范围
        norm_score = min(10, max(0, 5 + raw_score * 2.5))
        return norm_score
    
    def update_model(self, biome_map, user_score):
        """根据用户评分更新模型"""
        features = self.extract_features(biome_map)
        predicted = np.dot(features, self.weights) + self.bias
        
        # 归一化评分到模型范围
        normalized_score = (user_score / 10.0) * 4 - 2  # 映射到-2到2范围
        
        # 计算损失
        error = normalized_score - predicted
        
        # 梯度下降更新
        self.weights += self.lr * error * features
        self.bias += self.lr * error
        
        # 记录历史
        self.history.append({
            'features': features,
            'predicted': predicted,
            'actual': normalized_score,
            'error': error
        })
        
        return error
    
    def save_model(self, path="user_preference_model.pkl"):
        """保存模型到文件"""
        data = {
            'weights': self.weights,
            'bias': self.bias,
            'history_size': len(self.history),
            'version': 1.0
        }
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logging.error(f"保存用户偏好模型失败: {e}")
            return False
    
    def load_model(self, path="user_preference_model.pkl"):
        """从文件加载模型"""
        if not os.path.exists(path):
            return False
            
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.weights = data['weights']
                self.bias = data['bias']
                return True
        except Exception as e:
            logging.error(f"加载用户偏好模型失败: {e}")
            return False