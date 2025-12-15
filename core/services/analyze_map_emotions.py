#from __future__ import annotations
#标准库
import os
import json
import time
import warnings
import traceback
import collections.abc
import numpy as np
import joblib
from typing import Dict, Optional, List, Tuple

#数据处理与科学计算
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.ndimage import gaussian_filter, sobel, binary_dilation, label
from dataclasses import dataclass
import joblib
import traceback
import collections.abc

#图形界面与绘图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']  # 中文优先
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#网络与并发
from concurrent.futures import ProcessPoolExecutor

#其他工具
from dataclasses import dataclass, asdict, field
import warnings
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable


#####################################################
# 以下为情感模型集成的代码
#####################################################

#####################################################
# 高级情感模型集成系统
#####################################################

# 配置
EMOTION_MODEL_CONFIG = {
    "version": "2.0",
    "model_path": "./data/models/emotion_prediction_model.joblib",
    "cache_path": "./data/models/emotion_memory.dat",
    "feature_importance_path": "./data/configs/feature_importance.json",
    "enable_cache": True,
    "parallel_processing": True,
    "max_workers": 4,
    "gaussian_smoothing": True,
    "smoothing_sigma": 1.5,
    "adaptive_regions": True,
    "min_region_size": 8,
    "prediction_interval": True,
    "interval_confidence": 0.9,
    "enhanced_features": True
}

@dataclass
class EmotionFeatures:
    """地图情感特征的数据类"""
    plant_coverage: float
    building_density: float
    avg_height: float
    height_variance: float
    edge_density: float
    height_gradient: float
    water_proximity: float
    biome_diversity: float
    spatial_complexity: float
    symmetry_measure: float
    feature_vector: np.ndarray = None
    
    def as_array(self) -> np.ndarray:
        """将特征转换为numpy数组"""
        if self.feature_vector is not None:
            return self.feature_vector
            
        self.feature_vector = np.array([
            self.plant_coverage,
            self.building_density,
            self.avg_height,
            self.height_variance,
            self.edge_density,
            self.height_gradient,
            self.water_proximity,
            self.biome_diversity,
            self.spatial_complexity,
            self.symmetry_measure
        ]).reshape(1, -1)
        return self.feature_vector

@dataclass
class EmotionPrediction:
    """情感预测结果"""
    valence: float
    arousal: float
    valence_lower: Optional[float] = None
    valence_upper: Optional[float] = None
    arousal_lower: Optional[float] = None
    arousal_upper: Optional[float] = None
    dominance: Optional[float] = None
    feature_contributions: Optional[Dict[str, float]] = None

class EmotionModelCache:
    """情感模型的缓存系统"""
    
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache = {}
        self.feature_hash_map = {}
        self.hits = 0
        self.misses = 0
        self._load_cache()
        
    def _load_cache(self):
        """加载缓存"""
        if os.path.exists(self.cache_path):
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                self.cache = data['cache'].item() if 'cache' in data else {}
                self.feature_hash_map = data['feature_hash'].item() if 'feature_hash' in data else {}
            except Exception as e:
                warnings.warn(f"无法加载情感模型缓存: {e}")
                self.cache = {}
                self.feature_hash_map = {}
    
    def save_cache(self):
        """保存缓存到文件"""
        # 限制缓存大小，最多保存10000个预测
        if len(self.cache) > 10000:
            # 仅保留最近使用的项
            keys = list(self.cache.keys())
            for k in keys[:-10000]:
                del self.cache[k]
                if k in self.feature_hash_map:
                    del self.feature_hash_map[k]
                    
        np.savez_compressed(
            self.cache_path, 
            cache=self.cache,
            feature_hash=self.feature_hash_map
        )
    
    def get_prediction(self, features: EmotionFeatures) -> Optional[EmotionPrediction]:
        """从缓存获取预测"""
        # 使用特征值的哈希作为键
        feature_array = features.as_array().flatten()
        feature_hash = hash(feature_array.tobytes())
        
        # 检查是否命中缓存
        region_id = self.feature_hash_map.get(feature_hash)
        if region_id and region_id in self.cache:
            self.hits += 1
            return self.cache[region_id]
            
        self.misses += 1
        return None
        
    def store_prediction(self, features: EmotionFeatures, prediction: EmotionPrediction, region_id=None):
        """存储预测到缓存"""
        if region_id is None:
            region_id = f"region_{len(self.cache)}"
            
        # 计算特征哈希
        feature_array = features.as_array().flatten()
        feature_hash = hash(feature_array.tobytes())
        
        # 存储预测和特征哈希映射
        self.cache[region_id] = prediction
        self.feature_hash_map[feature_hash] = region_id

class EmotionModel:
    """增强的情感预测模型"""
    
    def __init__(self, config=None):
        self.config = config or EMOTION_MODEL_CONFIG
        self.model_path = self.config["model_path"]
        self.feature_importance_path = self.config["feature_importance_path"]
        self.model_valence = None
        self.model_arousal = None
        self.feature_names = [
            "plant_coverage", "building_density", "avg_height",
            "height_variance", "edge_density", "height_gradient",
            "water_proximity", "biome_diversity", "spatial_complexity",
            "symmetry_measure"
        ]
        
        # 特征重要性
        self.feature_importance = {name: 0.0 for name in self.feature_names}
        
        # 初始化缓存
        if self.config["enable_cache"]:
            self.cache = EmotionModelCache(self.config["cache_path"])
        else:
            self.cache = None
            
        # 特征处理器
        self.scaler = None
            
    def train_advanced_emotion_model(self):
        """训练高级情感模型"""
        print("训练高级情感模型...")
        start_time = time.time()
        
        # 生成更真实的训练数据
        n_samples = 1000
        rng = np.random.default_rng(42)
        
        # 生成基本特征
        features = {}
        features["plant_coverage"] = rng.uniform(0, 1, n_samples)
        features["building_density"] = rng.uniform(0, 0.5, n_samples)
        features["avg_height"] = rng.uniform(0, 50, n_samples)
        features["height_variance"] = rng.uniform(0, 100, n_samples) 
        features["edge_density"] = rng.uniform(0, 1, n_samples)
        features["height_gradient"] = rng.uniform(0, 10, n_samples)
        features["water_proximity"] = rng.uniform(0, 1, n_samples)
        features["biome_diversity"] = rng.uniform(0, 1, n_samples)
        features["spatial_complexity"] = rng.uniform(0, 1, n_samples)
        features["symmetry_measure"] = rng.uniform(0, 1, n_samples)
        
        # 创建交互特征
        feature_interaction = (
            features["plant_coverage"] * features["water_proximity"] * 0.3 + 
            features["building_density"] * features["avg_height"] * 0.2
        )
        
        # 构建情感响应模型，基于环境心理学研究
        # Valence (愉悦度): 植被和水面积极影响，高密度建筑和高度变化消极影响
        valence = (
            0.8 * features["plant_coverage"] +
            0.6 * features["water_proximity"] +
            0.3 * features["symmetry_measure"] -
            0.4 * features["building_density"] -
            0.2 * (features["height_variance"] / 100) +
            0.1 * features["biome_diversity"] +
            0.3 * feature_interaction +
            rng.normal(0, 0.05, n_samples)  # 噪声
        ).clip(-1, 1)  # 限制在[-1,1]范围
        
        # Arousal (唤醒度): 复杂度、非对称性、高度变化和边缘密度正向影响
        arousal = (
            0.5 * features["spatial_complexity"] +
            0.4 * (1 - features["symmetry_measure"]) +
            0.3 * features["edge_density"] +
            0.3 * (features["height_variance"] / 100) +
            0.2 * features["building_density"] -
            0.1 * features["plant_coverage"] +
            rng.normal(0, 0.05, n_samples)  # 噪声
        ).clip(-1, 1)  # 限制在[-1,1]范围
        
        # 构建特征矩阵和目标变量
        X = np.column_stack([features[name] for name in self.feature_names])
        y_valence = valence
        y_arousal = arousal
        
        # 标准化特征
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 使用RandomizedSearchCV优化模型
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
        }
        
        # 优化Valence模型
        self.model_valence = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=0), 
            param_grid, 
            n_iter=10,
            cv=3, 
            verbose=0, 
            random_state=0, 
            n_jobs=-1
        )
        self.model_valence.fit(X_scaled, y_valence)
        
        # 优化Arousal模型
        self.model_arousal = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=0), 
            param_grid, 
            n_iter=10,
            cv=3, 
            verbose=0, 
            random_state=0, 
            n_jobs=-1
        )
        self.model_arousal.fit(X_scaled, y_arousal)
        
        # 在保存模型前确保目录存在
        import os
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # 保存模型
        model_data = {
            'model_valence': self.model_valence.best_estimator_,
            'model_arousal': self.model_arousal.best_estimator_,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, self.model_path)
        
        # 记录特征重要性
        v_importance = self.model_valence.best_estimator_.feature_importances_
        a_importance = self.model_arousal.best_estimator_.feature_importances_
        
        for i, name in enumerate(self.feature_names):
            self.feature_importance[name] = {
                'valence': float(v_importance[i]),
                'arousal': float(a_importance[i])
            }
            
        # 保存特征重要性
        with open(self.feature_importance_path, 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
            
        print(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"Valence模型最佳参数: {self.model_valence.best_params_}")
        print(f"Arousal模型最佳参数: {self.model_arousal.best_params_}")

        
    def load_model(self):
        """加载情感预测模型"""
        try:
            if not os.path.exists(self.model_path):
                print("模型文件不存在，训练新模型")
                self.train_advanced_emotion_model()
                return
                    
            print(f"从 {self.model_path} 加载模型")
            model_data = joblib.load(self.model_path)
            
            # 详细调试输出
            print(f"加载的模型数据类型: {type(model_data)}")
            if isinstance(model_data, dict):
                print(f"模型数据键: {list(model_data.keys())}")
                
                # 确保模型正确加载
                if 'model_valence' in model_data:
                    model_valence = model_data['model_valence']
                    print(f"model_valence类型: {type(model_valence)}")
                    
                    # 确保是有效的预测器
                    if hasattr(model_valence, 'predict'):
                        self.model_valence = model_valence
                    else:
                        print(f"警告: model_valence不是有效的预测器, 需要重新训练")
                        self.model_valence = None
                else:
                    print("未找到model_valence键")
                    self.model_valence = None
                    
                if 'model_arousal' in model_data:
                    model_arousal = model_data['model_arousal']
                    print(f"model_arousal类型: {type(model_arousal)}")
                    
                    # 确保是有效的预测器
                    if hasattr(model_arousal, 'predict'):
                        self.model_arousal = model_arousal
                    else:
                        print(f"警告: model_arousal不是有效的预测器, 需要重新训练")
                        self.model_arousal = None
                else:
                    print("未找到model_arousal键")
                    self.model_arousal = None
                    
                # 加载其他组件
                if 'scaler' in model_data:
                    self.scaler = model_data['scaler']
                
                if 'feature_names' in model_data:
                    self.feature_names = model_data['feature_names']
                    
                # 如果模型无效，重新训练
                if self.model_valence is None or self.model_arousal is None:
                    print("模型加载失败，重新训练")
                    self.train_advanced_emotion_model()
            else:
                print(f"警告: 模型数据格式错误: {type(model_data)}")
                self.train_advanced_emotion_model()
                
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            print(traceback.format_exc())  # 打印完整堆栈
            print("将重新训练模型")
            self.train_advanced_emotion_model()
    
    def compute_enhanced_features(self, height_map, biome_map, vegetation, buildings, rivers, region=None) -> EmotionFeatures:
        """计算增强的地图特征"""
        # 确定区域范围
        if region:
            x_start, x_end, y_start, y_end = region
            # 确保区域至少有2x2的尺寸
            if x_end - x_start < 2:
                x_end = min(x_start + 2, len(height_map[0]))
            if y_end - y_start < 2:
                y_end = min(y_start + 2, len(height_map))
            
            sub_height = np.array([row[x_start:x_end] for row in height_map[y_start:y_end]])
            sub_biome = np.array([row[x_start:x_end] for row in biome_map[y_start:y_end]]) if biome_map else None
        else:
            sub_height = np.array(height_map)
            sub_biome = np.array(biome_map) if biome_map else None
            x_start, y_start = 0, 0
            x_end, y_end = len(sub_height[0]), len(sub_height)
            
        h_reg, w_reg = sub_height.shape
        total_cells = h_reg * w_reg

        # 计算区域内的植被和建筑 - 适配多维数据结构
        plant_coverage = 0
        if vegetation is not None and len(vegetation) > 0:  # 修复真值判断
            try:
                veg_in_region = []
                # 统一处理列表和numpy数组
                for veg in np.array(vegetation):  # 转换为numpy数组确保通用性
                    try:
                        # 根据数据结构提取前两个元素作为坐标
                        x, y = int(veg[0]), int(veg[1])  # 格式: (x, y, type, size, age)
                        if x_start <= x < x_end and y_start <= y < y_end:
                            veg_in_region.append((x, y))
                    except (IndexError, ValueError, TypeError):
                        continue  # 忽略格式错误的数据
                
                plant_coverage = len(veg_in_region) / total_cells if total_cells > 0 else 0
            except Exception as e:
                print(f"处理植被时出错: {e}")

        building_density = 0
        if buildings is not None and len(buildings) > 0:  # 防御空值
            try:
                bld_in_region = []
                # 统一处理列表和numpy数组
                for bld in np.array(buildings):  # 自动转换数据结构
                    try:
                        # 提取前两个元素作为坐标
                        x, y = int(bld[0]), int(bld[1])  # 格式: (x, y, type, orientation, importance)
                        if x_start <= x < x_end and y_start <= y < y_end:
                            bld_in_region.append((x, y))
                    except (IndexError, ValueError, TypeError):
                        continue
                
                building_density = len(bld_in_region) / total_cells if total_cells > 0 else 0
            except Exception as e:
                print(f"处理建筑时出错: {e}")
        
        # 基础统计特征
        avg_height = np.mean(sub_height)
        height_variance = np.var(sub_height)
        
        # 计算高度梯度（地形复杂度）
        height_gradient = 0
        if h_reg >= 2 and w_reg >= 2:
            try:
                gy, gx = np.gradient(sub_height)
                height_gradient = np.mean(np.sqrt(gx**2 + gy**2))
            except Exception as e:
                print(f"计算高度梯度时出错: {e}")
        
        # 边缘密度 - 使用Sobel算子
        edge_density = 0
        if h_reg >= 3 and w_reg >= 3:
            try:
                from scipy.ndimage import sobel
                edge_x = sobel(sub_height, axis=0)
                edge_y = sobel(sub_height, axis=1)
                edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
                edge_density = np.mean(edge_magnitude) / np.max(edge_magnitude) if np.max(edge_magnitude) > 0 else 0
            except Exception as e:
                print(f"计算边缘密度时出错: {e}")

        # 计算水域临近度
        water_proximity = 0
        if rivers is not None:
            try:
                river_points = []
                if isinstance(rivers, np.ndarray) and rivers.dtype == bool and rivers.size > 0:
                    # 处理布尔数组，提取为True的坐标
                    y_coords, x_coords = np.where(rivers)
                    for y_global, x_global in zip(y_coords, x_coords):
                        # 检查坐标是否在当前区域范围内
                        if x_start <= x_global < x_end and y_start <= y_global < y_end:
                            river_points.append((x_global, y_global))
                else:
                    # 处理其他类型的rivers数据（如坐标列表）
                    for r in rivers:
                        try:
                            if hasattr(r, 'x') and hasattr(r, 'y'):
                                x, y = int(r.x), int(r.y)
                            elif isinstance(r, (list, tuple)) and len(r) >= 2:
                                x, y = int(r[0]), int(r[1])
                            else:
                                continue
                            if x_start <= x < x_end and y_start <= y < y_end:
                                river_points.append((x, y))
                        except (ValueError, TypeError):
                            continue
                
                if river_points:
                    from scipy.spatial.distance import cdist
                    grid_points = [(i, j) for i in range(w_reg) for j in range(h_reg)]
                    distances = cdist(grid_points, river_points, 'euclidean')
                    min_distances = np.min(distances, axis=1).reshape(h_reg, w_reg)
                    normalized_dist = 1 - np.minimum(min_distances / max(w_reg, h_reg), 1)
                    water_proximity = np.mean(normalized_dist)
            except Exception as e:
                print(f"处理河流时出错: {e}")

        # 生物群系多样性
        biome_diversity = 0
        if sub_biome is not None:
            unique_biomes = np.unique(sub_biome)
            biome_diversity = len(unique_biomes) / 10  # 假设最多10种生物群系
        
        # 空间复杂度（使用熵）
        def calculate_entropy(arr):
            _, counts = np.unique(arr, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs))
        
        spatial_complexity = calculate_entropy(sub_height) / 8 if sub_height.size > 0 else 0
        
        # 对称性测量
        def measure_symmetry(arr):
            h, w = arr.shape
            h_flip = np.fliplr(arr)
            h_sym = 1 - np.mean(np.abs(arr - h_flip)) / np.mean(arr) if np.mean(arr) > 0 else 0
            v_flip = np.flipud(arr)
            v_sym = 1 - np.mean(np.abs(arr - v_flip)) / np.mean(arr) if np.mean(arr) > 0 else 0
            return (h_sym + v_sym) / 2
        
        symmetry_measure = measure_symmetry(sub_height) if sub_height.size > 0 else 0
        
        return EmotionFeatures(
            plant_coverage=plant_coverage,
            building_density=building_density,
            avg_height=avg_height,
            height_variance=height_variance,
            edge_density=edge_density,
            height_gradient=height_gradient,
            water_proximity=water_proximity,
            biome_diversity=biome_diversity,
            spatial_complexity=spatial_complexity,
            symmetry_measure=symmetry_measure
        )
    
    def predict_emotion(self, features: EmotionFeatures) -> EmotionPrediction:
        """预测地区情感值"""
        # 检查缓存
        if self.cache:
            cached = self.cache.get_prediction(features)
            if cached:
                return cached
                
        # 确保模型已加载
        if self.model_valence is None or self.model_arousal is None:
            self.load_model()  
        
        # 准备特征
        X = features.as_array()
        
        # 严格验证模型可用性
        if (self.model_valence is None or not hasattr(self.model_valence, 'predict') or
            self.model_arousal is None or not hasattr(self.model_arousal, 'predict')):
            print(f"无效模型对象，返回默认预测值 (valence类型:{type(self.model_valence)}, arousal类型:{type(self.model_arousal)})")
            return EmotionPrediction(
                valence=0.0,
                arousal=0.0,
                valence_lower=-0.2,
                valence_upper=0.2,
                arousal_lower=-0.2,
                arousal_upper=0.2
            )
        try:
            # 尝试缩放特征
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            print(f"特征缩放错误: {e}")
            # 回退到简单的标准化处理
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0) + 1e-6  # 避免除零
            X_scaled = (X - X_mean) / X_std
      
        # 预测情感值
        try:
            # 使用安全的预测方法
            if hasattr(self.model_valence, 'predict'):
                valence_pred = self.model_valence.predict(X_scaled)
                # 防御性处理 - 检查结果是数组还是标量
                if hasattr(valence_pred, '__len__') and len(valence_pred) > 0:
                    valence = float(valence_pred[0])
                else:
                    valence = float(valence_pred)  # 如果是标量，直接转换
            else:
                print(f"警告: model_valence ({type(self.model_valence)}) 没有predict方法")
                valence = 0.0
                
            if hasattr(self.model_arousal, 'predict'):
                arousal_pred = self.model_arousal.predict(X_scaled)
                # 防御性处理 - 检查结果是数组还是标量
                if hasattr(arousal_pred, '__len__') and len(arousal_pred) > 0:
                    arousal = float(arousal_pred[0])
                else:
                    arousal = float(arousal_pred)  # 如果是标量，直接转换
            else:
                print(f"警告: model_arousal ({type(self.model_arousal)}) 没有predict方法")
                arousal = 0.0
        except Exception as e:
            print(f"情感预测错误: {str(e)}")
            return EmotionPrediction(valence=0.0, arousal=0.0)
       
        # 计算预测区间（如果启用）
        valence_lower = valence_upper = arousal_lower = arousal_upper = None
        if self.config["prediction_interval"]:
            try:
                # 首先尝试使用估计器集合
                v_preds = []
                a_preds = []
                
                # 正确处理GradientBoostingRegressor的二维estimators_结构
                if hasattr(self.model_valence, 'estimators_'):
                    try:
                        # GradientBoostingRegressor的estimators_是二维结构 [n_stages, n_estimators]
                        for stage in self.model_valence.estimators_:
                            if isinstance(stage, np.ndarray) or isinstance(stage, list):
                                for est in stage:
                                    if hasattr(est, 'predict'):
                                        try:
                                            pred = est.predict(X_scaled)
                                            if hasattr(pred, '__len__') and len(pred) > 0:
                                                v_preds.append(pred[0])
                                            else:
                                                v_preds.append(float(pred))
                                        except Exception as e:
                                            pass
                    except Exception as e:
                        print(f"访问valence estimators时出错: {e}")
                        
                if hasattr(self.model_arousal, 'estimators_'):
                    try:
                        for stage in self.model_arousal.estimators_:
                            if isinstance(stage, np.ndarray) or isinstance(stage, list):
                                for est in stage:
                                    if hasattr(est, 'predict'):
                                        try:
                                            pred = est.predict(X_scaled)
                                            if hasattr(pred, '__len__') and len(pred) > 0:
                                                a_preds.append(pred[0])
                                            else:
                                                a_preds.append(float(pred))
                                        except Exception as e:
                                            pass
                    except Exception as e:
                        print(f"访问arousal estimators时出错: {e}")
                
                # 尝试使用staged_predict作为备选方法
                if len(v_preds) <= 1 and hasattr(self.model_valence, 'staged_predict'):
                    try:
                        for pred in self.model_valence.staged_predict(X_scaled):
                            if hasattr(pred, '__len__') and len(pred) > 0:
                                v_preds.append(pred[0])
                            else:
                                v_preds.append(float(pred))
                    except Exception as e:
                        print(f"使用staged_predict时出错: {e}")
                        
                if len(a_preds) <= 1 and hasattr(self.model_arousal, 'staged_predict'):
                    try:
                        for pred in self.model_arousal.staged_predict(X_scaled):
                            if hasattr(pred, '__len__') and len(pred) > 0:
                                a_preds.append(pred[0])
                            else:
                                a_preds.append(float(pred))
                    except Exception as e:
                        print(f"使用staged_predict时出错: {e}")
                
                # 如果没有足够的预测，使用简单的替代方法
                # 修复：检查列表长度而不是直接布尔评估
                if not v_preds or len(v_preds) <= 1:
                    # 使用固定比例的简单区间
                    v_preds = [valence * 0.9, valence, valence * 1.1]
                    print("使用简单替代方法计算valence预测区间")
                
                if not a_preds or len(a_preds) <= 1:
                    # 使用固定比例的简单区间
                    a_preds = [arousal * 0.9, arousal, arousal * 1.1]
                    print("使用简单替代方法计算arousal预测区间")
                
                # 计算区间
                alpha = 1 - self.config["interval_confidence"]
                valence_lower = float(np.quantile(v_preds, alpha/2))
                valence_upper = float(np.quantile(v_preds, 1-alpha/2))
                arousal_lower = float(np.quantile(a_preds, alpha/2))
                arousal_upper = float(np.quantile(a_preds, 1-alpha/2))
            
            except Exception as e:
                print(f"计算预测区间时出错: {e}")
                # 设置默认区间
                valence_lower = valence - 0.2
                valence_upper = valence + 0.2
                arousal_lower = arousal - 0.2
                arousal_upper = arousal + 0.2
       
        # 计算特征贡献（简化版SHAP）
        feature_contributions = None
        if self.feature_importance:
            try:
                # 确保X_scaled是二维数组
                if not isinstance(X_scaled, np.ndarray):
                    print(f"警告: X_scaled不是numpy数组，而是 {type(X_scaled)}")
                    # 尝试转换
                    try:
                        X_scaled = np.array(X_scaled).reshape(1, -1)
                    except:
                        print("无法转换X_scaled为二维数组")
                        X_scaled = None
                
                if isinstance(X_scaled, np.ndarray):
                    # 检查维度
                    if X_scaled.ndim == 1:
                        X_scaled = X_scaled.reshape(1, -1)
                    elif X_scaled.ndim > 2:
                        print(f"警告: X_scaled维度过高: {X_scaled.ndim}")
                        X_scaled = X_scaled.reshape(1, -1)
                        
                    # 检查形状与特征名称列表长度是否匹配
                    if X_scaled.shape[1] != len(self.feature_names):
                        print(f"警告: X_scaled形状 {X_scaled.shape} 与特征数量 {len(self.feature_names)} 不匹配")
                        # 如果不匹配，可能需要调整
                        if X_scaled.shape[1] < len(self.feature_names):
                            # 填充额外的0
                            padding = np.zeros((X_scaled.shape[0], len(self.feature_names) - X_scaled.shape[1]))
                            X_scaled = np.hstack([X_scaled, padding])
                        else:
                            # 截断多余的列
                            X_scaled = X_scaled[:, :len(self.feature_names)]
                    
                    # 现在安全地计算特征贡献
                    feature_contributions = {}
                    for i, name in enumerate(self.feature_names):
                        if name in self.feature_importance and i < X_scaled.shape[1]:
                            # 检查特征重要性的类型
                            if isinstance(self.feature_importance[name], dict):
                                val_imp = self.feature_importance[name].get('valence', 0)
                                aro_imp = self.feature_importance[name].get('arousal', 0)
                            else:
                                # 如果是浮点数或其他非字典类型，使用其作为重要性
                                val_imp = aro_imp = float(self.feature_importance[name])
                            
                            feature_contributions[name] = {
                                'valence': float(val_imp * X_scaled[0, i]),
                                'arousal': float(aro_imp * X_scaled[0, i])
                            }
            except Exception as e:
                print(f"计算特征贡献时出错: {e}")
                # 跳过特征贡献计算
                feature_contributions = None
     
        # 创建预测结果
        prediction = EmotionPrediction(
            valence=valence,
            arousal=arousal,
            valence_lower=valence_lower,
            valence_upper=valence_upper,
            arousal_lower=arousal_lower,
            arousal_upper=arousal_upper,
            feature_contributions=feature_contributions
        )
    
        # 缓存结果
        if self.cache:
            self.cache.store_prediction(features, prediction)
  
        return prediction
    
    def close(self):
        """清理资源"""
        if self.cache and self.config["enable_cache"]:
            self.cache.save_cache()

class MapEmotionAnalyzer:
    """地图情感分析器 - 高级情感分析系统"""    
    def __init__(self,  height_map, biome_map, vegetation, buildings, rivers, content_layout, caves, roads, roads_map,config=None):
        self.config = config or EMOTION_MODEL_CONFIG
        self.height_map = height_map
        self.biome_map = biome_map
        self.vegetation = vegetation
        self.model = EmotionModel(self.config)
        self.model.load_model()
        
    def _generate_adaptive_regions(self, height_map, min_size=None):
        """生成自适应区域划分"""
        try:
            min_size = min_size or self.config["min_region_size"]
            h = len(height_map)
            w = len(height_map[0])
            
            # 检查高度图的尺寸是否足够大
            if h < 3 or w < 3:  # 需要至少3x3来计算有意义的梯度
                print(f"高度图尺寸太小 ({w}x{h})，使用单个区域")
                return [("region_single", 0, w, 0, h)]
            
            # 分析地形变化，确定区域边界
            height_array = np.array(height_map)
            gy, gx = np.gradient(height_array)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            
            # 使用简单网格划分替代复杂逻辑
            grid_size = max(min_size, min(h, w) // 8)
            regions = []
            for y in range(0, h, grid_size):
                for x in range(0, w, grid_size):
                    x_end = min(x + grid_size, w)
                    y_end = min(y + grid_size, h)
                    regions.append((f"grid_{x//grid_size}_{y//grid_size}", 
                                x, x_end, y, y_end))
            
            return regions
            
        except Exception as e:
            print(f"计算区域划分时出错: {e}")
            # 出错时返回单个区域
            try:
                w = len(height_map[0]) if height_map and height_map[0] else 1
                h = len(height_map) if height_map else 1
                return [("region_fallback", 0, w, 0, h)]
            except:
                # 严重错误情况下返回空列表而不是None
                print("区域划分失败，返回空区域列表")
                return []
        
    def _compute_region_emotion_worker(self, args):
        """并行工作器函数"""
        region_name, x_start, x_end, y_start, y_end = args[0]
        height_map, biome_map, vegetation, buildings, rivers = args[1:]
        
        features = self.model.compute_enhanced_features(
            height_map, biome_map, vegetation, buildings, rivers, 
            (x_start, x_end, y_start, y_end)
        )
        
        emotion = self.model.predict_emotion(features)
        return region_name, emotion, (x_start, x_end, y_start, y_end)
    
    def analyze_map_emotions(self, height_map, biome_map, vegetation, buildings, rivers, content_layout, caves=None, roads=None, roads_map=None):
        """分析整个地图和区域情感"""
        # 正确检查高度图是否为空或太小
        if (self.height_map is None or 
            (isinstance(self.height_map, np.ndarray) and self.height_map.size == 0) or 
            (not isinstance(self.height_map, np.ndarray) and len(self.height_map) == 0) or
            len(self.height_map) < 2 or len(self.height_map[0]) < 2):
            print("高度图为空或尺寸过小，无法进行情感分析")
            return content_layout

        height_map = self.height_map  # 使用实例变量而不是参数
            
        start_time = time.time()

        # 计算全局情感
        global_features = self.model.compute_enhanced_features(
            height_map, biome_map, vegetation, buildings, rivers
        )

        global_emotion = self.model.predict_emotion(global_features)
        
        # 记录全局情感
        emotion_data = {
            "valence": global_emotion.valence,
            "arousal": global_emotion.arousal
        }

        # 检查content_layout的类型并相应处理
        if isinstance(content_layout, dict):
            # 如果是字典，直接添加键值对
            content_layout["map_emotion"] = emotion_data
        elif isinstance(content_layout, list):
            # 如果是列表，添加新的情感字典
            content_layout.append({"type": "map_emotion", "data": emotion_data})
        else:
            # 其他情况，打印警告
            print(f"警告: content_layout类型({type(content_layout)})不支持存储情感数据")
       
        # 如果启用了预测区间
        if global_emotion.valence_lower is not None:
            # 处理字典类型
            if isinstance(content_layout, dict) and "map_emotion" in content_layout:
                content_layout["map_emotion"]["valence_interval"] = [
                    global_emotion.valence_lower, 
                    global_emotion.valence_upper
                ]
                content_layout["map_emotion"]["arousal_interval"] = [
                    global_emotion.arousal_lower,
                    global_emotion.arousal_upper
                ]
            # 处理列表类型
            elif isinstance(content_layout, list):
                # 查找map_emotion项
                for item in content_layout:
                    if isinstance(item, dict) and item.get("type") == "map_emotion":
                        item["data"]["valence_interval"] = [
                            global_emotion.valence_lower, 
                            global_emotion.valence_upper
                        ]
                        item["data"]["arousal_interval"] = [
                            global_emotion.arousal_lower,
                            global_emotion.arousal_upper
                        ]
                        break

        # 区域情感分析
        regions = self._generate_adaptive_regions(height_map) if self.config["adaptive_regions"] else []
        region_emotions = {}

        # 准备并行计算参数
        task_args = [(region, height_map, biome_map, vegetation, buildings, rivers) 
                    for region in regions]

        # 并行处理区域情感计算
        if self.config["parallel_processing"]:
            with ProcessPoolExecutor(max_workers=self.config["max_workers"]) as executor:
                futures = [executor.submit(self._compute_region_emotion_worker, args) 
                          for args in task_args]
                for future in futures:
                    region_name, emotion, bbox = future.result()
                    region_emotions[region_name] = {
                        "emotion": emotion.__dict__,
                        "bounding_box": bbox
                    }
        else:
            for args in task_args:
                region_name, emotion, bbox = self._compute_region_emotion_worker(args)
                region_emotions[region_name] = {
                    "emotion": emotion.__dict__,
                    "bounding_box": bbox
                }

        # 应用高斯平滑（如果需要）
        if self.config["gaussian_smoothing"]:
            valence_matrix = np.zeros((len(height_map), len(height_map[0])))
            arousal_matrix = np.zeros_like(valence_matrix)
            
            for region_name, data in region_emotions.items():
                x_start, x_end, y_start, y_end = data["bounding_box"]
                valence_matrix[y_start:y_end, x_start:x_end] = data["emotion"]["valence"]
                arousal_matrix[y_start:y_end, x_start:x_end] = data["emotion"]["arousal"]
            
            smoothed_valence = gaussian_filter(valence_matrix, sigma=self.config["smoothing_sigma"])
            smoothed_arousal = gaussian_filter(arousal_matrix, sigma=self.config["smoothing_sigma"])
            
            # 更新区域情感数据
            for region_name, data in region_emotions.items():
                x_start, x_end, y_start, y_end = data["bounding_box"]
                data["emotion"]["valence"] = float(np.mean(smoothed_valence[y_start:y_end, x_start:x_end]))
                data["emotion"]["arousal"] = float(np.mean(smoothed_arousal[y_start:y_end, x_start:x_end]))

        # 将情感值转换为八种基本情感的具体点
        emotion_points = self._generate_emotion_points(height_map, global_emotion, region_emotions)
        
        # 将情感点添加到content_layout
        if isinstance(content_layout, dict):
            content_layout["emotions"] = emotion_points
        elif isinstance(content_layout, list):
            # 已经是一个列表，直接扩展它
            content_layout.extend(emotion_points)
        else:
            # 如果content_layout是其他类型，返回情感点列表作为新的content_layout
            content_layout = emotion_points
       
        # 保存缓存
        try:
            if hasattr(self.model, 'cache') and self.model.cache:
                self.model.close()
                print(f"情感分析完成，总耗时: {time.time() - start_time:.2f}秒")
                if hasattr(self.model.cache, 'hits'):
                    hit_rate = self.model.cache.hits / (self.model.cache.hits + self.model.cache.misses + 1e-6)
                    print(f"缓存命中率: {hit_rate:.1%}")
        except Exception as e:
            print(f"关闭情感模型时出错: {e}")

        return content_layout
    
    def _generate_emotion_points(self, height_map, global_emotion, region_emotions):
        """根据情感分析结果生成八种基本情感的点"""
        height, width = len(height_map), len(height_map[0])
        emotion_points = []
        
        # 定义八种基本情感在情感空间中的位置 (valence, arousal) 坐标
        emotion_mapping = {
            "joy": (0.8, 0.5),         # 高愉悦度，中等唤醒度
            "fear": (-0.8, 0.8),       # 低愉悦度，高唤醒度
            "anger": (-0.6, 0.7),      # 低愉悦度，高唤醒度
            "sadness": (-0.7, -0.4),   # 低愉悦度，低唤醒度
            "surprise": (0.2, 0.8),    # 中等愉悦度，高唤醒度
            "disgust": (-0.5, 0.0),    # 低愉悦度，中等唤醒度
            "anticipation": (0.5, 0.6), # 高愉悦度，高唤醒度
            "trust": (0.7, -0.2)       # 高愉悦度，低唤醒度
        }
        
        # 为每个区域生成情感点
        for region_name, data in region_emotions.items():
            x_start, x_end, y_start, y_end = data["bounding_box"]
            region_valence = data["emotion"]["valence"]
            region_arousal = data["emotion"]["arousal"]
            
            # 计算该区域的主要情感类型和强度
            main_emotion, intensity = self._get_emotion_from_valence_arousal(
                region_valence, region_arousal, emotion_mapping
            )
            
            # 选择区域中的特征点作为情感点
            # 简单方法：在区域的4个不同位置创建点
            emotion_points_count = max(1, min(4, (x_end - x_start) * (y_end - y_start) // 100))
            
            for i in range(emotion_points_count):
                # 计算情感点位置 - 分散在区域内
                if emotion_points_count == 1:
                    x = (x_start + x_end) // 2
                    y = (y_start + y_end) // 2
                else:
                    idx = i % 4
                    if idx == 0:  # 左上
                        x = x_start + (x_end - x_start) // 4
                        y = y_start + (y_end - y_start) // 4
                    elif idx == 1:  # 右上
                        x = x_end - (x_end - x_start) // 4
                        y = y_start + (y_end - y_start) // 4
                    elif idx == 2:  # 左下
                        x = x_start + (x_end - x_start) // 4
                        y = y_end - (y_end - y_start) // 4
                    else:  # 右下
                        x = x_end - (x_end - x_start) // 4
                        y = y_end - (y_end - y_start) // 4
                
                # 确保坐标在有效范围内
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                
                # 创建情感点
                emotion_point = {
                    "x": x,
                    "y": y,
                    "type": main_emotion,
                    "intensity": intensity
                }
                emotion_points.append(emotion_point)
            
            # 次要情感：找出第二个最接近的情感，确保情感多样性
            second_emotion, second_intensity = self._get_second_emotion(
                region_valence, region_arousal, emotion_mapping, main_emotion
            )
            
            # 添加次要情感点（数量较少）
            if second_intensity > 0.3:  # 只有当强度足够时才添加
                x = x_start + (x_end - x_start) // 2
                y = y_start + (y_end - y_start) // 2
                emotion_point = {
                    "x": x,
                    "y": y,
                    "type": second_emotion,
                    "intensity": second_intensity
                }
                emotion_points.append(emotion_point)
        
        # 添加一些全局情感点，确保覆盖整个地图
        global_valence = global_emotion.valence
        global_arousal = global_emotion.arousal
        global_emotion_type, global_intensity = self._get_emotion_from_valence_arousal(
            global_valence, global_arousal, emotion_mapping
        )
        
        # 在地图中均匀添加全局情感点
        grid_size = max(10, min(width, height) // 10)
        for y in range(grid_size // 2, height, grid_size):
            for x in range(grid_size // 2, width, grid_size):
                # 检查这个位置是否已经被区域情感点覆盖
                covered = False
                for point in emotion_points:
                    if abs(point["x"] - x) < grid_size // 2 and abs(point["y"] - y) < grid_size // 2:
                        covered = True
                        break
                
                if not covered:
                    # 随机选择一种情感，但倾向于全局主情感
                    if np.random.random() < 0.7:
                        emotion_type = global_emotion_type
                        intensity = global_intensity * (0.7 + 0.3 * np.random.random())
                    else:
                        # 随机选择一种情感
                        emotion_type = np.random.choice(list(emotion_mapping.keys()))
                        intensity = 0.3 + 0.4 * np.random.random()
                    
                    emotion_point = {
                        "x": x,
                        "y": y,
                        "type": emotion_type,
                        "intensity": intensity
                    }
                    emotion_points.append(emotion_point)
        
        return emotion_points

    def _get_emotion_from_valence_arousal(self, valence, arousal, emotion_mapping):
        """根据valence和arousal值确定最匹配的情感类型和强度"""
        # 找出最接近的情感类型
        closest_emotion = None
        min_distance = float('inf')
        
        for emotion, (e_valence, e_arousal) in emotion_mapping.items():
            # 计算欧几里得距离
            distance = np.sqrt((valence - e_valence)**2 + (arousal - e_arousal)**2)
            if distance < min_distance:
                min_distance = distance
                closest_emotion = emotion
        
        # 计算强度 - 距离越近，强度越大
        # 将距离映射到0.3-1.0的范围（确保即使很远的点也有最小强度）
        max_possible_distance = 2.0  # valence和arousal范围都是[-1,1]，所以最大距离是2
        intensity = 1.0 - (min_distance / max_possible_distance) * 0.7
        intensity = max(0.3, min(1.0, intensity))
        
        return closest_emotion, intensity

    def _get_second_emotion(self, valence, arousal, emotion_mapping, exclude_emotion):
        """获取第二接近的情感类型，排除已选的主情感"""
        distances = []
        
        for emotion, (e_valence, e_arousal) in emotion_mapping.items():
            if emotion == exclude_emotion:
                continue
            
            # 计算欧几里得距离
            distance = np.sqrt((valence - e_valence)**2 + (arousal - e_arousal)**2)
            distances.append((emotion, distance))
        
        # 按距离排序
        distances.sort(key=lambda x: x[1])
        
        # 获取最接近的情感
        second_emotion = distances[0][0]
        distance = distances[0][1]
        
        # 计算强度 - 距离越近，强度越大
        max_possible_distance = 2.0
        intensity = 1.0 - (distance / max_possible_distance) * 0.7
        intensity = max(0.3, min(0.8, intensity * 0.8))  # 比主情感稍弱一些
        
        return second_emotion, intensity