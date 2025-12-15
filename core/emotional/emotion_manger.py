import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from core.services.analyze_map_emotions import MapEmotionAnalyzer

class EmotionManager:
    """地图情感分析与可视化管理器"""
    
    def __init__(self, logger=None):
        """初始化情感管理器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.emotion_data = None
        self.last_analysis_map_data = None
        self.emotion_map = None
        self.primary_emotions = [
            "joy", "fear", "anger", "sadness", 
            "surprise", "disgust", "anticipation", "trust"
        ]
        
    def analyze_map_emotions(self, map_data):
        """分析地图情感

        Args:
            map_data: 地图数据对象

        Returns:
            bool: 分析是否成功
        """
        if not map_data or not map_data.is_valid():
            if self.logger:
                self.logger.log("无效的地图数据，无法进行情感分析", "ERROR")
            return False
            
        try:
            height_map = map_data.get_layer("height")
            biome_map = map_data.get_layer("biome")
            vegetation = map_data.layers.get("vegetation", [])
            buildings = map_data.layers.get("buildings", [])
            rivers = map_data.get_layer("rivers")
            content_layout = map_data.content_layout or {}
            cave_entrances = map_data.layers.get("caves", [])
            roads = map_data.layers.get("roads", [])
            roads_map = map_data.layers.get("roads_map", np.zeros_like(height_map, dtype=np.uint8))
            
            # 创建分析器实例
            analyzer = MapEmotionAnalyzer(
                height_map, biome_map, vegetation, buildings, 
                rivers, content_layout, cave_entrances, roads, roads_map
            )
            
            # 分析情感信息
            updated_content_layout = analyzer.analyze_map_emotions(
                biome_map, vegetation, buildings, rivers, 
                content_layout, cave_entrances, roads, roads_map
            )
            
            # 处理更新后的内容布局 - 修复列表/字典兼容性问题
            if isinstance(updated_content_layout, dict):
                # 如果是字典，直接使用
                map_data.content_layout = updated_content_layout
                self.emotion_data = updated_content_layout.get("emotions", {})
            elif isinstance(updated_content_layout, list):
                # 如果是列表，将其作为"emotions"字段放入内容布局中
                if not isinstance(map_data.content_layout, dict):
                    map_data.content_layout = {}
                map_data.content_layout["emotions"] = updated_content_layout
                self.emotion_data = {"default": updated_content_layout}
            else:
                # 其他情况，创建空情感数据
                if self.logger:
                    self.logger.log(f"情感分析返回了意外类型: {type(updated_content_layout)}", "WARNING")
                map_data.content_layout = map_data.content_layout or {}
                map_data.content_layout["emotions"] = {}
                self.emotion_data = {}
            
            self.last_analysis_map_data = map_data
            
            # 生成情感热力图
            self._generate_emotion_heatmaps(height_map.shape)
            
            if self.logger:
                self.logger.log("地图情感分析完成")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"情感分析过程中发生错误: {e}", "ERROR")
                import traceback
                self.logger.log(traceback.format_exc(), "DEBUG")
            return False
    
    def _generate_emotion_heatmaps(self, map_shape):
        """生成情感热力图
        
        Args:
            map_shape: 地图形状(height, width)
        """
        if not self.emotion_data:
            return
            
        height, width = map_shape
        # 创建情感热力图字典
        self.emotion_map = {}
        
        # 判断情感数据的格式
        if "default" in self.emotion_data and isinstance(self.emotion_data["default"], list):
            # 如果是列表格式存储在default键中
            self._generate_heatmap_from_list(self.emotion_data["default"], height, width)
        else:
            # 对每种情感生成热力图
            for emotion in self.primary_emotions:
                emotion_points = self.emotion_data.get(emotion, [])
                heatmap = np.zeros((height, width), dtype=np.float32)
                
                # 填充热力图
                for point in emotion_points:
                    if not isinstance(point, dict):
                        continue
                        
                    x, y, intensity = point.get("x", 0), point.get("y", 0), point.get("intensity", 0)
                    if 0 <= x < width and 0 <= y < height:
                        radius = int(intensity * 10)  # 根据强度确定影响半径
                        radius = max(3, min(20, radius))
                        
                        # 在点周围创建衰减影响
                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    # 使用高斯衰减
                                    dist = np.sqrt(dx**2 + dy**2)
                                    if dist <= radius:
                                        influence = intensity * np.exp(-(dist**2) / (2 * (radius/3)**2))
                                        heatmap[ny, nx] = max(heatmap[ny, nx], influence)
                
                self.emotion_map[emotion] = heatmap

    def _generate_heatmap_from_list(self, emotion_list, height, width):
        """从情感点列表生成热力图
        
        Args:
            emotion_list: 情感点列表
            height: 地图高度
            width: 地图宽度
        """
        # 为每种主要情感创建空热力图
        for emotion in self.primary_emotions:
            self.emotion_map[emotion] = np.zeros((height, width), dtype=np.float32)
        
        # 添加调试信息
        if self.logger:
            type_counts = {}
            for point in emotion_list:
                if isinstance(point, dict):
                    emotion_type = point.get("type", "joy")
                    if emotion_type not in type_counts:
                        type_counts[emotion_type] = 0
                    type_counts[emotion_type] += 1
            
            self.logger.log(f"情感点类型统计: {type_counts}")
        
        # 分类处理列表中的每个情感点
        for point in emotion_list:
            if not isinstance(point, dict):
                continue
                
            x = point.get("x", 0)
            y = point.get("y", 0)
            intensity = point.get("intensity", 0)
            emotion_type = point.get("type", "joy")  # 默认为joy
            
            # 确保情感类型在主要情感中
            if emotion_type not in self.primary_emotions:
                if self.logger:
                    self.logger.log(f"未知情感类型: {emotion_type}，使用默认值 'joy'", "WARNING")
                emotion_type = "joy"
            
            # 确保坐标在有效范围内
            if 0 <= x < width and 0 <= y < height:
                radius = int(intensity * 10)  # 根据强度确定影响半径
                radius = max(3, min(20, radius))
                
                # 在点周围创建衰减影响
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            # 使用高斯衰减
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= radius:
                                influence = intensity * np.exp(-(dist**2) / (2 * (radius/3)**2))
                                self.emotion_map[emotion_type][ny, nx] = max(
                                    self.emotion_map[emotion_type][ny, nx], influence
                                )
    
    def get_emotion_stats(self):
        """获取情感统计数据
        
        Returns:
            dict: 情感统计信息
        """
        if not self.emotion_data:
            return {}
            
        stats = {}
        
        # 判断情感数据的格式
        if "default" in self.emotion_data and isinstance(self.emotion_data["default"], list):
            # 从列表格式获取统计信息
            emotion_counts = {}
            emotion_intensities = {}
            
            for point in self.emotion_data["default"]:
                if not isinstance(point, dict):
                    continue
                    
                emotion_type = point.get("type", "joy")
                intensity = point.get("intensity", 0)
                
                if emotion_type not in self.primary_emotions:
                    emotion_type = "joy"
                    
                if emotion_type not in emotion_counts:
                    emotion_counts[emotion_type] = 0
                    emotion_intensities[emotion_type] = []
                    
                emotion_counts[emotion_type] += 1
                emotion_intensities[emotion_type].append(intensity)
            
            # 计算统计数据
            for emotion in self.primary_emotions:
                count = emotion_counts.get(emotion, 0)
                intensities = emotion_intensities.get(emotion, [])
                
                stats[emotion] = {
                    "count": count,
                    "avg_intensity": np.mean(intensities) if intensities else 0,
                    "max_intensity": np.max(intensities) if intensities else 0
                }
        else:
            # 从字典格式获取统计信息
            for emotion in self.primary_emotions:
                points = self.emotion_data.get(emotion, [])
                if not points:
                    stats[emotion] = {
                        "count": 0,
                        "avg_intensity": 0,
                        "max_intensity": 0
                    }
                    continue
                    
                intensities = [p.get("intensity", 0) for p in points if isinstance(p, dict)]
                stats[emotion] = {
                    "count": len(points),
                    "avg_intensity": np.mean(intensities) if intensities else 0,
                    "max_intensity": np.max(intensities) if intensities else 0
                }
        
        return stats
    
    def get_dominant_emotions(self, top_n=3):
        """获取主导情感
        
        Args:
            top_n: 返回前N个主导情感
            
        Returns:
            list: [(情感名称, 权重值)]格式的列表
        """
        if not self.emotion_data:
            return []
            
        emotion_weights = []
        
        # 判断情感数据的格式
        if "default" in self.emotion_data and isinstance(self.emotion_data["default"], list):
            # 从列表格式计算权重
            emotion_counts = {}
            emotion_intensities = {}
            
            for point in self.emotion_data["default"]:
                if not isinstance(point, dict):
                    continue
                    
                emotion_type = point.get("type", "joy")
                intensity = point.get("intensity", 0)
                
                if emotion_type not in self.primary_emotions:
                    emotion_type = "joy"
                    
                if emotion_type not in emotion_counts:
                    emotion_counts[emotion_type] = 0
                    emotion_intensities[emotion_type] = []
                    
                emotion_counts[emotion_type] += 1
                emotion_intensities[emotion_type].append(intensity)
            
            # 计算每种情感的总体权重
            for emotion in self.primary_emotions:
                count = emotion_counts.get(emotion, 0)
                intensities = emotion_intensities.get(emotion, [])
                
                weight = count * np.mean(intensities) if intensities else 0
                emotion_weights.append((emotion, weight))
        else:
            # 从字典格式计算权重
            for emotion in self.primary_emotions:
                points = self.emotion_data.get(emotion, [])
                if not points:
                    emotion_weights.append((emotion, 0))
                    continue
                    
                # 计算总体权重：考虑数量和强度
                intensities = [p.get("intensity", 0) for p in points if isinstance(p, dict)]
                weight = len(points) * np.mean(intensities) if intensities else 0
                emotion_weights.append((emotion, weight))
        
        # 按权重排序并返回前N个
        return sorted(emotion_weights, key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_emotion_heatmap(self, emotion_name):
        """获取指定情感的热力图
        
        Args:
            emotion_name: 情感名称
            
        Returns:
            numpy.ndarray: 热力图数组，如果不存在则返回None
        """
        if not self.emotion_map or emotion_name not in self.emotion_map:
            return None
        
        return self.emotion_map[emotion_name]
    
    def get_combined_emotion_heatmap(self, emotions=None, weights=None):
        """获取多种情感组合的热力图
        
        Args:
            emotions: 情感名称列表，如果为None则使用所有情感
            weights: 各情感权重，如果为None则均等权重
            
        Returns:
            numpy.ndarray: 组合热力图
        """
        if not self.emotion_map:
            return None
            
        if emotions is None:
            emotions = self.primary_emotions
        
        if weights is None:
            weights = [1] * len(emotions)
        elif len(weights) != len(emotions):
            weights = [1] * len(emotions)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1 / len(emotions)] * len(emotions)
        
        # 创建组合热力图
        first_map = next(iter(self.emotion_map.values()))
        combined_map = np.zeros_like(first_map)
        
        for emotion, weight in zip(emotions, weights):
            if emotion in self.emotion_map:
                combined_map += self.emotion_map[emotion] * weight
        
        return combined_map
        
    def save_emotion_heatmap(self, emotion_name, filepath):
        """保存情感热力图到文件
        
        Args:
            emotion_name: 情感名称，如果为"combined"则保存组合热力图
            filepath: 保存路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if emotion_name == "combined":
                heatmap = self.get_combined_emotion_heatmap()
                title = "组合情感热力图"
            else:
                heatmap = self.get_emotion_heatmap(emotion_name)
                title = f"{emotion_name} 情感热力图"
            
            if heatmap is None:
                if self.logger:
                    self.logger.log(f"找不到情感 '{emotion_name}' 的热力图", "WARNING")
                return False
            
            # 创建情感特定的颜色映射
            cmap = self._get_emotion_colormap(emotion_name)
            
            # 显示热力图
            im = ax.imshow(heatmap, cmap=cmap, interpolation='bilinear')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label="情感强度")
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.log(f"情感热力图已保存到 {filepath}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"保存情感热力图时出错: {e}", "ERROR")
            return False
    
    def _get_emotion_colormap(self, emotion_name):
        """获取情感对应的颜色映射
        
        Args:
            emotion_name: 情感名称
            
        Returns:
            matplotlib.colors.Colormap: 颜色映射
        """
        # 为每种情感定义特定的颜色方案
        emotion_colors = {
            "joy": [(0, 0, 0, 0), (1, 1, 0, 0.7), (1, 1, 0.5, 1)],  # 黄色
            "fear": [(0, 0, 0, 0), (0.5, 0, 0.5, 0.7), (0.7, 0, 0.7, 1)],  # 紫色
            "anger": [(0, 0, 0, 0), (1, 0, 0, 0.7), (1, 0.3, 0, 1)],  # 红色
            "sadness": [(0, 0, 0, 0), (0, 0, 0.8, 0.7), (0, 0.3, 1, 1)],  # 蓝色
            "surprise": [(0, 0, 0, 0), (1, 0.6, 0, 0.7), (1, 0.8, 0, 1)],  # 橙色
            "disgust": [(0, 0, 0, 0), (0.3, 0.5, 0, 0.7), (0.5, 0.8, 0, 1)],  # 橄榄绿
            "anticipation": [(0, 0, 0, 0), (1, 0.5, 0, 0.7), (1, 0.7, 0.3, 1)],  # 橙红色
            "trust": [(0, 0, 0, 0), (0, 0.7, 0.3, 0.7), (0, 1, 0.5, 1)],  # 绿色
            "combined": [(0, 0, 0, 0), (1, 1, 1, 0.7), (1, 1, 1, 1)]  # 白色
        }
        
        # 如果没有特定颜色方案，使用默认方案
        if emotion_name not in emotion_colors:
            emotion_name = "combined"
        
        return LinearSegmentedColormap.from_list(
            emotion_name, 
            emotion_colors[emotion_name], 
            N=256
        )