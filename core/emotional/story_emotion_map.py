import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Any, Optional
import json
import os

from core.emotional.analyze_story_emotion import EmotionAnalyzer, EmotionVector

class StoryEmotionMap:
    """管理故事情感映射和可视化的类"""
    
    def __init__(self, width=0, height=0, logger=None):
        """初始化情感地图
        
        Args:
            width: 地图宽度
            height: 地图高度
            logger: 日志记录器
        """
        self.width = width
        self.height = height
        self.logger = logger
        self.emotion_analyzer = EmotionAnalyzer()
        self.story_emotions = {}  # 故事情感数据
        self.emotion_heatmaps = {}  # 情感热力图
        self.event_emotions = []  # 事件情感数据
        self.emotion_stats = {}  # 情感统计数据
        
        # 情感颜色映射
        self.emotion_colors = {
            "joy": [(0, 0, 0, 0), (1, 1, 0, 0.7), (1, 1, 0.5, 1)],  # 黄色
            "trust": [(0, 0, 0, 0), (0, 0.7, 0.3, 0.7), (0, 1, 0.5, 1)],  # 绿色
            "fear": [(0, 0, 0, 0), (0.5, 0, 0.5, 0.7), (0.7, 0, 0.7, 1)],  # 紫色
            "surprise": [(0, 0, 0, 0), (1, 0.6, 0, 0.7), (1, 0.8, 0, 1)],  # 橙色
            "sadness": [(0, 0, 0, 0), (0, 0, 0.8, 0.7), (0, 0.3, 1, 1)],  # 蓝色
            "disgust": [(0, 0, 0, 0), (0.3, 0.5, 0, 0.7), (0.5, 0.8, 0, 1)],  # 橄榄绿
            "anger": [(0, 0, 0, 0), (1, 0, 0, 0.7), (1, 0.3, 0, 1)],  # 红色
            "anticipation": [(0, 0, 0, 0), (1, 0.5, 0, 0.7), (1, 0.7, 0.3, 1)],  # 橙红色
            "valence": [(0, 0, 0, 0), (0, 0, 1, 0.7), (1, 1, 1, 0.8), (1, 0, 0, 1)],  # 蓝-白-红
            "arousal": [(0, 0, 0, 0), (0, 0.5, 0, 0.7), (1, 1, 0, 1)]  # 绿-黄
        }
    
    def analyze_story_content(self, story_content, map_data=None):
        """分析故事内容的情感
        
        Args:
            story_content: 故事内容字典
            map_data: 可选的地图数据对象
            
        Returns:
            bool: 分析是否成功
        """
        if not story_content:
            if self.logger:
                self.logger.log("没有故事内容可分析", "WARNING")
            return False
        
        try:
            # 分析整体故事
            overall_story = story_content.get("overall_story", "")
            if overall_story:
                if self.logger:
                    self.logger.log("分析整体故事情感...")
                    
                overall_analysis = self.emotion_analyzer.analyze_story_emotions(overall_story)
                self.story_emotions["overall"] = overall_analysis
                
                # 提取全局情感向量
                global_emotion = overall_analysis.global_emotion
                valence, arousal = global_emotion.to_valence_arousal()
                
                if self.logger:
                    self.logger.log(f"整体故事情感: valence={valence:.2f}, arousal={arousal:.2f}")
                    self.logger.log(f"情感复杂度: {overall_analysis.emotional_complexity:.2f}")
            
            # 分析各个事件
            self.event_emotions = []
            expanded_stories = story_content.get("expanded_stories", [])
            
            for i, event in enumerate(expanded_stories):
                event_content = event.get("expanded_content", "")
                original_event = event.get("original_event", {})
                
                if event_content:
                    if self.logger:
                        self.logger.log(f"分析事件 {i+1} 情感...")
                        
                    event_analysis = self.emotion_analyzer.analyze_story_emotions(event_content)
                    
                    # 记录事件情感数据
                    self.event_emotions.append({
                        "index": i,
                        "event": original_event,
                        "position": (original_event.get("x", 0), original_event.get("y", 0)),
                        "analysis": event_analysis
                    })
            
            # 如果有地图数据，生成情感热力图
            if map_data and hasattr(map_data, 'width') and hasattr(map_data, 'height'):
                self.width = map_data.width
                self.height = map_data.height
                self._generate_emotion_heatmaps()
            
            # 计算情感统计数据
            self._calculate_emotion_stats()
            
            if self.logger:
                self.logger.log("故事情感分析完成")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"故事情感分析出错: {e}", "ERROR")
                import traceback
                self.logger.log(traceback.format_exc(), "DEBUG")
            return False
    
    def _generate_emotion_heatmaps(self):
        """生成情感热力图"""
        if not self.event_emotions or self.width == 0 or self.height == 0:
            return
        
        # 创建基础热力图 - valence和arousal
        valence_map = np.zeros((self.height, self.width), dtype=np.float32)
        arousal_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        # 创建情绪分类热力图
        emotion_categories = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
        category_maps = {emotion: np.zeros((self.height, self.width), dtype=np.float32) for emotion in emotion_categories}
        
        # 填充热力图
        for event in self.event_emotions:
            x, y = event["position"]
            analysis = event["analysis"]
            
            if 0 <= x < self.width and 0 <= y < self.height:
                # 计算影响半径
                valence, arousal = analysis.global_emotion.to_valence_arousal()
                intensity = (valence + arousal) / 2  # 综合强度
                radius = int(20 * intensity)  # 最大半径为20
                radius = max(5, min(30, radius))  # 限制半径范围
                
                # 在每个事件点周围创建情感影响区域
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = int(x) + dx, int(y) + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            # 使用二维高斯函数计算影响强度
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= radius:
                                weight = np.exp(-(dist**2) / (2 * (radius/3)**2))
                                
                                # 更新valence和arousal热力图
                                valence_map[ny, nx] = max(valence_map[ny, nx], valence * weight)
                                arousal_map[ny, nx] = max(arousal_map[ny, nx], arousal * weight)
                                
                                # 更新情感分类热力图
                                for emotion in emotion_categories:
                                    score = analysis.global_emotion.emotion_categories.get(emotion, 0)
                                    category_maps[emotion][ny, nx] = max(
                                        category_maps[emotion][ny, nx], 
                                        score * weight
                                    )
        
        # 存储生成的热力图
        self.emotion_heatmaps["valence"] = valence_map
        self.emotion_heatmaps["arousal"] = arousal_map
        for emotion in emotion_categories:
            self.emotion_heatmaps[emotion] = category_maps[emotion]
    
    def _calculate_emotion_stats(self):
        """计算情感统计数据"""
        if not self.story_emotions or not self.event_emotions:
            return
        
        # 整体情感统计
        if "overall" in self.story_emotions:
            overall = self.story_emotions["overall"]
            valence, arousal = overall.global_emotion.to_valence_arousal()
            
            self.emotion_stats["overall"] = {
                "valence": valence,
                "arousal": arousal,
                "complexity": overall.emotional_complexity,
                "variance": overall.emotional_variance,
                "intensity": overall.language_intensity
            }
        
        # 事件情感统计
        event_valence = []
        event_arousal = []
        event_emotions = {}
        
        for event in self.event_emotions:
            analysis = event["analysis"]
            valence, arousal = analysis.global_emotion.to_valence_arousal()
            
            event_valence.append(valence)
            event_arousal.append(arousal)
            
            # 统计各情感类别
            for emotion, score in analysis.global_emotion.emotion_categories.items():
                if emotion not in event_emotions:
                    event_emotions[emotion] = []
                event_emotions[emotion].append(score)
        
        # 计算平均值和变化程度
        if event_valence:
            self.emotion_stats["events"] = {
                "avg_valence": np.mean(event_valence),
                "avg_arousal": np.mean(event_arousal),
                "valence_variance": np.var(event_valence),
                "arousal_variance": np.var(event_arousal),
                "emotion_categories": {
                    emotion: np.mean(scores) for emotion, scores in event_emotions.items()
                }
            }
    
    def get_emotion_heatmap(self, emotion_name):
        """获取指定情感的热力图
        
        Args:
            emotion_name: 情感名称
            
        Returns:
            numpy.ndarray: 热力图，如果不存在则返回None
        """
        return self.emotion_heatmaps.get(emotion_name)
    
    def get_dominant_emotions(self):
        """获取主导情感
        
        Returns:
            List[Tuple[str, float]]: 主导情感及其权重
        """
        if not self.emotion_stats or "events" not in self.emotion_stats:
            return []
        
        categories = self.emotion_stats["events"].get("emotion_categories", {})
        if not categories:
            return []
        
        # 返回按权重排序的情感类别
        sorted_emotions = sorted(
            [(emotion, weight) for emotion, weight in categories.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions
    
    def create_emotion_heatmap_figure(self, emotion_name="valence", figsize=(8, 6)):
        """创建情感热力图图表
        
        Args:
            emotion_name: 情感名称
            figsize: 图形大小
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        heatmap = self.get_emotion_heatmap(emotion_name)
        if heatmap is None:
            fig = Figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"没有可用的{emotion_name}情感热力图", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # 创建情感特定的颜色映射
        cmap = self._get_emotion_colormap(emotion_name)
        
        # 显示热力图
        im = ax.imshow(heatmap, cmap=cmap, interpolation='bilinear')
        ax.set_title(f"{emotion_name.capitalize()} 情感分布")
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("情感强度")
        
        # 添加事件标记
        for event in self.event_emotions:
            x, y = event["position"]
            if 0 <= x < self.width and 0 <= y < self.height:
                ax.plot(x, y, 'o', markersize=8, markeredgecolor='black', 
                       markerfacecolor='white', alpha=0.8)
                ax.annotate(str(event["index"] + 1), (x, y), 
                           ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 设置坐标轴
        ax.set_xlim(0, self.width - 1)
        ax.set_ylim(self.height - 1, 0)  # 反转Y轴
        ax.set_xticks([])
        ax.set_yticks([])
        
        fig.tight_layout()
        return fig
    
    def create_emotion_comparison_chart(self, figsize=(10, 6)):
        """创建事件情感对比图表
        
        Args:
            figsize: 图形大小
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        if not self.event_emotions:
            ax.text(0.5, 0.5, "没有事件情感数据可供对比", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        # 准备数据
        event_indices = [f"事件 {e['index']+1}" for e in self.event_emotions]
        valence_values = []
        arousal_values = []
        complexity_values = []
        
        for event in self.event_emotions:
            analysis = event["analysis"]
            valence, arousal = analysis.global_emotion.to_valence_arousal()
            valence_values.append(valence)
            arousal_values.append(arousal)
            complexity_values.append(analysis.emotional_complexity)
        
        # 绘制柱状图
        x = np.arange(len(event_indices))
        width = 0.25
        
        bars1 = ax.bar(x - width, valence_values, width, label='效价 (Valence)', color='salmon')
        bars2 = ax.bar(x, arousal_values, width, label='唤醒度 (Arousal)', color='skyblue')
        bars3 = ax.bar(x + width, complexity_values, width, label='复杂度', color='lightgreen')
        
        # 添加数据标签
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        
        # 格式化图表
        ax.set_xlabel('事件')
        ax.set_ylabel('得分')
        ax.set_title('事件情感对比')
        ax.set_xticks(x)
        ax.set_xticklabels(event_indices)
        ax.legend()
        
        fig.tight_layout()
        return fig
    
    def create_emotional_arc_chart(self, figsize=(10, 6)):
        """创建情感弧线图表
        
        Args:
            figsize: 图形大小
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        if "overall" not in self.story_emotions:
            ax.text(0.5, 0.5, "没有情感弧线数据", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        overall = self.story_emotions["overall"]
        arc_data = overall.emotional_arc
        
        if not arc_data:
            ax.text(0.5, 0.5, "没有情感弧线数据", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        # 准备数据
        segments = range(len(arc_data))
        valence_values = [point[0] for point in arc_data]
        arousal_values = [point[1] for point in arc_data]
        
        # 绘制情感弧线
        ax.plot(segments, valence_values, 'r-', label='情感效价 (Valence)', marker='o', linewidth=2)
        ax.plot(segments, arousal_values, 'b-', label='情感唤醒度 (Arousal)', marker='s', linewidth=2)
        
        # 标记关键情感点
        key_points = overall.key_moments
        for point in key_points:
            idx = point.get("index", 0)
            if 0 <= idx < len(arc_data):
                point_type = point.get("type", "")
                if "valence" in point_type:
                    ax.plot(idx, valence_values[idx], 'ro', markersize=10)
                    ax.annotate(point_type.replace("_", " ").title(), 
                                (idx, valence_values[idx]),
                                xytext=(0, 10),
                                textcoords='offset points',
                                ha='center',
                                fontsize=8)
                elif "arousal" in point_type:
                    ax.plot(idx, arousal_values[idx], 'bs', markersize=10)
                    ax.annotate(point_type.replace("_", " ").title(), 
                                (idx, arousal_values[idx]),
                                xytext=(0, -15),
                                textcoords='offset points',
                                ha='center',
                                fontsize=8)
        
        # 格式化图表
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        ax.set_xlabel('故事段落')
        ax.set_ylabel('情感值')
        ax.set_ylim(0, 1)
        ax.set_title('故事情感弧线')
        
        fig.tight_layout()
        return fig
    
    def generate_emotion_insights(self):
        """生成情感洞见和建议
        
        Returns:
            List[str]: 洞见和建议列表
        """
        insights = []
        
        if not self.emotion_stats:
            return ["没有足够的分析数据生成洞见"]
        
        # 全局情感洞见
        if "overall" in self.emotion_stats:
            overall = self.emotion_stats["overall"]
            
            # 情感变化评估
            if overall["variance"] < 0.1:
                insights.append("故事情感变化较少，可能缺乏起伏感。建议增加情感高潮和低谷，提高情感对比度。")
            elif overall["variance"] > 0.3:
                insights.append("故事情感变化丰富，有较好的起伏感，有助于维持玩家的兴趣。")
            
            # 情感复杂度评估
            if overall["complexity"] < 0.3:
                insights.append("故事情感相对简单，可以考虑增加更丰富的情感层次和冲突。")
            elif overall["complexity"] > 0.6:
                insights.append("故事情感复杂丰富，可以给玩家带来深刻的情感体验。")
            
            # 整体情感基调评估
            if overall["valence"] < 0.4:
                insights.append("故事整体情感基调偏向消极，请确保这符合您的游戏风格和目标受众。")
            elif overall["valence"] > 0.7:
                insights.append("故事整体情感基调较为积极，适合轻松愉快的游戏体验。")
            
            if overall["arousal"] < 0.4:
                insights.append("故事激活度较低，可能缺乏紧张感和刺激性。考虑增加更多动态和紧张元素。")
            elif overall["arousal"] > 0.7:
                insights.append("故事激活度较高，富有紧张感和刺激性，可能会让玩家保持高度投入。")
        
        # 事件情感连贯性评估
        if "events" in self.emotion_stats and len(self.event_emotions) >= 2:
            events = self.emotion_stats["events"]
            
            # 情感变化评估
            if events["valence_variance"] < 0.05:
                insights.append("事件之间的情感效价变化较小，可能导致故事体验单调。建议增加事件间的情感对比。")
            elif events["valence_variance"] > 0.2:
                insights.append("事件之间的情感效价变化较大，有助于创造起伏感，但请确保这些变化是合理的。")
            
            if events["arousal_variance"] < 0.05:
                insights.append("事件之间的情感唤醒度变化较小，可能导致故事节奏平淡。建议添加高低起伏的情感节点。")
            elif events["arousal_variance"] > 0.2:
                insights.append("事件之间的情感唤醒度变化较大，有助于创造紧张与放松的节奏感。")
            
            # 主导情感评估
            dominant_emotions = self.get_dominant_emotions()
            if dominant_emotions:
                top_emotion, top_score = dominant_emotions[0]
                if top_score > 0.4:
                    insights.append(f"'{top_emotion}' 是故事中的主导情感。请确保这与您期望的游戏体验一致。")
        
        # 如果没有洞见，添加默认洞见
        if not insights:
            insights.append("故事情感分析完成，但未发现显著的情感模式。请结合游戏目标自行评估。")
        
        return insights
    
    def export_emotion_analysis(self, filepath):
        """导出情感分析报告
        
        Args:
            filepath: 文件保存路径
            
        Returns:
            bool: 是否成功导出
        """
        try:
            # 准备导出数据
            export_data = {
                "overall_stats": self.emotion_stats.get("overall", {}),
                "events_stats": self.emotion_stats.get("events", {}),
                "dominant_emotions": self.get_dominant_emotions(),
                "insights": self.generate_emotion_insights()
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # 保存为JSON文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            if self.logger:
                self.logger.log(f"情感分析报告已导出至 {filepath}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"导出情感分析报告失败: {e}", "ERROR")
            return False
    
    def _get_emotion_colormap(self, emotion_name):
        """获取情感对应的颜色映射
        
        Args:
            emotion_name: 情感名称
            
        Returns:
            matplotlib.colors.Colormap: 颜色映射
        """
        # 使用预定义的颜色方案
        if emotion_name not in self.emotion_colors:
            emotion_name = "valence"  # 默认使用valence的颜色方案
        
        return LinearSegmentedColormap.from_list(
            emotion_name, 
            self.emotion_colors[emotion_name], 
            N=256
        )