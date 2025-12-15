import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Any, Optional
import re

# 导入情感分析模块
from core.emotional.analyze_story_emotion import EmotionAnalyzer
from core.emotional.plutchik_get_emotion_model import PlutchikEmotion

class StoryEmotionAnalyzer:
    """负责分析故事剧情的情感内容和弧线"""
    
    def __init__(self, logger=None):
        """初始化故事情感分析器"""
        self.logger = logger
        self.emotion_analyzer = EmotionAnalyzer()
        self.analysis_results = {}
        
    def analyze_story_content(self, story_content: Dict[str, Any]) -> Dict[str, Any]:
        """分析故事内容的情感
        
        Args:
            story_content: 包含整体故事和各事件详情的字典
            
        Returns:
            情感分析结果字典
        """
        if not story_content:
            if self.logger:
                self.logger.log("没有可分析的故事内容", "WARNING")
            return {}
            
        if self.logger:
            self.logger.log("开始分析故事情感...")
        
        analysis_results = {
            "overall": None,
            "events": [],
            "emotional_arc": [],
            "key_emotional_points": []
        }
        
        # 分析整体故事
        overall_story = story_content.get("overall_story", "")
        if overall_story:
            if self.logger:
                self.logger.log("分析整体故事情感...")
            overall_analysis = self.emotion_analyzer.analyze_story_emotions(overall_story)
            analysis_results["overall"] = overall_analysis
            analysis_results["emotional_arc"] = overall_analysis.emotional_arc
            analysis_results["key_emotional_points"] = overall_analysis.key_moments
            
            if self.logger:
                valence, arousal = overall_analysis.global_emotion.to_valence_arousal()
                self.logger.log(f"整体故事情感: valence={valence:.2f}, arousal={arousal:.2f}")
                self.logger.log(f"情感复杂度: {overall_analysis.emotional_complexity:.2f}")
        
        # 分析各个事件
        expanded_stories = story_content.get("expanded_stories", [])
        for i, event in enumerate(expanded_stories):
            event_content = event.get("expanded_content", "")
            if event_content:
                if self.logger:
                    self.logger.log(f"分析事件 {i+1} 情感...")
                event_analysis = self.emotion_analyzer.analyze_story_emotions(event_content)
                
                # 提取事件的元信息
                original_event = event.get("original_event", {})
                event_type = self._extract_event_type(event_content)
                
                # 记录分析结果
                analysis_results["events"].append({
                    "index": i,
                    "event": original_event,
                    "type": event_type,
                    "analysis": event_analysis,
                    "position": (original_event.get("x", 0), original_event.get("y", 0))
                })
        
        # 保存分析结果
        self.analysis_results = analysis_results
        
        if self.logger:
            self.logger.log(f"故事情感分析完成: 分析了1个整体故事和{len(analysis_results['events'])}个事件")
            
        return analysis_results
    
    def _extract_event_type(self, content: str) -> str:
        """从事件内容中提取事件类型"""
        # 查找"任务类型"行
        match = re.search(r"任务类型[：:]\s*(.+?)[\n\r]", content)
        if match:
            return match.group(1).strip()
            
        # 如果没找到，尝试其他可能的标识
        if "主线" in content:
            return "主线"
        elif "支线" in content:
            return "支线"
        elif "隐藏" in content:
            return "隐藏"
        else:
            return "未知"
    
    def create_emotional_arc_chart(self, figsize=(10, 6)) -> Figure:
        """创建情感弧线图表
        
        Returns:
            包含情感弧线的matplotlib图表
        """
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        if "emotional_arc" not in self.analysis_results or not self.analysis_results["emotional_arc"]:
            ax.text(0.5, 0.5, "没有情感弧线数据", ha='center', va='center')
            fig.suptitle("情感弧线")
            return fig
            
        # 提取情感弧线数据
        arc_data = self.analysis_results["emotional_arc"]
        
        # 分离valence和arousal数据
        segments = range(len(arc_data))
        valence_values = [point[0] for point in arc_data]
        arousal_values = [point[1] for point in arc_data]
        
        # 绘制valence曲线
        ax.plot(segments, valence_values, 'r-', label='情感效价 (Valence)', marker='o', linewidth=2)
        
        # 绘制arousal曲线
        ax.plot(segments, arousal_values, 'b-', label='情感唤醒度 (Arousal)', marker='s', linewidth=2)
        
        # 标记关键情感点
        key_points = self.analysis_results.get("key_emotional_points", [])
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
        
        # 添加网格线、图例和标签
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        ax.set_xlabel('故事段落')
        ax.set_ylabel('情感值')
        ax.set_ylim(0, 1)
        ax.set_xticks(segments)
        
        fig.suptitle('故事情感弧线分析', fontsize=14)
        fig.tight_layout()
        
        return fig
    
    def create_emotion_comparison_chart(self, figsize=(10, 6)) -> Figure:
        """创建事件情感对比图表
        
        Returns:
            包含事件情感对比的matplotlib图表
        """
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        events = self.analysis_results.get("events", [])
        if not events:
            ax.text(0.5, 0.5, "没有事件情感数据", ha='center', va='center')
            fig.suptitle("事件情感对比")
            return fig
            
        event_names = [f"事件{e['index']+1}" for e in events]
        valence_values = []
        arousal_values = []
        complexity_values = []
        
        for event in events:
            analysis = event.get("analysis")
            if analysis:
                v, a = analysis.global_emotion.to_valence_arousal()
                valence_values.append(v)
                arousal_values.append(a)
                complexity_values.append(analysis.emotional_complexity)
            else:
                valence_values.append(0)
                arousal_values.append(0)
                complexity_values.append(0)
        
        # 设置柱状图的宽度和位置
        x = np.arange(len(event_names))
        width = 0.25
        
        # 绘制三组柱状图
        bars1 = ax.bar(x - width, valence_values, width, label='情感效价', color='salmon')
        bars2 = ax.bar(x, arousal_values, width, label='情感唤醒度', color='skyblue')
        bars3 = ax.bar(x + width, complexity_values, width, label='情感复杂度', color='lightgreen')
        
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
        
        # 设置图表标题、标签等
        ax.set_xlabel('事件')
        ax.set_ylabel('情感值')
        ax.set_title('事件情感对比分析')
        ax.set_xticks(x)
        ax.set_xticklabels(event_names)
        ax.legend()
        
        fig.tight_layout()
        
        return fig
    
    def generate_emotional_insights(self) -> List[str]:
        """生成基于情感分析的洞见和建议
        
        Returns:
            洞见和建议列表
        """
        insights = []
        
        if not self.analysis_results or "overall" not in self.analysis_results:
            return ["没有足够的分析数据生成洞见"]
            
        overall = self.analysis_results.get("overall")
        if overall:
            # 情感弧线评估
            if overall.emotional_variance < 0.1:
                insights.append("故事情感变化较少，可能缺乏起伏感。建议增加情感高潮和低谷，提高情感对比度。")
            elif overall.emotional_variance > 0.3:
                insights.append("故事情感变化丰富，有较好的起伏感，有助于维持玩家的兴趣。")
                
            # 情感复杂度评估
            if overall.emotional_complexity < 0.3:
                insights.append("故事情感相对简单，可以考虑增加更丰富的情感层次和冲突。")
            elif overall.emotional_complexity > 0.6:
                insights.append("故事情感复杂丰富，可以给玩家带来深刻的情感体验。")
                
            # 整体情感基调评估
            valence, arousal = overall.global_emotion.to_valence_arousal()
            if valence < 0.4:
                insights.append("故事整体情感基调偏向消极，请确保这符合您的游戏风格和目标受众。")
            elif valence > 0.7:
                insights.append("故事整体情感基调较为积极，适合轻松愉快的游戏体验。")
                
            if arousal < 0.4:
                insights.append("故事激活度较低，可能缺乏紧张感和刺激性。考虑增加更多动态和紧张元素。")
            elif arousal > 0.7:
                insights.append("故事激活度较高，富有紧张感和刺激性，可能会让玩家保持高度投入。")
        
        # 事件间情感连贯性评估
        events = self.analysis_results.get("events", [])
        if len(events) >= 2:
            emotion_shifts = []
            for i in range(1, len(events)):
                prev_event = events[i-1].get("analysis")
                curr_event = events[i].get("analysis")
                if prev_event and curr_event:
                    prev_v, prev_a = prev_event.global_emotion.to_valence_arousal()
                    curr_v, curr_a = curr_event.global_emotion.to_valence_arousal()
                    shift = ((curr_v - prev_v)**2 + (curr_a - prev_a)**2)**0.5
                    emotion_shifts.append(shift)
            
            avg_shift = sum(emotion_shifts) / len(emotion_shifts) if emotion_shifts else 0
            if avg_shift < 0.2:
                insights.append("事件之间的情感变化较小，可能导致游戏体验单调。建议增加事件间的情感对比。")
            elif avg_shift > 0.5:
                insights.append("事件之间的情感变化较大，可能会给玩家带来情感上的起伏感，但请确保这些变化是合理的。")
        
        # 如果没有生成任何洞见，添加一个默认洞见
        if not insights:
            insights.append("故事情感分析完成，但未发现明显的情感模式。请根据游戏目标进行调整。")
            
        return insights