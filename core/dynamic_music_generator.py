import numpy as np
import pygame
from typing import Dict, List, Tuple, Any, Optional
import os
import time
import threading
from queue import Queue
import logging
from scipy.interpolate import interp1d
from core.generate_music import *

class DynamicMusicGenerator:
    """基于区域情感的动态音乐生成系统"""
    
    def __init__(self, map_width=0, map_height=0, music_resource_path="data/music", logger=None):
        """初始化音乐生成器
        
        Args:
            map_width: 地图宽度
            map_height: 地图高度
            music_resource_path: 音乐资源文件夹路径
            logger: 日志记录器
        """
        self.map_width = map_width
        self.map_height = map_height
        self.logger = logger
        self.resource_path = music_resource_path

        # 预先初始化重要属性
        self._last_volumes = {}
        self._last_mix_time = time.time()
        self._mix_frame_count = 0
        self._adaptation_rate = 0.1
        
        # 情感数据
        self.emotion_regions = {}  # 地图情感区域数据
        self.story_event_emotions = {}  # 故事事件情感数据
        
        # 音乐相关变量
        self.audio_mixer = None  # 音频混合器
        self.current_position = (0, 0)  # 当前位置
        self.is_playing = False  # 是否正在播放
        self.music_layers = {}  # 音乐层级
        
        # 音乐参数
        self.music_parameters = {
            "tempo": 100,  # BPM
            "scale": "major",  # 调式
            "instruments": [],  # 乐器配置
            "density": 0.5,  # 音符密度
            "reverb": 0.3,  # 混响
        }
        
        # 初始化混合权重
        self.current_weights = {}
        self.target_weights = {}
        
        # 音频处理线程和队列
        self.audio_queue = Queue(maxsize=10)
        self.audio_thread = None
        self.thread_active = False
        
        # 情感到音乐映射表
        self.emotion_music_mapping = self._initialize_emotion_mapping()
        
        # 初始化音频系统
        self._initialize_audio_system()
        
        # 添加: 生成模式相关属性
        self._generation_mode = "parallel"  # 默认为并行生成模式
        self.blend_factor = 0.5  # 混合因子默认值
        self.feedback_enabled = False  # 环境反馈默认禁用
        self.active_models = ["MusicVAE", "GrooVAE"]  # 默认激活的模型
        self.generation_order = ["structure", "melody", "harmony", "arrangement"]  # 默认生成顺序
        self.current_generation_stage = 0  # 当前生成阶段
        self.feedback_history = []  # 反馈历史记录
        self.last_feedback_time = time.time()  # 上次反馈时间
        
        if self.logger:
            self.logger.log("动态音乐生成器初始化完成")

    def set_blend_factor(self, blend_factor):
        """设置混合模式的融合因子
        
        Args:
            blend_factor: 融合因子 (0.0-1.0)，控制不同模型输出的混合比例
            
        Returns:
            bool: 是否成功设置
        """
        try:
            # 确保输入值在有效范围内
            self.blend_factor = max(0.0, min(1.0, blend_factor))
            
            if self.logger:
                self.logger.log(f"已设置融合因子: {self.blend_factor:.2f}")
            
            # 如果当前是混合模式，立即应用新的融合因子
            if hasattr(self, '_generation_mode') and self._generation_mode == 'hybrid':
                # 更新混合参数，可以在这里添加更多逻辑
                pass
                
            return True
        except Exception as e:
            if self.logger:
                self.logger.log(f"设置融合因子时出错: {e}", "ERROR")
        return False

    def enable_feedback(self, enable=True):
        """启用或禁用环境反馈机制，用于强化学习协作模式
        
        Args:
            enable: 是否启用反馈
            
        Returns:
            bool: 是否成功设置
        """
        try:
            self.feedback_enabled = enable
            
            if self.logger:
                status = "启用" if enable else "禁用"
                self.logger.log(f"已{status}环境反馈机制")
            
            # 根据反馈状态调整算法行为
            if hasattr(self, '_generation_mode') and self._generation_mode == 'reinforcement':
                # 初始化或重置反馈状态追踪
                self.feedback_history = []
                self.last_feedback_time = time.time()
            
            return True
        except Exception as e:
            if self.logger:
                self.logger.log(f"设置环境反馈时出错: {e}", "ERROR")
            return False

    def set_active_models(self, active_models):
        """设置并行生成模式中要使用的活跃模型
        
        Args:
            active_models: 模型名称列表
            
        Returns:
            bool: 是否成功设置
        """
        try:
            # 验证模型名称
            valid_models = ["MusicVAE", "GrooVAE", "Transformer", "GANSynth", "NSynth"]
            self.active_models = [model for model in active_models if model in valid_models]
            
            if self.logger:
                self.logger.log(f"已设置活跃模型: {', '.join(self.active_models)}")
            
            # 应用配置变更，可能需要调整音频通道和处理逻辑
            if hasattr(self, '_generation_mode') and self._generation_mode == 'parallel':
                # 根据活跃模型调整音频处理策略
                pass
            
            return True
        except Exception as e:
            if self.logger:
                self.logger.log(f"设置活跃模型时出错: {e}", "ERROR")
            return False

    def set_generation_order(self, order):
        """设置分层生成模式中音乐生成的顺序
        
        Args:
            order: 生成阶段的顺序列表，如 ["structure", "melody", "harmony", "arrangement"]
            
        Returns:
            bool: 是否成功设置
        """
        try:
            valid_stages = ["structure", "melody", "harmony", "rhythm", "bass", "arrangement"]
            self.generation_order = [stage for stage in order if stage in valid_stages]
            
            if self.logger:
                self.logger.log(f"已设置生成顺序: {' → '.join(self.generation_order)}")
            
            # 应用新的生成顺序
            if hasattr(self, '_generation_mode') and self._generation_mode == 'layered':
                # 根据生成顺序调整处理流程
                self.current_generation_stage = 0  # 重置生成阶段
            
            return True
        except Exception as e:
            if self.logger:
                self.logger.log(f"设置生成顺序时出错: {e}", "ERROR")
            return False

    def set_track_volume(self, track_name, volume):
        """设置特定音轨的音量
        
        Args:
            track_name: 音轨名称
            volume: 音量值 (0.0-1.0)
            
        Returns:
            bool: 是否成功设置
        """
        if not self.music_channels or track_name not in self.music_channels:
            if self.logger:
                self.logger.log(f"找不到音轨: {track_name}", "WARNING")
            return False
            
        try:
            channel = self.music_channels[track_name]
            channel.set_volume(volume)
            
            if self.logger:
                self.logger.log(f"已设置音轨 '{track_name}' 音量为 {volume:.2f}", "DEBUG")
            return True
        except Exception as e:
            if self.logger:
                self.logger.log(f"设置音轨 '{track_name}' 音量时出错: {e}", "ERROR")
            return False
    
    def _initialize_emotion_mapping(self):
        """初始化情感到音乐参数的映射表"""
        # 创建不同情感类型到音乐特征的映射
        mapping = {
            "joy": {
                "scale": "major",
                "tempo_factor": 0.8,
                "brightness": 0.9,
                "instruments": ["piano", "strings", "flute"],
                "rhythm_complexity": 0.7,
                "reverb": 0.3
            },
            "trust": {
                "scale": "major",
                "tempo_factor": 0.6,
                "brightness": 0.7,
                "instruments": ["piano", "guitar", "strings"],
                "rhythm_complexity": 0.4,
                "reverb": 0.4
            },
            "fear": {
                "scale": "minor",
                "tempo_factor": 0.7,
                "brightness": 0.3,
                "instruments": ["strings", "synth"],
                "rhythm_complexity": 0.6,
                "reverb": 0.6
            },
            "surprise": {
                "scale": "lydian",
                "tempo_factor": 0.7,
                "brightness": 0.6,
                "instruments": ["piano", "synth"],
                "rhythm_complexity": 0.8,
                "reverb": 0.4
            },
            "sadness": {
                "scale": "minor",
                "tempo_factor": 0.4,
                "brightness": 0.2,
                "instruments": ["piano", "strings"],
                "rhythm_complexity": 0.3,
                "reverb": 0.7
            },
            "disgust": {
                "scale": "phrygian",
                "tempo_factor": 0.5,
                "brightness": 0.3,
                "instruments": ["synth"],
                "rhythm_complexity": 0.5,
                "reverb": 0.5
            },
            "anger": {
                "scale": "minor",
                "tempo_factor": 0.9,
                "brightness": 0.5,
                "instruments": ["synth", "guitar"],
                "rhythm_complexity": 0.8,
                "reverb": 0.3
            },
            "anticipation": {
                "scale": "mixolydian",
                "tempo_factor": 0.6,
                "brightness": 0.7,
                "instruments": ["piano", "flute"],
                "rhythm_complexity": 0.6,
                "reverb": 0.4
            }
        }
        
        # 添加效价/唤醒度映射
        valence_arousal_map = {
            # 高效价高唤醒 (愉快兴奋)
            (0.7, 0.7): {
                "scale": "major",
                "tempo_factor": 0.9,
                "brightness": 0.9,
                "instruments": ["piano", "strings", "flute"],
                "rhythm_complexity": 0.8,
                "reverb": 0.3
            },
            # 高效价低唤醒 (放松满足)
            (0.7, 0.3): {
                "scale": "major",
                "tempo_factor": 0.5,
                "brightness": 0.8,
                "instruments": ["piano", "guitar"],
                "rhythm_complexity": 0.3,
                "reverb": 0.5
            },
            # 低效价高唤醒 (愤怒恐惧)
            (0.3, 0.7): {
                "scale": "minor",
                "tempo_factor": 0.8,
                "brightness": 0.3,
                "instruments": ["synth", "strings"],
                "rhythm_complexity": 0.7,
                "reverb": 0.4
            },
            # 低效价低唤醒 (悲伤抑郁)
            (0.3, 0.3): {
                "scale": "minor",
                "tempo_factor": 0.4,
                "brightness": 0.2,
                "instruments": ["piano", "strings"],
                "rhythm_complexity": 0.2,
                "reverb": 0.7
            },
        }
        
        return {"emotions": mapping, "valence_arousal": valence_arousal_map}
    
    def _select_instruments_by_emotion(self, valence, arousal):
        """根据效价-唤醒值选择合适的乐器组合
        
        Args:
            valence: 效价值 (0-1)
            arousal: 唤醒值 (0-1)
            
        Returns:
            List[str]: 乐器标签列表
        """
        instruments = []
        
        # 基于效价选择乐器
        if valence > 0.7:  # 高效价
            instruments.extend(["piano", "flute", "strings"])
        elif valence < 0.3:  # 低效价
            instruments.extend(["synth", "strings"])
        else:  # 中等效价
            instruments.extend(["piano", "guitar"])
        
        # 基于唤醒度添加乐器
        if arousal > 0.7:  # 高唤醒
            if "synth" not in instruments:
                instruments.append("synth")
        elif arousal < 0.3:  # 低唤醒
            if "piano" not in instruments:
                instruments.append("piano")
        
        # 确保至少有2种乐器
        if len(instruments) < 2:
            if "piano" not in instruments:
                instruments.append("piano")
            if "strings" not in instruments and len(instruments) < 2:
                instruments.append("strings")
        
        return instruments
    
    def set_map_size(self, width, height):
        """设置地图尺寸
        
        Args:
            width: 地图宽度
            height: 地图高度
        """
        self.map_width = width
        self.map_height = height
    
    def load_emotion_data(self, emotion_manager=None, story_emotion_map=None):
        """从情感管理器和故事情感地图加载情感数据
        
        Args:
            emotion_manager: EmotionManager实例
            story_emotion_map: StoryEmotionMap实例
            
        Returns:
            bool: 是否成功加载数据
        """
        try:
            # 清空现有数据
            self.emotion_regions = {}
            self.story_event_emotions = {}
            
            # 加载地图情感数据
            if emotion_manager and emotion_manager.emotion_map:
                if self.logger:
                    self.logger.log("加载地图情感数据...")
                
                # 获取地图尺寸
                if emotion_manager.last_analysis_map_data:
                    map_data = emotion_manager.last_analysis_map_data
                    self.map_width = map_data.width
                    self.map_height = map_data.height
                
                # 处理各种情感热力图
                for emotion_name, heatmap in emotion_manager.emotion_map.items():
                    # 找出情感强度高的区域
                    threshold = np.max(heatmap) * 0.7  # 使用最大值的70%作为阈值
                    significant_indices = np.where(heatmap > threshold)
                    
                    if len(significant_indices[0]) > 0:
                        # 对每个重要区域创建一个区域条目
                        for i in range(len(significant_indices[0])):
                            y, x = significant_indices[0][i], significant_indices[1][i]
                            region_id = f"{emotion_name}_{x}_{y}"
                            
                            # 计算区域边界（向外扩展5个单位）
                            x_start, x_end = max(0, x-5), min(self.map_width-1, x+5)
                            y_start, y_end = max(0, y-5), min(self.map_height-1, y+5)
                            
                            intensity = heatmap[y, x]
                            self.emotion_regions[region_id] = {
                                "emotion": emotion_name,
                                "valence": 0.7 if emotion_name in ["joy", "trust", "surprise", "anticipation"] else 0.3,
                                "arousal": 0.7 if emotion_name in ["fear", "anger", "surprise", "anticipation"] else 0.3,
                                "intensity": float(intensity),
                                "bounding_box": (x_start, x_end, y_start, y_end)
                            }
            
            # 加载故事事件情感数据
            if story_emotion_map and story_emotion_map.event_emotions:
                if self.logger:
                    self.logger.log("加载故事事件情感数据...")
                
                # 获取地图尺寸
                self.map_width = story_emotion_map.width or self.map_width
                self.map_height = story_emotion_map.height or self.map_height
                
                # 处理每个故事事件
                for i, event in enumerate(story_emotion_map.event_emotions):
                    x, y = event["position"]
                    analysis = event["analysis"]
                    
                    if 0 <= x < self.map_width and 0 <= y < self.map_height:
                        # 计算效价和唤醒度
                        valence, arousal = analysis.global_emotion.to_valence_arousal()
                        
                        # 获取主导情感
                        dominant_emotion = ""
                        max_score = 0
                        for emotion, score in analysis.global_emotion.emotion_categories.items():
                            if score > max_score:
                                max_score = score
                                dominant_emotion = emotion
                        
                        # 创建事件情感区域
                        event_id = f"story_event_{i}"
                        # 区域范围（按事件强度调整）
                        radius = 10 + int(max_score * 10)
                        x_start, x_end = max(0, int(x)-radius), min(self.map_width-1, int(x)+radius)
                        y_start, y_end = max(0, int(y)-radius), min(self.map_height-1, int(y)+radius)
                        
                        self.story_event_emotions[event_id] = {
                            "emotion": dominant_emotion,
                            "valence": float(valence),
                            "arousal": float(arousal),
                            "intensity": float(max_score),
                            "complexity": float(analysis.emotional_complexity),
                            "variance": float(analysis.emotional_variance),
                            "bounding_box": (x_start, x_end, y_start, y_end)
                        }
            
            # 整合数据，故事事件优先级高于地图情感
            combined_regions = {}
            combined_regions.update(self.emotion_regions)
            combined_regions.update(self.story_event_emotions)
            self.emotion_regions = combined_regions
            
            # 生成音乐参数
            self._generate_music_parameters()
            
            if self.logger:
                total_regions = len(self.emotion_regions)
                self.logger.log(f"加载了 {total_regions} 个情感区域数据")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"加载情感数据出错: {e}", "ERROR")
                import traceback
                self.logger.log(traceback.format_exc(), "DEBUG")
            return False
    
    def _generate_music_parameters(self):
        """为每个情感区域生成音乐参数"""
        if not self.emotion_regions:
            return
            
        # 缓存常用情感类型参数
        emotion_param_cache = {}
        valence_arousal_cache = {}
        
        for region_id, region_data in self.emotion_regions.items():
            emotion_type = region_data.get("emotion", "")
            valence = region_data.get("valence", 0.5)
            arousal = region_data.get("arousal", 0.5)
            intensity = region_data.get("intensity", 0.5)
            
            # 尝试从缓存获取参数
            cache_key = emotion_type if emotion_type else f"va_{valence:.1f}_{arousal:.1f}"
            
            if cache_key in emotion_param_cache:
                params = emotion_param_cache[cache_key]
            else:
                # 获取基础音乐参数映射
                if emotion_type and emotion_type in self.emotion_music_mapping["emotions"]:
                    # 直接从情感映射中获取参数
                    params = self.emotion_music_mapping["emotions"][emotion_type].copy()
                    
                    # 一次性添加所有缺失的键
                    defaults = {
                        "instrument_tags": params.get("instruments", ["piano"]),
                        "density_factor": params.get("rhythm_complexity", 0.5),
                        "reverb_factor": params.get("reverb", 0.3)
                    }
                    params.update({k: v for k, v in defaults.items() if k not in params})
                    
                    # 缓存结果
                    emotion_param_cache[cache_key] = params
                else:
                    # 使用效价-唤醒度查找最接近的参数
                    va_key = f"{valence:.1f}_{arousal:.1f}"
                    if va_key in valence_arousal_cache:
                        params = valence_arousal_cache[va_key]
                    else:
                        params = self._find_closest_valence_arousal(valence, arousal)
                        valence_arousal_cache[va_key] = params
            
            # 预先计算最终参数
            base_tempo = 100  # 基础速度
            tempo = base_tempo * params["tempo_factor"]
            
            # 避免重复选择音乐文件
            filename_key = f"{params['scale']}_{'-'.join(params['instrument_tags'])}_{params['tempo_factor']:.1f}"
            if not hasattr(self, '_music_file_cache'):
                self._music_file_cache = {}
                
            if filename_key not in self._music_file_cache:
                selected_files = self._select_music_files(
                    params["scale"],
                    params["instrument_tags"],
                    params["tempo_factor"]
                )
                self._music_file_cache[filename_key] = selected_files
            else:
                selected_files = self._music_file_cache[filename_key]
            
            # 直接修改region_data，避免创建新字典
            region_data["music_params"] = {
                "tempo": tempo,
                "scale": params["scale"],
                "instruments": params["instrument_tags"],
                "density": params["density_factor"] * intensity,
                "reverb": params["reverb_factor"],
                "filename": selected_files
            }
            
    def _find_closest_valence_arousal(self, valence, arousal):
        """查找最接近给定效价-唤醒度的音乐参数映射，使用高效空间索引
        
        Args:
            valence: 效价值 (0-1)
            arousal: 唤醒值 (0-1)
            
        Returns:
            dict: 音乐参数
        """
        va_map = self.emotion_music_mapping["valence_arousal"]
        if not va_map:
            return {
                "scale": "major",
                "tempo_factor": 0.6,
                "brightness": 0.5,
                "instruments": ["piano"],
                "rhythm_complexity": 0.5,
                "reverb": 0.3,
                "instrument_tags": ["piano"],
                "density_factor": 0.5,
                "reverb_factor": 0.3
            }
        
        # 初始化空间索引结构（仅第一次调用）
        if not hasattr(self, '_va_space_index'):
            # 构建包含点和对应参数的数据结构
            self._va_points = list(va_map.keys())
            self._va_params = list(va_map.values())
            
            # 简单实现：对于有限的点集，使用预计算距离表比构建KD树等复杂结构更快
            # 如果VA点非常多，可以改用scipy.spatial.KDTree
        
        # 查找最近的VA点
        min_distance = float('inf')
        closest_idx = 0
        
        # 使用更高效的距离计算：提前计算平方，避免开平方根
        # 因为我们只关心相对大小，不需要精确距离值
        for idx, (v, a) in enumerate(self._va_points):
            # 使用平方距离代替欧几里得距离
            # 避免了平方根计算，性能更好
            dist_squared = (v - valence)**2 + (a - arousal)**2
            if dist_squared < min_distance:
                min_distance = dist_squared
                closest_idx = idx
        
        # 获取最近点的参数
        closest_params = self._va_params[closest_idx]
        
        # 直接使用字典推导式高效处理所有默认值
        result = closest_params.copy()
        defaults = {
            "instrument_tags": result.get("instruments", ["piano"]),
            "density_factor": result.get("rhythm_complexity", 0.5),
            "reverb_factor": result.get("reverb", 0.3)
        }
        result.update({k: v for k, v in defaults.items() if k not in result})
        
        return result
        
    def _select_music_files(self, scale_type, instrument_tags, tempo_factor):
        """根据音乐参数选择适合的音乐文件
        
        Args:
            scale_type: 音阶类型（如 'major', 'minor'）
            instrument_tags: 乐器标签列表
            tempo_factor: 速度因子 (0.0-1.0)，决定音乐速度
            
        Returns:
            dict: 包含各个音轨的音乐文件路径
        """
        # 检查资源路径
        if not os.path.exists(self.resource_path):
            if self.logger:
                self.logger.log(f"音乐资源目录不存在: {self.resource_path}")
            return {}
        
        # 初始化文件系统缓存（只在首次调用时执行）
        if not hasattr(self, '_track_files_cache'):
            self._track_files_cache = {}
            self._available_files_by_dir = {}
        
        # 根据速度因子确定速度类型
        speed = "medium"
        if tempo_factor > 0.7:
            speed = "fast"
        elif tempo_factor < 0.4:
            speed = "slow"
        
        # 确定情感类型与音阶的映射关系
        mood = "neutral"
        if scale_type == "major":
            mood = "positive"
        elif scale_type == "minor":
            mood = "negative"
        
        # 需要选择的音轨类型
        track_types = {
            "melody": os.path.join(self.resource_path, "melody"),
            "harmony": os.path.join(self.resource_path, "harmony"),
            "percussion": os.path.join(self.resource_path, "percussion"),
            "ambient": os.path.join(self.resource_path, "ambient"),
            "instruments": os.path.join(self.resource_path, "instruments")
        }
        
        # 最终选择的文件
        selected_files = {}
        
        # 使用一个查询模板字典，避免重复构造相同的查询条件
        query_templates = {
            "melody": [
                f"{mood}_melody_{speed}",  # 首选：匹配情绪和速度
                f"{mood}_melody_",         # 次选：匹配情绪
                "default_melody.wav"       # 兜底：默认旋律
            ],
            "harmony": [
                f"{mood}_harmony_",        # 首选：匹配情绪
                "default_harmony.wav"      # 兜底：默认和声
            ],
            "percussion": [
                f"percussion_{speed}",     # 首选：匹配速度
                "percussion_",             # 次选：任意速度
                "default_percussion.wav"   # 兜底：默认打击乐
            ],
            "ambient": [
                f"ambient_{mood}",         # 首选：匹配情绪
                "ambient_"                 # 次选：任意情绪
            ],
            "instruments": []              # 乐器会动态构建查询
        }
        
        import random
        from functools import lru_cache
        
        # 使用LRU缓存加速文件查找
        @lru_cache(maxsize=128)
        def find_matching_files(track_path, prefix):
            """缓存文件查找结果"""
            # 获取目录中的可用文件（只读取一次）
            if track_path not in self._available_files_by_dir:
                if not os.path.exists(track_path):
                    self._available_files_by_dir[track_path] = []
                else:
                    self._available_files_by_dir[track_path] = [
                        f for f in os.listdir(track_path) if f.endswith('.wav')
                    ]
                    
            available_files = self._available_files_by_dir[track_path]
            
            # 如果是精确文件名匹配
            if prefix.endswith('.wav'):
                return [prefix] if prefix in available_files else []
                
            # 否则进行前缀匹配
            return [f for f in available_files if f.startswith(prefix)]
        
        # 处理每个音轨类型
        for track_type, track_path in track_types.items():
            # 跳过不存在的目录
            if not os.path.exists(track_path):
                if self.logger:
                    self.logger.log(f"音轨目录不存在: {track_path}", "WARNING")
                continue
            
            matching_files = []
            
            # 处理乐器特殊情况
            if track_type == "instruments":
                # 为每种乐器构建查询
                for tag in instrument_tags:
                    tag_prefix = f"{tag}_{mood}"
                    tag_files = find_matching_files(track_path, tag_prefix)
                    matching_files.extend(tag_files)
                    
                # 如果没有匹配到任何文件，尝试任意乐器
                if not matching_files:
                    available_files = self._available_files_by_dir.get(track_path, [])
                    if available_files:
                        matching_files = [random.choice(available_files)]
            else:
                # 使用查询模板顺序查找
                for prefix in query_templates[track_type]:
                    matching_files = find_matching_files(track_path, prefix)
                    if matching_files:
                        break
            
                # 如果找到了匹配文件，随机选择一个并添加到结果中
            if matching_files:
                selected_file = random.choice(matching_files)
                selected_files[track_type] = os.path.join(track_path, selected_file)
                # 修复日志级别检查
                try:
                    if self.logger and hasattr(self.logger, 'log_level') and self.logger.log_level <= 1:
                        self.logger.log(f"为 {track_type} 选择了音频文件: {selected_file}")
                    elif self.logger:
                        # 在没有log_level属性时，也记录详细信息
                        self.logger.log(f"为 {track_type} 选择了音频文件: {selected_file}", "DEBUG")
                except Exception:
                    # 异常安全的日志记录
                    if self.logger:
                        self.logger.log(f"为 {track_type} 选择了音频文件: {selected_file}")
        
        return selected_files
    
    def update_position(self, x, y):
        """更新当前位置并重新计算音乐混合权重
        
        Args:
            x: X坐标
            y: Y坐标
        """
        if not self.emotion_regions:
            return
            
        # 如果位置变化不大，跳过更新（提高性能）
        if hasattr(self, 'current_position'):
            old_x, old_y = self.current_position
            distance_squared = (x - old_x)**2 + (y - old_y)**2
            
            # 如果移动距离平方小于阈值，不更新
            if distance_squared < 1.0:  
                return
        
        # 更新位置
        self.current_position = (x, y)
        
        # 计算新的权重
        self.target_weights = self._calculate_region_weights(x, y)
        
        # 额外优化：标记只更新受影响的区域，而不是全部区域
        if not hasattr(self, '_affected_regions'):
            self._affected_regions = set(self.emotion_regions.keys())
        else:
            # 计算可能受到影响的区域（基于距离）
            self._affected_regions = self._find_affected_regions(x, y)

    def _find_affected_regions(self, x, y, max_distance=50.0):
        """找出可能受位置变化影响的情感区域
        
        Args:
            x: X坐标
            y: Y坐标
            max_distance: 最大影响距离
            
        Returns:
            set: 受影响的区域ID集合
        """
        affected = set()
        max_dist_squared = max_distance * max_distance
        
        for region_id, region_data in self.emotion_regions.items():
            bb = region_data.get("bounding_box", (0, 0, 0, 0))
            x_start, x_end, y_start, y_end = bb
            
            # 计算区域中心
            x_center = (x_start + x_end) / 2
            y_center = (y_start + y_end) / 2
            
            # 计算到区域中心的距离平方
            dist_squared = (x - x_center)**2 + (y - y_center)**2
            
            # 如果在影响范围内，添加到受影响集合
            if dist_squared <= max_dist_squared:
                affected.add(region_id)
        
        return affected

    def _calculate_region_weights(self, x, y, sigma=20.0):
        """计算各区域的影响权重，使用优化的高斯衰减
        
        Args:
            x: X坐标
            y: Y坐标
            sigma: 高斯函数的sigma参数，影响衰减速度
            
        Returns:
            Dict[str, float]: 区域ID到权重的映射
        """
        weights = {}
        
        # 预计算高斯衰减常量
        neg_half_sigma_squared_inv = -0.5 / (sigma * sigma)
        
        # 初始化空间索引（如果还没有）
        if not hasattr(self, '_region_spatial_index'):
            self._build_spatial_index()
        
        # 计算每个区域的权重
        for region_id, region_data in self.emotion_regions.items():
            bb = region_data.get("bounding_box", (0, 0, 0, 0))
            x_start, x_end, y_start, y_end = bb
            
            # 计算区域中心
            x_center = (x_start + x_end) / 2
            y_center = (y_start + y_end) / 2
            
            # 计算区域半径
            radius = max(x_end - x_start, y_end - y_start) / 2
            
            # 计算欧氏距离平方（避免开平方根）
            distance_squared = (x - x_center)**2 + (y - y_center)**2
            distance = distance_squared**0.5  # 只在必要时计算平方根
            
            # 应用高斯衰减，优化数学计算
            if distance <= radius:  # 如果在区域内
                # 区域内权重为1.0
                weight = 1.0
            else:
                # 区域外权重根据距离衰减
                # 使用预计算常量优化指数计算
                normalized_distance = (distance - radius)
                weight = np.exp(neg_half_sigma_squared_inv * normalized_distance * normalized_distance)
            
            # 考虑情感强度
            intensity = region_data.get("intensity", 0.5)
            weight *= intensity
            
            # 仅在权重显著时考虑该区域（微优化）
            if weight > 0.01:
                weights[region_id] = weight
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            # 直接修改weights字典，避免创建新字典
            for k in weights:
                weights[k] /= total_weight
        
        return weights

    def _build_spatial_index(self):
        """构建情感区域的空间索引，以加速区域查找"""
        # 简单实现：对于中等数量的区域，使用网格索引足够高效
        # 如果区域非常多，可以考虑使用四叉树等更复杂的空间索引结构
        
        # 创建网格索引
        grid_size = 50  # 网格尺寸
        self._region_grid = {}
        
        for region_id, region_data in self.emotion_regions.items():
            bb = region_data.get("bounding_box", (0, 0, 0, 0))
            x_start, x_end, y_start, y_end = bb
            
            # 计算区域覆盖的网格单元
            grid_x_start = max(0, x_start // grid_size)
            grid_x_end = max(0, x_end // grid_size + 1)
            grid_y_start = max(0, y_start // grid_size)
            grid_y_end = max(0, y_end // grid_size + 1)
            
            # 将区域添加到相应的网格单元
            for gx in range(grid_x_start, grid_x_end):
                for gy in range(grid_y_start, grid_y_end):
                    grid_key = (gx, gy)
                    if grid_key not in self._region_grid:
                        self._region_grid[grid_key] = set()
                    self._region_grid[grid_key].add(region_id)
    
    def load_music_resources(self):
        """加载音乐资源文件"""
        if not self.emotion_regions:
            if self.logger:
                self.logger.log("没有情感区域数据，无法加载音乐资源", "WARNING")
            return False
            
        try:
            # 确保音频系统已初始化
            if not self.audio_mixer:
                self._initialize_audio_system()
                if not self.audio_mixer:
                    if self.logger:
                        self.logger.log("音频系统未初始化，无法加载音乐资源", "ERROR")
                    return False
            
            self.music_layers = {}
            
            if self.logger:
                self.logger.log("开始加载音乐资源...")
            
            for region_id, region_data in self.emotion_regions.items():
                # 获取音乐参数
                music_params = region_data.get("music_params", {})
                filenames = music_params.get("filename", {})
                
                if not filenames:
                    continue
                    
                # 加载音乐层
                region_layers = {}
                for layer_name, file_path in filenames.items():
                    # 检查文件是否存在
                    if os.path.exists(file_path):
                        try:
                            # 使用pygame加载音频
                            sound = pygame.mixer.Sound(file_path)
                            region_layers[layer_name] = {
                                "sound": sound,
                                "channel": None,
                                "volume": 0.0
                            }
                            if self.logger:
                                self.logger.log(f"加载音频: {file_path}")
                        except Exception as e:
                            if self.logger:
                                self.logger.log(f"无法加载音频 {file_path}: {e}", "WARNING")
                    else:
                        if self.logger:
                            self.logger.log(f"音频文件不存在: {file_path}", "WARNING")
                
                # 存储区域的音乐层
                if region_layers:
                    self.music_layers[region_id] = region_layers
            
            if self.logger:
                num_regions = len(self.music_layers)
                total_layers = sum(len(layers) for layers in self.music_layers.values())
                self.logger.log(f"音乐资源加载完成: {num_regions}个区域, {total_layers}个音频层")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"加载音乐资源时出错: {e}", "ERROR")
                import traceback
                self.logger.log(traceback.format_exc(), "DEBUG")
            return False
    
    def start_playback(self):
        """开始播放音乐"""
        if not self.music_layers:
            if self.logger:
                self.logger.log("没有加载音乐资源，无法开始播放", "WARNING")
            return False
            
        try:
            # 如果已经在播放，先停止
            if self.is_playing:
                self.stop_playback()
            
            # 启动音频处理线程
            self.thread_active = True
            self.audio_thread = threading.Thread(target=self._audio_process_thread)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # 设置播放状态
            self.is_playing = True
            
            if self.logger:
                self.logger.log("开始播放动态音乐")

            # 查找一个有效的区域资源用于初始播放
            if self.current_weights:
                # 如果已有权重，使用权重最高的区域
                dominant_region = max(self.current_weights.items(), key=lambda x: x[1])[0]
            elif self.music_layers:
                # 否则使用任意一个已加载的区域
                dominant_region = next(iter(self.music_layers))
            else:
                if self.logger:
                    self.logger.log("没有可用的音乐区域，无法开始播放", "WARNING")
                return False
                
            # 获取区域的音乐资源
            region_layers = self.music_layers.get(dominant_region, {})
            
            # 强制所有轨道开始播放
            for track_name, channel in self.music_channels.items():
                if track_name in region_layers:
                    sound = region_layers[track_name]["sound"]
                    channel.set_volume(0.7)  # 设置初始音量
                    channel.play(sound, loops=-1)
                    self._last_volumes[track_name] = 0.7
                    if self.logger:
                        self.logger.log(f"强制开始播放 {track_name} 音轨", "INFO")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"开始播放时出错: {e}", "ERROR")
            return False
    
    def stop_playback(self):
        """停止播放音乐"""
        if not self.is_playing:
            return
            
        try:
            # 停止所有声音
            pygame.mixer.stop()
            
            # 停止音频处理线程
            self.thread_active = False
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1.0)
            
            # 重置播放状态
            self.is_playing = False
            
            if self.logger:
                self.logger.log("停止播放音乐")
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"停止播放时出错: {e}", "ERROR")
    
    def _audio_process_thread(self):
        """音频处理线程，负责动态调整音轨混合"""
        if self.logger:
            self.logger.log("音频处理线程已启动")
        
        last_update_time = time.time()
        update_interval = 0.1  # 每0.1秒更新一次
        
        while self.thread_active:
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                self._update_audio_mix()
                last_update_time = current_time
            
            # 检查队列中是否有音频更新任务
            try:
                if not self.audio_queue.empty():
                    task = self.audio_queue.get(block=False)
                    # 处理音频任务...
                    self.audio_queue.task_done()
            except Exception:
                pass
            
            time.sleep(0.01)  # 避免CPU占用过高
        
        if self.logger:
            self.logger.log("音频处理线程已停止")
    
    def _update_audio_mix(self):
        """更新音频混合，使用高效的淡入淡出算法"""
        if not self.is_playing or not self.audio_mixer:
            return
        
        # 应用自适应淡变速率
        current_time = time.time()
        if not hasattr(self, '_last_mix_time'):
            self._last_mix_time = current_time
            self._mix_frame_count = 0
            self._adaptation_rate = 0.1  # 初始淡变速率
        
        # 计算时间增量，用于自适应调整
        delta_time = current_time - self._last_mix_time
        self._last_mix_time = current_time
        
        # 增加帧计数
        self._mix_frame_count += 1
        
        # 每30帧调整一次淡变速率
        if self._mix_frame_count % 30 == 0:
            # 如果帧率高于60fps，降低淡变速率，否则提高速率
            fps = 1.0 / max(delta_time, 0.001)  # 防止除零
            if fps > 60:
                self._adaptation_rate = max(0.05, self._adaptation_rate * 0.95)
            else:
                self._adaptation_rate = min(0.3, self._adaptation_rate * 1.05)
        
        # 直接使用目标权重更新当前权重
        fade_factor = self._adaptation_rate
        
        # 深度优化：仅更新有显著变化的权重
        weight_changes = False
        
        for region_id, target_weight in self.target_weights.items():
            current = self.current_weights.get(region_id, 0.0)
            if abs(target_weight - current) > 0.01:
                new_weight = current + (target_weight - current) * fade_factor
                self.current_weights[region_id] = new_weight
                weight_changes = True
        
        # 处理应该淡出的区域
        regions_to_fade_out = [r for r in self.current_weights 
                            if r not in self.target_weights and self.current_weights[r] > 0.01]
        for region_id in regions_to_fade_out:
            self.current_weights[region_id] *= (1 - fade_factor)
            if self.current_weights[region_id] < 0.01:
                self.current_weights[region_id] = 0
            weight_changes = True
        
        if weight_changes and self.logger:
            self.logger.log(f"权重已更新: 主导区域 {region_id}, 权重 {target_weight:.2f}", "DEBUG")
        
        # 如果权重有显著变化，更新混合
        if weight_changes:
            # 根据权重选择和混合音轨
            self._select_and_mix_tracks_optimized()

    def _select_and_mix_tracks_optimized(self):
        """优化版音轨选择与混合，使用更高效的算法"""
        if not self.music_channels or not self.current_weights:
            return
        
        # 使用音量变化阈值避免频繁的微小调整
        volume_threshold = 0.005
        
        # 查找主导情感区域
        dominant_region = None
        max_weight = 0.0
        
        for region_id, weight in self.current_weights.items():
            if weight > max_weight and region_id in self.emotion_regions:
                max_weight = weight
                dominant_region = region_id
        
        if not dominant_region or dominant_region not in self.emotion_regions:
            return
        
        # 获取区域的音乐层
        if dominant_region not in self.music_layers:
            # 如果是首次处理该区域，延迟加载其音频
            if not self._lazy_load_region_audio(dominant_region):
                return
        
        region_layers = self.music_layers[dominant_region]
        
        # 缓存当前音量状态，用于检测变化
        if not hasattr(self, '_last_volumes'):
            self._last_volumes = {}
        
        # 基础音量和音量因子
        volume_base = 0.7
        volume_factor = max_weight
        
        # 音量变化速率（根据情感转变速度动态调整）
        volume_change_rate = 0.2
        
        # 优化：按音轨类型的重要性排序进行处理
        track_priorities = ["melody", "harmony", "percussion", "ambient", "instruments"]
        
        for track_type in track_priorities:
            if track_type in region_layers and track_type in self.music_channels:
                layer_data = region_layers[track_type]
                channel = self.music_channels[track_type]
                sound = layer_data["sound"]
                
                # 计算目标音量（每个轨道的相对重要性不同）
                if track_type == "melody":
                    target_volume = volume_base * volume_factor
                elif track_type == "harmony":
                    target_volume = volume_base * 0.7 * volume_factor
                elif track_type == "percussion":
                    target_volume = volume_base * 0.8 * volume_factor
                elif track_type == "ambient":
                    target_volume = volume_base * 0.5 * volume_factor
                else:
                    target_volume = volume_base * 0.6 * volume_factor
                
                # 获取当前音量
                current_volume = self._last_volumes.get(track_type, 0)
                
                # 日志记录
                if self.logger:
                    self.logger.log(f"处理音轨 {track_type}, 目标音量 {target_volume:.2f}, 当前音量 {current_volume:.2f}", "DEBUG")

                # 添加：确保初始音量足够高，避免长时间的淡入过程
                if current_volume == 0 and target_volume > 0.1:
                    current_volume = 0.3  # 设置一个合理的初始音量
                    self._last_volumes[track_type] = current_volume
                
                # 仅当音量变化显著时才更新
                if abs(target_volume - current_volume) > volume_threshold:
                    # 平滑过渡到目标音量
                    new_volume = current_volume + (target_volume - current_volume) * volume_change_rate
                    new_volume = max(0.0, min(1.0, new_volume))  # 确保在合法范围内
                    
                    try:
                        # 设置新音量
                        channel.set_volume(new_volume)
                        self._last_volumes[track_type] = new_volume
                        
                        # 如果需要开始播放
                        if new_volume > 0.01 and not channel.get_busy():  # 降低阈值，更容易触发播放
                            # 首次播放时强制音量更高以便听到
                            first_volume = max(0.3, new_volume)
                            channel.set_volume(first_volume)
                            channel.play(sound, loops=-1)
                            if self.logger:
                                self.logger.log(f"开始播放 {track_type} 音轨，音量: {first_volume:.2f}", "DEBUG")
                        
                        # 如果音量降为0，考虑停止播放
                        elif new_volume < 0.05 and channel.get_busy():
                            channel.fadeout(300)  # 平滑淡出
                            
                    except Exception as e:
                        if self.logger:
                            self.logger.log(f"调整 {track_type} 音轨出错: {str(e)}", "WARNING")

    def _lazy_load_region_audio(self, region_id):
        """延迟加载区域音频资源"""
        if region_id not in self.emotion_regions:
            return False
            
        region_data = self.emotion_regions[region_id]
        music_params = region_data.get("music_params", {})
        filenames = music_params.get("filename", {})
        
        if not filenames:
            return False
            
        # 加载音乐层
        region_layers = {}
        for layer_name, file_path in filenames.items():
            if os.path.exists(file_path):
                try:
                    # 使用pygame加载音频
                    sound = pygame.mixer.Sound(file_path)
                    region_layers[layer_name] = {
                        "sound": sound,
                        "channel": None,
                        "volume": 0.0
                    }
                except Exception:
                    continue
        
        # 存储区域的音乐层
        if region_layers:
            self.music_layers[region_id] = region_layers
            return True
        
        return False

    
    def get_currently_playing_regions(self):
        """获取当前正在播放的区域信息
        
        Returns:
            List[Dict]: 区域信息列表
        """
        if not self.is_playing:
            return []
            
        playing_regions = []
        for region_id, weight in self.current_weights.items():
            if weight > 0.01 and region_id in self.emotion_regions:
                region_data = self.emotion_regions[region_id]
                playing_regions.append({
                    "id": region_id,
                    "weight": weight,
                    "emotion": region_data.get("emotion", ""),
                    "valence": region_data.get("valence", 0),
                    "arousal": region_data.get("arousal", 0),
                    "music_params": region_data.get("music_params", {})
                })
        
        # 按权重排序
        return sorted(playing_regions, key=lambda x: x["weight"], reverse=True)
    
    def get_current_music_params(self):
        """获取当前混合后的音乐参数
        
        Returns:
            Dict: 音乐参数
        """
        if not self.is_playing or not self.current_weights:
            return {}
            
        # 从各区域的权重计算混合参数
        tempo = 0
        scales = {}
        instruments = set()
        density = 0
        reverb = 0
        total_weight = 0
        
        for region_id, weight in self.current_weights.items():
            if weight <= 0.01 or region_id not in self.emotion_regions:
                continue
                
            total_weight += weight
            region_data = self.emotion_regions[region_id]
            music_params = region_data.get("music_params", {})
            
            # 累加加权参数
            tempo += music_params.get("tempo", 100) * weight
            
            # 累加调式权重
            scale = music_params.get("scale", "major")
            if scale not in scales:
                scales[scale] = 0
            scales[scale] += weight
            
            # 累加乐器标签
            instruments.update(music_params.get("instruments", []))
            
            # 累加密度和混响
            density += music_params.get("density", 0.5) * weight
            reverb += music_params.get("reverb", 0.3) * weight
        
        # 防止除零错误
        if total_weight <= 0:
            return {}
            
        # 计算最终参数
        dominant_scale = max(scales.items(), key=lambda x: x[1])[0] if scales else "major"
        
        return {
            "tempo": tempo / total_weight,
            "scale": dominant_scale,
            "instruments": list(instruments),
            "density": density / total_weight,
            "reverb": reverb / total_weight
        }
    
    def create_region_map_visualization(self, figsize=(10, 8)):
        """创建区域地图可视化
        
        Args:
            figsize: 图形大小
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.figure import Figure
        
        if not self.emotion_regions:
            # 创建空白图
            fig = Figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "没有可用的情感区域数据", 
                   ha='center', va='center', fontsize=12)
            return fig
        
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # 绘制地图背景
        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.set_title("音乐情感区域图")
        
        # 绘制情感区域
        for region_id, region_data in self.emotion_regions.items():
            bb = region_data.get("bounding_box", (0, 0, 0, 0))
            x_start, x_end, y_start, y_end = bb
            
            # 获取区域情感
            emotion = region_data.get("emotion", "")
            valence = region_data.get("valence", 0.5)
            arousal = region_data.get("arousal", 0.5)
            
            # 为不同情感类型设置不同颜色
            color = "lightblue"
            if emotion in ["joy", "trust"]:
                color = "lightgreen"
            elif emotion in ["fear", "anger"]:
                color = "salmon"
            elif emotion in ["sadness", "disgust"]:
                color = "lightblue"
            elif emotion in ["surprise", "anticipation"]:
                color = "yellow"
            
            # 如果是故事事件
            if region_id.startswith("story_event"):
                # 边框加粗，半透明填充
                rect = Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                               fill=True, alpha=0.6, color=color, 
                               linewidth=2, edgecolor='black')
            else:
                # 普通情感区域
                rect = Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                               fill=True, alpha=0.3, color=color, 
                               linewidth=1, edgecolor='gray')
            
            ax.add_patch(rect)
            
            # 添加区域标签
            center_x = (x_start + x_end) / 2
            center_y = (y_start + y_end) / 2
            
            # 标签内容
            label = emotion
            if region_id.startswith("story_event"):
                event_index = region_id.split("_")[-1]
                label = f"事件 {event_index}: {emotion}"
            
            ax.text(center_x, center_y, label, 
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        # 绘制当前位置
        if self.current_position:
            x, y = self.current_position
            ax.plot(x, y, 'ro', markersize=8, markeredgecolor='black')
            ax.text(x, y+1, "当前位置", ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="red", alpha=0.9))
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', edgecolor='gray', alpha=0.3, label='积极情感'),
            Patch(facecolor='salmon', edgecolor='gray', alpha=0.3, label='紧张情感'),
            Patch(facecolor='lightblue', edgecolor='gray', alpha=0.3, label='消极情感'),
            Patch(facecolor='yellow', edgecolor='gray', alpha=0.3, label='惊奇/期待'),
            Patch(facecolor='white', edgecolor='black', alpha=1.0, label='故事事件')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 添加坐标轴标签
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        fig.tight_layout()
        return fig
    
    def save_region_map_visualization(self, filepath):
        """保存区域地图可视化到文件
        
        Args:
            filepath: 文件保存路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            fig = self.create_region_map_visualization()
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            
            if self.logger:
                self.logger.log(f"区域地图可视化已保存到 {filepath}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"保存区域地图可视化时出错: {e}", "ERROR")
            return False
    def generate_music_from_text(self, text, output_dir="data/music"):
        """根据文本内容生成音乐"""
        if not text or text.strip() == "":
            if self.logger:
                self.logger.log("无法从空文本生成音乐")
            return False
            
        # 使用generate_music.py中的extract_emotion_from_text分析情感
        try:
            from core.generate_music import extract_emotion_from_text, MusicGenerator
        except ImportError:
            if self.logger:
                self.logger.log("无法导入音乐生成模块")
            return False
        
        emotion_params = extract_emotion_from_text(text)
        
        if self.logger:
            self.logger.log(f"从文本提取的情感: {emotion_params['emotion']}, " 
                        f"效价: {emotion_params['valence']:.2f}, "
                        f"唤醒度: {emotion_params['arousal']:.2f}")
        
        # 使用MusicGenerator生成音乐
        generator = MusicGenerator(
            emotion=emotion_params["emotion"],
            valence=emotion_params["valence"],
            arousal=emotion_params["arousal"],
            intensity=emotion_params["intensity"]
        )
        
        # 生成音乐
        audio_tracks = generator.generate(duration=10.0, collab_mode="hybrid")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名前缀
        prefix = emotion_params["emotion"]
        
        # 确定速度
        if emotion_params["arousal"] > 0.7:
            speed = "fast"
        elif emotion_params["arousal"] < 0.3:
            speed = "slow"
        else:
            speed = "medium"
        
        # 确定情绪
        if emotion_params["valence"] > 0.6:
            mood = "positive"
        elif emotion_params["valence"] < 0.4:
            mood = "negative"
        else:
            mood = "neutral"
        
        # 保存各轨道
        melody_path = os.path.join(output_dir, "melody", f"{mood}_melody_{speed}_1.wav")
        harmony_path = os.path.join(output_dir, "harmony", f"{mood}_harmony_1.wav")
        percussion_path = os.path.join(output_dir, "percussion", f"percussion_{speed}_1.wav")
        ambient_path = os.path.join(output_dir, "ambient", f"ambient_{mood}.wav")
        # 添加第5个轨道 - 乐器音色
        instruments_path = os.path.join(output_dir, "instruments", f"{emotion_params['emotion']}_instruments_1.wav")
        
        # 创建子目录
        for dir_name in ["melody", "harmony", "percussion", "ambient", "instruments"]:
            os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
        
        # 保存文件
        generator.save_audio(audio_tracks["melody"], melody_path)
        generator.save_audio(audio_tracks["harmony"], harmony_path)
        generator.save_audio(audio_tracks["percussion"], percussion_path)
        generator.save_audio(audio_tracks["ambient"], ambient_path)
        # 保存第5个轨道的音频
        if "instruments" in audio_tracks:
            generator.save_audio(audio_tracks["instruments"], instruments_path)
        
        if self.logger:
            self.logger.log(f"已为文本生成音乐文件: {prefix} ({mood}, {speed})")
        
        return True

    def _initialize_audio_system(self):
        """初始化音频系统，添加资源管理功能"""
        try:
            import pygame.mixer
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            self.audio_mixer = pygame.mixer
            
            # 添加更多通道以支持复杂混合
            pygame.mixer.set_num_channels(16)  # 增加可用通道数
            
            # 修改通道名称，确保与轨道类型一致
            self.music_channels = {
                "melody": pygame.mixer.Channel(0),
                "harmony": pygame.mixer.Channel(1),
                "percussion": pygame.mixer.Channel(2),
                "ambient": pygame.mixer.Channel(3),
                "instruments": pygame.mixer.Channel(4)  # 修改从"effects"到"instruments"
            }
            
            # 添加音频资源管理
            self._audio_resources = {}  # 存储所有已加载的音频资源
            self._audio_usage_count = {}  # 跟踪资源使用次数
            self._audio_last_used = {}  # 跟踪资源上次使用时间
            
            # 设置资源管理参数
            self._max_audio_cache_size = 64  # 最大缓存音频数
            self._audio_cache_cleanup_interval = 60  # 清理间隔（秒）
            self._last_cache_cleanup = time.time()
            
            if self.logger:
                self.logger.log("音频系统初始化成功，已配置资源管理")
                
        except Exception as e:
            self.audio_mixer = None
            if self.logger:
                self.logger.log(f"音频系统初始化失败: {str(e)}", "ERROR")

    def _get_audio(self, file_path):
        """获取音频资源，使用缓存机制
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            pygame.mixer.Sound 或 None
        """
        # 检查是否已缓存
        if file_path in self._audio_resources:
            # 更新使用统计
            self._audio_usage_count[file_path] += 1
            self._audio_last_used[file_path] = time.time()
            return self._audio_resources[file_path]
        
        # 加载新资源
        try:
            if os.path.exists(file_path):
                sound = pygame.mixer.Sound(file_path)
                
                # 缓存资源
                self._audio_resources[file_path] = sound
                self._audio_usage_count[file_path] = 1
                self._audio_last_used[file_path] = time.time()
                
                # 检查是否需要清理缓存
                self._check_audio_cache_cleanup()
                
                return sound
        except Exception as e:
            if self.logger:
                self.logger.log(f"加载音频失败 {file_path}: {e}", "WARNING")
        
        return None

    def _check_audio_cache_cleanup(self):
        """检查并清理音频缓存"""
        current_time = time.time()
        
        # 每隔一段时间执行一次清理
        if len(self._audio_resources) > self._max_audio_cache_size or (
                current_time - self._last_cache_cleanup > self._audio_cache_cleanup_interval):
            
            # 如果缓存超过限制，清理最不常用的资源
            if len(self._audio_resources) > self._max_audio_cache_size:
                # 按使用频率和最后使用时间排序
                sorted_resources = sorted(
                    self._audio_resources.keys(),
                    key=lambda x: (self._audio_usage_count.get(x, 0), -self._audio_last_used.get(x, 0))
                )
                
                # 删除约20%最不常用的资源
                resources_to_remove = sorted_resources[:len(sorted_resources) // 5]
                
                for path in resources_to_remove:
                    if path in self._audio_resources:
                        del self._audio_resources[path]
                        if path in self._audio_usage_count:
                            del self._audio_usage_count[path]
                        if path in self._audio_last_used:
                            del self._audio_last_used[path]
                
                if self.logger:
                    self.logger.log(f"已清理 {len(resources_to_remove)} 个不常用的音频资源", "DEBUG")
            
            self._last_cache_cleanup = current_time

    def _blend_emotion_parameters(self, prev_emotion, next_emotion, blend_factor):
        """融合两种情感的音乐参数
        
        Args:
            prev_emotion: 前一个情感的音乐参数字典
            next_emotion: 下一个情感的音乐参数字典
            blend_factor: 融合因子 (0.0-1.0)，0为完全前一个情感，1为完全下一个情感
            
        Returns:
            dict: 融合后的参数
        """
        # 如果任一参数为空，返回另一个
        if not prev_emotion:
            return next_emotion
        if not next_emotion:
            return prev_emotion
        
        # 创建融合参数
        blended = {}
        
        # 数值型参数线性插值
        numeric_keys = ["tempo", "density", "reverb"]
        for key in numeric_keys:
            if key in prev_emotion and key in next_emotion:
                blended[key] = prev_emotion[key] * (1-blend_factor) + next_emotion[key] * blend_factor
        
        # 离散参数（如音阶）根据权重选择
        if "scale" in prev_emotion and "scale" in next_emotion:
            # 如果权重超过阈值，使用新的音阶，否则保留旧的音阶
            scale_change_threshold = 0.7
            blended["scale"] = next_emotion["scale"] if blend_factor > scale_change_threshold else prev_emotion["scale"]
        
        # 乐器集合合并，随融合因子增加而添加新乐器
        if "instruments" in prev_emotion and "instruments" in next_emotion:
            prev_instruments = set(prev_emotion["instruments"])
            next_instruments = set(next_emotion["instruments"])
            
            # 根据融合因子决定使用多少新乐器
            if blend_factor < 0.3:
                # 保持原有乐器为主
                blended["instruments"] = list(prev_instruments)
            elif blend_factor > 0.7:
                # 以新乐器为主
                blended["instruments"] = list(next_instruments)
            else:
                # 混合使用
                common_instruments = prev_instruments.intersection(next_instruments)
                unique_next = next_instruments - prev_instruments
                
                # 从新乐器中选择一部分
                num_new_instruments = int(len(unique_next) * blend_factor)
                selected_new = list(unique_next)[:num_new_instruments]
                
                blended["instruments"] = list(common_instruments) + selected_new
        
        return blended
            
    def set_generation_mode(self, mode):
        """设置音乐生成模式
        
        Args:
            mode: 生成模式 ('parallel', 'layered', 'hybrid', 'reinforcement')
        
        Returns:
            bool: 是否成功设置
        """
        valid_modes = ['parallel', 'layered', 'hybrid', 'reinforcement']
        if mode not in valid_modes:
            if self.logger:
                self.logger.log(f"无效的生成模式: {mode}", "ERROR")
            return False
        
        self._generation_mode = mode
        
        if self.logger:
            self.logger.log(f"已设置生成模式: {mode}")
        return True

    def set_parameters(self, params):
        """设置音乐参数
        
        Args:
            params: 参数字典
            
        Returns:
            bool: 是否成功设置
        """
        try:
            # 更新基本参数
            if "tempo" in params:
                self.music_parameters["tempo"] = params["tempo"]
            
            if "scale" in params:
                self.music_parameters["scale"] = params["scale"]
                
            if "density" in params:
                self.music_parameters["density"] = params["density"]
                
            # 更新效果参数
            if "effects" in params:
                effects = params["effects"]
                if "reverb" in effects:
                    self.music_parameters["reverb"] = effects["reverb"]
                
                # 可添加其他效果参数...
            
            # 实时应用参数
            if self.is_playing:
                # 调整各个音轨的效果
                for channel_name, channel in self.music_channels.items():
                    # 应用音量调整等
                    pass
            
            if self.logger:
                self.logger.log(f"已更新音乐参数: {params}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"设置音乐参数时出错: {e}", "ERROR")
            return False