import numpy as np
import pygame
from typing import Dict, List, Tuple, Any, Optional
import os
import time
import threading
from queue import Queue
import logging
from scipy.interpolate import interp1d
from core.generate_music import extract_emotion_from_text, MusicGenerator
from core.dynamic_music_generator import DynamicMusicGenerator

class EnhancedMusicTransitionSystem:
    """情感区域驱动的音乐过渡系统，实现平滑的音乐变化效果"""
    
    def __init__(self, map_width=0, map_height=0, music_resource_path="data/music", logger=None):
        """初始化增强型音乐系统
        
        Args:
            map_width: 地图宽度
            map_height: 地图高度
            music_resource_path: 音乐资源文件夹路径
            logger: 日志记录器
        """
        # 使用现有的DynamicMusicGenerator作为基础
        self.music_generator = DynamicMusicGenerator(
            map_width=map_width, 
            map_height=map_height,
            music_resource_path=music_resource_path,
            logger=logger
        )
        self.logger = logger
        
        # 玩家追踪数据
        self.player_position = (0, 0)
        self.player_velocity = (0, 0)  # 移动速度和方向
        self.player_history = []  # 最近位置历史
        
        # 过渡控制参数
        self.transition_speed = 0.05  # 默认过渡速度
        self.current_transition = 0.0  # 当前过渡进度 (0.0-1.0)
        self.previous_region = None  # 之前的区域ID
        self.target_region = None  # 目标区域ID
        
        # 高级乐器混合配置
        self.instrument_transition_config = {
            # 从欢快到恐怖的过渡
            "joy_to_fear": {
                "fade_out_first": ["flute", "piano"],  # 先淡出这些乐器
                "fade_in_last": ["synth", "strings"],   # 最后淡入这些乐器
                "transition_order": ["percussion", "ambient", "harmony", "melody"]  # 音轨过渡顺序
            },
            # 从恐怖到欢快的过渡
            "fear_to_joy": {
                "fade_out_first": ["synth"],
                "fade_in_last": ["flute", "piano"],
                "transition_order": ["ambient", "harmony", "melody", "percussion"]
            },
            # 从平静到紧张的过渡
            "trust_to_anger": {
                "fade_out_first": ["piano", "guitar"],
                "fade_in_last": ["synth", "percussion"],
                "transition_order": ["ambient", "harmony", "melody", "percussion"]
            }
        }
        
        # 预定义情感区域 (可以从地图数据自动生成)
        self.emotion_regions = {}
        
        # 自定义单个乐器的音量控制
        self.instrument_volumes = {}
        
        # 音乐历史分析和预测
        self.music_history = []
        self.predicted_emotions = []
        
        if self.logger:
            self.logger.log("情感音乐过渡系统已初始化")
    
    def define_emotion_region(self, region_id, emotion_type, bounding_box, 
                            intensity=0.8, custom_instruments=None):
        """定义一个情感区域
        
        Args:
            region_id: 区域唯一标识
            emotion_type: 情感类型 (joy, fear, sadness 等)
            bounding_box: 区域边界 (x_min, x_max, y_min, y_max)
            intensity: 情感强度 (0.0-1.0)
            custom_instruments: 可选的自定义乐器列表
            
        Returns:
            bool: 是否成功添加区域
        """
        if region_id in self.emotion_regions:
            if self.logger:
                self.logger.log(f"情感区域 {region_id} 已存在，将被更新")
        
        # 获取情感的效价和唤醒度
        valence, arousal = self._get_valence_arousal_for_emotion(emotion_type)
        
        # 创建区域数据
        self.emotion_regions[region_id] = {
            "emotion": emotion_type,
            "valence": valence,
            "arousal": arousal,
            "intensity": intensity,
            "bounding_box": bounding_box,
            "custom_instruments": custom_instruments
        }
        
        if self.logger:
            self.logger.log(f"已添加情感区域 {region_id}: {emotion_type}")
        
        return True
    
    def load_emotion_map(self, emotion_manager=None, story_emotion_map=None):
        """从情感管理器和故事情感地图加载情感区域数据"""
        success = self.music_generator.load_emotion_data(emotion_manager, story_emotion_map)
        
        if success:
            # 复制情感区域数据以便进行扩展处理
            self.emotion_regions = self.music_generator.emotion_regions.copy()
            
            # 增强情感区域数据
            self._enhance_emotion_regions()
            
            if self.logger:
                self.logger.log(f"已加载并增强 {len(self.emotion_regions)} 个情感区域")
        
        return success
    
    def _enhance_emotion_regions(self):
        """增强情感区域数据，添加更多音乐过渡相关信息"""
        for region_id, region_data in self.emotion_regions.items():
            emotion_type = region_data.get("emotion", "")
            
            # 添加默认连接属性 (用于自然过渡)
            if "connections" not in region_data:
                region_data["connections"] = {}
            
            # 添加过渡速率属性
            if "transition_rate" not in region_data:
                # 基于情感类型自动设置过渡速率
                if emotion_type in ["surprise", "fear", "anger"]:
                    # 突然的情感应该有更快的过渡
                    region_data["transition_rate"] = 0.1
                elif emotion_type in ["sadness", "disgust"]:
                    # 消极情感过渡缓慢
                    region_data["transition_rate"] = 0.03
                else:
                    # 默认中等过渡速率
                    region_data["transition_rate"] = 0.05
            
            # 添加情感相容性映射
            if "compatibility" not in region_data:
                region_data["compatibility"] = self._generate_emotion_compatibility(emotion_type)
    
    def _generate_emotion_compatibility(self, emotion_type):
        """生成情感相容性映射，用于更自然的音乐过渡
        
        Args:
            emotion_type: 情感类型
            
        Returns:
            dict: 与其他情感类型的相容性评分 (0.0-1.0)
        """
        # 情感相容性矩阵 - 值越高表示过渡越自然
        compatibility = {
            "joy": {"joy": 1.0, "trust": 0.8, "anticipation": 0.7, "surprise": 0.5, 
                    "sadness": 0.2, "disgust": 0.1, "anger": 0.1, "fear": 0.2},
                    
            "trust": {"joy": 0.8, "trust": 1.0, "anticipation": 0.7, "surprise": 0.4, 
                     "sadness": 0.3, "disgust": 0.2, "anger": 0.1, "fear": 0.2},
                     
            "anticipation": {"joy": 0.7, "trust": 0.7, "anticipation": 1.0, "surprise": 0.6, 
                            "sadness": 0.3, "disgust": 0.2, "anger": 0.4, "fear": 0.5},
                            
            "surprise": {"joy": 0.5, "trust": 0.4, "anticipation": 0.6, "surprise": 1.0, 
                        "sadness": 0.3, "disgust": 0.3, "anger": 0.5, "fear": 0.7},
                        
            "sadness": {"joy": 0.2, "trust": 0.3, "anticipation": 0.3, "surprise": 0.3, 
                       "sadness": 1.0, "disgust": 0.6, "anger": 0.4, "fear": 0.5},
                       
            "disgust": {"joy": 0.1, "trust": 0.2, "anticipation": 0.2, "surprise": 0.3, 
                       "sadness": 0.6, "disgust": 1.0, "anger": 0.7, "fear": 0.5},
                       
            "anger": {"joy": 0.1, "trust": 0.1, "anticipation": 0.4, "surprise": 0.5, 
                     "sadness": 0.4, "disgust": 0.7, "anger": 1.0, "fear": 0.6},
                     
            "fear": {"joy": 0.2, "trust": 0.2, "anticipation": 0.5, "surprise": 0.7, 
                    "sadness": 0.5, "disgust": 0.5, "anger": 0.6, "fear": 1.0}
        }
        
        # 返回当前情感与所有情感的相容性
        if emotion_type in compatibility:
            return compatibility[emotion_type]
        else:
            # 如果找不到确切的情感类型，返回默认相容性
            return {emo: 0.5 for emo in compatibility.keys()}
    
    def _get_valence_arousal_for_emotion(self, emotion_type):
        """获取情感类型对应的效价和唤醒度值
        
        Args:
            emotion_type: 情感类型
            
        Returns:
            tuple: (valence, arousal) 值，每个值范围为 0.0-1.0
        """
        # 情感到效价-唤醒度的映射
        emotion_va_map = {
            "joy": (0.8, 0.7),           # 高效价高唤醒
            "trust": (0.7, 0.4),         # 高效价中唤醒
            "anticipation": (0.7, 0.6),  # 高效价中唤醒
            "surprise": (0.6, 0.8),      # 中效价高唤醒
            "sadness": (0.2, 0.3),       # 低效价低唤醒
            "disgust": (0.2, 0.5),       # 低效价中唤醒
            "anger": (0.2, 0.9),         # 低效价高唤醒
            "fear": (0.3, 0.8)           # 低效价高唤醒
        }
        
        # 返回对应值，如果找不到则返回中性值
        return emotion_va_map.get(emotion_type.lower(), (0.5, 0.5))
    
    def update_player_position(self, x, y):
        """更新玩家位置并处理音乐过渡
        
        Args:
            x: X坐标
            y: Y坐标
        """
        # 保存前一个位置
        prev_x, prev_y = self.player_position
        
        # 更新位置
        self.player_position = (x, y)
        
        # 计算移动速度向量
        self.player_velocity = (x - prev_x, y - prev_y)
        
        # 保存位置历史 (最多保留10个位置)
        self.player_history.append((x, y))
        if len(self.player_history) > 10:
            self.player_history.pop(0)
        
        # 查找当前区域
        current_region = self._find_region_at_position(x, y)
        
        # 区域变化检测
        if current_region != self.target_region:
            # 如果从一个区域移到另一个区域
            if self.target_region is not None:
                self.previous_region = self.target_region
            
            self.target_region = current_region
            self.current_transition = 0.0  # 重置过渡进度
            
            if self.logger and self.target_region:
                region_data = self.emotion_regions.get(self.target_region, {})
                emotion = region_data.get("emotion", "unknown")
                self.logger.log(f"进入新区域: {self.target_region} ({emotion})")
        
        # 设置音乐生成器的位置
        self.music_generator.update_position(x, y)
        
        # 处理高级过渡逻辑
        self._handle_advanced_transition()
    
    def _handle_advanced_transition(self):
        """处理高级过渡逻辑，实现更自然的音乐变化"""
        if not self.previous_region or not self.target_region:
            return
        
        # 获取区域数据
        prev_data = self.emotion_regions.get(self.previous_region, {})
        target_data = self.emotion_regions.get(self.target_region, {})
        
        # 获取情感类型
        prev_emotion = prev_data.get("emotion", "")
        target_emotion = target_data.get("emotion", "")
        
        # 创建过渡配置键
        transition_key = f"{prev_emotion}_to_{target_emotion}"
        
        # 获取自定义过渡配置或使用默认配置
        transition_config = self.instrument_transition_config.get(transition_key, {})
        
        # 确定过渡速率 - 使用区域指定速率或默认速率
        transition_rate = target_data.get("transition_rate", self.transition_speed)
        
        # 应用情感相容性因子调整过渡速率
        compatibility = prev_data.get("compatibility", {}).get(target_emotion, 0.5)
        adjusted_rate = transition_rate * (2.0 - compatibility)  # 相容性低时过渡更慢
        
        # 更新过渡进度
        self.current_transition += adjusted_rate
        if self.current_transition > 1.0:
            self.current_transition = 1.0
        
        # TODO: 根据过渡配置和当前进度，实现乐器的智能淡入淡出效果
        # 这部分需要与DynamicMusicGenerator的音轨混合系统集成
    
    def _find_region_at_position(self, x, y):
        """查找玩家当前位置所在的情感区域
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            str 或 None: 区域ID，如果不在任何区域内则返回None
        """
        # 优先考虑移动预测的区域
        predicted_region = self._predict_next_region()
        if predicted_region:
            return predicted_region
        
        # 优先级排序：当前包含位置的区域 > 临近区域
        contained_regions = []
        nearby_regions = []
        
        for region_id, region_data in self.emotion_regions.items():
            bb = region_data.get("bounding_box", (0, 0, 0, 0))
            x_min, x_max, y_min, y_max = bb
            
            # 检查位置是否在区域内
            if x_min <= x <= x_max and y_min <= y <= y_max:
                region_size = (x_max - x_min) * (y_max - y_min)
                contained_regions.append((region_id, region_size))
            else:
                # 计算到区域边界的距离
                dx = max(0, x_min - x, x - x_max)
                dy = max(0, y_min - y, y - y_max)
                distance = (dx**2 + dy**2)**0.5
                
                # 只考虑较近的区域
                if distance < 50:  # 可调整的阈值
                    nearby_regions.append((region_id, distance))
        
        # 如果有多个包含当前位置的区域，选择最小的区域（最具体的）
        if contained_regions:
            return min(contained_regions, key=lambda x: x[1])[0]
        
        # 如果没有包含位置的区域，选择最近的区域
        if nearby_regions:
            return min(nearby_regions, key=lambda x: x[1])[0]
        
        # 如果没有区域，保持当前区域不变
        return self.target_region
    
    def _predict_next_region(self):
        """预测玩家下一个可能进入的区域，实现提前过渡
        
        Returns:
            str 或 None: 预测的区域ID
        """
        # 如果位置历史不足，无法预测
        if len(self.player_history) < 3:
            return None
        
        # 计算移动方向和速度
        vx, vy = self.player_velocity
        speed = (vx**2 + vy**2)**0.5
        
        # 如果几乎没有移动，不进行预测
        if speed < 0.5:
            return None
        
        # 预测未来位置
        future_steps = int(min(5, max(1, speed / 2)))  # 基于速度调整预测步数
        future_x = self.player_position[0] + vx * future_steps
        future_y = self.player_position[1] + vy * future_steps
        
        # 查找预测位置的区域
        for region_id, region_data in self.emotion_regions.items():
            bb = region_data.get("bounding_box", (0, 0, 0, 0))
            x_min, x_max, y_min, y_max = bb
            
            # 检查预测位置是否在区域内
            if x_min <= future_x <= x_max and y_min <= future_y <= y_max:
                # 只有当预测区域与当前区域不同时才返回
                if region_id != self.target_region:
                    return region_id
        
        return None
    
    def load_music_resources(self):
        """加载音乐资源"""
        return self.music_generator.load_music_resources()
    
    def start_playback(self):
        """开始播放音乐"""
        return self.music_generator.start_playback()
    
    def stop_playback(self):
        """停止播放音乐"""
        self.music_generator.stop_playback()
    
    def get_current_music_info(self):
        """获取当前播放的音乐信息
        
        Returns:
            dict: 音乐信息，包括情感、区域、参数等
        """
        # 获取基础信息
        playing_regions = self.music_generator.get_currently_playing_regions()
        current_params = self.music_generator.get_current_music_params()
        
        # 添加过渡信息
        result = {
            "playing_regions": playing_regions,
            "current_params": current_params,
            "transition_progress": self.current_transition,
            "current_region": self.target_region,
            "previous_region": self.previous_region,
            "player_position": self.player_position
        }
        
        # 如果有当前区域，添加其情感信息
        if self.target_region and self.target_region in self.emotion_regions:
            region_data = self.emotion_regions[self.target_region]
            result["current_emotion"] = region_data.get("emotion", "")
            result["valence"] = region_data.get("valence", 0.5)
            result["arousal"] = region_data.get("arousal", 0.5)
        
        return result
    
    def set_volume(self, track_type=None, volume=1.0):
        """设置音轨音量
        
        Args:
            track_type: 音轨类型 ('melody', 'harmony', 'percussion', 'ambient')，
                      如果为None则设置主音量
            volume: 音量大小 (0.0-1.0)
        """
        # 修改音乐生成器中对应的音轨音量
        if self.music_generator and self.music_generator.music_channels:
            if track_type and track_type in self.music_generator.music_channels:
                channel = self.music_generator.music_channels[track_type]
                channel.set_volume(volume)
            elif track_type is None:
                # 设置所有音轨的音量
                for channel in self.music_generator.music_channels.values():
                    channel.set_volume(volume)
    
    def generate_custom_emotion_music(self, emotion_text, duration=10.0, export_path=None):
        """根据文本生成自定义情感音乐
        
        Args:
            emotion_text: 情感文本描述
            duration: 音乐时长(秒)
            export_path: 导出路径(可选)
            
        Returns:
            dict: 生成的音频数据
        """
        # 分析文本情感
        emotion_params = extract_emotion_from_text(emotion_text)
        
        # 创建音乐生成器
        music_gen = MusicGenerator(
            emotion=emotion_params["emotion"],
            valence=emotion_params["valence"],
            arousal=emotion_params["arousal"],
            intensity=emotion_params["intensity"]
        )
        
        # 生成音乐
        audio_tracks = music_gen.generate(duration=duration, collab_mode="hybrid")
        
        # 如果指定了导出路径，保存音频文件
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            music_gen.save_audio(audio_tracks["combined"], export_path)
        
        return audio_tracks
    
    def create_visualization(self, figsize=(10, 8)):
        """创建情感区域和音乐过渡可视化
        
        Args:
            figsize: 图形大小
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        return self.music_generator.create_region_map_visualization(figsize)