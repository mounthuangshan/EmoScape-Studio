import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from threading import Thread
import logging
from collections import defaultdict

class MusicTransitionManager:
    """音乐过渡管理器 - 实现平滑的区域间音乐过渡"""
    
    def __init__(self, music_generator, logger=None):
        """初始化过渡管理器
        
        Args:
            music_generator: 音乐生成器对象
            logger: 日志记录器（可选）
        """
        self.music_generator = music_generator
        self.logger = logger
        
        # 过渡状态
        self.previous_region = None
        self.target_region = None
        self.transition_progress = 0.0
        self.in_transition = False
        
        # 过渡参数
        self.base_transition_speed = 0.05  # 基础过渡速度
        self.transition_speed_multiplier = 1.0  # 速度调节因子
        self.transition_buffer_distance = 20  # 过渡开始缓冲距离
        
        # 玩家移动数据
        self.player_position = (0, 0)
        self.player_velocity = (0, 0)
        self.position_history = []  # 记录最近的位置历史
        self.max_history = 10       # 历史记录最大长度
        
        # 过渡计时
        self.last_update_time = time.time()
        
        # 乐器过渡配置
        self.instrument_transition_config = self._create_instrument_transition_config()
        
        # 情感过渡配置
        self.emotion_transition_compatibility = self._create_emotion_compatibility_map()
        
        # 调试信息
        self.debug_info = {
            "current_region": None,
            "target_region": None,
            "transition_progress": 0.0,
            "predicted_region": None,
            "compatibility_factor": 1.0
        }
    
    def _create_instrument_transition_config(self):
        """创建乐器过渡配置表"""
        return {
            # 从欢快到恐怖的过渡
            "joy_to_fear": {
                "fade_out_first": ["flute", "piano"],  # 先淡出这些乐器
                "fade_in_last": ["strings", "synth"],   # 最后淡入这些乐器
                "track_order": ["percussion", "ambient", "harmony", "melody"]  # 音轨过渡顺序
            },
            # 从恐怖到欢快的过渡
            "fear_to_joy": {
                "fade_out_first": ["synth", "strings"],
                "fade_in_last": ["flute", "piano"],
                "track_order": ["ambient", "harmony", "melody", "percussion"]
            },
            # 从平静到紧张的过渡
            "trust_to_anger": {
                "fade_out_first": ["piano", "guitar"],
                "fade_in_last": ["synth", "percussion"],
                "track_order": ["ambient", "harmony", "melody", "percussion"]
            },
            # 从悲伤到惊奇的过渡
            "sadness_to_surprise": {
                "fade_out_first": ["strings"],
                "fade_in_last": ["piano", "synth"],
                "track_order": ["harmony", "melody", "percussion", "ambient"]
            }
        }
    
    def _create_emotion_compatibility_map(self):
        """创建情感相容性映射，控制过渡速度"""
        # 情感相容性矩阵 - 值越高表示过渡越自然
        return {
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
    
    def update_position(self, x, y):
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
        
        # 保存位置历史
        self.position_history.append((x, y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        # 查找当前区域
        current_region = self._find_region_at_position(x, y)
        
        # 预测可能进入的区域
        predicted_region = self._predict_next_region()
        
        # 区域变化检测
        if current_region != self.target_region:
            # 开始新的过渡
            self.previous_region = self.target_region
            self.target_region = current_region
            
            # 仅当有前一个区域时才启动过渡
            if self.previous_region is not None and self.target_region is not None:
                self.in_transition = True
                self.transition_progress = 0.0
                self._log_transition_start()
            else:
                # 首次进入区域，直接设为当前区域
                self.in_transition = False
                self.transition_progress = 1.0
        
        # 更新音乐生成器位置 - 平滑过渡处理
        self._update_music_with_transition(predicted_region)
    
    def _update_music_with_transition(self, predicted_region=None):
        """处理音乐更新，应用平滑过渡效果
        
        Args:
            predicted_region: 预测的下一个区域（可选）
        """
        # 获取当前时间
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # 如果不在过渡中，直接更新位置
        if not self.in_transition:
            self.music_generator.update_position(*self.player_position)
            return
        
        # 计算过渡速度
        transition_speed = self._calculate_transition_speed()
        
        # 更新过渡进度
        self.transition_progress += transition_speed * time_delta * 2.0  # 调整系数使过渡更平滑
        if self.transition_progress >= 1.0:
            self.in_transition = False
            self.transition_progress = 1.0
        
        # 应用插值来平滑过渡
        self._apply_transition_interpolation(predicted_region)
    
    def _calculate_transition_speed(self):
        """计算当前过渡速度，根据情感相容性和速度调整"""
        base_speed = self.base_transition_speed * self.transition_speed_multiplier
        
        # 如果没有完整的区域信息，使用默认速度
        if not self.previous_region or not self.target_region:
            return base_speed
        
        # 获取情感类型
        prev_emotion = self._get_region_emotion(self.previous_region)
        target_emotion = self._get_region_emotion(self.target_region)
        
        if not prev_emotion or not target_emotion:
            return base_speed
        
        # 获取情感相容性
        compatibility = self._get_emotion_compatibility(prev_emotion, target_emotion)
        
        # 根据相容性调整速度：相容性高 = 更快的过渡
        # 使用非线性映射，让高相容性时过渡显著加快
        speed_factor = 0.5 + (compatibility * 1.5)
        
        # 保存调试信息
        self.debug_info["compatibility_factor"] = speed_factor
        
        return base_speed * speed_factor
    
    def _get_region_emotion(self, region_id):
        """获取区域的主要情感类型"""
        if not region_id or not hasattr(self.music_generator, 'emotion_regions'):
            return None
        
        region_data = self.music_generator.emotion_regions.get(region_id)
        if region_data:
            return region_data.get("emotion", "").lower()
        return None
    
    def _get_emotion_compatibility(self, from_emotion, to_emotion):
        """获取两种情感之间的相容性系数"""
        if not from_emotion or not to_emotion:
            return 0.5  # 默认中等相容性
        
        # 查找相容性映射
        from_map = self.emotion_transition_compatibility.get(from_emotion, {})
        compatibility = from_map.get(to_emotion, 0.5)  # 默认0.5
        
        return compatibility
    
    def _apply_transition_interpolation(self, predicted_region=None):
        """应用过渡插值，实现平滑的音乐变化
        
        Args:
            predicted_region: 预测的下一个区域（可选）
        """
        if not self.previous_region or not self.target_region:
            # 没有完整的过渡信息，直接更新当前位置
            self.music_generator.update_position(*self.player_position)
            return
        
        # 获取情感类型
        prev_emotion = self._get_region_emotion(self.previous_region)
        target_emotion = self._get_region_emotion(self.target_region)
        
        # 构建过渡配置键
        transition_key = f"{prev_emotion}_to_{target_emotion}"
        transition_config = self.instrument_transition_config.get(transition_key, {})
        
        # 为音乐生成器创建自定义权重
        custom_weights = {
            self.previous_region: max(0, 1.0 - self.transition_progress),
            self.target_region: min(1.0, self.transition_progress)
        }
        
        # 如果有预测区域和低过渡进度，添加预期权重
        if predicted_region and self.transition_progress < 0.3:
            prediction_weight = 0.1  # 小权重以轻微影响音乐
            if predicted_region not in custom_weights:
                custom_weights[predicted_region] = prediction_weight
        
        # 应用自定义权重
        if hasattr(self.music_generator, '_apply_custom_region_weights'):
            # 如果音乐生成器有这个方法，使用它应用自定义权重
            self.music_generator._apply_custom_region_weights(custom_weights, transition_config)
        else:
            # 否则，使用基本位置更新作为回退
            self.music_generator.update_position(*self.player_position)
        
        # 更新调试信息
        self.debug_info["transition_progress"] = self.transition_progress
    
    def _find_region_at_position(self, x, y):
        """查找玩家当前位置所在的情感区域
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            str or None: 区域ID，如果不在任何区域内则返回None
        """
        if not hasattr(self.music_generator, 'emotion_regions'):
            return None
        
        # 当前包含位置的区域
        contained_regions = []
        
        # 检查所有区域
        for region_id, region_data in self.music_generator.emotion_regions.items():
            bb = region_data.get("bounding_box", (0, 0, 0, 0))
            x_min, x_max, y_min, y_max = bb
            
            # 检查位置是否在区域内
            if x_min <= x <= x_max and y_min <= y <= y_max:
                # 计算区域大小，较小的区域有更高优先级
                region_size = (x_max - x_min) * (y_max - y_min)
                contained_regions.append((region_id, region_size))
        
        # 如果有多个包含当前位置的区域，选择最小的区域
        if contained_regions:
            return min(contained_regions, key=lambda x: x[1])[0]
        
        # 查找最近的区域
        nearest_region = None
        min_distance = float('inf')
        
        for region_id, region_data in self.music_generator.emotion_regions.items():
            bb = region_data.get("bounding_box", (0, 0, 0, 0))
            x_min, x_max, y_min, y_max = bb
            
            # 计算到区域边界的距离
            dx = max(0, x_min - x, x - x_max)
            dy = max(0, y_min - y, y - y_max)
            distance = (dx**2 + dy**2)**0.5
            
            if distance < min_distance and distance < self.transition_buffer_distance:
                min_distance = distance
                nearest_region = region_id
        
        return nearest_region
    
    def _predict_next_region(self):
        """预测玩家下一个可能进入的区域
        
        Returns:
            str or None: 预测的区域ID
        """
        # 如果位置历史不足，无法预测
        if len(self.position_history) < 3:
            return None
        
        # 计算移动方向和速度
        vx, vy = self.player_velocity
        speed = (vx**2 + vy**2)**0.5
        
        # 如果几乎没有移动，不进行预测
        if speed < 0.5:
            return None
        
        # 预测未来位置 - 基于当前速度和方向
        future_steps = int(min(5, max(1, speed / 2)))
        future_x = self.player_position[0] + vx * future_steps
        future_y = self.player_position[1] + vy * future_steps
        
        # 寻找预测位置所在的区域
        predicted_region = self._find_region_at_position(future_x, future_y)
        
        # 仅当预测区域与当前区域不同时返回
        if predicted_region != self.target_region:
            self.debug_info["predicted_region"] = predicted_region
            return predicted_region
        
        return None
    
    def set_transition_speed(self, speed_multiplier):
        """设置过渡速度乘数
        
        Args:
            speed_multiplier: 速度乘数 (0.1-2.0)
        """
        self.transition_speed_multiplier = max(0.1, min(2.0, speed_multiplier))
    
    def get_transition_info(self):
        """获取当前过渡状态信息
        
        Returns:
            dict: 过渡状态信息
        """
        return {
            "in_transition": self.in_transition,
            "progress": self.transition_progress,
            "from_region": self.previous_region,
            "to_region": self.target_region,
            "from_emotion": self._get_region_emotion(self.previous_region),
            "to_emotion": self._get_region_emotion(self.target_region),
            "speed_multiplier": self.transition_speed_multiplier
        }
    
    def _log_transition_start(self):
        """记录过渡开始的日志信息"""
        if not self.logger:
            return
        
        prev_emotion = self._get_region_emotion(self.previous_region)
        target_emotion = self._get_region_emotion(self.target_region)
        
        self.logger.log(f"音乐过渡开始: {prev_emotion} ({self.previous_region}) → {target_emotion} ({self.target_region})")